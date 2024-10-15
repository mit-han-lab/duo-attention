#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>
#include <ATen/ATen.h>

__global__ void dequantize_int4_with_zero_point_per_group_kernel(
    const uint8_t* __restrict__ q_packed, 
    const at::Half* __restrict__ scale,     
    const at::Half* __restrict__ zero_point,                      
    int group_size,                       
    at::Half* buffer                        
)
{
    size_t group_idx = blockIdx.x;

    int group_size_packed = group_size / 2;
    
    int thread_idx = threadIdx.x;
    int blockDim_x = blockDim.x;
    
    const uint8_t* q_group = q_packed + group_idx * group_size_packed;
    at::Half* out_group = buffer + group_idx * group_size;
    
    at::Half s = scale[group_idx];
    at::Half z = zero_point[group_idx];
    
    for(int i = thread_idx; i < group_size_packed; i += blockDim_x){
        uint8_t byte = q_group[i];

        uint8_t val0 = (byte >> 4) & 0x0F;
        uint8_t val1 = byte & 0x0F;

        at::Half f0 = __float2half((float)val0);
        at::Half f1 = __float2half((float)val1);

        out_group[i << 1] = __hadd(__hmul(f0, s), z);
        out_group[i << 1 | 1] = __hadd(__hmul(f1, s), z);
    }
}

void dequantize_int4_with_zero_point_per_group(
    torch::Tensor q_packed,       
    torch::Tensor scale,          
    torch::Tensor zero_point,     
    int group_size,               
    torch::Tensor buffer,         
    int N
) {
    int thread_per_group = 8;
    const int max_blocks = 1048576;

    int chunks = (N + max_blocks - 1) / max_blocks;

    for (int chunk_idx = 0; chunk_idx < chunks; ++chunk_idx) {
        int start_group = chunk_idx * max_blocks;
        int end_group = min(start_group + max_blocks, N);
        
        int num_groups = end_group - start_group;

        dequantize_int4_with_zero_point_per_group_kernel<<<num_groups, thread_per_group>>>(
            q_packed.data_ptr<uint8_t>() + (size_t)start_group * (group_size / 2),
            scale.data_ptr<at::Half>() + start_group,
            zero_point.data_ptr<at::Half>() + start_group,
            group_size,
            buffer.data_ptr<at::Half>() + (size_t)start_group * group_size
        );
    }
}

__global__ void quantize_int4_with_zero_point_per_group_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scale,
    half* __restrict__ zero_point,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int group_size,
    const int64_t __restrict__ stride_batch, 
    const int64_t __restrict__ stride_seq,
    const int64_t __restrict__ stride_head
)
{
    int num_groups = head_dim / group_size;
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_groups = batch_size * seq_len * num_heads * num_groups;
    if (group_idx >= total_groups)
        return;

    int group_in_head_idx = group_idx % num_groups;
    int tmp = group_idx / num_groups;
    int head_idx = tmp % num_heads;
    tmp = tmp / num_heads;
    int seq_idx = tmp % seq_len;
    int batch_idx = tmp / seq_len;

    int64_t input_offset = batch_idx * stride_batch + seq_idx * stride_seq + head_idx * stride_head + group_in_head_idx * group_size;

    float group_min = FLT_MAX;
    float group_max = -FLT_MAX;

    for (int i = 0; i < group_size; ++i)
    {
        float x = __half2float(input[input_offset + i]);
        group_min = fminf(group_min, x);
        group_max = fmaxf(group_max, x);
    }

    float eps = 1e-8f;
    float scale_value = (group_max - group_min) / 15.0f + eps;
    float zero_point_value = group_min;

    int scale_offset = (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * num_groups) + group_in_head_idx;
    scale[scale_offset] = __float2half(scale_value);
    zero_point[scale_offset] = __float2half(zero_point_value);

    int packed_group_size = group_size / 2;
    for (int i = 0; i < packed_group_size; ++i)
    {
        float x0 = __half2float(input[input_offset + (i << 1)]);
        float qf0 = (x0 - zero_point_value) / scale_value;
        qf0 = roundf(qf0);
        qf0 = fminf(fmaxf(qf0, 0.0f), 15.0f);
        uint8_t q0 = static_cast<uint8_t>(qf0);

        float x1 = __half2float(input[input_offset + (i << 1 | 1)]);
        float qf1 = (x1 - zero_point_value) / scale_value;
        qf1 = roundf(qf1);
        qf1 = fminf(fmaxf(qf1, 0.0f), 15.0f);
        uint8_t q1 = static_cast<uint8_t>(qf1);

        uint8_t packed_value = (q0 << 4) | q1;

        int total_packed_size = num_groups * packed_group_size;
        int output_offset = (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * total_packed_size) + group_in_head_idx * packed_group_size + i;

        output[output_offset] = packed_value;
    }
}

void quantize_int4_with_zero_point_per_group(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor zero_point,
    int group_size
)
{
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int num_heads = input.size(2);
    int head_dim = input.size(3);
    int num_groups = head_dim / group_size;
    int num_total_groups = batch_size * seq_len * num_heads * num_groups;

    int threads_per_block = 256;
    int blocks = (num_total_groups + threads_per_block - 1) / threads_per_block;

    quantize_int4_with_zero_point_per_group_kernel<<<blocks, threads_per_block>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scale.data_ptr<at::Half>()), 
        reinterpret_cast<half*>(zero_point.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        group_size,
        input.strides()[0], 
        input.strides()[1],
        input.strides()[2]
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4_with_zero_point_per_group", &quantize_int4_with_zero_point_per_group, "Quantize INT4 with zero point per group (CUDA)");
    m.def("dequantize_int4_with_zero_point_per_group", &dequantize_int4_with_zero_point_per_group, "Dequantize int4 data with zero point per group (CUDA)");
}
