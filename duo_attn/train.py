import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import wandb
import matplotlib.pyplot as plt
from duo_attn.utils import (
    get_model,
    parse_args,
    get_tokenizer,
    visualize_pruned_attention_heads,
    full_attention_heads_to_list,
    save_full_attention_heads,
    seed_everything,
)
from duo_attn.data import (
    get_dataset,
    MultiplePasskeyRetrievalDataset,
    get_supervised_dataloader,
)
from duo_attn.patch import (
    enable_duo_attention_training,
    get_full_attention_heads,
    set_full_attention_heads,
    map_full_attention_heads,
    load_full_attention_heads,
)

from duo_attn.loss import l1_loss


import torch.distributed as dist

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
import types

from transformers import AutoModelForCausalLM, AutoConfig

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def apply_fsdp(model: torch.nn.Module, mesh, mp_policy, modules_to_shard):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}

    for module in model.modules():
        if any([isinstance(module, m) for m in modules_to_shard]):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)


def train(
    args, model, rank, world_size, train_dataloader, optimizer, scheduler, resume_step
):
    model.train()

    if rank == 0:
        pbar = tqdm(range(args.num_steps))

    local_rank = int(os.environ["LOCAL_RANK"])

    global_step = 0
    local_step = 0

    while True:
        if global_step >= args.num_steps:
            break
        for step, batch in enumerate(train_dataloader):
            if global_step <= resume_step:
                global_step += 1
                if rank == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"Skipping step {global_step} to resume to {resume_step}"
                    )
                continue

            @torch.no_grad()
            def clamp_(x, min_val, max_val):
                x.clamp_(min_val, max_val)

            map_full_attention_heads(model, func=lambda x: clamp_(x, 0, 1))

            batch = {k: v.to(f"cuda:{local_rank}") for k, v in batch.items()}

            # duplicate for the two way forward
            input_ids = torch.cat([batch["input_ids"], batch["input_ids"]], dim=0)

            seq_len = input_ids.shape[1]
            seq_parallel_chunk_size = seq_len // world_size
            seq_parallel_chunk_start = seq_parallel_chunk_size * rank
            seq_parallel_chunk_end = seq_parallel_chunk_start + seq_parallel_chunk_size
            position_ids = torch.arange(
                seq_parallel_chunk_start,
                seq_parallel_chunk_end,
                device=input_ids.device,
            ).unsqueeze(0)

            outputs = model(
                input_ids=input_ids[:, seq_parallel_chunk_start:seq_parallel_chunk_end],
                position_ids=position_ids,
            )

            hidden_states = outputs[0]

            original_hidden_states = hidden_states[: args.batch_size]
            pruned_hidden_states = hidden_states[args.batch_size :]

            labels = batch["labels"][:, seq_parallel_chunk_start:seq_parallel_chunk_end]
            label_mask = labels != -100
            num_labels = label_mask.sum()
            global_num_labels = num_labels.clone().detach()
            dist.all_reduce(global_num_labels)

            # filter out label == IGNORE_INDEX (-100)
            original_hidden_states = original_hidden_states[label_mask].float()
            pruned_hidden_states = pruned_hidden_states[label_mask].float()

            distill_loss = (
                (original_hidden_states - pruned_hidden_states)
                .pow(2)
                .mean(dim=-1)
                .sum()
                * world_size
                / global_num_labels
            )

            full_attention_heads = get_full_attention_heads(model)
            full_attention_heads = [
                h.full_tensor().to(original_hidden_states.device)
                for h in full_attention_heads
            ]

            reg_loss = l1_loss(torch.cat(full_attention_heads).float())

            loss = distill_loss + args.reg_weight * reg_loss

            loss.backward()

            local_step = (local_step + 1) % args.gradient_accumulation_steps

            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(distill_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(reg_loss, op=dist.ReduceOp.AVG)

            if local_step != 0:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if rank == 0:
                full_attention_heads_list = full_attention_heads_to_list(
                    full_attention_heads
                )

                if not args.disable_wandb:
                    fig = visualize_pruned_attention_heads(full_attention_heads_list)

                    sample_len = batch["input_ids"].shape[1]
                    wandb.log(
                        {
                            "distill_loss": distill_loss.item(),
                            "reg_loss": reg_loss.item(),
                            "attn_heads": fig,
                            "step": global_step,
                            "sample_len": sample_len,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                    plt.close(fig)

                pbar.set_description(
                    f"Len={seq_len}/{global_num_labels}|Dloss={distill_loss.item():.3f}|Rloss={reg_loss.item():.3f}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

            if args.output_dir is not None and global_step % args.save_steps == 0:
                if rank == 0:
                    save_full_attention_heads(
                        full_attention_heads_list,
                        os.path.join(
                            args.output_dir,
                            f"full_attention_heads_step={global_step}.tsv",
                        ),
                    )
                    os.system(f"rm {args.output_dir}/full_attention_heads_latest.tsv")
                    os.system(
                        f"cp {args.output_dir}/full_attention_heads_step={global_step}.tsv {args.output_dir}/full_attention_heads_latest.tsv"
                    )

                # save scheduler and optimizer state
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(
                        args.output_dir,
                        f"optimizer_scheduler_state-step={global_step}-rank={rank}.pt",
                    ),
                )

                # copy the full_attention_heads and optimizer_scheduler_state to the latest state, replacing the old one
                # remove the previous latest state
                os.system(
                    f"rm {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )
                os.system(
                    f"cp {args.output_dir}/optimizer_scheduler_state-step={global_step}-rank={rank}.pt {args.output_dir}/optimizer_scheduler_state_latest-rank={rank}.pt"
                )

            if global_step >= args.num_steps:
                break

    if rank == 0:
        pbar.close()


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.model_name)

    if args.config_name is not None:
        config = AutoConfig.from_pretrained(args.config_name)
    else:
        config = AutoConfig.from_pretrained(args.model_name)

    if args.rope_theta is not None:
        print(f"Setting rope_theta from {config.rope_theta} to {args.rope_theta}")
        config.rope_theta = args.rope_theta

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    enable_duo_attention_training(
        model,
        args.sink_size,
        args.recent_size,
        args.max_length,
        initial_value=args.initial_value,
        enable_ulysses_attention=True,
        streaming_attn_implementation=args.streaming_attn_implementation,
    )

    model = model.model

    for param in model.parameters():
        param.requires_grad = False

    num_attn_heads = 0
    for name, param in model.named_parameters():
        if "full_attention_heads" in name:
            param.requires_grad = True
            num_attn_heads += param.numel()

    setup()

    torch.cuda.set_device(local_rank)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    apply_activation_checkpointing(model)

    # mesh = None
    mesh = DeviceMesh(device_type="cuda", mesh=[i for i in range(world_size)])

    apply_fsdp(
        model,
        mesh,
        mp_policy,
        modules_to_shard={LlamaDecoderLayer, MistralDecoderLayer},
    )

    if rank == 0:
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(
                    f"Trainable parameter: {name} with shape {param.shape}, dtype {param.dtype}, device {param.device}"
                )

    haystack_dataset = get_dataset(args.dataset_name, split="train")

    if args.dataset_format == "multiple_passkey":
        train_dataset = MultiplePasskeyRetrievalDataset(
            haystack_dataset,
            tokenizer,
            max_length=args.max_length,
            min_depth_ratio=args.min_needle_depth_ratio,
            max_depth_ratio=args.max_needle_depth_ratio,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            depth_ratio_num_intervals=args.depth_ratio_num_intervals,
            num_passkeys=args.num_passkeys,
        )
    else:
        raise ValueError(f"Invalid dataset format: {args.dataset_format}")

    train_dataloader = get_supervised_dataloader(
        train_dataset, tokenizer, args.batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1,
            max((step + 1) / (args.num_steps // 5), 0.1),
            max((args.num_steps - step) / (args.num_steps // 5), 0.1),
        ),
    )
    if rank == 0:
        experiment_config = vars(args)
        if not args.disable_wandb:
            wandb.init(project="DuoAttention", config=experiment_config)
            if args.exp_name is not None:
                wandb.run.name = args.exp_name

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "config.json"), "w") as f:
                json.dump(experiment_config, f)

    # if resume and link exists, load the latest state
    if args.resume and os.path.exists(
        os.path.join(
            args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
        )
    ):
        # load the latest state in the output_dir
        state = torch.load(
            os.path.join(
                args.output_dir, f"optimizer_scheduler_state_latest-rank={rank}.pt"
            )
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        full_attention_heads = load_full_attention_heads(
            args.output_dir, filename="full_attention_heads_latest.tsv"
        )
        set_full_attention_heads(model, full_attention_heads)
        resume_step = state["global_step"]
        print(f"Resuming from step {resume_step}")
    else:
        resume_step = -1

    train(
        args,
        model,
        rank,
        world_size,
        train_dataloader,
        optimizer,
        scheduler,
        resume_step,
    )

    full_attention_heads = get_full_attention_heads(model)
    full_attention_heads = [h.full_tensor() for h in full_attention_heads]

    if rank == 0:
        print("Training finished")
        if args.output_dir is not None:
            full_attention_heads_list = full_attention_heads_to_list(
                full_attention_heads
            )
            # save the full attention heads as tsv
            save_full_attention_heads(
                full_attention_heads_list,
                os.path.join(args.output_dir, "full_attention_heads.tsv"),
            )

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)
