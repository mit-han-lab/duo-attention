"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""

import os
import glob
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import numpy as np
import argparse
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

from datetime import datetime, timezone
import time
import torch
from duo_attn.patch import enable_duo_attention_eval

from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(
        self,
        args,
        needle="\n\nRemember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n",
        haystack_dir="PaulGrahamEssays",
        retrieval_question="what is the best thing to do in San Francisco?\n\nAnswer: The best thing to do in San Francisco is",
        results_version=1,
        context_lengths_min=1000,
        context_lengths_max=1048000,
        context_lengths_num_intervals=40,
        context_lengths=None,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_intervals=10,
        document_depth_percents=None,
        document_depth_percent_interval_type="linear",
        model_provider="LLaMa",
        model_name="",
        model_name_suffix=None,
        num_concurrent_requests=1,
        save_results=True,
        save_contexts=True,
        final_context_length_buffer=200,
        seconds_to_sleep_between_completions=None,
        print_ongoing_status=True,
        attn_load_dir=None,
        sparsity=0.5,
        simulation_length=50,
    ):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )

        self.args = args
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []

        if "/" in model_name:
            self.model_version = model_name.split("/")[-1]
        else:
            self.model_version = model_name
        if model_name_suffix is not None:
            self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if (
                context_lengths_min is None
                or context_lengths_max is None
                or context_lengths_num_intervals is None
            ):
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied."
                )
            else:
                self.context_lengths = np.round(
                    np.linspace(
                        context_lengths_min,
                        context_lengths_max,
                        num=context_lengths_num_intervals,
                        endpoint=True,
                    )
                ).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if (
                document_depth_percent_min is None
                or document_depth_percent_max is None
                or document_depth_percent_intervals is None
            ):
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied."
                )
            else:
                if document_depth_percent_interval_type == "linear":
                    self.document_depth_percents = np.round(
                        np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            num=document_depth_percent_intervals,
                            endpoint=True,
                        )
                    ).astype(int)
                elif document_depth_percent_interval_type == "sigmoid":
                    self.document_depth_percents = [
                        self.logistic(x)
                        for x in np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            document_depth_percent_intervals,
                        )
                    ]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError(
                "document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals"
            )

        self.model_name = model_name

        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.eos_token_ids = self.generation_config.eos_token_id
        if not isinstance(self.eos_token_ids, list):
            self.eos_token_ids = [self.eos_token_ids]

        if self.enc.pad_token_id is None:
            if self.enc.eos_token_id is not None:
                self.enc.pad_token_id = self.enc.eos_token_id
            else:
                self.enc.pad_token_id = 0
        print("Loading from %s" % model_name)

        self.model_to_test = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval()

        print(f"attn_load_dir: {attn_load_dir}")

        if attn_load_dir is not None:
            print(
                f"Loading attention pattern from {attn_load_dir} with sparsity {sparsity}"
            )
            full_attention_heads, sink_size, recent_size = load_attn_pattern(
                attn_load_dir
            )
            if args.sink_size is not None:
                sink_size = args.sink_size
            if args.recent_size is not None:
                recent_size = args.recent_size
            full_attention_heads, sparsity = sparsify_attention_heads(
                full_attention_heads, None, sparsity
            )
            enable_duo_attention_eval(
                self.model_to_test,
                full_attention_heads,
                sink_size,
                recent_size,
            )

        # list all usable GPU devices using torch
        device_list = [i for i in range(torch.cuda.device_count())]
        self.model_to_test = to_device(self.model_to_test, device_list, enable_tp=True)

        self.model_to_test_description = model_name

        self.evaluation_model = None
        self.debug = "debug"
        self.simulation_length = simulation_length
        model_name = model_name.split("/")[-1]

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len:
                continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        test_format = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n\nQuestion: Based on the content of the book, {self.retrieval_question}"
        return test_format

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)

        test_start_time = time.time()

        # Simulate multiround conversation
        prompt = self.enc(prompt, return_tensors="pt")

        prompt_input_ids = prompt["input_ids"].to(self.model_to_test.device)

        simulation_start_idx = prompt_input_ids.size(1) - self.simulation_length

        question_input_ids = prompt_input_ids[:, simulation_start_idx:]
        prompt_input_ids = prompt_input_ids[:, :simulation_start_idx]

        with torch.no_grad():
            if self.args.prefilling_chunk_size is not None:
                past_key_values = None
                for i in range(
                    0, prompt_input_ids.size(1), self.args.prefilling_chunk_size
                ):
                    chunk = prompt_input_ids[:, i : i + self.args.prefilling_chunk_size]
                    output = self.model_to_test(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
            else:
                output = self.model_to_test(
                    input_ids=prompt_input_ids, past_key_values=None, use_cache=True
                )
                past_key_values = output.past_key_values

            for input_id in question_input_ids[0]:
                output = self.model_to_test(
                    input_ids=input_id.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values

            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(50):
                outputs = self.model_to_test(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in self.eos_token_ids:
                    break

        response = self.enc.decode(generated_content, skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        score = scorer.score(self.needle, response)["rouge1"].fmeasure * 10

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            "model": self.model_to_test_description,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "version": self.results_version,
            "needle": self.needle,
            "model_response": response,
            "score": score,
            "test_duration_seconds": test_elapsed_time,
            "test_timestamp_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S%z"
            ),
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}%")
            print(f"Score: {score}")
            print(f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results["file_name"] = context_file_location

            # Save the context to file for retesting
            if not os.path.exists("contexts"):
                os.makedirs("contexts")

            if not os.path.exists(f"contexts/{self.model_version}"):
                os.makedirs(f"contexts/{self.model_version}")

            with open(
                f"contexts/{self.model_version}/{context_file_location}_context.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(context)

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists("results"):
                os.makedirs("results")

            if not os.path.exists(f"results/{self.model_version}"):
                os.makedirs(f"results/{self.model_version}")

            # Save the result to file for retesting
            p = f"results/{self.model_version}/{context_file_location}_results.json"
            print("Writing at %s" % p)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = "results/" + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    context_length_met = result["context_length"] == context_length
                    depth_percent_met = result["depth_percent"] == depth_percent
                    version_met = result.get("version", 1) == self.results_version
                    model_met = result["model"] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if (
                        context_length_met
                        and depth_percent_met
                        and version_met
                        and model_met
                    ):
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text, add_special_tokens=False)

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            tokens_new_context = tokens_context[:insertion_point]

            print(f"Insertion at {insertion_point} / {len(tokens_context)}")
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, "r") as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.enc.encode(context)

    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length], skip_special_tokens=True)

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        # asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--s_len", metavar="N", type=int, help="a number")
    parser.add_argument("-e", "--e_len", metavar="N", type=int, help="a number")
    parser.add_argument("--model_path", type=str, default=None, help="path to model")
    parser.add_argument("--model_name", type=str, default=None, help="name of model")
    parser.add_argument(
        "--model_name_suffix", type=str, default=None, help="name of model"
    )
    parser.add_argument(
        "--model_provider", type=str, default="LLaMA", help="which model to use"
    )
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="attention pattern directory"
    )
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument("--simulation_length", type=int, default=50)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=40)
    parser.add_argument("--document_depth_percent_intervals", type=int, default=10)
    parser.add_argument("--context_lengths_min", type=int, default=1000)
    parser.add_argument("--context_lengths_max", type=int, default=1048000)
    parser.add_argument("--document_depth_percent_min", type=int, default=0)
    parser.add_argument("--document_depth_percent_max", type=int, default=100)

    parser.add_argument("--prefilling_chunk_size", type=int, default=None)

    parser.add_argument("--sparsity", type=float, default=0.5)

    parser.add_argument(
        "--method",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.model_path is not None:
        assert args.model_name is None
        model_name = args.model_path
    else:
        assert args.model_name is not None
        model_name = args.model_name

    ht = LLMNeedleHaystackTester(
        args=args,
        model_name=model_name,
        model_name_suffix=args.model_name_suffix,
        model_provider=args.model_provider,
        save_contexts=True,
        save_results=True,
        attn_load_dir=args.attn_load_dir,
        sparsity=args.sparsity,
        simulation_length=args.simulation_length,
        context_lengths_min=args.context_lengths_min,
        context_lengths_max=args.context_lengths_max,
        context_lengths_num_intervals=args.context_lengths_num_intervals,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        document_depth_percent_min=args.document_depth_percent_min,
        document_depth_percent_max=args.document_depth_percent_max,
    )

    ht.start_test(args)
