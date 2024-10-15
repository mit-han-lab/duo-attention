import torch
from datasets import load_dataset
from dataclasses import dataclass
from typing import Sequence, Dict

import torch
import transformers
from torch.utils.data import Dataset, IterableDataset


def get_dataset(dataset_name, split="train", size=None):
    dataset = load_dataset("json", data_files=dataset_name, split=split)
    if size is not None:
        dataset = dataset.select(range(size))
    return dataset


class MultiplePasskeyRetrievalDataset(Dataset):
    PASSKEY_ALPHABET = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliett",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
    ]

    ORDINAL_NUMBERS = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
    ]

    def __init__(
        self,
        haystack_dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length=None,
        passkey_length=32,
        num_passkeys=10,
        needle="Remeber this sequence of words, it's the {ordinal_number} passkey to the vault: ",
        retrieval_question="Based on the content of the book, what is the {ordinal_number} passkey to the vault?\nPasskey: ",
        prompt1="<|im_start|> This is a very long story book: <book> ",
        prompt2=" </book>.\n\n",
        buffer_size=300,
        seperator="\n\n",
        min_depth_ratio=0.1,
        max_depth_ratio=0.9,
        context_lengths_num_intervals=20,
        depth_ratio_num_intervals=20,
        context_length_min=None,
        context_length_max=None,
        pad_to_multiple_of=16,
    ):
        super(MultiplePasskeyRetrievalDataset, self).__init__()

        self.tokenizer = tokenizer

        self.max_length = (
            max_length if max_length is not None else tokenizer.model_max_length
        )
        self.max_depth_ratio = max_depth_ratio
        self.min_depth_ratio = min_depth_ratio
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.depth_ratio_num_intervals = depth_ratio_num_intervals

        if context_length_min is None or context_length_max is None:
            self.context_length_min = self.context_length_max = self.max_length
        else:
            self.context_length_min = context_length_min
            self.context_length_max = context_length_max

        self.context_length_intervals = torch.linspace(
            self.context_length_min,
            self.context_length_max,
            context_lengths_num_intervals,
            dtype=torch.int,
        )

        self.depth_ratio_intervals = torch.linspace(
            min_depth_ratio, max_depth_ratio, depth_ratio_num_intervals
        )

        self.passkey_length = passkey_length

        self.num_passkeys = num_passkeys

        self.haystack = ""

        for sample in haystack_dataset["text"]:
            if self._get_token_nums(self.haystack) >= self.context_length_max:
                break
            self.haystack += sample

        self.haystack = self._trim(self.haystack, self.context_length_max)

        self.needle = needle
        self.needle_tokens_list = [
            self.tokenizer.encode(
                self.needle.format(ordinal_number=ordinal_number),
                add_special_tokens=False,
            )
            for ordinal_number in self.ORDINAL_NUMBERS[: self.num_passkeys]
        ]
        self.retrieval_question_tokens_list = [
            self.tokenizer.encode(
                retrieval_question.format(ordinal_number=ordinal_number),
                add_special_tokens=False,
            )
            for ordinal_number in self.ORDINAL_NUMBERS[: self.num_passkeys]
        ]

        self.haystack_tokens = self.tokenizer.encode(
            self.haystack, add_special_tokens=False
        )
        self.seperator_tokens = self.tokenizer.encode(
            seperator, add_special_tokens=False
        )
        self.prompt1_tokens = self.tokenizer.encode(prompt1, add_special_tokens=True)
        self.prompt2_tokens = self.tokenizer.encode(prompt2, add_special_tokens=False)

        passkey = self._generate_passkey()
        passkey_tokens = self.tokenizer.encode(passkey, add_special_tokens=False)
        needle_tokens = self.needle_tokens_list[0] + passkey_tokens

        other_input_len = (
            len(self.prompt1_tokens)
            + len(self.prompt2_tokens)
            + (
                len(self.seperator_tokens)
                + len(needle_tokens)
                + len(self.seperator_tokens)
                + len(self.retrieval_question_tokens_list[0])
                + len(passkey_tokens)
            )
            * self.num_passkeys
        )
        if (
            len(self.haystack_tokens) + other_input_len
            > self.context_length_max - buffer_size
        ):
            self.haystack_tokens = self.haystack_tokens[
                : self.context_length_max - buffer_size - other_input_len
            ]

    def _generate_passkey(self):
        random_seq = torch.randint(
            0, len(self.PASSKEY_ALPHABET), (self.passkey_length,)
        )
        passkey = " ".join([self.PASSKEY_ALPHABET[i] for i in random_seq])
        return passkey

    def __len__(self):
        return len(self.context_length_intervals)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        context_length = self.context_length_intervals[i]
        # randomly sample self.num_passkeys depth ratios in self.depth_ratio_intervals
        depth_ratios = (
            self.depth_ratio_intervals[
                torch.randperm(self.depth_ratio_num_intervals)[: self.num_passkeys]
            ]
            .sort()
            .values
        )
        passkey_tokens_list = [
            self.tokenizer.encode(self._generate_passkey(), add_special_tokens=False)
            for _ in range(self.num_passkeys)
        ]
        context = self._insert_needle(context_length, depth_ratios, passkey_tokens_list)
        return self._construct_input(context, passkey_tokens_list)

    def _trim(self, context, context_length):
        tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        return context

    def _get_token_nums(self, context):
        return len(self.tokenizer.encode(context))

    def _insert_needle(self, context_length, depth_ratios, passkey_tokens_list):
        haystack_tokens = self.haystack_tokens[:context_length]

        context = []
        last_insertion_point = 0

        for i, (depth_ratio, passkey_tokens) in enumerate(
            zip(depth_ratios, passkey_tokens_list)
        ):
            insertion_point = int(len(haystack_tokens) * depth_ratio)

            needle_tokens = self.needle_tokens_list[i] + passkey_tokens

            context += (
                haystack_tokens[last_insertion_point:insertion_point]
                + self.seperator_tokens
                + needle_tokens
                + self.seperator_tokens
            )
            last_insertion_point = insertion_point

        context += haystack_tokens[last_insertion_point:]

        return context

    def _construct_input(self, context_tokens, passkey_tokens_list):
        qa_tokens = []
        for i, (passkey_tokens, retrieval_question_tokens) in enumerate(
            zip(passkey_tokens_list, self.retrieval_question_tokens_list)
        ):
            qa_tokens += (
                retrieval_question_tokens + passkey_tokens + self.seperator_tokens
            )

        context_tokens = self.prompt1_tokens + context_tokens

        # pad to multiple of 16
        if len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens) % 16 != 0:
            pad_len = (
                16
                - (len(context_tokens) + len(qa_tokens) + len(self.prompt2_tokens)) % 16
            )
            context_tokens += self.haystack_tokens[-pad_len:]

        context_tokens += self.prompt2_tokens

        input_ids = torch.tensor(context_tokens + qa_tokens)

        assert input_ids.size(0) % 16 == 0

        labels = torch.tensor([-100] * len(context_tokens) + qa_tokens)
        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        ret_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        for key in instances[0].keys():
            if key not in ret_dict:
                ret_dict[key] = torch.stack([instance[key] for instance in instances])
        return ret_dict


def get_supervised_dataloader(
    dataset, tokenizer, batch_size, num_workers=4, shuffle=True, sampler=None
):
    collator = DataCollator(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=None if sampler is not None else shuffle,
        sampler=sampler,
    )
    return dataloader
