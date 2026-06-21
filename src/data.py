"""Dataset loading and tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from .formatting import build_prompt, build_training_text


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for loading an Alpaca-style dataset."""

    name: str
    split: str = "train"
    validation_size: float = 0.02
    seed: int = 42


def load_alpaca_dataset(config: DatasetConfig) -> DatasetDict:
    """Load an Alpaca-style dataset and create train/eval splits."""
    dataset = load_dataset(config.name, split=config.split)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset for split {config.split!r}, got {type(dataset)!r}")
    return dataset.train_test_split(test_size=config.validation_size, seed=config.seed)


def tokenize_example(example: dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_seq_length: int) -> dict[str, list[int]]:
    """Tokenize one example and mask prompt tokens from the loss."""
    eos_token = tokenizer.eos_token or ""
    prompt = build_prompt(example["instruction"], example.get("input"))
    full_text = build_training_text(
        example["instruction"],
        example["output"],
        example.get("input"),
        eos_token=eos_token,
    )

    tokenized_prompt = tokenizer(prompt, add_special_tokens=False)
    tokenized_full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    prompt_length = min(len(tokenized_prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase, max_seq_length: int) -> DatasetDict:
    """Tokenize train/eval splits and remove raw text columns."""
    remove_columns = dataset["train"].column_names
    return dataset.map(
        lambda example: tokenize_example(example, tokenizer, max_seq_length),
        remove_columns=remove_columns,
        desc="Tokenizing Alpaca examples",
    )
