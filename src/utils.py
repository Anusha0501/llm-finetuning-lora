"""Shared configuration and model-loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    """Convert a config dtype string into a torch dtype."""
    if dtype_name is None:
        return None
    normalized = dtype_name.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    """Create a bitsandbytes 4-bit config when QLoRA mode is enabled."""
    if not config.get("use_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=resolve_torch_dtype(config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    """Load a tokenizer and ensure it has a padding token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_config: dict[str, Any], quantization_config: dict[str, Any]):
    """Load the base causal language model."""
    quantization = build_quantization_config(quantization_config)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "device_map": "auto",
    }
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(model_config["name"], **model_kwargs)
