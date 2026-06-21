"""Prompt formatting helpers for Alpaca-style instruction tuning."""

from __future__ import annotations

ALPACA_PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_PROMPT_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def build_prompt(instruction: str, input_text: str | None = None) -> str:
    """Create the user-visible prompt used during training and inference."""
    cleaned_input = (input_text or "").strip()
    if cleaned_input:
        return ALPACA_PROMPT_WITH_INPUT.format(
            instruction=instruction.strip(),
            input=cleaned_input,
        )
    return ALPACA_PROMPT_WITHOUT_INPUT.format(instruction=instruction.strip())


def build_training_text(instruction: str, response: str, input_text: str | None = None, eos_token: str = "") -> str:
    """Create a complete causal-LM training string including the target response."""
    return f"{build_prompt(instruction, input_text)}{response.strip()}{eos_token}"
