"""Generate text from a trained LoRA adapter."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel

from .formatting import build_prompt
from .utils import load_base_model, load_tokenizer, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tinyllama_lora.yaml", help="Path to YAML config used for training.")
    parser.add_argument("--adapter-path", required=True, help="Directory containing the saved LoRA adapter.")
    parser.add_argument("--instruction", required=True, help="Instruction to send to the model.")
    parser.add_argument("--input", default="", help="Optional input/context for the instruction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    tokenizer = load_tokenizer(config["model"]["name"], config["model"].get("trust_remote_code", False))
    model = load_base_model(config["model"], config["quantization"])
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    prompt = build_prompt(args.instruction, args.input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config["generation"]["max_new_tokens"],
            temperature=config["generation"]["temperature"],
            top_p=config["generation"]["top_p"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
