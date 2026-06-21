"""Train a TinyLlama LoRA adapter on Alpaca-style instructions."""

from __future__ import annotations

import argparse

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from .data import DatasetConfig, load_alpaca_dataset, tokenize_dataset
from .utils import load_base_model, load_tokenizer, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tinyllama_lora.yaml", help="Path to YAML training config.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional validation cap for quick experiments.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training_config = config["training"]
    if args.max_steps is not None:
        training_config["max_steps"] = args.max_steps

    tokenizer = load_tokenizer(config["model"]["name"], config["model"].get("trust_remote_code", False))
    dataset = load_alpaca_dataset(DatasetConfig(**config["dataset"]))
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples is not None:
        dataset["test"] = dataset["test"].select(range(min(args.max_eval_samples, len(dataset["test"]))))
    tokenized = tokenize_dataset(dataset, tokenizer, training_config["max_seq_length"])

    model = load_base_model(config["model"], config["quantization"])
    if config["quantization"].get("use_4bit", False):
        model = prepare_model_for_kbit_training(model)
    if training_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        max_steps=training_config["max_steps"],
        warmup_ratio=training_config["warmup_ratio"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        bf16=training_config["bf16"],
        fp16=training_config["fp16"],
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=training_config["report_to"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])


if __name__ == "__main__":
    main()
