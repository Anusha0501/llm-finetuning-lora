# Fine-Tuning TinyLlama with LoRA on Alpaca

This repository is a compact, runnable learning project for understanding how companies customize foundation models without retraining every parameter. It fine-tunes [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on an Alpaca-style instruction dataset using Hugging Face Transformers, Datasets, PEFT, and LoRA.

## What you will learn

- **Fine-tuning:** adapting a pretrained foundation model to follow a target domain, tone, or task format.
- **Full fine-tuning:** updating every model parameter, which is flexible but expensive in memory, compute, and storage.
- **LoRA:** freezing the base model and training small low-rank adapter matrices inside selected linear layers.
- **QLoRA:** combining quantized base-model loading with LoRA adapters to reduce GPU memory requirements.
- **Quantization:** representing model weights with fewer bits, such as 8-bit or 4-bit, to reduce memory usage.
- **Tokenization:** converting text prompts and answers into token IDs the model can process.
- **Training loop:** batching tokenized examples, computing language-model loss, backpropagating through adapter weights, evaluating, and saving the adapter.
- **Evaluation:** measuring validation loss/perplexity and manually checking generated answers.

## Project layout

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── tinyllama_lora.yaml
└── src/
    ├── data.py
    ├── evaluate.py
    ├── formatting.py
    ├── train_lora.py
    └── utils.py
```

## Setup

> A CUDA GPU is strongly recommended. CPU execution is useful for reading the code and smoke tests, but training TinyLlama on CPU will be very slow.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: authenticate with Hugging Face if your environment requires it.

```bash
huggingface-cli login
```

## Train a LoRA adapter

The default configuration uses the Alpaca-cleaned dataset and TinyLlama chat model:

```bash
python -m src.train_lora --config configs/tinyllama_lora.yaml
```

For a quick local smoke run, override the sample count and training steps:

```bash
python -m src.train_lora \
  --config configs/tinyllama_lora.yaml \
  --max-train-samples 64 \
  --max-eval-samples 16 \
  --max-steps 3
```

The trainer saves LoRA adapter weights, tokenizer files, metrics, and trainer state under `outputs/tinyllama-alpaca-lora/` by default.

## Evaluate and generate responses

After training:

```bash
python -m src.evaluate \
  --adapter-path outputs/tinyllama-alpaca-lora \
  --instruction "Explain LoRA to a product manager." \
  --input "Keep it concise."
```

## How companies use this pattern

1. **Start with a foundation model** that already knows language, reasoning patterns, and broad world knowledge.
2. **Collect task-specific data** such as support tickets, policy documents, product documentation, labeled examples, or expert demonstrations.
3. **Format examples consistently** so the model sees the same instruction/response contract it will receive in production.
4. **Fine-tune adapters** with LoRA or QLoRA to cheaply specialize behavior while preserving the base model.
5. **Evaluate offline and with humans** using validation loss, task metrics, safety checks, and red-team prompts.
6. **Deploy base model + adapter** so teams can swap adapters by customer, domain, language, or task.

## LoRA vs. full fine-tuning vs. QLoRA

| Method | What trains? | Pros | Tradeoffs |
| --- | --- | --- | --- |
| Full fine-tuning | All model weights | Maximum flexibility | Expensive, large checkpoints, higher overfitting risk |
| LoRA | Small adapter matrices | Cheap, modular, fast iteration | Slightly less expressive than full fine-tuning |
| QLoRA | LoRA adapters with quantized base weights | Much lower memory usage | Requires compatible hardware/software and careful dtype settings |

## Notes

- The code automatically masks prompt tokens with `-100` labels so loss is computed only on the assistant response.
- Set `quantization.use_4bit: true` in the YAML config to try QLoRA-style loading with bitsandbytes.
- Always review dataset licenses, base model licenses, and privacy requirements before training on company data.
