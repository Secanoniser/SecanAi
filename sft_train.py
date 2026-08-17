"""Supervised fine-tuning for the LOCAL SecanAi checkpoint (level-up step).

Rewritten (2026-08-16) to replace the SmolLM2-only hardcoded version:
  * base model is configurable - defaults to the locally pre-trained
    checkpoint (``output_model``), but SmolLM2 works too (--base),
  * trains on the 100K-row thinking dataset (``json info/
    sft_thinking_dataset.jsonl``) with proper label masking (only the
    assistant completion contributes to loss),
  * chat-template aware: if the base tokenizer has a chat template, samples
    are wrapped with ``apply_chat_template``; otherwise the dataset's native
    ``User: ... / Assistant:`` format is kept (matches the server fallback),
  * stratified subsampling keeps CPU training feasible (full 100K rows would
    take ~16 hours on CPU; 5K rows is a sensible default).

Usage:
    python sft_train.py                         # local checkpoint, 5000 samples
    python sft_train.py --base HuggingFaceTB/SmolLM2-135M --max-examples 10000
    python sft_train.py --epochs 3 --lr 2e-5 --max-length 192
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).resolve().parent
THINKING_DATASET = BASE_DIR / "json info" / "sft_thinking_dataset.jsonl"


def load_thinking_samples(path: Path, max_examples: int) -> list[dict[str, str]]:
    """Load the thinking dataset, stratified by ``type`` (math/qa/...)."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not data.get("prompt") or not data.get("completion"):
                continue
            rows.append(
                {
                    "prompt": data["prompt"].strip(),
                    "completion": data["completion"].strip(),
                    "instruction": data.get("instruction", "").strip(),
                    "thinking": data.get("thinking", "").strip(),
                    "response": data.get("response", "").strip(),
                    "type": data.get("type", "qa"),
                }
            )

    types = Counter(row["type"] for row in rows)
    print(f"[*] Dataset: {len(rows):,} rows, types: {dict(types)}")

    per_type_cap = max(1, max_examples // max(len(types), 1))
    selected: list[dict[str, str]] = []
    per_type: Counter[str] = Counter()
    for row in rows:
        if per_type[row["type"]] >= per_type_cap:
            continue
        selected.append(row)
        per_type[row["type"]] += 1
        if len(selected) >= max_examples:
            break
    print(f"[*] Selected {len(selected):,} samples for training (cap {per_type_cap}/type).")
    return selected


def build_messages(sample: dict[str, str], tokenizer) -> str:
    """Format one sample as a prompt string, respecting the chat template."""
    instruction = sample["instruction"] or sample["prompt"].split("\nAssistant:")[0].removeprefix("User: ").strip()
    completion = sample["completion"]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are SecanAi, a small locally-run assistant. Be direct, accurate, and concise."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": completion},
            ],
            tokenize=False,
        )
    return f"User: {instruction}\nAssistant: {completion}"


def tokenize_with_mask(examples: dict[str, list[str]], tokenizer, max_length: int) -> dict:
    """Tokenize and mask prompt tokens so only completions train the model."""
    inputs_list = []
    labels_list = []
    for text in examples["text"]:
        if "Assistant:" in text:
            prompt_part, completion_part = text.split("Assistant:", 1)
            prompt_part = prompt_part + "Assistant:"
        else:
            prompt_part, completion_part = text, ""

        prompt_ids = tokenizer(prompt_part, truncation=True, max_length=max_length)["input_ids"]
        full_ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]

        labels = [-100] * len(full_ids)
        if completion_part:
            completion_ids = tokenizer(completion_part, truncation=True, max_length=max_length)["input_ids"]
            labels[len(prompt_ids) : len(prompt_ids) + len(completion_ids)] = completion_ids[: max_length - len(prompt_ids)]

        inputs_list.append(full_ids)
        labels_list.append(labels)

    return {"input_ids": inputs_list, "labels": labels_list, "attention_mask": [[1] * len(ids) for ids in inputs_list]}


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT the local SecanAi checkpoint on the thinking dataset.")
    parser.add_argument("--base", default=str(BASE_DIR / "output_model"), help="Base model or checkpoint to fine-tune.")
    parser.add_argument("--dataset", type=Path, default=THINKING_DATASET)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "output_model" / "sft_local_output")
    parser.add_argument("--max-examples", type=int, default=int(os.getenv("SFT_MAX_EXAMPLES", "5000")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("SFT_EPOCHS", "3")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("SFT_LR", "2e-5")))
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Thinking dataset not found: {args.dataset}")

    print(f"[*] Loading base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32, device_map="auto")
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    samples = load_thinking_samples(args.dataset, args.max_examples)
    texts = [build_messages(sample, tokenizer) for sample in samples]
    raw = Dataset.from_dict({"text": texts})
    tokenized = raw.map(
        lambda batch: tokenize_with_mask(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
    )

    steps_per_epoch = max(len(tokenized) // args.batch_size, 1)
    estimate_minutes = round(steps_per_epoch * args.epochs * 2.3 / 60, 1)  # ~2.3s/step on CPU
    print(f"[*] Starting SFT: {len(tokenized):,} samples, {args.epochs} epochs, "
          f"~{steps_per_epoch} steps/epoch, estimated {estimate_minutes} min on CPU.")
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"[+] SFT complete. Model saved to {args.output}")
    print("[*] Evaluate with: python eval_harness.py --source local-checkpoint")
    print("[*] Serve with:   set MODEL_SOURCE=local-checkpoint && python server.py")


if __name__ == "__main__":
    main()