"""Fine-tune a causal language model on validated SecanAi canonical JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from settings import get_settings


def build_features(example: dict, tokenizer, max_length: int) -> dict[str, list[int]]:
    """Tokenize one conversation and mask system/user tokens and padding in labels."""
    messages = example["messages"]
    prompt_messages = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    full_ids = full_ids[:max_length]
    labels = list(full_ids)
    for index in range(min(len(prompt_ids), len(labels))):
        labels[index] = -100
    attention_mask = [1] * len(full_ids)
    padding = max_length - len(full_ids)
    if padding > 0:
        full_ids += [tokenizer.pad_token_id] * padding
        labels += [-100] * padding
        attention_mask += [0] * padding
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.processed_data_dir / "sft.jsonl")
    parser.add_argument("--base-model", default=settings.base_model_id)
    parser.add_argument("--output", type=Path, default=settings.artifacts_dir / "models" / "sft")
    parser.add_argument("--run-dir", type=Path, default=settings.runs_dir / "sft")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str)
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Processed dataset not found: {args.data}. Run prepare_sft_data.py first.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    source_dataset = load_dataset("json", data_files=str(args.data))["train"]
    dataset = {
        "train": source_dataset.filter(lambda row: row["split"] == "train"),
        "validation": source_dataset.filter(lambda row: row["split"] == "validation"),
    }
    if not len(dataset["train"]) or not len(dataset["validation"]):
        raise ValueError("Dataset must contain non-empty train and validation splits. Run prepare_sft_data.py first.")
    tokenized = {name: split.map(lambda record: build_features(record, tokenizer, args.max_length), remove_columns=split.column_names)
                 for name, split in dataset.items()}
    training_args = TrainingArguments(
        output_dir=str(args.run_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        load_best_model_at_end=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"])
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fine-tuned model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
