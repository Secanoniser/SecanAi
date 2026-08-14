"""Compare a base model and a candidate on the held-out canonical test split."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import get_settings


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def load_examples(path: Path, limit: int) -> list[dict]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") == "test":
            examples.append(row)
            if len(examples) >= limit:
                break
    if not examples:
        raise ValueError("No test examples found. Run prepare_sft_data.py first.")
    return examples


def score(model_source: str, examples: list[dict], max_new_tokens: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_source)
    model.eval()
    results = []
    exact = 0
    for example in examples:
        prompt, expected = (message["content"] for message in example["messages"])
        actual = generate(model, tokenizer, prompt, max_new_tokens)
        match = normalize(actual) == normalize(expected)
        exact += match
        results.append({"source_id": example["source_id"], "expected": expected, "actual": actual, "exact_match": match})
    return {"model": model_source, "examples": len(results), "exact_match": exact / len(results), "results": results}


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.processed_data_dir / "sft.jsonl")
    parser.add_argument("--base-model", default=settings.base_model_id)
    parser.add_argument("--candidate", default=str(settings.model_path))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=settings.runs_dir / "evaluation.json")
    args = parser.parse_args()
    if not Path(args.candidate).exists():
        raise FileNotFoundError(f"Candidate model not found: {args.candidate}")
    examples = load_examples(args.data, args.limit)
    report = {"base": score(args.base_model, examples, args.max_new_tokens), "candidate": score(args.candidate, examples, args.max_new_tokens)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: {k: value for k, value in result.items() if k != "results"} for key, result in report.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
