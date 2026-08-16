"""Offline evaluation harness for SecanAi checkpoints.

Runs a fixed prompt suite against a chosen model source (identical routing to
the chat server via ``model_router``), records dated outputs, and prints a
compact comparison table. This is the roadmap item that runs *alongside*
training - it never touches ``/api/chat``.

Usage:
    python eval_harness.py --source smollm2-baseline --quick
    python eval_harness.py --source local-checkpoint
    python eval_harness.py --compare
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from model_router import MODEL_SOURCES, load_model

PROJECT_DIR = Path(__file__).resolve().parent
EVAL_DIR = PROJECT_DIR / "eval_results"

# The harness evaluates the model, not the full chat stack, so it uses a
# neutral system prompt and no retrieval. Keep it aligned with the server's
# persona wording where it matters (identity + safety boundary).
SYSTEM_PROMPT = (
    "You are SecanAi, a small locally-run assistant. Be direct, accurate, and concise. "
    "If you are uncertain or the supplied context is insufficient, say so. "
    "Do not provide instructions that would facilitate violence, wrongdoing, or harm."
)

# Fixed suite: (category, prompt, expected_keyword). Keywords are deliberately
# lenient single terms - the harness measures broad capability signals, not
# exact answers, and is designed for 24M-135M parameter models.
EVAL_SUITE: list[dict[str, str]] = [
    # --- Python (8) ---
    {"category": "python", "prompt": "What is a Python list comprehension?", "keyword": "list"},
    {"category": "python", "prompt": "Write a Python function that adds two numbers.", "keyword": "def"},
    {"category": "python", "prompt": "What is a Python dictionary and how do you use it?", "keyword": "key"},
    {"category": "python", "prompt": "How do you open and read a file in Python?", "keyword": "open"},
    {"category": "python", "prompt": "What does the 'if __name__ == \"__main__\"' line do?", "keyword": "main"},
    {"category": "python", "prompt": "Explain what a for loop does in Python.", "keyword": "loop"},
    {"category": "python", "prompt": "What is a function argument in Python?", "keyword": "argument"},
    {"category": "python", "prompt": "How do you install a package with pip?", "keyword": "pip"},
    # --- Math (8) ---
    {"category": "math", "prompt": "Solve 2x + 3 = 7 for x.", "keyword": "x"},
    {"category": "math", "prompt": "What is 12 times 8?", "keyword": "96"},
    {"category": "math", "prompt": "What is the square root of 144?", "keyword": "12"},
    {"category": "math", "prompt": "If a triangle has angles 60 and 90 degrees, what is the third angle?", "keyword": "30"},
    {"category": "math", "prompt": "What is 25 percent of 80?", "keyword": "20"},
    {"category": "math", "prompt": "What is the area of a rectangle 4 by 6?", "keyword": "24"},
    {"category": "math", "prompt": "Simplify the fraction 8/12.", "keyword": "2/3"},
    {"category": "math", "prompt": "What comes next: 2, 4, 6, 8, ...?", "keyword": "10"},
    # --- Science (6) ---
    {"category": "science", "prompt": "What is the chemical symbol for water?", "keyword": "H2O"},
    {"category": "science", "prompt": "What planet is closest to the Sun?", "keyword": "Mercury"},
    {"category": "science", "prompt": "What force pulls objects toward the Earth?", "keyword": "gravity"},
    {"category": "science", "prompt": "What gas do plants absorb from the air?", "keyword": "carbon"},
    {"category": "science", "prompt": "What is the boiling point of water in Celsius?", "keyword": "100"},
    {"category": "science", "prompt": "What is photosynthesis?", "keyword": "light"},
    # --- AI (6) ---
    {"category": "ai", "prompt": "What is a transformer in machine learning?", "keyword": "attention"},
    {"category": "ai", "prompt": "What is a neural network?", "keyword": "neuron"},
    {"category": "ai", "prompt": "What is supervised learning?", "keyword": "label"},
    {"category": "ai", "prompt": "What is a token in a language model?", "keyword": "token"},
    {"category": "ai", "prompt": "What is a large language model?", "keyword": "language"},
    {"category": "ai", "prompt": "What does fine-tuning a model mean?", "keyword": "train"},
    # --- General (6) ---
    {"category": "general", "prompt": "What is the capital of France?", "keyword": "Paris"},
    {"category": "general", "prompt": "How many days are in a leap year?", "keyword": "366"},
    {"category": "general", "prompt": "What is the largest ocean on Earth?", "keyword": "Pacific"},
    {"category": "general", "prompt": "What language is spoken in Japan?", "keyword": "Japanese"},
    {"category": "general", "prompt": "What is the currency of the United States?", "keyword": "dollar"},
    {"category": "general", "prompt": "Who wrote the play Romeo and Juliet?", "keyword": "Shakespeare"},
]

REFUSAL_MARKERS = ("can't help with that", "i can't help", "cannot assist", "i can't provide", "i'm sorry, i can't")


def _format_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    body = "\n\n".join(f"{labels[message['role']]}: {message['content']}" for message in messages)
    return f"{body}\n\nAssistant:"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_keyword(response: str, keyword: str) -> bool:
    return _normalize(keyword) in _normalize(response)


def run_eval(source: str, quick: bool = False, max_tokens: int = 96) -> dict[str, Any]:
    tokenizer, model = load_model(source)
    device = getattr(model, "device", torch.device("cpu"))
    suite = EVAL_SUITE[:5] if quick else EVAL_SUITE

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    for item in suite:
        formatted = _format_prompt(
            tokenizer,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item["prompt"]}],
        )
        tokenizer.truncation_side = "left"
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
        tokenizer.truncation_side = "right"
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        refused = any(marker in _normalize(response) for marker in REFUSAL_MARKERS)
        keyword_hit = _contains_keyword(response, item["keyword"])
        results.append(
            {
                "category": item["category"],
                "prompt": item["prompt"],
                "keyword": item["keyword"],
                "keyword_hit": keyword_hit,
                "refused": refused,
                "characters": len(response),
                "response": response,
            }
        )
        print(
            f"[{'HIT' if keyword_hit else '  -'}] ({item['category']:>8}) {item['prompt'][:60]}"
            f"  ->  {response[:70].replace(chr(10), ' ')!r}"
        )

    elapsed_seconds = round(time.perf_counter() - started, 1)
    report: dict[str, Any] = {
        "source": source,
        "timestamp": datetime.now(UTC).isoformat(),
        "suite_size": len(results),
        "quick": quick,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed_seconds,
        "summary": _summarize(results),
        "results": results,
    }
    return report


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({item["category"] for item in results})
    by_category: dict[str, dict[str, int]] = {}
    for category in categories:
        items = [item for item in results if item["category"] == category]
        by_category[category] = {
            "hits": sum(1 for item in items if item["keyword_hit"]),
            "refused": sum(1 for item in items if item["refused"]),
            "total": len(items),
        }
    return {
        "hits": sum(1 for item in results if item["keyword_hit"]),
        "refused": sum(1 for item in results if item["refused"]),
        "total": len(results),
        "hit_rate": round(sum(1 for item in results if item["keyword_hit"]) / max(len(results), 1), 3),
        "by_category": by_category,
    }


def _write_report(report: dict[str, Any]) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_DIR / f"eval_{stamp}_{report['source']}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report["summary"]
    lines = [
        f"# Evaluation Report - {report['source']}",
        "",
        f"- Timestamp: {report['timestamp']}",
        f"- Parameters: {report['parameter_count'] / 1e6:.1f}M",
        f"- Suite size: {report['suite_size']} prompts (quick={report['quick']})",
        f"- Elapsed: {report['elapsed_seconds']}s",
        "",
        "## Summary",
        "",
        f"- Keyword hit rate: **{summary['hit_rate']:.1%}** ({summary['hits']}/{summary['total']})",
        f"- Refusals: {summary['refused']}/{summary['total']}",
        "",
        "| Category | Hits | Refused | Total |",
        "| --- | --- | --- | --- |",
    ]
    for category, counts in summary["by_category"].items():
        lines.append(f"| {category} | {counts['hits']} | {counts['refused']} | {counts['total']} |")
    lines.append("")
    lines.append("## Prompt-by-prompt output")
    lines.append("")
    for item in report["results"]:
        status = "HIT" if item["keyword_hit"] else ("REFUSED" if item["refused"] else "miss")
        lines.append(f"### [{status}] ({item['category']}) {item['prompt']}")
        lines.append("")
        lines.append(f"Expected keyword: `{item['keyword']}`")
        lines.append("")
        lines.append(f"```\n{item['response']}\n```")
        lines.append("")
    md_path = json_path.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Saved eval report: {md_path}")
    return md_path


def list_reports() -> None:
    if not EVAL_DIR.exists():
        print("No eval results yet. Run an evaluation first.")
        return
    reports = sorted(EVAL_DIR.glob("eval_*.json"))
    if not reports:
        print("No eval results yet. Run an evaluation first.")
        return
    print(f"{'timestamp':<22} {'source':<22} {'hit rate':>10} {'hits':>6} {'refused':>8} {'prompts':>8}")
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data["summary"]
        stamp = re.sub(r"eval_|\.json", "", path.name)
        print(
            f"{stamp:<22} {data['source']:<22} {summary['hit_rate']:>9.1%} {summary['hits']:>6} "
            f"{summary['refused']:>8} {summary['total']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SecanAi checkpoint against the fixed prompt suite.")
    parser.add_argument("--source", choices=sorted(MODEL_SOURCES), default=None, help="Model source to evaluate.")
    parser.add_argument("--quick", action="store_true", help="Run only the first 5 prompts for a fast smoke test.")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--compare", action="store_true", help="List previously saved eval reports.")
    args = parser.parse_args()

    if args.compare:
        list_reports()
        return

    source = args.source
    if source is None:
        # Resolve the same default the server would choose.
        from model_router import resolve_model_source

        source, _ = resolve_model_source()
    report = run_eval(source, quick=args.quick, max_tokens=args.max_tokens)
    _write_report(report)


if __name__ == "__main__":
    main()