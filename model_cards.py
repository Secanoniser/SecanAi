"""Model card writer for SecanAi checkpoints.

Roadmap item P4: maintain a MODEL_CARD.md for released checkpoints (data
sources, date, parameter count, eval result, intended use, limitations).
Runs offline, alongside training - it never touches /api/chat.

Usage:
    python model_cards.py --source smollm2-baseline
    python model_cards.py --source local-checkpoint --eval-json eval_results/eval_20260816_120000_local-checkpoint.json
    python model_cards.py --list
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_router import MODEL_SOURCES, LOCAL_SOURCE_NAMES, load_model, resolve_model_source

PROJECT_DIR = Path(__file__).resolve().parent
CARDS_DIR = PROJECT_DIR / "model_cards"


def _read_eval_summary(eval_json: Path | None) -> dict[str, Any] | None:
    if eval_json is None or not eval_json.exists():
        return None
    try:
        data = json.loads(eval_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = data.get("summary")
    if not summary:
        return None
    return {
        "timestamp": data.get("timestamp"),
        "source": data.get("source"),
        "hit_rate": summary.get("hit_rate"),
        "hits": summary.get("hits"),
        "refused": summary.get("refused"),
        "total": summary.get("total"),
        "by_category": summary.get("by_category", {}),
    }


def write_model_card(source: str, eval_summary: dict[str, Any] | None = None) -> Path:
    tokenizer, model = load_model(source)
    model_id = MODEL_SOURCES[source]
    config = getattr(model, "config", None)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    config_dict: dict[str, Any] = {}
    if config is not None:
        try:
            config_dict = config.to_dict()
        except Exception:
            config_dict = {}

    card_name = "smollm2-baseline" if source not in LOCAL_SOURCE_NAMES else "local-checkpoint"
    safe_name = re_sub_safe(card_name)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    path = CARDS_DIR / f"MODEL_CARD_{safe_name}.md"

    lines = [
        f"# Model Card: {card_name}",
        "",
        f"- **Checkpoint source:** `{source}`",
        f"- **Model identifier:** `{model_id}`",
        f"- **Card generated:** {datetime.now(UTC).isoformat()}",
        "",
        "## Overview",
        "",
        "A small locally runnable causal language model served by the SecanAi chat stack "
        "(FastAPI + Transformers). It is intended for educational use, local prototyping, "
        "and experiments with retrieval-augmented chat on a single CPU/GPU machine.",
        "",
        "## Architecture",
        "",
        f"- Parameters: **{parameter_count / 1e6:.1f}M**",
        f"- Model type: `{config_dict.get('model_type', 'unknown')}`",
        f"- Hidden size: `{config_dict.get('hidden_size', 'n/a')}`",
        f"- Layers: `{config_dict.get('num_hidden_layers', 'n/a')}`",
        f"- Attention heads: `{config_dict.get('num_attention_heads', 'n/a')}`",
        f"- Context window: `{config_dict.get('max_position_embeddings', 'n/a')}` tokens",
        f"- Chat template: `{'yes' if getattr(tokenizer, 'chat_template', None) else 'no (fallback format)'}`",
        "",
        "## Training data",
        "",
        "- `smollm2-baseline`: `HuggingFaceTB/SmolLM2-135M-Instruct` (Hugging Face), served as-is.",
        "- `local-checkpoint`: locally trained/fine-tuned weights in `output_model/` built from "
        "`scrape_data.py` (Wikipedia article introductions) plus `json info/sft_thinking_dataset.jsonl`. "
        "See `train.py` / `sft_train.py` for the exact recipe used for this checkpoint.",
        "",
        "## Evaluation",
        "",
    ]

    if eval_summary:
        by_category = eval_summary.get("by_category", {})
        lines.append(f"- Timestamp: {eval_summary.get('timestamp', 'n/a')}")
        lines.append(f"- Keyword hit rate: **{eval_summary['hit_rate']:.1%}** ({eval_summary['hits']}/{eval_summary['total']})")
        lines.append(f"- Refusals: {eval_summary['refused']}/{eval_summary['total']}")
        lines.append("")
        lines.append("| Category | Hits | Total |")
        lines.append("| --- | --- | --- |")
        for category, counts in by_category.items():
            lines.append(f"| {category} | {counts.get('hits', 0)} | {counts.get('total', 0)} |")
        lines.append("")
        lines.append("Run `python eval_harness.py --source <source>` to refresh these numbers.")
    else:
        lines.append("No eval report attached. Run `python eval_harness.py --source <source>` first.")
    lines.append("")

    lines.extend(
        [
            "## Intended use",
            "",
            "- Local, offline chat experiments (see `server.py` + `index.html`).",
            "- Studying how retrieval (RAG) changes answer quality for small models.",
            "- Benchmarking SFT vs. baseline behavior with `eval_harness.py`.",
            "",
            "## Limitations",
            "",
            "- Very small parameter count: limited factual recall, weak long-form reasoning.",
            "- CPU-only serving in this repo; generation is slow for long replies.",
            "- The safety filter is a transparent first-pass blocklist, not a classifier.",
            "- The local checkpoint was trained on a small curated corpus; it should not be "
            "treated as a general-knowledge assistant.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {path}")
    return path


def re_sub_safe(name: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def list_cards() -> None:
    if not CARDS_DIR.exists():
        print("No model cards yet.")
        return
    for path in sorted(CARDS_DIR.glob("MODEL_CARD_*.md")):
        print(f"- {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a model card for a SecanAi checkpoint.")
    parser.add_argument("--source", choices=sorted(MODEL_SOURCES), default=None)
    parser.add_argument("--eval-json", type=Path, default=None, help="Optional eval report JSON from eval_harness.py.")
    parser.add_argument("--list", action="store_true", help="List existing model cards.")
    args = parser.parse_args()

    if args.list:
        list_cards()
        return

    source = args.source
    if source is None:
        source, _ = resolve_model_source()
    eval_summary = _read_eval_summary(args.eval_json)
    write_model_card(source, eval_summary)


if __name__ == "__main__":
    main()