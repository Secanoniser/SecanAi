"""Normalize, deduplicate, and deterministically split SecanAi source SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from settings import get_settings


def clean_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    # Repair the common UTF-8-decoded-as-Latin-1 pattern only when round-tripping succeeds.
    if any(marker in value for marker in ("Â", "Ã", "â€")):
        try:
            repaired = value.encode("latin-1").decode("utf-8")
            if repaired:
                value = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"[ \t]+", " ", value).strip()


def split_for(key: str, validation_percent: int, test_percent: int) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + validation_percent:
        return "validation"
    return "train"


def prepare(input_path: Path, output_path: Path, validation_percent: int = 10, test_percent: int = 10) -> dict[str, Any]:
    if validation_percent < 0 or test_percent < 0 or validation_percent + test_percent >= 100:
        raise ValueError("validation and test percentages must be non-negative and total less than 100")
    stats: dict[str, Any] = Counter(input_rows=0, kept_rows=0, invalid_rows=0, duplicate_rows=0, repaired_rows=0)
    stats["splits"] = Counter()
    seen: set[tuple[str, str]] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            stats["input_rows"] += 1
            try:
                record = json.loads(line)
                original_instruction = record["instruction"]
                original_completion = record["completion"]
                instruction = clean_text(original_instruction)
                completion = clean_text(original_completion)
            except (json.JSONDecodeError, KeyError, TypeError):
                stats["invalid_rows"] += 1
                continue
            if not instruction or not completion:
                stats["invalid_rows"] += 1
                continue
            if instruction != original_instruction or completion != original_completion:
                stats["repaired_rows"] += 1
            key = (" ".join(instruction.split()).casefold(), " ".join(completion.split()).casefold())
            if key in seen:
                stats["duplicate_rows"] += 1
                continue
            seen.add(key)
            identity = "\n".join(key)
            split = split_for(identity, validation_percent, test_percent)
            canonical = {
                "source_id": f"{input_path.stem}:{line_number}",
                "source_type": record.get("type", "unknown"),
                "split": split,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": completion},
                ],
            }
            destination.write(json.dumps(canonical, ensure_ascii=False) + "\n")
            stats["kept_rows"] += 1
            stats["splits"][split] += 1
    stats["splits"] = dict(stats["splits"])
    return dict(stats)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.repository_root / "json info" / "sft_thinking_dataset.jsonl")
    parser.add_argument("--output", type=Path, default=settings.processed_data_dir / "sft.jsonl")
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--test-percent", type=int, default=10)
    parser.add_argument("--report", type=Path, default=settings.processed_data_dir / "sft_report.json")
    args = parser.parse_args()
    report = prepare(args.input, args.output, args.validation_percent, args.test_percent)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
