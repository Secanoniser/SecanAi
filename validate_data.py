"""Validate canonical or source instruction JSONL files without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

MOJIBAKE = re.compile(r"(?:Â|Ã|â€)")
SOURCE_REQUIRED = ("instruction", "completion")
CANONICAL_REQUIRED = ("messages", "source_id", "split")


def normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def validate_jsonl(path: Path, canonical: bool = False) -> dict[str, Any]:
    required = CANONICAL_REQUIRED if canonical else SOURCE_REQUIRED
    report: dict[str, Any] = {
        "path": str(path), "canonical": canonical, "rows": 0, "invalid_json": 0,
        "missing_required": 0, "empty_required": 0, "mojibake_rows": 0,
        "duplicate_rows": 0, "types": Counter(), "splits": Counter(), "errors": [],
    }
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                report["invalid_json"] += 1
                report["errors"].append(f"line {line_number}: {exc.msg}")
                continue
            report["rows"] += 1
            if any(field not in record for field in required):
                report["missing_required"] += 1
                continue
            if canonical:
                messages = record["messages"]
                is_valid_messages = isinstance(messages, list) and len(messages) == 2 and all(
                    isinstance(message, dict) and message.get("role") in {"user", "assistant"}
                    and isinstance(message.get("content"), str) and message["content"].strip()
                    for message in messages
                )
                values = [record.get("source_id", ""), record.get("split", "")]
                if not is_valid_messages:
                    report["missing_required"] += 1
                    continue
                values.extend(message["content"] for message in messages)
                key = (normalized(messages[0]["content"]), normalized(messages[1]["content"]))
                report["splits"][record.get("split", "")] += 1
            else:
                values = [record.get(field, "") for field in required]
                key = (normalized(str(record.get("instruction", ""))), normalized(str(record.get("completion", ""))))
                report["types"][record.get("type", "unknown")] += 1
            if any(not isinstance(value, str) or not value.strip() for value in values):
                report["empty_required"] += 1
            if any(MOJIBAKE.search(value) for value in values if isinstance(value, str)):
                report["mojibake_rows"] += 1
            if key in seen:
                report["duplicate_rows"] += 1
            seen.add(key)
    report["types"] = dict(report["types"])
    report["splits"] = dict(report["splits"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args()
    report = validate_jsonl(args.input, canonical=args.canonical)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report["invalid_json"] or report["missing_required"] or report["empty_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
