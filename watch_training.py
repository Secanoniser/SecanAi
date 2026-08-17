"""Real-time training progress bar for the local LLM.

Polls the training log written by ``train.py`` (logs/train.log) and renders a
live progress bar with step, speed, elapsed, ETA, latest loss, and saved
checkpoints. Refresh happens in-place (single line), so it can run alongside
the training in any terminal.

Usage:
    python watch_training.py                  # defaults: logs/train.log
    python watch_training.py --log custom.log
    python watch_training.py --refresh 5      # slower refresh (seconds)

Ctrl+C stops the watcher; it does NOT touch the training process.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Force UTF-8 output so the block characters render on any Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

BAR_FILL, BAR_EMPTY = "█", "░"


def _bar_chars() -> tuple[str, str]:
    """Fall back to ASCII if the console cannot encode block characters."""
    try:
        ("█" * 3).encode(sys.stdout.encoding or "utf-8")
        return "█", "░"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


TQDM_LINE = re.compile(
    r"(\d+)/(\d+)\s+\[((?:\d+:)?\d+:\d+)<((?:\d+:)?\d+:\d+),\s+([\d.]+)s/it\]"
)
LOSS_LINE = re.compile(r"'loss': '([\d.]+)'.*?'epoch': '([\d.]+)'")


def parse_eta(eta: str) -> int:
    """'1:35:13' or '35:13' -> seconds."""
    parts = [int(part) for part in eta.split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def read_log(path: Path) -> str:
    """Read the log even while the training process holds it open.

    ``newline=""`` disables universal-newline translation: tqdm's carriage
    returns (\\r) must survive intact so the newest progress line can be
    picked out of the overwritten segments.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read()
    except OSError:
        return ""


def render_bar(fraction: float, width: int = 30) -> str:
    global BAR_FILL, BAR_EMPTY
    filled = int(fraction * width)
    fill, empty = BAR_FILL, BAR_EMPTY
    try:
        bar = "[" + fill * filled + empty * (width - filled) + "]"
        bar.encode(sys.stdout.encoding or "utf-8")
        return bar
    except (UnicodeEncodeError, LookupError):
        BAR_FILL, BAR_EMPTY = "#", "-"
        return "[" + "#" * filled + "-" * (width - filled) + "]"


def collect_checkpoints() -> list[str]:
    names = []
    for path in sorted(
        (PROJECT_DIR / "output_model").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    ):
        if (path / "trainer_state.json").exists():
            names.append(path.name.replace("checkpoint-", ""))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Live progress bar for train.py.")
    parser.add_argument("--log", type=Path, default=PROJECT_DIR / "logs" / "train.log")
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Render one frame and exit (for scripted checks).")
    args = parser.parse_args()

    print(f"[*] Watching {args.log}  (Ctrl+C to stop; training keeps running)\n")
    last_seen = (0, 0.0)  # (step, log mtime)
    stuck_warning = False

    def render_frame() -> bool:
        """Render one frame; return True when training is finished."""
        nonlocal last_seen, stuck_warning
        raw = read_log(args.log)
        lines = raw.split("\n")

        # Newest MAIN training bar. The log also contains secondary tqdm bars
        # (e.g. "Writing model shards: 1/1") whose totals differ - only bars
        # matching the main training total count for progress/finish logic.
        candidates = []
        for segment in raw.split("\r"):
            candidate = TQDM_LINE.search(segment)
            if candidate:
                candidates.append(candidate)
        main_total = 0
        for candidate in candidates:
            if int(candidate.group(2)) > main_total:
                main_total = int(candidate.group(2))
        match = None
        if main_total >= 100:  # main training bar (3190); ignore 1/1 shard bars
            for candidate in reversed(candidates):
                if int(candidate.group(2)) == main_total:
                    match = candidate
                    break
        elif candidates:
            match = candidates[-1]  # very early log: no main bar yet

        loss = None
        for line in lines:
            loss_match = LOSS_LINE.search(line)
            if loss_match:
                loss = (float(loss_match.group(1)), float(loss_match.group(2)))

        checkpoints = collect_checkpoints()

        if match:
            step, total = int(match.group(1)), int(match.group(2))
            elapsed = parse_eta(match.group(3))  # mm:ss or h:mm:ss
            eta_seconds = parse_eta(match.group(4))
            speed = float(match.group(5))
            fraction = step / total if total else 0.0

            loss_text = f"loss {loss[0]:.3f} (ep {loss[1]:.2f})" if loss else "loss (buffered)"
            ckpt_text = f"ckpt: {', '.join(checkpoints)}" if checkpoints else "ckpt: none yet"

            line = (
                f"\r{render_bar(fraction)} {fraction * 100:5.1f}% "
                f"| {step:,}/{total:,} | {speed:.2f}s/it "
                f"| elapsed {fmt_duration(elapsed)} | ETA {fmt_duration(eta_seconds)} "
                f"| {loss_text} | {ckpt_text}    "
            )
            sys.stdout.write(line)
            sys.stdout.flush()

            last_seen = (step, os.path.getmtime(args.log) if args.log.exists() else 0.0)

            if step >= total or "Pre-training complete" in raw:
                print("\n\n[+] Training finished!")
                return True
        else:
            # No progress line yet - show initialization status.
            age = time.time() - (args.log.stat().st_mtime if args.log.exists() else time.time())
            sys.stdout.write(
                f"\r{render_bar(0.0)} init: "
                f"tokenizing corpus / loading model (log {int(age)}s old)    "
            )
            sys.stdout.flush()

        # Stuck detection: log untouched for > 3 min while training should run.
        if args.log.exists() and time.time() - args.log.stat().st_mtime > 180 and match:
            if not stuck_warning:
                print("\n[!] No log activity for 3+ minutes (model saving pauses this; else the run may have died).")
                stuck_warning = True

        return False

    try:
        while True:
            if render_frame():
                return
            if args.once:
                print()
                return
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\n[*] Watcher stopped. Training continues in the background.")


if __name__ == "__main__":
    main()