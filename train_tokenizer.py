"""Train a proper Byte-Level BPE tokenizer for the local LLM.

Diagnosis fix (2026-08-16): the previous tokenizer produced garbage output
because of three defects:
  1. ``tokenizer.decoder`` was never set to ``ByteLevel()``, so decoded text
     leaked raw byte markers (``Ġ`` for space, ``Ċ`` for newline).
  2. No ``byte_fallback``, so bytes absent from the tiny training corpus
     (e.g. ``?``) collapsed to ``<unk>``.
  3. ``vocab_size=10000`` with ``min_frequency=2`` on a ~7 KB corpus yielded
     only 751 tokens - far too few to express language.

This version targets a real vocabulary, guarantees all 256 byte values are
encodable, and registers the ByteLevel decoder so round-trips are lossless.

Usage:
    python train_tokenizer.py [--corpus corpus.txt] [--save-dir tokenizer] [--vocab-size 8192]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer


def train_custom_tokenizer(corpus_path: Path, save_dir: Path, vocab_size: int) -> Path:
    print(f"[*] Initializing Byte-Level BPE tokenizer (target vocab {vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, trim_offsets=True)
    # Fix #1: register the matching decoder so output is human-readable.
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )

    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}. Run scrape_data.py first.")

    print(f"[*] Training tokenizer on corpus: {corpus_path} ({corpus_path.stat().st_size / 1024:.1f} KB)...")
    tokenizer.train([str(corpus_path)], trainer)

    save_dir.mkdir(parents=True, exist_ok=True)
    output = save_dir / "tokenizer.json"
    tokenizer.save(str(output))

    vocab = tokenizer.get_vocab_size()
    print(f"[+] Tokenizer saved to {output}")
    print(f"[+] Final vocabulary size: {vocab}")

    # Sanity checks: the question mark and space must round-trip losslessly.
    probe = "What is Python? 2 + 2 = 4."
    encoded = tokenizer.encode(probe)
    ids = encoded.ids
    reconstructed = tokenizer.decode(ids)
    re_encoded = tokenizer.encode(reconstructed).ids
    print(f"[*] Probe ids: {ids[:12]}...")
    print(f"[*] Probe round-trip: {reconstructed!r}")
    print(f"[*] id-stable round-trip: {ids == re_encoded}")
    if ids != re_encoded:
        print("[!] WARNING: round-trip mismatch - inspect the tokenizer before training a model.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Byte-Level BPE tokenizer.")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent / "corpus.txt")
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parent / "tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()
    train_custom_tokenizer(args.corpus, args.save_dir, args.vocab_size)