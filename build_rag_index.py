"""Build the optional semantic FAISS index for ``corpus.txt``.

Install ``requirements-rag.txt`` first.  The chat server still offers a
lexical fallback if this script has not been run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval import CorpusRetriever, chunk_corpus


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local semantic RAG index.")
    parser.add_argument("--corpus", type=Path, default=PROJECT_DIR / "corpus.txt")
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "rag_index")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies first: pip install -r requirements-rag.txt") from exc

    if not args.corpus.exists():
        raise SystemExit(f"Corpus not found: {args.corpus}")

    chunks = chunk_corpus(args.corpus.read_text(encoding="utf-8", errors="replace"))
    if not chunks:
        raise SystemExit("Corpus has no indexable text.")

    print(f"[*] Loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model)
    print(f"[*] Embedding {len(chunks)} chunks...")
    vectors = np.asarray(embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=True), dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    args.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_dir / "index.faiss"))
    (args.index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (args.index_dir / "metadata.json").write_text(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "chunk_count": len(chunks),
                "embedding_model": args.embedding_model,
                "normalized_embeddings": True,
                "similarity": "inner_product",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] Wrote semantic index to {args.index_dir}")


if __name__ == "__main__":
    main()
