"""Local corpus retrieval used by the chat server.

The retriever works immediately with a lexical fallback.  Installing the
optional RAG dependencies and running ``build_rag_index.py`` upgrades it to
semantic FAISS retrieval without changing the server API.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORD_PATTERN = re.compile(r"[a-z0-9]{2,}")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_OVERLAP = 160


@dataclass(frozen=True)
class RetrievedChunk:
    """A corpus excerpt selected for a user query."""

    content: str
    source: str
    score: float


def chunk_corpus(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Split corpus text into overlapping, readable chunks.

    Paragraph boundaries are preferred so retrieved context remains useful to
    both a person reading it and a small local model.
    """
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalized) if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
        else:
            # A single large paragraph is split deterministically rather than
            # discarded.  The next chunk carries a small overlap for context.
            for start in range(0, len(paragraph), chunk_size - overlap):
                piece = paragraph[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            current = ""

    if current:
        chunks.append(current)
    return chunks


def _tokens(text: str) -> Counter[str]:
    return Counter(WORD_PATTERN.findall(text.lower()))


class CorpusRetriever:
    """Retrieve relevant chunks from a local corpus.

    A FAISS index is used only when all optional artifacts are available.  The
    deterministic lexical fallback makes the feature usable on a fresh clone
    and is intentionally exposed in ``status()`` instead of pretending it is
    semantic retrieval.
    """

    def __init__(
        self,
        corpus_path: Path,
        index_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self._chunks: list[str] | None = None
        self._faiss_index: Any | None = None
        self._embedder: Any | None = None
        self._semantic_error: str | None = None

    def _load_chunks(self) -> list[str]:
        if self._chunks is not None:
            return self._chunks

        chunks_file = self.index_dir / "chunks.json"
        if chunks_file.exists():
            try:
                loaded = json.loads(chunks_file.read_text(encoding="utf-8"))
                if isinstance(loaded, list) and all(isinstance(chunk, str) for chunk in loaded):
                    self._chunks = loaded
                    return self._chunks
            except (OSError, json.JSONDecodeError):
                # The source corpus remains the fallback if an interrupted
                # index build left invalid metadata behind.
                pass

        if not self.corpus_path.exists():
            self._chunks = []
            return self._chunks

        self._chunks = chunk_corpus(self.corpus_path.read_text(encoding="utf-8", errors="replace"))
        return self._chunks

    def _load_semantic_index(self) -> bool:
        if self._faiss_index is not None and self._embedder is not None:
            return True
        if self._semantic_error is not None:
            return False

        index_file = self.index_dir / "index.faiss"
        chunks_file = self.index_dir / "chunks.json"
        if not index_file.exists() or not chunks_file.exists():
            self._semantic_error = "Semantic index has not been built."
            return False

        try:
            import faiss  # type: ignore[import-not-found]
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._load_chunks()
            self._faiss_index = faiss.read_index(str(index_file))
            self._embedder = SentenceTransformer(self.embedding_model)
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._semantic_error = f"Semantic retrieval unavailable: {exc}"
            return False

    def retrieve(self, query: str, limit: int = 3) -> tuple[list[RetrievedChunk], str]:
        """Return up to ``limit`` chunks and the retrieval mode used."""
        chunks = self._load_chunks()
        if not chunks or not query.strip():
            return [], "disabled"

        if self._load_semantic_index():
            try:
                import numpy as np

                query_vector = self._embedder.encode([query], normalize_embeddings=True)
                distances, ids = self._faiss_index.search(np.asarray(query_vector, dtype="float32"), min(limit, len(chunks)))
                results = [
                    RetrievedChunk(content=chunks[index], source=f"corpus chunk {index + 1}", score=float(distance))
                    for distance, index in zip(distances[0], ids[0])
                    if 0 <= index < len(chunks)
                ]
                return results, "semantic"
            except (RuntimeError, ValueError, IndexError) as exc:
                self._semantic_error = f"Semantic query failed: {exc}"

        query_terms = _tokens(query)
        if not query_terms:
            return [], "lexical"

        scored: list[tuple[float, int]] = []
        for index, chunk in enumerate(chunks):
            terms = _tokens(chunk)
            overlap = sum(min(count, terms[term]) for term, count in query_terms.items())
            # Prefer chunks that match several distinct query terms and avoid
            # favoring a long chunk just because it repeats one word.
            diversity = sum(1 for term in query_terms if term in terms)
            score = float(overlap + diversity * 0.5)
            if score:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(content=chunks[index], source=f"corpus chunk {index + 1}", score=score)
            for score, index in scored[:limit]
        ], "lexical"

    def status(self) -> dict[str, Any]:
        chunks = self._load_chunks()
        semantic_ready = (self.index_dir / "index.faiss").exists() and (self.index_dir / "chunks.json").exists()
        return {
            "enabled": bool(chunks),
            "corpus_path": str(self.corpus_path),
            "chunk_count": len(chunks),
            "semantic_index_ready": semantic_ready,
            "semantic_error": self._semantic_error,
        }
