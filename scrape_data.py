"""Reproducible, append-safe Wikipedia corpus builder for the local LLM.

Improvements (2026-08-16, second pass):
  * retry with exponential backoff on HTTP 429/5xx, honoring ``Retry-After``,
    so a full topic list completes instead of failing at the rate limit,
  * a contact address in the User-Agent (Wikipedia prioritizes polite bots),
  * a per-article size cap so a single huge article (e.g. the 251 KB
    "Fourier transform") cannot dominate the corpus,
  * ``--rebalance`` rewrites ``corpus.txt`` applying the cap to existing
    articles too, so old oversized articles are trimmed in place.

Usage:
    python scrape_data.py                # fetch all topics, append, then rebalance
    python scrape_data.py --topics 30
    python scrape_data.py --rebalance    # only trim existing articles to the cap
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_DIR / "corpus.txt"
API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "SecanAiLocalLLM/1.1 (Educational Project; mailto:secanai.local@example.com)"}

MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "100000"))
MAX_RETRIES = 5

# Curated educational topics spanning AI, programming, math, science,
# history, and general knowledge.
TOPICS = [
    # AI & computing
    "Artificial intelligence", "Machine learning", "Deep learning", "Neural network",
    "Transformer (deep learning architecture)", "Large language model", "Natural language processing",
    "Computer vision", "Reinforcement learning", "Supervised learning", "Unsupervised learning",
    "Python (programming language)", "Programming language", "Compiler", "Operating system",
    "Computer science", "Algorithm", "Data structure", "Database", "Web search engine",
    "Computer network", "Information retrieval", "Automated theorem proving", "TensorFlow", "PyTorch",
    # Mathematics
    "Calculus", "Algebra", "Geometry", "Trigonometry", "Probability", "Statistics",
    "Pythagorean theorem", "Linear algebra", "Differential equation", "Number theory",
    "Mathematical logic", "Discrete mathematics", "Graph theory", "Fourier transform",
    "Bayes' theorem", "Prime number", "Golden ratio", "Set theory", "Topology", "Complex number",
    # Science
    "Physics", "Chemistry", "Biology", "Astronomy", "Quantum mechanics", "Thermodynamics",
    "Electromagnetism", "Photosynthesis", "DNA", "Evolution", "Cell (biology)", "Solar System",
    "Gravity", "Atom", "Molecule", "Plate tectonics", "Climate change", "Electricity",
    # History & geography
    "History of science", "Industrial Revolution", "Renaissance", "Roman Empire",
    "World War II", "Geography", "Earth", "Ocean", "Mountain",
    # General knowledge
    "Economics", "Psychology", "Philosophy", "Ethics", "Logic", "Music", "Art",
    "Literature", "Money", "Language", "English language",
]


def fetch_article(title: str, sleep_base: float) -> str | None:
    """Return the plain-text body of one Wikipedia article (size-capped).

    Retries on 429/5xx with exponential backoff, honoring ``Retry-After``.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
        "format": "json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else sleep_base * (2 ** (attempt - 1))
                print(f"[retry {attempt}/{MAX_RETRIES}] '{title}' -> HTTP {response.status_code}, waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    return cap_article(extract.strip())
            return None
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"[!] Giving up on '{title}': {exc}")
                return None
            delay = sleep_base * (2 ** (attempt - 1))
            print(f"[retry {attempt}/{MAX_RETRIES}] '{title}' -> {type(exc).__name__}, waiting {delay:.1f}s...")
            time.sleep(delay)
    return None


def cap_article(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    """Trim an article to ``max_chars`` on a paragraph boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = cut.rfind("\n\n")
    if boundary > max_chars // 2:
        cut = cut[:boundary]
    return f"{cut}\n[article truncated to {max_chars} characters for corpus balance]"


def load_existing_markers() -> set[str]:
    if not CORPUS_PATH.exists():
        return set()
    markers: set[str] = set()
    for line in CORPUS_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("--- Article: "):
            markers.add(line.removeprefix("--- Article: ").removesuffix(" ---").strip())
    return markers


def rebalance_corpus() -> None:
    """Apply the size cap to every existing article in the corpus."""
    if not CORPUS_PATH.exists():
        return
    text = CORPUS_PATH.read_text(encoding="utf-8", errors="replace")
    articles = re.findall(r"--- Article: (.*?) ---\n(.*?)(?=\n--- Article: |\Z)", text, re.S)
    if not articles:
        return
    rebuilt = []
    for name, body in articles:
        rebuilt.append(f"--- Article: {name} ---\n{cap_article(body.strip())}")
    CORPUS_PATH.write_text("\n\n".join(rebuilt) + "\n", encoding="utf-8")
    print(f"[+] Rebalanced {len(articles)} articles to max {MAX_ARTICLE_CHARS:,} chars each.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the training corpus from Wikipedia.")
    parser.add_argument("--topics", type=int, default=len(TOPICS), help="Number of topics to fetch (default: all).")
    parser.add_argument("--sleep", type=float, default=1.2, help="Base seconds between requests (rate limiting).")
    parser.add_argument("--rebalance", action="store_true", help="Only trim existing articles to the size cap.")
    args = parser.parse_args()

    if args.rebalance:
        rebalance_corpus()
        return

    topics = TOPICS[: args.topics]
    existing = load_existing_markers()
    print(f"[*] Corpus file: {CORPUS_PATH} (existing articles: {len(existing)})")
    print(f"[*] Fetching {len(topics)} articles (cap {MAX_ARTICLE_CHARS:,} chars/article, sleep {args.sleep}s)...")

    fetched: list[str] = []
    skipped = 0
    failures = 0

    for index, topic in enumerate(topics, start=1):
        if topic in existing:
            print(f"[skip] Already in corpus: {topic}")
            skipped += 1
            continue
        print(f"[{index}/{len(topics)}] Fetching: {topic}...")
        extract = fetch_article(topic, args.sleep)
        if extract is None:
            failures += 1
        else:
            fetched.append(f"--- Article: {topic} ---\n{extract}\n")
            print(f"[+] {len(extract):,} chars")
        time.sleep(args.sleep)

    if fetched:
        old_chars = CORPUS_PATH.stat().st_size if CORPUS_PATH.exists() else 0
        with CORPUS_PATH.open("a", encoding="utf-8") as corpus_file:
            if old_chars:
                corpus_file.write("\n\n")
            corpus_file.write("\n\n".join(fetched))
        new_chars = CORPUS_PATH.stat().st_size
        print(f"[+] Corpus updated: {old_chars / 1024:.1f} KB -> {new_chars / 1024:.1f} KB "
              f"(+{len(fetched)} articles, skipped {skipped}, failed {failures}).")
    else:
        print(f"[!] No new articles (skipped {skipped}, failed {failures}).")

    # Always rebalance afterwards so oversized articles never dominate.
    rebalance_corpus()
    print(f"[*] Final corpus: {CORPUS_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()