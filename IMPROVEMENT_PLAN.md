# Advanced Local LLM Improvement Plan: Scaling Beyond Synthetic Data

## 1. Architectural & Data Limitations (What Is Lacking)
* **Data Volume & Diversity:** Synthetic corpora (~30KB) are too small. Real frontier LLMs require billions to trillions of tokens across diverse domains (books, code repositories, research papers, encyclopedias).
* **Tokenization Granularity:** Standard BPE tokenizers trained on tiny corpora miss rare words and fail at precise numerical representations.
* **Model Parameter Scale:** 24M parameters act as a compressed lookup table rather than a generalized reasoner. True reasoning requires $\ge 1\text{B}$ to $7\text{B}+$ parameters.
* **Lack of Web Scraped Knowledge:** Real-world knowledge requires automated web scraping and data filtration pipelines to ingest clean educational text.

---

## 2. Updated Task Roadmap

### [x] Task 1: Create Initial Improvement Plan
### [x] Task 2: Scale Model Architecture (~24M Parameters)
### [x] Task 3: Multi-Domain Synthetic Corpus Generation
### [x] Task 4: Supervised Fine-Tuning (SFT) & Chain-of-Thought
### [ ] Task 5: Automated Web Scraping Data Collector (`scrape_data.py`)
- Implement a Python scraper leveraging the Wikipedia API to fetch real encyclopedic articles on Mathematics, Artificial Intelligence, Physics, and Computer Science to build a massive, real-world training corpus.
