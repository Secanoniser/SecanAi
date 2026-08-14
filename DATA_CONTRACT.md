# SecanAi SFT Data Contract

The training-ready dataset is UTF-8 JSONL. Each row has exactly one user turn and one assistant turn:

```json
{"source_id":"dataset:42","source_type":"math","split":"train","messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

- `source_id`: non-empty immutable identifier tracing the source row.
- `source_type`: optional category string; use `unknown` when unavailable.
- `split`: one of `train`, `validation`, or `test`. Splits are assigned deterministically from the normalized user/assistant pair.
- `messages`: exactly two non-empty message objects in `user`, then `assistant` order.

Only assistant content is a supervised target. Input records are normalized to UTF-8, exact duplicate normalized pairs are removed, and source records with invalid JSON or missing/empty canonical text are rejected. Keep original sources separate from `data/processed` and document their provenance and licence in a dataset card before use.
