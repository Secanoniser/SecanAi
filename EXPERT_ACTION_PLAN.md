# Expert Action Plan: Fixing Local LLM Gibberish & Scaling to Coherent Generation

## 1. Root Cause Diagnosis
1. **Undertrained & Tiny Model:** 24M parameters trained on 7.5KB of text for 3 epochs is far too small to model English grammar.
2. **Tiny SFT Dataset:** 9 examples caused severe under-fitting to chat completion patterns.
3. **Missing Padding/EOS Tokens:** Generation lacked explicit `eos_token_id` and `pad_token_id` handling, resulting in decoding artifacts.
4. **Lack of Anti-Repetition Constraints:** Missing `no_repeat_ngram_size=3` and attention masks caused looping loops.

---

## 2. Execution Steps

### Step 1: Quick Sanity Check (`debug_gen.py`)
Run greedy decoding to inspect raw model weights.

### Step 2: Massive SFT Expansion (`sft_train.py`)
- Expand SFT dataset to 50+ diverse Q&A pairs (AI, Python, Math, Science, General).
- Train for 15 epochs with lower learning rate (`5e-5`) and batch size 4.

### Step 3: Architecture Scaling (`train.py`)
- Scale model from 24M to ~100M parameters (`hidden_size=768`, `num_hidden_layers=12`, `num_attention_heads=12`).

### Step 4: Generation Argument Fixes (`server.py` & `test_inference.py`)
- Explicitly set `pad_token = eos_token`.
- Add `attention_mask`, `no_repeat_ngram_size=3`, and stop-word cleaning (`User:`, `Assistant:`).
