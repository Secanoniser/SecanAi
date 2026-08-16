"""Debug Tests B+C+D: tokenizer round-trip + raw generation at low temperature.

Run: python debug_pipeline.py [checkpoint-or-model-id]
"""
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1] if len(sys.argv) > 1 else "output_model"
print(f"=== Testing checkpoint: {model_id} ===\n")

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()
print(f"[tokenizer] vocab_size: {tok.vocab_size}")
print(f"[tokenizer] special tokens: {tok.special_tokens_map}")
print(f"[tokenizer] has chat_template: {bool(getattr(tok, 'chat_template', None))}")
print(f"[model] params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print()

# --- Test B: round-trip (id-stable; leading space from add_prefix_space is fine) ---
text = "What is Python?"
ids = tok(text)["input_ids"]
re_ids = tok(tok.decode(ids))["input_ids"]
print(f"[Test B] '{text}' -> token ids: {ids}")
print(f"[Test B] decoded back: {tok.decode(ids)!r}")
print(f"[Test B] id-stable round-trip: {'OK' if ids == re_ids else 'MISMATCH!'}")
print()

# --- Test D: exact prompt that reaches the model ---
if getattr(tok, "chat_template", None):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "You are SecanAi, a small locally-run assistant."},
         {"role": "user", "content": "What is Python?"}],
        tokenize=False, add_generation_prompt=True,
    )
else:
    prompt = "System: You are SecanAi, a small locally-run assistant.\n\nUser: What is Python?\n\nAssistant:"
print(f"[Test D] prompt passed to model ({len(prompt)} chars):")
print(repr(prompt[:300]))
print()

# --- Test C: raw generation, no RAG/template/streaming/safety ---
for label, do_sample, temp in [("greedy (temp n/a)", False, None), ("temp=0.3", True, 0.3), ("temp=0.7", True, 0.7)]:
    inputs = tok("What is 2 + 2?", return_tensors="pt")
    kwargs = dict(
        max_new_tokens=48,
        do_sample=do_sample,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    if do_sample:
        kwargs["temperature"] = temp
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    elapsed = round(time.perf_counter() - t0, 1)
    generated = out[0][inputs["input_ids"].shape[1]:]
    decoded = tok.decode(generated, skip_special_tokens=True)
    print(f"[Test C] {label} ({elapsed}s): {decoded!r}")
print()
print("=== DONE ===")