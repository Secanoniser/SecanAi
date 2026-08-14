import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "C:\\Users\\Nyxentra\\Desktop\\local_llm\\output_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = "User: What is Python?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=40,
        do_sample=False,                    # greedy decoding
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
clean = decoded.replace("Ġ", " ").replace("Ċ", "\n")
print("=== DEBUG GENERATION ===")
print(clean)
print("========================")
