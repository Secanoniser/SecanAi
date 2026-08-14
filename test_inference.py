import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_direct():
    model_dir = "C:\\Users\\Nyxentra\\Desktop\\local_llm\\output_model"
    if not os.path.exists(model_dir):
        print(f"[!] Model directory {model_dir} not found.")
        return

    print(f"[*] Loading fine-tuned math model directly from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    test_queries = [
        "User: Solve the linear equation: 2x + 3 = 7\nAssistant:",
        "User: What is Python?\nAssistant:",
        "User: Explain Pythagorean theorem in a right triangle.\nAssistant:",
        "User: What is the derivative of x^2?\nAssistant:"
    ]

    print("\n================ EXPERT GENERATION INFERENCE ================\n")
    for query in test_queries:
        inputs = tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(query):].strip()
        for stop in ["User:", "\nUser:", "Assistant:"]:
            if stop in response:
                response = response.split(stop)[0].strip()
        clean_text = response.replace("Ġ", " ").replace("Ċ", "\n")
        print(f"--- QUERY ---\n{query.strip()}")
        print(f"--- RESPONSE ---\n{clean_text}\n")
        print("=" * 60)

if __name__ == "__main__":
    test_model_direct()
