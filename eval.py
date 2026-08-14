import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model():
    model_dir = "C:\\Users\\Nyxentra\\Desktop\\local_llm\\output_model"
    if not os.path.exists(model_dir):
        print(f"[!] Trained model not found at {model_dir}. Please run train.py first.")
        return

    print(f"[*] Loading model and tokenizer for evaluation from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    test_prompts = [
        "Artificial Intelligence",
        "Python is",
        "The transformer architecture",
        "Machine Learning"
    ]

    print("\n================ EVALUATION BENCHMARK ================\n")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=40,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up byte-level BPE space representations if any remain
        clean_text = generated_text.replace("Ġ", " ").replace("Ċ", "\n")
        print(f"Prompt: '{prompt}'")
        print(f"Output: '{clean_text}'")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_model()
