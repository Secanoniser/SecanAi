import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from settings import get_settings

def run_local_llm():
    model_dir = get_settings().model_path
    
    if not os.path.exists(model_dir):
        print(f"[!] Model directory {model_dir} not found. Please run train.py first.")
        return

    print(f"[*] Loading locally trained model with anti-repetition parameters from: {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # Pipeline with anti-repetition controls (repetition_penalty, top_k, top_p)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.2,
        top_k=50,
        top_p=0.9
    )
    
    print("\n[+] Upgraded Local LLM Ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            prompt = input("User > ")
            if prompt.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not prompt.strip():
                continue
            
            formatted_prompt = f"User: {prompt}\nAssistant:"
            outputs = pipe(formatted_prompt)
            response = outputs[0]["generated_text"][len(formatted_prompt):].strip()
            
            print(f"\nAssistant > {response}\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    run_local_llm()
