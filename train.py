import os
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)
from dataset import CausalTextDataset

def main():
    print("[*] Starting Scaled (~100M Parameter) Local LLM Pre-training Pipeline...")
    
    base_dir = "C:\\Users\\Nyxentra\\Desktop\\local_llm"
    corpus_path = os.path.join(base_dir, "corpus.txt")
    tokenizer_path = os.path.join(base_dir, "tokenizer", "tokenizer.json")
    output_dir = os.path.join(base_dir, "output_model")

    if not os.path.exists(tokenizer_path):
        print("[!] Tokenizer not found. Please run train_tokenizer.py first.")
        return

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>"
    )

    dataset = CausalTextDataset(corpus_path, tokenizer, block_size=128)

    # Scaled Model Configuration (~80-100M parameters)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
    )

    model = LlamaForCausalLM(config)
    print(f"[+] Initialized Scaled Llama model from scratch with {model.num_parameters():,} parameters.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=5,
        save_steps=20,
        logging_steps=10,
        learning_rate=3e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("[*] Starting pre-training on ~100M model...")
    trainer.train()

    print(f"[+] Pre-training complete! Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
