import os
from pathlib import Path

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)
from dataset import CausalTextDataset

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("[*] Starting Scaled (~100M Parameter) Local LLM Pre-training Pipeline...")
    
    corpus_path = BASE_DIR / "corpus.txt"
    tokenizer_path = BASE_DIR / "tokenizer" / "tokenizer.json"
    output_dir = BASE_DIR / "output_model"

    epochs = int(os.getenv("TRAIN_EPOCHS", "1"))
    save_steps = int(os.getenv("TRAIN_SAVE_STEPS", "500"))
    block_size = int(os.getenv("TRAIN_BLOCK_SIZE", "256"))
    learning_rate = float(os.getenv("TRAIN_LR", "3e-4"))
    resume = os.getenv("TRAIN_RESUME", "").lower() in {"1", "true", "yes"}

    if not tokenizer_path.exists():
        print("[!] Tokenizer not found. Please run train_tokenizer.py first.")
        return

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>"
    )

    dataset = CausalTextDataset(str(corpus_path), tokenizer, block_size=block_size)

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
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        save_steps=save_steps,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"[*] Starting pre-training on ~100M model (epochs={epochs}, block_size={block_size}, lr={learning_rate})...")
    checkpoint_dir = None
    if resume:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if checkpoints:
            checkpoint_dir = str(checkpoints[-1])
            print(f"[*] Resuming from {checkpoint_dir}")
        else:
            print("[*] TRAIN_RESUME set but no checkpoint found; starting fresh.")
    trainer.train(resume_from_checkpoint=checkpoint_dir)

    print(f"[+] Pre-training complete! Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

if __name__ == "__main__":
    main()
