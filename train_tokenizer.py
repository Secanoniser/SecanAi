import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from settings import get_settings

def train_custom_tokenizer(corpus_path: str, save_dir: str):
    print("[*] Initializing Byte-Level BPE Tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=10000,
        min_frequency=2,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
    )

    if not os.path.exists(corpus_path):
        print(f"[!] Corpus path {corpus_path} not found. Creating a sample training corpus...")
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("Hello world! Building a local LLM from scratch is a fascinating engineering challenge.\n" * 100)

    print(f"[*] Training tokenizer on corpus: {corpus_path}...")
    tokenizer.train([corpus_path], trainer)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"[+] Tokenizer saved successfully to {save_dir}/tokenizer.json")

if __name__ == "__main__":
    settings = get_settings()
    train_custom_tokenizer(str(settings.raw_data_dir / "corpus.txt"), str(settings.artifacts_dir / "tokenizer"))
