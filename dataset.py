import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class CausalTextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, block_size: int = 512):
        print(f"[*] Loading and tokenizing dataset from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize entire corpus
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].squeeze(0)

        # Chunk into blocks of block_size
        self.examples = []
        total_length = len(input_ids)
        
        # Drop remainder or pad
        for i in range(0, total_length - block_size + 1, block_size):
            self.examples.append(input_ids[i : i + block_size])
            
        if len(self.examples) == 0:
            # Fallback if corpus is smaller than block_size, pad it
            padded = torch.nn.functional.pad(
                input_ids, 
                (0, block_size - len(input_ids)), 
                value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
            self.examples.append(padded)

        print(f"[+] Created {len(self.examples)} training blocks of size {block_size}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        # For causal LM, input and labels are the same (shifted inside transformers loss)
        return {"input_ids": chunk, "labels": chunk.clone()}

if __name__ == "__main__":
    from transformers import PreTrainedTokenizerFast
    print("[+] dataset.py module loaded successfully.")
