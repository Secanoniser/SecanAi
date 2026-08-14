import json
import tempfile
import unittest
from pathlib import Path

from prepare_sft_data import clean_text, prepare
from sft_train import build_features
from validate_data import validate_jsonl


class DataPipelineTests(unittest.TestCase):
    def test_clean_text_repairs_common_mojibake(self):
        self.assertEqual(clean_text("aÂ²"), "a²")

    def test_prepare_deduplicates_and_creates_canonical_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join([
                    json.dumps({"instruction": "What is 2 + 2?", "completion": "4", "type": "math"}),
                    json.dumps({"instruction": "What is 2 + 2?", "completion": "4", "type": "math"}),
                    json.dumps({"instruction": "Square aÂ²", "completion": "Use a²", "type": "qa"}),
                ]) + "\n",
                encoding="utf-8",
            )
            output = root / "processed.jsonl"
            report = prepare(source, output, validation_percent=10, test_percent=10)
            self.assertEqual(report["kept_rows"], 2)
            self.assertEqual(report["duplicate_rows"], 1)
            validation = validate_jsonl(output, canonical=True)
            self.assertEqual(validation["invalid_json"], 0)
            self.assertEqual(validation["missing_required"], 0)
            self.assertEqual(validation["mojibake_rows"], 0)

    def test_training_labels_mask_prompt_and_padding(self):
        class Tokenizer:
            pad_token_id = 0

            @staticmethod
            def apply_chat_template(messages, tokenize, add_generation_prompt):
                return [10, 11, 12] if len(messages) == 1 else [10, 11, 12, 13, 14]

        example = {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
        features = build_features(example, Tokenizer(), max_length=8)
        self.assertEqual(features["labels"], [-100, -100, -100, 13, 14, -100, -100, -100])
        self.assertEqual(features["attention_mask"], [1, 1, 1, 1, 1, 0, 0, 0])
