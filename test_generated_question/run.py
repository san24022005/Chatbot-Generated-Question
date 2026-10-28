from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from model_loader import load_local_t5_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a question from a paragraph using the trained model and tokenizer."
    )
    parser.add_argument("--text", help="Paragraph to generate a question from.")
    args = parser.parse_args()
    text = (args.text if args.text is not None else input("Input paragraph: ")).strip()
    if not text:
        parser.error("Please provide a paragraph.")

    model = load_local_t5_model(
        model_dir=PROJECT_DIR / "T5-question-generated-model",
        tokenizer_dir=PROJECT_DIR / "T5-question-generated-tokenizer",
    )
    input_text = f"generate questions: {text.strip()}"
    inputs = model.tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=model.max_input_tokens,
    )
    outputs = model.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    print("Generated question:")
    print(model.tokenizer.decode(outputs[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
