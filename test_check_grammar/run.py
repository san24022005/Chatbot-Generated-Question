from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from model_loader import generate_text, load_local_t5_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct grammar using the trained model and tokenizer."
    )
    parser.add_argument("--text", help="Paragraph to correct.")
    args = parser.parse_args()
    text = (args.text if args.text is not None else input("Input paragraph: ")).strip()
    if not text:
        parser.error("Please provide a paragraph.")

    model = load_local_t5_model(
        model_dir=PROJECT_DIR / "T5-check-grammar-model",
        tokenizer_dir=PROJECT_DIR / "T5-check-grammar-tokenizer",
    )
    print("Corrected text:")
    print(generate_text(model, text))


if __name__ == "__main__":
    main()
