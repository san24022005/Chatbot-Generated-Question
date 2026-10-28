from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration


PROJECT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: Any
    max_input_tokens: int


_LOADED_MODELS: dict[str, LoadedModel] = {}
_LOAD_ERRORS: dict[str, str] = {}
_LOAD_LOCK = Lock()


MODEL_PATHS = {
    # Mô hình kiểm tra ngữ pháp hiện có
    "grammar": {
        "model_dir": PROJECT_DIR / "T5-check-grammar-model",
        "tokenizer_dir": PROJECT_DIR / "T5-check-grammar-tokenizer",
    },
    # Mô hình sinh câu hỏi
    "question": {
        "model_dir": PROJECT_DIR / "T5-question-generated-model",
        "tokenizer_dir": PROJECT_DIR / "T5-question-generated-tokenizer",
    },
}


def load_local_t5_model(model_dir: Path, tokenizer_dir: Path | None = None) -> LoadedModel:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục mô hình: {model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config mô hình: {config_path}")

    if tokenizer_dir is None:
        tokenizer_dir = model_dir

    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục tokenizer: {tokenizer_dir}")
    if not (tokenizer_dir / "tokenizer.json").is_file() and not (
        tokenizer_dir / "tokenizer_config.json"
    ).is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tokenizer trong {tokenizer_dir}; cần tokenizer.json hoặc tokenizer_config.json."
        )

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "t5":
        raise RuntimeError(
            f"Mô hình ở {model_dir} không phải kiến trúc T5; model_type nhận được: {config.model_type}."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        use_fast=True,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    max_input_tokens = int(
        getattr(config, "n_positions", None)
        or getattr(config, "max_position_embeddings", None)
        or 512
    )

    model.to(DEVICE)
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        max_input_tokens=max_input_tokens,
    )


def generate_text(loaded_model: LoadedModel, text: str, max_output_tokens: int = 128) -> str:
    encoded = loaded_model.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=loaded_model.max_input_tokens,
    )
    encoded = {key: value.to(loaded_model.model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output_ids = loaded_model.model.generate(
            **encoded,
            max_length=min(loaded_model.max_input_tokens, max_output_tokens),
            num_beams=4,
            do_sample=False,
        )
    return loaded_model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def check_grammar(text: str) -> str:
    loaded = get_grammar_model()
    return generate_text(loaded, text)


def generate_questions(context: str) -> str:
    loaded = get_question_model()
    input_text = f"generate questions: {context.strip()}"
    encoded = loaded.tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=loaded.max_input_tokens,
    )
    encoded = {key: value.to(loaded.model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output_ids = loaded.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return loaded.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def load_models() -> tuple[dict[str, LoadedModel], dict[str, str]]:
    with _LOAD_LOCK:
        if _LOADED_MODELS or _LOAD_ERRORS:
            return dict(_LOADED_MODELS), dict(_LOAD_ERRORS)

        for name, paths in MODEL_PATHS.items():
            try:
                _LOADED_MODELS[name] = load_local_t5_model(
                    model_dir=paths["model_dir"],
                    tokenizer_dir=paths["tokenizer_dir"],
                )
            except Exception as exc:
                LOGGER.exception("Failed to load %s model", name)
                _LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        return dict(_LOADED_MODELS), dict(_LOAD_ERRORS)


def get_grammar_model() -> LoadedModel:
    try:
        return load_models()[0]["grammar"]
    except KeyError as exc:
        raise RuntimeError("The grammar model is not available.") from exc


def get_question_model() -> LoadedModel:
    try:
        return load_models()[0]["question"]
    except KeyError as exc:
        raise RuntimeError("The question model is not available.") from exc
