from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration


PROJECT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: Any
    max_input_tokens: int


MODEL_PATHS = {
    "grammar": {
        "model_dir": PROJECT_DIR / "T5-check-grammar-model",
        "tokenizer_dir": PROJECT_DIR / "T5-check-grammar-tokenizer" / "T5-base-tokenizer",
    },
}


def _load_local_t5_model(model_dir: Path, tokenizer_dir: Path | None = None) -> LoadedModel:
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


def load_models() -> tuple[dict[str, LoadedModel], dict[str, str]]:
    loaded: dict[str, LoadedModel] = {}
    errors: dict[str, str] = {}
    for name, paths in MODEL_PATHS.items():
        try:
            loaded[name] = _load_local_t5_model(
                model_dir=paths["model_dir"],
                tokenizer_dir=paths["tokenizer_dir"],
            )
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return loaded, errors
