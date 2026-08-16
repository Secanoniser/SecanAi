"""Load-time model selection for the SecanAi serving stack.

Layer 1 of the feature roadmap: instead of hardcoding which checkpoint
``server.py`` serves, this module decides at load time between the locally
trained NanoLLM/Llama checkpoint and the SmolLM2 baseline.

The eval harness and model-card tooling import the same resolver so every
consumer agrees on what "the active model" means.

Selection precedence (highest first):
  1. ``MODEL_SOURCE`` environment variable (``smollm2-baseline``,
     ``nanollm``, ``local-checkpoint``).
  2. The SmolLM2 baseline by default.

The local checkpoint is NOT auto-selected anymore. Diagnosis (2026-08-16)
proved the 105M checkpoint produced byte-level garbage (751-token tokenizer,
~8K training tokens). It is only served when explicitly requested via
``MODEL_SOURCE``; once retrained with a proper tokenizer, it should also pass
``eval_harness.py`` before being adopted as the default (quality gate).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_CHECKPOINT = Path(os.getenv("LOCAL_MODEL_PATH", PROJECT_DIR / "output_model"))
BASELINE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Aliases are kept for compatibility with earlier configuration examples.
MODEL_SOURCES: dict[str, str] = {
    "smollm2-baseline": BASELINE_MODEL_ID,
    "nanollm": str(LOCAL_CHECKPOINT),
    "local-checkpoint": str(LOCAL_CHECKPOINT),
    "local": str(LOCAL_CHECKPOINT),
}

LOCAL_SOURCE_NAMES = {"nanollm", "local-checkpoint", "local"}


def _checkpoint_is_trainable(path: Path) -> bool:
    """A local checkpoint is servable when it has a config and weights."""
    if not (path / "config.json").exists():
        return False
    return any(path.glob("model.safetensors*")) or any(path.glob("pytorch_model*.bin")) or (path / "model.pt").exists()


def resolve_model_source() -> tuple[str, str]:
    """Return ``(source_name, model_id)`` for the checkpoint to serve.

    Raises ``RuntimeError`` when an explicit configuration points at a
    checkpoint that does not exist or is not servable.
    """
    configured = os.getenv("MODEL_SOURCE", "").strip().lower()
    if configured:
        if configured not in MODEL_SOURCES:
            valid = ", ".join(sorted(MODEL_SOURCES))
            raise RuntimeError(f"Unknown MODEL_SOURCE '{configured}'. Choose one of: {valid}.")
        model_id = MODEL_SOURCES[configured]
        if configured in LOCAL_SOURCE_NAMES and not _checkpoint_is_trainable(Path(model_id)):
            raise RuntimeError(
                f"Local checkpoint selected ({model_id}) but it has no servable weights. "
                "Train a model first (train.py / sft_train.py) or set MODEL_SOURCE=smollm2-baseline."
            )
        return configured, model_id

    if _checkpoint_is_trainable(LOCAL_CHECKPOINT):
        # Serveable, but NOT preferred: the last trained checkpoint produced
        # incoherent output (broken tokenizer). Keep it behind an explicit
        # MODEL_SOURCE until it passes the eval quality gate.
        pass
    return "smollm2-baseline", BASELINE_MODEL_ID


def load_model(source: str | None = None) -> tuple[Any, Any]:
    """Load ``(tokenizer, model)`` for the resolved (or explicitly given) source.

    Both the FastAPI server and the offline eval harness use this entry point
    so the served model and the evaluated model can never drift apart.
    """
    active_source, model_id = resolve_model_source() if source is None else (source, MODEL_SOURCES[source])
    if source is not None and source not in MODEL_SOURCES:
        raise RuntimeError(f"Unknown model source '{source}'.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=os.getenv("DEVICE_MAP", "auto"),
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model


def describe_model(tokenizer: Any, model: Any, source: str, model_id: str) -> dict[str, Any]:
    """Structured identity used by ``GET /api/model`` and model cards."""
    config = getattr(model, "config", None)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "source": source,
        "model_id": model_id,
        "display_name": "Local checkpoint" if source in LOCAL_SOURCE_NAMES else "SmolLM2 baseline",
        "parameter_count": parameter_count,
        "parameter_count_human": f"{parameter_count / 1e6:.1f}M",
        "hidden_size": int(getattr(config, "hidden_size", 0) or 0),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0) or 0),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0) or 0),
        "max_position_embeddings": int(getattr(config, "max_position_embeddings", 0) or 0),
        "model_type": str(getattr(config, "model_type", "unknown")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "dtype": str(getattr(model, "dtype", "float32")),
        "device": str(getattr(model, "device", "cpu")),
    }


if __name__ == "__main__":
    source, model_id = resolve_model_source()
    print(f"[+] Active model source: {source}")
    print(f"[+] Model id: {model_id}")