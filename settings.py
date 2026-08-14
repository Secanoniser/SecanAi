"""Central, repository-relative configuration for SecanAi."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


@dataclass(frozen=True)
class Settings:
    repository_root: Path = REPOSITORY_ROOT
    data_dir: Path = field(default_factory=lambda: _path_from_env("SECANAI_DATA_DIR", REPOSITORY_ROOT / "data"))
    artifacts_dir: Path = field(default_factory=lambda: _path_from_env("SECANAI_ARTIFACTS_DIR", REPOSITORY_ROOT / "artifacts"))
    base_model_id: str = field(default_factory=lambda: os.getenv("SECANAI_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"))
    model_path: Path | None = None
    host: str = field(default_factory=lambda: os.getenv("SECANAI_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("SECANAI_PORT", "8000")))
    max_prompt_characters: int = field(default_factory=lambda: int(os.getenv("SECANAI_MAX_PROMPT_CHARS", "8000")))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("SECANAI_MAX_NEW_TOKENS", "256")))

    def __post_init__(self) -> None:
        if self.model_path is None:
            object.__setattr__(self, "model_path", _path_from_env("SECANAI_MODEL_PATH", self.artifacts_dir / "models" / "sft"))

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"

    def ensure_directories(self) -> None:
        for directory in (self.raw_data_dir, self.processed_data_dir, self.artifacts_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
