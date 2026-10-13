"""Environment-backed configuration for the refund agent."""

import os
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required agent configuration is unavailable."""


def load_model_id(env_file: Path | None = None) -> str:
    """Load ``.env`` and return the required LiteLLM model identifier."""

    load_dotenv(dotenv_path=env_file, override=False)
    model_id = os.getenv("MODEL_ID", "").strip()
    if not model_id:
        raise ConfigurationError("MODEL_ID is required; add it to .env or the process environment")
    return model_id
