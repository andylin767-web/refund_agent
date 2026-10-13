from pathlib import Path

import pytest

from refund_agent.agent import RefundAgent
from refund_agent.config import ConfigurationError, load_model_id


def test_load_model_id_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_ID", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_ID=test/provider-model\n", encoding="utf-8")

    agent = RefundAgent(model=load_model_id(env_file))

    assert agent.model == "test/provider-model"


def test_process_environment_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_ID", "environment/model")
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_ID=file/model\n", encoding="utf-8")

    assert load_model_id(env_file) == "environment/model"


def test_missing_model_id_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_ID", raising=False)

    with pytest.raises(ConfigurationError, match="MODEL_ID is required"):
        load_model_id(tmp_path / "missing.env")
