import pytest


@pytest.fixture(autouse=True)
def disable_external_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent unit tests from exporting telemetry with developer credentials."""

    monkeypatch.setenv("REFUND_AGENT_OTEL_EXPORT", "false")
