import asyncio
import builtins
from collections.abc import Iterator

import pytest

import refund_agent.main as main_module
from refund_agent.agent import AgentResult
from refund_agent.main import main, read_user_message, run_chat


class FakeAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def run_agent(self, user_message: str) -> AgentResult:
        self.messages.append(user_message)
        return AgentResult(
            status=None,
            message=f"handled {user_message}",
        )


def terminal_input(monkeypatch: pytest.MonkeyPatch, messages: list[str]) -> None:
    responses: Iterator[str] = iter(messages)
    monkeypatch.setattr(builtins, "input", lambda prompt: next(responses))


def test_read_user_message_strips_terminal_input(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal_input(monkeypatch, ["  Refund order 123  "])

    assert read_user_message() == "Refund order 123"


def test_chat_processes_messages_until_quit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = FakeAgent()
    terminal_input(monkeypatch, ["", "Refund order 123", "Check order 456", "quit"])

    asyncio.run(run_chat(agent))  # type: ignore[arg-type]

    assert agent.messages == ["Refund order 123", "Check order 456"]
    assert capsys.readouterr().out.splitlines() == [
        "Please enter a message, or type 'quit' or 'exit'.",
        "Refund Agent: handled Refund order 123",
        "Refund Agent: handled Check order 456",
        "Goodbye.",
    ]


@pytest.mark.parametrize("command", ["quit", "QUIT", "exit", " Exit "])
def test_chat_accepts_exit_commands(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = FakeAgent()
    terminal_input(monkeypatch, [command])

    asyncio.run(run_chat(agent))  # type: ignore[arg-type]

    assert agent.messages == []


def test_chat_handles_terminal_end_of_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def end_of_file(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", end_of_file)

    asyncio.run(run_chat(FakeAgent()))  # type: ignore[arg-type]

    assert capsys.readouterr().out == "\nGoodbye.\n"


def test_main_loads_model_and_starts_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_models: list[str] = []

    class ConfiguredFakeAgent(FakeAgent):
        def __init__(self, model: str) -> None:
            super().__init__()
            constructed_models.append(model)

    terminal_input(monkeypatch, ["exit"])
    monkeypatch.setattr(main_module, "load_model_id", lambda: "test/model")
    monkeypatch.setattr(main_module, "RefundAgent", ConfiguredFakeAgent)

    main()

    assert constructed_models == ["test/model"]
