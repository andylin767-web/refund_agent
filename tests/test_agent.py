import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from pydantic import BaseModel

import refund_agent.agent as agent_module
from refund_agent.agent import (
    RefundAgent,
    ToolErrorOutput,
    execute_tool,
    refund_request_needs_order_id,
)
from refund_agent.order import OrderStatus
from refund_agent.order_service import OrderNotFoundError
from refund_agent.tools import OrderOutput, RefundOutcome, RefundOutput


class ToolResult(BaseModel):
    value: str


class FakeMessage:
    def __init__(self, content: str | None, tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, object]:
        del exclude_none
        return {"role": "assistant", "content": self.content}


def response(
    message: FakeMessage,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def refund_result(outcome: RefundOutcome) -> RefundOutput:
    """Build a complete refund-tool result for agent tests."""

    return RefundOutput(
        order=OrderOutput(
            order_id="123",
            delivered_days_ago=1,
            status=OrderStatus.DELIVERED,
            price=Decimal("10.00"),
        ),
        outcome=outcome,
        message="result",
    )


def test_litellm_provider_help_output_is_suppressed() -> None:
    assert litellm.suppress_debug_info is True


def test_execute_tool_dispatches_and_serializes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        agent_module.TOOL_HANDLERS,
        "example",
        lambda order_id: ToolResult(value=order_id),
    )

    result = execute_tool("example", '{"order_id": "123"}')

    assert result.model_dump() == {"value": "123"}


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("missing", "{}"), ("get_order", "not-json")],
)
def test_execute_tool_returns_structured_error(name: str, arguments: str) -> None:
    assert "error" in execute_tool(name, arguments).model_dump()


def test_execute_tool_identifies_missing_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_order(order_id: str) -> BaseModel:
        raise OrderNotFoundError(f"order {order_id!r} was not found")

    monkeypatch.setitem(agent_module.TOOL_HANDLERS, "get_order", missing_order)

    result = execute_tool("get_order", '{"order_id": "999"}')

    assert isinstance(result, ToolErrorOutput)
    assert result.outcome is RefundOutcome.ORDER_NOT_FOUND


def test_agent_returns_text_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        assert kwargs["model"] == "test/model"
        return response(FakeMessage("Done"))

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)

    result = asyncio.run(RefundAgent("test/model").run_agent("Help"))

    assert result.status is None
    assert result.message == "Done"


@pytest.mark.parametrize(
    "user_message",
    [
        "Please refund my order",
        "I need a refund",
        "Can this order be refunded?",
    ],
)
def test_agent_returns_needs_order_id_without_calling_model(
    user_message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unexpected_completion(**kwargs: object) -> SimpleNamespace:
        pytest.fail(f"LLM should not be called: {kwargs}")

    monkeypatch.setattr(agent_module, "litellm_acompletion", unexpected_completion)

    result = asyncio.run(RefundAgent("test/model").run_agent(user_message))

    assert result.status is RefundOutcome.NEEDS_ORDER_ID
    assert result.message == "Please provide the order ID you want to refund."


@pytest.mark.parametrize(
    "user_message",
    [
        "Refund order 123",
        "Please refund order #456",
        "Refund 789",
        "Refund order ID: 999",
    ],
)
def test_refund_request_detects_numeric_order_id(user_message: str) -> None:
    assert not refund_request_needs_order_id(user_message)


def test_agent_executes_tool_then_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="example", arguments='{"order_id": "123"}'),
    )
    replies = iter([response(FakeMessage(None, [tool_call])), response(FakeMessage("Done"))])

    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return next(replies)

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)
    monkeypatch.setitem(
        agent_module.TOOL_HANDLERS,
        "example",
        lambda order_id: ToolResult(value=order_id),
    )

    result = asyncio.run(RefundAgent("test/model").run_agent("Refund 123"))

    assert result.status is None
    assert result.message == "Done"


def test_agent_preserves_refund_tool_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="refund", arguments="{}"),
    )
    replies = iter(
        [
            response(
                FakeMessage(None, [tool_call]),
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
            ),
            response(
                FakeMessage("Refund complete"),
                prompt_tokens=20,
                completion_tokens=4,
                total_tokens=0,
            ),
        ]
    )

    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return next(replies)

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)
    monkeypatch.setitem(
        agent_module.TOOL_HANDLERS,
        "refund",
        lambda: refund_result(RefundOutcome.REFUNDED),
    )

    result = asyncio.run(RefundAgent("test/model").run_agent("Refund order 123"))

    assert result.status is RefundOutcome.REFUNDED
    assert result.message == "Refund complete"
    assert result.metrics.llm_calls == 2
    assert result.metrics.tool_calls == 1
    assert result.metrics.tool_counts == {"refund": 1}
    assert result.metrics.tool_sequence == ("refund",)
    assert result.metrics.prompt_tokens == 30
    assert result.metrics.completion_tokens == 7
    assert result.metrics.total_tokens == 37


@pytest.mark.parametrize(
    ("user_message", "expected"),
    [
        ("Refund order 789", RefundOutcome.ALREADY_REFUNDED),
        ("Check order 789", None),
    ],
)
def test_agent_maps_refunded_order_lookup_for_refund_request(
    user_message: str,
    expected: RefundOutcome | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="get_order", arguments='{"order_id": "789"}'),
    )
    replies = iter(
        [response(FakeMessage(None, [tool_call])), response(FakeMessage("Already refunded"))]
    )

    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return next(replies)

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)
    monkeypatch.setitem(
        agent_module.TOOL_HANDLERS,
        "get_order",
        lambda order_id: OrderOutput(
            order_id=order_id,
            delivered_days_ago=5,
            status=OrderStatus.REFUNDED,
            price=Decimal("15.00"),
        ),
    )

    result = asyncio.run(RefundAgent("test/model").run_agent(user_message))

    assert result.status is expected
    assert result.message == "Already refunded"


@pytest.mark.parametrize(
    ("user_message", "expected"),
    [
        ("Refund order 999", RefundOutcome.ORDER_NOT_FOUND),
        ("Check order 999", None),
    ],
)
def test_agent_maps_missing_order_for_refund_request(
    user_message: str,
    expected: RefundOutcome | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="get_order", arguments='{"order_id": "999"}'),
    )
    replies = iter(
        [response(FakeMessage(None, [tool_call])), response(FakeMessage("Order not found"))]
    )

    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return next(replies)

    def missing_order(order_id: str) -> BaseModel:
        raise OrderNotFoundError(f"order {order_id!r} was not found")

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)
    monkeypatch.setitem(agent_module.TOOL_HANDLERS, "get_order", missing_order)

    result = asyncio.run(RefundAgent("test/model").run_agent(user_message))

    assert result.status is expected
    assert result.message == "Order not found"


@pytest.mark.parametrize(
    ("model", "max_attempts", "message"),
    [("", 5, "model"), ("test/model", 0, "max_attempts")],
)
def test_agent_rejects_invalid_configuration(model: str, max_attempts: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RefundAgent(model, max_attempts)
