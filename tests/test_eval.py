import asyncio
from pathlib import Path
from typing import ClassVar

import eval.run_eval as eval_module
import pytest

from refund_agent.agent import AgentMetrics, AgentResult
from refund_agent.tools import RefundOutcome


class FakeAgent:
    """Return predetermined outcomes without making LLM calls."""

    outcomes: ClassVar[list[RefundOutcome]] = []

    def __init__(self, model: str) -> None:
        assert model == "test/model"
        self._outcomes = iter(self.outcomes)

    async def run_agent(self, user_message: str) -> AgentResult:
        return AgentResult(
            status=next(self._outcomes),
            message=f"handled {user_message}",
            metrics=AgentMetrics(
                llm_calls=2,
                tool_calls=1,
                tool_counts={"get_order": 1},
                tool_sequence=("get_order",),
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )


def test_eval_scores_properties() -> None:
    scores = eval_module.EvalScores(
        total_cases=2,
        attempts_per_case=3,
        passed_attempts=5,
        pass_at_k_cases=2,
        pass_power_k_cases=1,
    )

    assert scores.total_attempts == 6
    assert scores.single_run_accuracy == pytest.approx(5 / 6)
    assert scores.pass_at_k == 1.0
    assert scores.pass_power_k == 0.5
    assert scores.average_llm_calls == 0.0
    assert scores.average_tool_calls == 0.0
    assert scores.average_total_tokens == 0.0


def test_empty_eval_scores_are_zero() -> None:
    scores = eval_module.EvalScores(0, 3, 0, 0, 0)

    assert scores.total_attempts == 0
    assert scores.single_run_accuracy == 0.0
    assert scores.pass_at_k == 0.0
    assert scores.pass_power_k == 0.0
    assert scores.average_llm_calls == 0.0
    assert scores.average_tool_calls == 0.0
    assert scores.average_total_tokens == 0.0


def test_run_evaluations_reports_pass_at_k_and_pass_power_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    orders_path = tmp_path / "orders.json"
    orders_path.write_text("{}\n", encoding="utf-8")
    FakeAgent.outcomes = [
        RefundOutcome.REFUNDED,
        RefundOutcome.DENIED,
        RefundOutcome.REFUNDED,
        RefundOutcome.DENIED,
        RefundOutcome.DENIED,
        RefundOutcome.DENIED,
    ]
    monkeypatch.setattr(
        eval_module,
        "load_cases",
        lambda: [
            {
                "input": "Refund order 1",
                "expected": "refunded",
                "expected_tools": ["get_order"],
                "max_tool_calls": 1,
            },
            {
                "input": "Refund order 2",
                "expected": "denied",
                "expected_tools": ["get_order"],
                "max_tool_calls": 1,
            },
        ],
    )
    monkeypatch.setattr(eval_module, "ORDERS_PATH", orders_path)
    monkeypatch.setattr(eval_module, "load_model_id", lambda: "test/model")
    monkeypatch.setattr(eval_module, "RefundAgent", FakeAgent)

    scores = asyncio.run(eval_module.run_evaluations(k=3, color=True))

    assert scores.single_run_accuracy == pytest.approx(5 / 6)
    assert scores.pass_at_k == 1.0
    assert scores.pass_power_k == 0.5
    assert scores.total_llm_calls == 12
    assert scores.total_tool_calls == 6
    assert scores.tool_counts == {"get_order": 6}
    assert scores.total_prompt_tokens == 600
    assert scores.total_completion_tokens == 120
    assert scores.total_tokens == 720
    assert scores.max_tokens_in_run == 120
    assert scores.average_llm_calls == 2.0
    assert scores.average_tool_calls == 1.0
    assert scores.average_total_tokens == 120.0
    output = capsys.readouterr().out
    assert "pass@3: 2/2 (100.0%)" in output
    assert "pass^3: 1/2 (50.0%)" in output
    assert "Tool usage: get_order=6" in output
    assert "\033[31mFAIL\033[0m" in output
    assert "\033[31mfailures: outcome 'denied' != 'refunded'\033[0m" in output


def test_red_returns_plain_text_when_color_is_disabled() -> None:
    assert eval_module.red("FAIL", enabled=False) == "FAIL"


def test_evaluate_result_enforces_tool_expectations() -> None:
    case: eval_module.EvalCase = {
        "input": "Refund order 1",
        "expected": "refunded",
        "expected_tools": ["get_order", "issue_refund"],
        "max_tool_calls": 1,
    }
    result = AgentResult(
        status=RefundOutcome.REFUNDED,
        message="done",
        metrics=AgentMetrics(
            tool_calls=2,
            tool_sequence=("issue_refund", "get_order"),
            total_tokens=150,
        ),
    )

    success, failures = eval_module.evaluate_result(case, result)

    assert not success
    assert failures == [
        "tools ['issue_refund', 'get_order'] != ['get_order', 'issue_refund']",
        "tool calls 2 > 1",
    ]


def test_evaluate_result_does_not_use_tokens_for_pass_or_fail() -> None:
    case: eval_module.EvalCase = {
        "input": "Refund order 1",
        "expected": "refunded",
        "expected_tools": ["issue_refund"],
        "max_tool_calls": 1,
    }
    result = AgentResult(
        status=RefundOutcome.REFUNDED,
        message="done",
        metrics=AgentMetrics(
            tool_calls=1,
            tool_sequence=("issue_refund",),
            total_tokens=1_000_000,
        ),
    )

    success, failures = eval_module.evaluate_result(case, result)

    assert success
    assert failures == []


def test_run_evaluations_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be at least one"):
        asyncio.run(eval_module.run_evaluations(k=0))
