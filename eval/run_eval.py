"""Run repeated refund-agent evaluations and report reliability metrics."""

import argparse
import asyncio
import json
import os
import shutil
import sys
from collections import Counter
from contextlib import chdir
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NotRequired, TypedDict, cast

from refund_agent.agent import AgentResult, RefundAgent
from refund_agent.config import load_model_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "eval" / "eval.json"
ORDERS_PATH = PROJECT_ROOT / "orders.json"
DEFAULT_K = 3
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


def terminal_supports_color() -> bool:
    """Return whether evaluation output should use ANSI colors."""

    return (
        sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.getenv("TERM", "").casefold() != "dumb"
    )


def red(text: str, *, enabled: bool) -> str:
    """Render text in red when terminal coloring is enabled."""

    return f"{ANSI_RED}{text}{ANSI_RESET}" if enabled else text


class EvalCase(TypedDict):
    input: str
    expected: str
    expected_tools: NotRequired[list[str]]
    max_tool_calls: NotRequired[int]


@dataclass(frozen=True)
class EvalScores:
    """Aggregate counts and rates from a repeated evaluation run."""

    total_cases: int
    attempts_per_case: int
    passed_attempts: int
    pass_at_k_cases: int
    pass_power_k_cases: int
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    max_tokens_in_run: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_attempts(self) -> int:
        """Return the number of individual agent runs."""

        return self.total_cases * self.attempts_per_case

    @property
    def single_run_accuracy(self) -> float:
        """Return accuracy across all individual runs."""

        return self.passed_attempts / self.total_attempts if self.total_attempts else 0.0

    @property
    def pass_at_k(self) -> float:
        """Return the fraction of cases with at least one passing run."""

        return self.pass_at_k_cases / self.total_cases if self.total_cases else 0.0

    @property
    def pass_power_k(self) -> float:
        """Return the fraction of cases for which every run passed."""

        return self.pass_power_k_cases / self.total_cases if self.total_cases else 0.0

    @property
    def average_llm_calls(self) -> float:
        """Return mean model calls per individual run."""

        return self.total_llm_calls / self.total_attempts if self.total_attempts else 0.0

    @property
    def average_tool_calls(self) -> float:
        """Return mean tool calls per individual run."""

        return self.total_tool_calls / self.total_attempts if self.total_attempts else 0.0

    @property
    def average_total_tokens(self) -> float:
        """Return mean total tokens per individual run."""

        return self.total_tokens / self.total_attempts if self.total_attempts else 0.0


def load_cases() -> list[EvalCase]:
    """Load evaluation cases from disk."""

    with CASES_PATH.open(encoding="utf-8") as file:
        cases: object = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("eval.json must contain a JSON list")
    return cast(list[EvalCase], cases)


def evaluate_result(case: EvalCase, result: AgentResult) -> tuple[bool, list[str]]:
    """Check one result against its outcome, tool, and optional budget expectations."""

    failures: list[str] = []
    actual = result.status.value if result.status is not None else None
    if actual != case["expected"]:
        failures.append(f"outcome {actual!r} != {case['expected']!r}")

    expected_tools = case.get("expected_tools")
    if expected_tools is not None and list(result.metrics.tool_sequence) != expected_tools:
        failures.append(f"tools {list(result.metrics.tool_sequence)!r} != {expected_tools!r}")

    max_tool_calls = case.get("max_tool_calls")
    if max_tool_calls is not None and result.metrics.tool_calls > max_tool_calls:
        failures.append(f"tool calls {result.metrics.tool_calls} > {max_tool_calls}")

    return not failures, failures


async def run_evaluations(k: int = DEFAULT_K, *, color: bool | None = None) -> EvalScores:
    """Run each case ``k`` times and return aggregate reliability scores."""

    if k < 1:
        raise ValueError("k must be at least one")

    use_color = terminal_supports_color() if color is None else color

    cases = load_cases()
    agent = RefundAgent(model=load_model_id())
    passed_attempts = 0
    pass_at_k_cases = 0
    pass_power_k_cases = 0
    total_llm_calls = 0
    total_tool_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    max_tokens_in_run = 0
    tool_counts: Counter[str] = Counter()

    for case_index, case in enumerate(cases, start=1):
        attempt_results: list[bool] = []

        for attempt in range(1, k + 1):
            # Every attempt receives a fresh order store. This prevents one
            # refund from changing later attempts or evaluation cases.
            with TemporaryDirectory() as temporary_directory:
                working_directory = Path(temporary_directory)
                shutil.copy2(ORDERS_PATH, working_directory / "orders.json")

                with chdir(working_directory):
                    result = await agent.run_agent(case["input"])

            actual = result.status.value if result.status is not None else None
            success, failures = evaluate_result(case, result)
            attempt_results.append(success)
            passed_attempts += int(success)
            total_llm_calls += result.metrics.llm_calls
            total_tool_calls += result.metrics.tool_calls
            total_prompt_tokens += result.metrics.prompt_tokens
            total_completion_tokens += result.metrics.completion_tokens
            total_tokens += result.metrics.total_tokens
            max_tokens_in_run = max(max_tokens_in_run, result.metrics.total_tokens)
            tool_counts.update(result.metrics.tool_counts)

            marker = "PASS" if success else red("FAIL", enabled=use_color)
            tool_sequence = " -> ".join(result.metrics.tool_sequence) or "none"
            failure_message = f"failures: {'; '.join(failures)}"
            failure_text = f"\n  {red(failure_message, enabled=use_color)}" if failures else ""
            print(
                f"[{marker}] {case_index}.{attempt}/{k} {case['input']}\n"
                f"  expected: {case['expected']}\n"
                f"  actual:   {actual}\n"
                f"  tools:    {tool_sequence} ({result.metrics.tool_calls} calls)\n"
                f"  LLM calls: {result.metrics.llm_calls}\n"
                f"  tokens: prompt={result.metrics.prompt_tokens}, "
                f"completion={result.metrics.completion_tokens}, "
                f"total={result.metrics.total_tokens}\n"
                f"  response: {result.message}{failure_text}"
            )

        case_pass_at_k = any(attempt_results)
        case_pass_power_k = all(attempt_results)
        pass_at_k_cases += int(case_pass_at_k)
        pass_power_k_cases += int(case_pass_power_k)
        pass_at_k_label = "PASS" if case_pass_at_k else red("FAIL", enabled=use_color)
        pass_power_k_label = "PASS" if case_pass_power_k else red("FAIL", enabled=use_color)
        print(f"  case metrics: pass@{k}={pass_at_k_label}, pass^{k}={pass_power_k_label}")

    scores = EvalScores(
        total_cases=len(cases),
        attempts_per_case=k,
        passed_attempts=passed_attempts,
        pass_at_k_cases=pass_at_k_cases,
        pass_power_k_cases=pass_power_k_cases,
        total_llm_calls=total_llm_calls,
        total_tool_calls=total_tool_calls,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        max_tokens_in_run=max_tokens_in_run,
        tool_counts=dict(tool_counts),
    )
    tool_usage = (
        ", ".join(f"{tool_name}={count}" for tool_name, count in sorted(scores.tool_counts.items()))
        or "none"
    )
    print(
        "\nEvaluation summary\n"
        f"  Single-run accuracy: {scores.passed_attempts}/{scores.total_attempts} "
        f"({scores.single_run_accuracy:.1%})\n"
        f"  pass@{k}: {scores.pass_at_k_cases}/{scores.total_cases} "
        f"({scores.pass_at_k:.1%})\n"
        f"  pass^{k}: {scores.pass_power_k_cases}/{scores.total_cases} "
        f"({scores.pass_power_k:.1%})\n"
        f"  Average LLM calls: {scores.average_llm_calls:.2f}\n"
        f"  Average tool calls: {scores.average_tool_calls:.2f}\n"
        f"  Average total tokens: {scores.average_total_tokens:.1f}\n"
        f"  Tokens: prompt={scores.total_prompt_tokens}, "
        f"completion={scores.total_completion_tokens}, total={scores.total_tokens}, "
        f"max/run={scores.max_tokens_in_run}\n"
        f"  Tool usage: {tool_usage}"
    )
    return scores


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"runs per evaluation case (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="colorize failures (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluations from the command line."""

    arguments = parse_args()
    color = None if arguments.color == "auto" else arguments.color == "always"
    asyncio.run(run_evaluations(k=arguments.k, color=color))


if __name__ == "__main__":
    main()
