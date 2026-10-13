"""LiteLLM-powered refund agent and local tool dispatcher."""

import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

import litellm
from litellm import acompletion as litellm_acompletion
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field

from refund_agent.order import OrderStatus
from refund_agent.order_service import OrderNotFoundError
from refund_agent.system_message import REFUND_AGENT_SYSTEM_MESSAGE
from refund_agent.telemetry import GenAITelemetry, get_gen_ai_telemetry
from refund_agent.tools import (
    GetOrderInput,
    IssueRefundInput,
    OrderOutput,
    RefundOutcome,
    RefundOutput,
    get_order,
    issue_refund,
)

litellm.suppress_debug_info = True

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Load an order by its exact order ID.",
            "parameters": GetOrderInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Apply the refund policy and record an eligible mock refund.",
            "parameters": IssueRefundInput.model_json_schema(),
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., BaseModel]] = {
    "get_order": get_order,
    "issue_refund": issue_refund,
}

_REFUND_INTENT_PATTERN = re.compile(r"\brefund(?:ed|ing)?\b", re.IGNORECASE)
_ORDER_ID_PATTERN = re.compile(
    r"(?:\border(?:\s+id)?\s*(?::|#|is)?|\brefund\s*(?::|#)?)\s*(\d+)\b",
    re.IGNORECASE,
)
_SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?:api[-_]?key|authorization|password|secret|token)", re.IGNORECASE
)
_TRACE_CONTENT_MAX_CHARS = 16_384


def _trace_content_enabled() -> bool:
    """Return whether potentially sensitive model content may be traced."""

    return os.getenv("REFUND_AGENT_TRACE_CONTENT", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_json_value(value: object) -> object:
    """Deserialize JSON strings when possible for structured trace content."""

    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _redact_trace_value(value: object) -> object:
    """Redact values stored under common credential field names."""

    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _SENSITIVE_FIELD_PATTERN.search(str(key))
                else _redact_trace_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_trace_value(item) for item in value]
    return value


def _trace_json(value: object) -> str:
    """Serialize redacted trace content and bound the resulting attribute size."""

    serialized = json.dumps(
        _redact_trace_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    if len(serialized) <= _TRACE_CONTENT_MAX_CHARS:
        return serialized

    # The preview is embedded in another JSON string, where quotes and
    # backslashes can double in size. Half the limit leaves room for escaping
    # and truncation metadata while keeping the final attribute bounded.
    preview_length = (_TRACE_CONTENT_MAX_CHARS - 200) // 2
    return json.dumps(
        {
            "truncated": True,
            "original_characters": len(serialized),
            "preview": serialized[:preview_length],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _gen_ai_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style message to the GenAI message schema."""

    role = str(message.get("role", "unknown"))
    parts: list[dict[str, Any]] = []
    content = message.get("content")
    if content is not None:
        parts.append({"type": "text", "content": content})

    if role == "tool":
        parts = [
            {
                "type": "tool_call_response",
                "id": message.get("tool_call_id", ""),
                "result": _parse_json_value(content),
            }
        ]
    else:
        for tool_call in message.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            parts.append(
                {
                    "type": "tool_call",
                    "id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": _parse_json_value(function.get("arguments")),
                }
            )
    return {"role": role, "parts": parts}


def _message_dict(message: object) -> dict[str, Any]:
    """Return a serializable dictionary for a provider message."""

    if isinstance(message, dict):
        return message
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return {"role": "unknown", "content": str(message)}


def _set_input_content_attributes(span: Any, messages: list[dict[str, Any]]) -> None:
    """Attach system instructions and input messages to a model span."""

    system_instructions: list[dict[str, object]] = []
    input_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            system_instructions.append({"type": "text", "content": message.get("content", "")})
        else:
            input_messages.append(_gen_ai_message(message))
    if system_instructions:
        span.set_attribute("gen_ai.system_instructions", _trace_json(system_instructions))
    if input_messages:
        span.set_attribute("gen_ai.input.messages", _trace_json(input_messages))


def _set_output_content_attribute(span: Any, response: Any) -> None:
    """Attach provider output choices to a model span."""

    output_messages: list[dict[str, Any]] = []
    for choice in response.choices:
        output_message = _gen_ai_message(_message_dict(choice.message))
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            output_message["finish_reason"] = str(finish_reason)
        output_messages.append(output_message)
    if output_messages:
        span.set_attribute("gen_ai.output.messages", _trace_json(output_messages))


def refund_request_needs_order_id(user_message: str) -> bool:
    """Return whether a refund request lacks an explicit numeric order ID."""

    return bool(_REFUND_INTENT_PATTERN.search(user_message)) and not bool(
        _ORDER_ID_PATTERN.search(user_message)
    )


class ToolErrorOutput(BaseModel):
    """Structured error returned when a tool cannot be executed."""

    model_config = ConfigDict(frozen=True)

    error: str
    outcome: RefundOutcome | None = None


def execute_tool(name: str, arguments: str) -> BaseModel:
    """Validate and execute one model-requested local tool."""

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return ToolErrorOutput(error=f"unknown tool: {name}")

    try:
        parsed_arguments: dict[str, Any] = json.loads(arguments)
        return handler(**parsed_arguments)
    except OrderNotFoundError as error:
        return ToolErrorOutput(
            error=str(error),
            outcome=RefundOutcome.ORDER_NOT_FOUND,
        )
    except (json.JSONDecodeError, LookupError, OSError, TypeError, ValueError) as error:
        return ToolErrorOutput(error=str(error))


def _tool_arguments_json(arguments: str) -> str:
    """Return compact JSON for a tool span, preserving invalid input verbatim."""

    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return json.dumps(parsed_arguments, separators=(",", ":"), sort_keys=True)


class AgentMetrics(BaseModel):
    """Observable LLM and tool usage from one agent run."""

    model_config = ConfigDict(frozen=True)

    llm_calls: int = 0
    tool_calls: int = 0
    tool_counts: dict[str, int] = Field(default_factory=dict)
    tool_sequence: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentResult(BaseModel):
    """Structured status and user-facing response from one agent run."""

    model_config = ConfigDict(frozen=True)

    status: RefundOutcome | None
    message: str
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)


def _token_count(usage: object, field: str) -> int:
    """Read one non-negative token count from a LiteLLM usage object."""

    value = getattr(usage, field, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _optional_token_count(usage: object, field: str) -> int | None:
    """Read a token count while preserving the difference between zero and missing."""

    value = getattr(usage, field, None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _provider_name(model: str) -> str:
    """Return the low-cardinality LiteLLM provider prefix when available."""

    provider, separator, _ = model.partition("/")
    return provider if separator else "unknown"


class RefundAgent:
    """Run a bounded LiteLLM conversation with refund tools."""

    def __init__(
        self,
        model: str,
        max_attempts: int = 5,
        *,
        telemetry: GenAITelemetry | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.model = model
        self.max_attempts = max_attempts
        self.telemetry = telemetry or get_gen_ai_telemetry()

    async def run_agent(self, user_message: str) -> AgentResult:
        """Run one traced refund-agent interaction."""

        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": _provider_name(self.model),
            "gen_ai.request.model": self.model,
            "langsmith.span.kind": "chain",
            "langsmith.trace.name": "refund_agent",
        }
        with self.telemetry.tracer.start_as_current_span(
            "refund_agent",
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        ) as span:
            result = await self._run_agent(user_message)
            if result.status is not None:
                span.set_attribute("refund.outcome", result.status.value)
            span.set_attribute("refund_agent.llm_calls", result.metrics.llm_calls)
            span.set_attribute("refund_agent.tool_calls", result.metrics.tool_calls)
            span.set_attribute("refund_agent.total_tokens", result.metrics.total_tokens)
            return result

    async def _complete(self, messages: list[dict[str, Any]]) -> Any:
        """Call LiteLLM and emit one GenAI client span and metric measurements."""

        attributes: dict[str, str] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": _provider_name(self.model),
            "gen_ai.request.model": self.model,
            "langsmith.span.kind": "llm",
        }
        metric_attributes = dict(attributes)
        metric_attributes.pop("langsmith.span.kind")
        started_at = time.perf_counter()

        with self.telemetry.tracer.start_as_current_span(
            f"chat {self.model}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
        ) as span:
            trace_content = _trace_content_enabled()
            if trace_content:
                _set_input_content_attributes(span, messages)
            try:
                response = await litellm_acompletion(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as error:
                error_type = type(error).__qualname__
                metric_attributes["error.type"] = error_type
                span.set_attribute("error.type", error_type)
                span.set_status(Status(StatusCode.ERROR, str(error)))
                self.telemetry.operation_duration.record(
                    time.perf_counter() - started_at,
                    attributes=metric_attributes,
                )
                raise

            response_model = getattr(response, "model", None)
            if isinstance(response_model, str) and response_model:
                span.set_attribute("gen_ai.response.model", response_model)
                metric_attributes["gen_ai.response.model"] = response_model

            response_id = getattr(response, "id", None)
            if isinstance(response_id, str) and response_id:
                span.set_attribute("gen_ai.response.id", response_id)

            finish_reasons = [
                str(choice.finish_reason)
                for choice in response.choices
                if getattr(choice, "finish_reason", None) is not None
            ]
            if finish_reasons:
                span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
            if trace_content:
                _set_output_content_attribute(span, response)

            usage = getattr(response, "usage", None)
            input_tokens = _optional_token_count(usage, "prompt_tokens")
            output_tokens = _optional_token_count(usage, "completion_tokens")
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                self.telemetry.token_usage.record(
                    input_tokens,
                    attributes={**metric_attributes, "gen_ai.token.type": "input"},
                )
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                self.telemetry.token_usage.record(
                    output_tokens,
                    attributes={**metric_attributes, "gen_ai.token.type": "output"},
                )
            self.telemetry.operation_duration.record(
                time.perf_counter() - started_at,
                attributes=metric_attributes,
            )
            return response

    async def _run_agent(self, user_message: str) -> AgentResult:
        """Run until the model returns a structured result or attempts expire."""

        if refund_request_needs_order_id(user_message):
            return AgentResult(
                status=RefundOutcome.NEEDS_ORDER_ID,
                message="Please provide the order ID you want to refund.",
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REFUND_AGENT_SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        refund_requested = bool(_REFUND_INTENT_PATTERN.search(user_message))
        status: RefundOutcome | None = None
        llm_calls = 0
        tool_counts: dict[str, int] = {}
        tool_sequence: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for llm_calls in range(1, self.max_attempts + 1):
            response = await self._complete(messages)
            usage = getattr(response, "usage", None)
            call_prompt_tokens = _token_count(usage, "prompt_tokens")
            call_completion_tokens = _token_count(usage, "completion_tokens")
            call_total_tokens = _token_count(usage, "total_tokens")
            prompt_tokens += call_prompt_tokens
            completion_tokens += call_completion_tokens
            total_tokens += call_total_tokens or call_prompt_tokens + call_completion_tokens
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump(exclude_none=True))

            if not assistant_message.tool_calls:
                return AgentResult(
                    status=status,
                    message=assistant_message.content or "",
                    metrics=AgentMetrics(
                        llm_calls=llm_calls,
                        tool_calls=len(tool_sequence),
                        tool_counts=tool_counts,
                        tool_sequence=tuple(tool_sequence),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    ),
                )

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_arguments = tool_call.function.arguments
                tool_sequence.append(tool_name)
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                with self.telemetry.tracer.start_as_current_span(
                    f"execute_tool {tool_name}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": tool_name,
                        "gen_ai.tool.type": "function",
                        "gen_ai.tool.call.id": tool_call.id,
                        "gen_ai.tool.call.arguments": _tool_arguments_json(tool_arguments),
                        "langsmith.span.kind": "tool",
                    },
                ) as tool_span:
                    tool_result = execute_tool(
                        tool_name,
                        tool_arguments,
                    )
                    tool_result_json = tool_result.model_dump_json(exclude_none=True)
                    tool_span.set_attribute("gen_ai.tool.call.result", tool_result_json)
                    if isinstance(tool_result, ToolErrorOutput):
                        error_type = (
                            tool_result.outcome.value
                            if tool_result.outcome is not None
                            else "tool_error"
                        )
                        tool_span.set_attribute("error.type", error_type)
                        tool_span.set_status(Status(StatusCode.ERROR, tool_result.error))
                if isinstance(tool_result, RefundOutput):
                    status = tool_result.outcome
                elif (
                    refund_requested
                    and isinstance(tool_result, ToolErrorOutput)
                    and tool_result.outcome is RefundOutcome.ORDER_NOT_FOUND
                ):
                    status = RefundOutcome.ORDER_NOT_FOUND
                elif (
                    refund_requested
                    and isinstance(tool_result, OrderOutput)
                    and tool_result.status is OrderStatus.REFUNDED
                ):
                    status = RefundOutcome.ALREADY_REFUNDED
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result_json,
                    }
                )

        raise RuntimeError(f"refund agent exceeded its maximum of {self.max_attempts} attempts")
