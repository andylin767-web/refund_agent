import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

import refund_agent.agent as agent_module
import refund_agent.telemetry as telemetry_module
from refund_agent.agent import RefundAgent
from refund_agent.telemetry import create_gen_ai_telemetry


class FakeMessage:
    def __init__(self, content: str | None = "Done", tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, object]:
        del exclude_none
        return {"role": "assistant", "content": self.content}


class FakeToolResult(BaseModel):
    order_id: str
    status: str


def test_langsmith_otlp_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
        "LANGSMITH_WORKSPACE_ID",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "REFUND_AGENT_OTEL_EXPORT",
    ):
        monkeypatch.delenv(variable, raising=False)

    assert telemetry_module._signal_endpoint("traces") is None
    assert telemetry_module._signal_endpoint("metrics") is None
    assert telemetry_module._langsmith_headers() is None

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "refund-tests")
    monkeypatch.setenv("LANGSMITH_WORKSPACE_ID", "workspace-id")

    assert (
        telemetry_module._signal_endpoint("traces")
        == "https://api.smith.langchain.com/otel/v1/traces"
    )
    assert telemetry_module._signal_endpoint("metrics") is None
    assert telemetry_module._langsmith_headers() == {
        "x-api-key": "test-key",
        "Langsmith-Project": "refund-tests",
        "x-tenant-id": "workspace-id",
    }

    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "https://metrics.example.com/v1/metrics",
    )
    assert telemetry_module._signal_endpoint("metrics") == "https://metrics.example.com/v1/metrics"
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example.com/otlp")
    custom_endpoint = telemetry_module._signal_endpoint("traces")
    assert custom_endpoint == "https://collector.example.com/otlp/v1/traces"
    assert (
        telemetry_module._signal_endpoint("metrics")
        == "https://collector.example.com/otlp/v1/metrics"
    )
    assert telemetry_module._langsmith_headers(custom_endpoint) is None


def test_agent_emits_gen_ai_spans_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFUND_AGENT_TRACE_CONTENT", raising=False)
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    telemetry = create_gen_ai_telemetry(tracer_provider, meter_provider)

    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        assert kwargs["model"] == "openai/test-model"
        return SimpleNamespace(
            id="response-123",
            model="test-model-2026",
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
            ),
            choices=[
                SimpleNamespace(
                    message=FakeMessage(),
                    finish_reason="stop",
                )
            ],
        )

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)

    result = asyncio.run(RefundAgent("openai/test-model", telemetry=telemetry).run_agent("Hello"))

    assert result.metrics.total_tokens == 15
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["chat openai/test-model", "refund_agent"]
    model_span = spans[0]
    root_span = spans[1]
    assert root_span.context is not None
    assert model_span.parent is not None
    assert model_span.parent.span_id == root_span.context.span_id
    assert model_span.attributes is not None
    assert model_span.attributes["gen_ai.operation.name"] == "chat"
    assert model_span.attributes["gen_ai.provider.name"] == "openai"
    assert model_span.attributes["gen_ai.request.model"] == "openai/test-model"
    assert model_span.attributes["gen_ai.response.model"] == "test-model-2026"
    assert model_span.attributes["gen_ai.response.id"] == "response-123"
    assert model_span.attributes["gen_ai.usage.input_tokens"] == 12
    assert model_span.attributes["gen_ai.usage.output_tokens"] == 3
    assert "gen_ai.input.messages" not in model_span.attributes
    assert "gen_ai.output.messages" not in model_span.attributes

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    metrics_by_name: dict[str, Any] = {
        metric.name: metric
        for resource_metrics in metrics_data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    assert set(metrics_by_name) == {
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    }
    token_points = metrics_by_name["gen_ai.client.token.usage"].data.data_points
    assert metrics_by_name["gen_ai.client.token.usage"].unit == "{token}"
    token_usage = {point.attributes["gen_ai.token.type"]: point.sum for point in token_points}
    assert token_usage == {"input": 12, "output": 3}
    duration_points = metrics_by_name["gen_ai.client.operation.duration"].data.data_points
    assert metrics_by_name["gen_ai.client.operation.duration"].unit == "s"
    assert len(duration_points) == 1
    assert duration_points[0].attributes["gen_ai.response.model"] == "test-model-2026"
    assert duration_points[0].count == 1
    assert duration_points[0].sum >= 0


def test_model_span_records_opt_in_input_and_output_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFUND_AGENT_TRACE_CONTENT", "true")
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    telemetry = create_gen_ai_telemetry(tracer_provider, MeterProvider())

    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=FakeMessage("I can help."), finish_reason="stop")]
        )

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)

    asyncio.run(RefundAgent("openai/test-model", telemetry=telemetry).run_agent("Check order 123"))

    model_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "chat openai/test-model"
    )
    assert model_span.attributes is not None
    system_attribute = model_span.attributes["gen_ai.system_instructions"]
    input_attribute = model_span.attributes["gen_ai.input.messages"]
    output_attribute = model_span.attributes["gen_ai.output.messages"]
    assert isinstance(system_attribute, str)
    assert isinstance(input_attribute, str)
    assert isinstance(output_attribute, str)
    system_instructions = json.loads(system_attribute)
    input_messages = json.loads(input_attribute)
    output_messages = json.loads(output_attribute)
    assert system_instructions[0]["content"].startswith("You are Refund Agent")
    assert input_messages == [
        {"role": "user", "parts": [{"type": "text", "content": "Check order 123"}]}
    ]
    assert output_messages == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "I can help."}],
            "finish_reason": "stop",
        }
    ]


def test_trace_content_is_redacted_and_truncated() -> None:
    redacted = json.loads(agent_module._trace_json({"api_key": "secret", "value": "safe"}))
    assert redacted == {"api_key": "[REDACTED]", "value": "safe"}

    truncated_json = agent_module._trace_json('"' * 20_000)
    truncated = json.loads(truncated_json)
    assert truncated["truncated"] is True
    assert truncated["original_characters"] > 20_000
    assert len(truncated_json) <= agent_module._TRACE_CONTENT_MAX_CHARS


def test_tool_span_records_input_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    telemetry = create_gen_ai_telemetry(tracer_provider, MeterProvider())
    tool_call = SimpleNamespace(
        id="call-123",
        function=SimpleNamespace(name="get_order", arguments='{ "order_id": "123" }'),
    )
    replies = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=FakeMessage(None, [tool_call]),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=FakeMessage("Found"), finish_reason="stop")]
            ),
        ]
    )

    async def fake_completion(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return next(replies)

    monkeypatch.setattr(agent_module, "litellm_acompletion", fake_completion)
    monkeypatch.setattr(
        agent_module,
        "execute_tool",
        lambda name, arguments: FakeToolResult(order_id="123", status="delivered"),
    )

    asyncio.run(RefundAgent("openai/test-model", telemetry=telemetry).run_agent("Order 123"))

    tool_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "execute_tool get_order"
    )
    assert tool_span.attributes is not None
    assert tool_span.attributes["gen_ai.tool.type"] == "function"
    assert tool_span.attributes["gen_ai.tool.call.arguments"] == '{"order_id":"123"}'
    assert tool_span.attributes["gen_ai.tool.call.result"] == (
        '{"order_id":"123","status":"delivered"}'
    )
