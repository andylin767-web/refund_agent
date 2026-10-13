"""OpenTelemetry GenAI instrumentation with LangSmith trace export."""

import os
import threading
from dataclasses import dataclass
from typing import Final

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

INSTRUMENTATION_NAME: Final = "refund_agent"
SERVICE_NAME: Final = "refund-agent"
DEFAULT_LANGSMITH_ENDPOINT: Final = "https://api.smith.langchain.com"


@dataclass(frozen=True)
class GenAITelemetry:
    """Tracer and GenAI metric instruments used by the refund agent."""

    tracer: Tracer
    meter: Meter
    token_usage: Histogram
    operation_duration: Histogram


def create_gen_ai_telemetry(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
) -> GenAITelemetry:
    """Create GenAI instruments from caller-owned providers."""

    tracer = tracer_provider.get_tracer(INSTRUMENTATION_NAME)
    meter = meter_provider.get_meter(INSTRUMENTATION_NAME)
    return GenAITelemetry(
        tracer=tracer,
        meter=meter,
        token_usage=meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used by GenAI operations.",
        ),
        operation_duration=meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="Duration of GenAI client operations.",
        ),
    )


def _signal_endpoint(signal: str) -> str | None:
    export_enabled = os.getenv("REFUND_AGENT_OTEL_EXPORT", "true").strip().casefold()
    if export_enabled in {"0", "false", "no", "off"}:
        return None

    explicit_endpoint = os.getenv(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT", "").strip()
    if explicit_endpoint:
        return explicit_endpoint

    base_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not base_endpoint:
        if not os.getenv("LANGSMITH_API_KEY", "").strip():
            return None
        if signal != "traces":
            return None
        langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT).rstrip("/")
        base_endpoint = f"{langsmith_endpoint}/otel"

    return f"{base_endpoint.rstrip('/')}/v1/{signal}"


def _langsmith_headers(endpoint: str | None = None) -> dict[str, str] | None:
    if os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").strip():
        return None

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        return None

    langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT).rstrip("/")
    if endpoint is not None and not endpoint.startswith(f"{langsmith_endpoint}/"):
        return None

    headers = {"x-api-key": api_key}
    project = os.getenv("LANGSMITH_PROJECT", "").strip()
    workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID", "").strip()
    if project:
        headers["Langsmith-Project"] = project
    if workspace_id:
        headers["x-tenant-id"] = workspace_id
    return headers


def _configure_telemetry() -> GenAITelemetry:
    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", SERVICE_NAME),
            "service.version": "0.1.0",
        }
    )
    trace_provider = TracerProvider(resource=resource)
    traces_endpoint = _signal_endpoint("traces")
    if traces_endpoint is not None:
        trace_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=traces_endpoint,
                    headers=_langsmith_headers(traces_endpoint),
                )
            )
        )

    metric_readers: list[PeriodicExportingMetricReader] = []
    metrics_endpoint = _signal_endpoint("metrics")
    if metrics_endpoint is not None:
        metric_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=metrics_endpoint,
                    headers=_langsmith_headers(metrics_endpoint),
                )
            )
        )
    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    return create_gen_ai_telemetry(trace_provider, meter_provider)


_telemetry: GenAITelemetry | None = None
_telemetry_lock = threading.Lock()


def get_gen_ai_telemetry() -> GenAITelemetry:
    """Return the process-wide telemetry instruments, configuring them once."""

    global _telemetry
    if _telemetry is None:
        with _telemetry_lock:
            if _telemetry is None:
                _telemetry = _configure_telemetry()
    return _telemetry
