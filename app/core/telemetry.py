from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import structlog
from app.core.config import settings

log = structlog.get_logger()

def setup_telemetry():
    resource = Resource.create(attributes={"service.name": settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    if settings.OTEL_ENABLED and settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            processor = BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
            )
            provider.add_span_processor(processor)
            log.info("OpenTelemetry tracing enabled", endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        except Exception as e:
            log.warning("Failed to setup OTEL exporter, tracing disabled", error=str(e))
    else:
        log.info("OpenTelemetry tracing disabled (OTEL_ENABLED=False)")

    trace.set_tracer_provider(provider)
