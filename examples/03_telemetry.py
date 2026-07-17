"""Telemetry: view Damper's OpenTelemetry spans with a local console exporter.

Run::

    python examples/03_telemetry.py

Damper depends only on ``opentelemetry-api`` at runtime; when no OpenTelemetry
SDK/exporter is configured, its spans are a safe no-op. To *see* the spans you
configure an SDK yourself. ``opentelemetry-sdk`` is NOT a Damper runtime
dependency -- it ships here as a dev extra. If it is not installed this example
prints guidance and exits successfully.

The console exporter writes span JSON to stdout only; there is no network. The
request is served by a tiny always-successful in-file fake, so no API key is
needed.
"""

from __future__ import annotations

import sys
from typing import Any


class _FakeResponse:
    """Attribute-accepting stand-in for an Anthropic ``Message``."""

    def __init__(self) -> None:
        self.id = "telemetry-demo"


class _OkMessages:
    def __init__(self, provider: _OkAnthropic) -> None:
        self._provider = provider

    def create(self, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse()


class _OkAnthropic:
    """Always-successful fake client (retry-ownership seam + one create)."""

    def __init__(self) -> None:
        self.messages = _OkMessages(self)

    def with_options(self, **kwargs: Any) -> _OkAnthropic:
        return self


def main() -> int:
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        print(
            "This example needs an OpenTelemetry SDK to display spans.\n"
            "opentelemetry-sdk is NOT a Damper runtime dependency; install the\n"
            "dev extra to run this demo:\n\n"
            "    pip install -e \".[dev]\"    # or: pip install opentelemetry-sdk\n\n"
            "Without an SDK configured, Damper's telemetry is a safe no-op."
        )
        return 0

    from opentelemetry import trace

    from damper import Policy, resilient

    # Local SDK setup: this is the user's responsibility, not Damper's. The
    # console exporter prints spans to stdout; nothing leaves the process.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    client = resilient(_OkAnthropic(), policy=Policy())
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16,
        messages=[{"role": "user", "content": "hello"}],
    )

    print(f"\nResponse served; damper.outcome={resp.damper.outcome}.")
    print(
        "Above: one damper.request span and its child damper.attempt span, with "
        "damper.* attributes (and gen_ai.* compatibility mappings)."
    )
    provider.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
