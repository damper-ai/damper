"""Telemetry tests (SESSION 7, SPEC section 18).

Uses a LOCAL ``TracerProvider`` + ``SimpleSpanProcessor`` +
``InMemorySpanExporter`` injected through ``telemetry.make_trace_factory`` and
the executor's internal ``trace_factory`` seam. No global tracer-provider
mutation, no ``set_tracer_provider``, no network.

Proves: request span parents attempt spans; the attempt span wraps the real
provider call; backoff is outside the attempt span but inside the request span;
the SPEC attributes and outcomes are emitted; unknown provider/model is omitted
safely; and -- crucially -- telemetry never changes retry behavior (identical
provider-attempt counts and results/exceptions with telemetry recording,
disabled, or deliberately failing).
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from damper import (
    Policy,
    RetriesExhausted,
    RetryBudgetExhausted,
    RetryCostCeilingHit,
    resilient,
    telemetry,
)
from damper._executor import execute, execute_async
from damper.budget import RetryBudget
from tests.fakes import (
    Err,
    FakeAnthropic,
    FakeAsyncAnthropic,
    FakeMessage,
    FakeProviderError,
    Ok,
    ScriptedFake,
    StreamThenReset,
)

FAST = Policy(backoff_base=0.0, backoff_max=0.0)
REQ: dict[str, Any] = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class RngCounter:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.value


def new_budget(*, ratio: float = 0.1, min_tokens: int = 10) -> RetryBudget:
    return RetryBudget(
        ratio=ratio, min_tokens=min_tokens, window=60.0, clock=lambda: 0.0
    )


def local_recording() -> tuple[Any, InMemorySpanExporter]:
    """Build a LOCAL provider + in-memory exporter and a bound trace factory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    factory = telemetry.make_trace_factory(provider.get_tracer("damper.test"))
    return factory, exporter


def run_execute(
    sequence: list[Any],
    *,
    factory: Any,
    policy: Policy = FAST,
    provider: str | None = "anthropic",
    request: dict[str, Any] | None = None,
    budget: RetryBudget | None = None,
    sleep: Any = None,
    rng: RngCounter | None = None,
    clock: Any = None,
) -> Any:
    fake = ScriptedFake.from_sequence(list(sequence))
    result = execute(
        lambda: fake.next_outcome(),
        policy=policy,
        budget=budget if budget is not None else new_budget(),
        request=REQ if request is None else request,
        sleep=sleep if sleep is not None else (lambda s: None),
        clock=clock if clock is not None else FakeClock(),
        rng=rng if rng is not None else RngCounter(),
        provider=provider,
        trace_factory=factory,
    )
    return result, fake


def spans_by_name(exporter: InMemorySpanExporter, name: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


# --------------------------------------------------------------------------- #
# No SDK configured
# --------------------------------------------------------------------------- #


def test_no_sdk_configured_is_safe_noop() -> None:
    # The default factory uses the global tracer; with no provider configured it
    # is a non-recording no-op. The request must still succeed normally.
    result, fake = run_execute(
        [Ok("r")], factory=telemetry.default_trace_factory
    )
    assert result.response == "r"
    assert fake.calls == 1


# --------------------------------------------------------------------------- #
# Span structure
# --------------------------------------------------------------------------- #


def test_request_span_parents_attempt_spans() -> None:
    factory, exporter = local_recording()
    run_execute([Err(529), Err(529), Ok("r")], factory=factory)

    req = spans_by_name(exporter, "damper.request")
    attempts = spans_by_name(exporter, "damper.attempt")
    assert len(req) == 1
    assert len(attempts) == 3
    for att in attempts:
        assert att.parent is not None
        assert att.parent.span_id == req[0].context.span_id


def test_attempt_span_wraps_the_provider_call() -> None:
    # An attempt that takes real time yields a non-trivial attempt-span duration,
    # proving the span brackets the actual provider call (not created after it).
    factory, exporter = local_recording()

    def slow_attempt() -> Any:
        time.sleep(0.01)
        return "r"

    execute(
        slow_attempt,
        policy=FAST,
        budget=new_budget(),
        request=REQ,
        sleep=lambda s: None,
        clock=time.monotonic,
        rng=RngCounter(),
        provider="anthropic",
        trace_factory=factory,
    )
    att = spans_by_name(exporter, "damper.attempt")[0]
    duration_s = (att.end_time - att.start_time) / 1e9
    assert duration_s >= 0.01


def test_backoff_is_outside_attempt_span_but_inside_request_span() -> None:
    factory, exporter = local_recording()
    names_at_sleep: list[list[str]] = []

    def sleep_fn(_seconds: float) -> None:
        # At the backoff sleep the just-failed attempt span has ended (exported)
        # while the request span is still open (not yet exported).
        names_at_sleep.append(sorted(s.name for s in exporter.get_finished_spans()))

    run_execute([Err(529), Ok("r")], factory=factory, sleep=sleep_fn)
    assert names_at_sleep == [["damper.attempt"]]


# --------------------------------------------------------------------------- #
# Attributes and outcomes
# --------------------------------------------------------------------------- #


def test_success_attributes_present() -> None:
    factory, exporter = local_recording()
    run_execute([Ok("r")], factory=factory)

    req = spans_by_name(exporter, "damper.request")[0]
    attrs = req.attributes
    assert attrs["damper.outcome"] == "ok"
    assert attrs["damper.attempts"] == 1
    assert attrs["damper.provider"] == "anthropic"
    assert attrs["damper.model"] == "claude-sonnet-4-6"
    assert "damper.retry_budget.balance" in attrs
    assert "damper.retry_budget.ratio" in attrs
    assert attrs["damper.tokens.output"] == 16
    assert "damper.cost.estimate_usd" in attrs
    # GenAI compatibility mappings.
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert attrs["gen_ai.operation.name"] == "chat"

    att = spans_by_name(exporter, "damper.attempt")[0]
    assert att.attributes["damper.attempt.n"] == 1
    assert "damper.attempt.latency_s" in att.attributes


def test_retry_attempt_attributes_include_classification_and_backoff() -> None:
    factory, exporter = local_recording()
    run_execute([Err(529), Ok("r")], factory=factory)

    attempts = spans_by_name(exporter, "damper.attempt")
    first = attempts[0].attributes
    assert first["damper.attempt.error_class"] == "retryable"
    assert first["damper.attempt.status_code"] == 529
    assert first["damper.attempt.backoff_s"] == 0.0  # FAST policy


def test_retries_exhausted_outcome() -> None:
    factory, exporter = local_recording()
    with pytest.raises(RetriesExhausted):
        run_execute([Err(529), Err(529), Err(529)], factory=factory)
    req = spans_by_name(exporter, "damper.request")[0]
    assert req.attributes["damper.outcome"] == "retries_exhausted"


def test_budget_exhausted_outcome() -> None:
    factory, exporter = local_recording()
    with pytest.raises(RetryBudgetExhausted):
        run_execute(
            [Err(529), Ok("r")],
            factory=factory,
            budget=new_budget(ratio=0.0, min_tokens=0),
        )
    req = spans_by_name(exporter, "damper.request")[0]
    assert req.attributes["damper.outcome"] == "budget_exhausted"


def test_cost_ceiling_outcome() -> None:
    factory, exporter = local_recording()
    policy = Policy(backoff_base=0.0, backoff_max=0.0, max_retry_cost_usd=0.0000001)
    with pytest.raises(RetryCostCeilingHit):
        run_execute([Err(529), Ok("r")], factory=factory, policy=policy)
    req = spans_by_name(exporter, "damper.request")[0]
    assert req.attributes["damper.outcome"] == "cost_ceiling"


def test_not_retryable_outcome() -> None:
    factory, exporter = local_recording()
    with pytest.raises(FakeProviderError):
        run_execute([Err(400)], factory=factory)
    req = spans_by_name(exporter, "damper.request")[0]
    assert req.attributes["damper.outcome"] == "not_retryable"


def test_unknown_provider_and_model_are_omitted() -> None:
    factory, exporter = local_recording()
    run_execute(
        [Ok("r")],
        factory=factory,
        provider=None,
        request={"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    )
    req = spans_by_name(exporter, "damper.request")[0]
    assert "damper.provider" not in req.attributes
    assert "damper.model" not in req.attributes
    # No priced model -> no cost estimate, but tokens still estimated.
    assert "damper.cost.estimate_usd" not in req.attributes
    assert req.attributes["damper.tokens.output"] == 8


# --------------------------------------------------------------------------- #
# Telemetry must not change retry behavior
# --------------------------------------------------------------------------- #


def test_enabled_and_disabled_have_identical_attempt_counts_and_result() -> None:
    factory, _ = local_recording()
    enabled, fake_on = run_execute([Err(529), Ok("r")], factory=factory)
    disabled, fake_off = run_execute(
        [Err(529), Ok("r")], factory=telemetry.noop_trace_factory
    )
    assert fake_on.calls == fake_off.calls == 2
    assert enabled.response == disabled.response == "r"
    assert enabled.metadata.attempts == disabled.metadata.attempts == 2
    assert enabled.metadata.outcome == disabled.metadata.outcome == "ok"


def test_provider_exception_remains_visible_with_recording() -> None:
    factory, exporter = local_recording()
    with pytest.raises(FakeProviderError) as excinfo:
        run_execute([Err(400)], factory=factory)
    assert excinfo.value.status_code == 400


# --------------------------------------------------------------------------- #
# Defensive isolation: failing tracer / span never changes behavior
# --------------------------------------------------------------------------- #


class _FailingTracer:
    def start_span(self, name: str, context: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("boom: start_span")


class _FailingSpan:
    def __init__(self, *, fail_on: str) -> None:
        self._fail_on = fail_on

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key: str, value: Any) -> None:
        if self._fail_on == "set_attribute":
            raise RuntimeError("boom: set_attribute")

    def set_status(self, status: Any) -> None:
        pass

    def get_span_context(self) -> Any:
        from opentelemetry.trace import INVALID_SPAN_CONTEXT

        return INVALID_SPAN_CONTEXT

    def end(self) -> None:
        if self._fail_on == "end":
            raise RuntimeError("boom: end")


class _FailingSpanTracer:
    def __init__(self, *, fail_on: str) -> None:
        self._fail_on = fail_on

    def start_span(self, name: str, context: Any = None, **kwargs: Any) -> Any:
        return _FailingSpan(fail_on=self._fail_on)


def _factory_for(tracer: Any) -> Any:
    return telemetry.make_trace_factory(tracer)


@pytest.mark.parametrize(
    "tracer",
    [
        _FailingTracer(),
        _FailingSpanTracer(fail_on="set_attribute"),
        _FailingSpanTracer(fail_on="end"),
    ],
)
def test_failing_telemetry_does_not_change_success(tracer: Any) -> None:
    result, fake = run_execute(
        [Err(529), Ok("r")], factory=_factory_for(tracer)
    )
    assert result.response == "r"
    assert fake.calls == 2
    assert result.metadata.outcome == "ok"


@pytest.mark.parametrize(
    "tracer",
    [
        _FailingTracer(),
        _FailingSpanTracer(fail_on="set_attribute"),
        _FailingSpanTracer(fail_on="end"),
    ],
)
def test_failing_telemetry_preserves_provider_exception(tracer: Any) -> None:
    with pytest.raises(FakeProviderError) as excinfo:
        run_execute([Err(400)], factory=_factory_for(tracer))
    assert excinfo.value.status_code == 400


# --------------------------------------------------------------------------- #
# Async parity
# --------------------------------------------------------------------------- #


async def _run_async(
    sequence: list[Any], *, factory: Any, policy: Policy = FAST
) -> Any:
    fake = ScriptedFake.from_sequence(list(sequence))

    async def attempt() -> Any:
        return fake.next_outcome()

    async def sleep(_s: float) -> None:
        return None

    result = await execute_async(
        attempt,
        policy=policy,
        budget=new_budget(),
        request=REQ,
        sleep=sleep,
        clock=FakeClock(),
        rng=RngCounter(),
        provider="anthropic",
        trace_factory=factory,
    )
    return result, fake


async def test_async_request_span_parents_attempt_spans() -> None:
    factory, exporter = local_recording()
    await _run_async([Err(529), Ok("r")], factory=factory)

    req = spans_by_name(exporter, "damper.request")
    attempts = spans_by_name(exporter, "damper.attempt")
    assert len(req) == 1
    assert len(attempts) == 2
    assert req[0].attributes["damper.outcome"] == "ok"
    for att in attempts:
        assert att.parent.span_id == req[0].context.span_id


async def test_async_failing_tracer_does_not_change_behavior() -> None:
    result, fake = await _run_async(
        [Err(529), Ok("r")], factory=_factory_for(_FailingTracer())
    )
    assert result.response == "r"
    assert fake.calls == 2


# --------------------------------------------------------------------------- #
# Streaming span boundary (SPEC section 15 + the v0.1 boundary documented in
# damper/telemetry.py). The request span covers retry-controlled establishment
# only; it is ended before the proxy reaches the caller.
# --------------------------------------------------------------------------- #


def install_local_tracer(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Route the default trace factory to a LOCAL provider for wrapper tests.

    ``resilient()`` deliberately does not expose the executor's ``trace_factory``
    seam, so the streaming path always builds its recorder from
    ``telemetry.default_trace_factory``, which resolves its tracer lazily via
    ``telemetry._get_tracer``. Patching that private accessor keeps the provider
    local to this test: it never calls ``set_tracer_provider`` and never mutates
    process-global OpenTelemetry state.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("damper.test")
    monkeypatch.setattr(telemetry, "_get_tracer", lambda: tracer)
    return exporter


def test_streaming_request_span_closes_before_proxy_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = install_local_tracer(monkeypatch)
    client = FakeAnthropic([Ok(FakeMessage(id="final"))])
    wrapped = resilient(client, policy=FAST)

    with wrapped.messages.stream(**REQ) as stream:
        # The proxy is in hand and nothing has been consumed yet, so a closed
        # request span here proves the span ended at establishment. "ok" means
        # establishment succeeded -- not that the caller consumed the stream.
        req = spans_by_name(exporter, "damper.request")
        assert len(req) == 1
        assert req[0].end_time is not None
        assert req[0].attributes["damper.outcome"] == "ok"
        end_time_at_enter = req[0].end_time
        spans_at_enter = len(exporter.get_finished_spans())

        events = list(stream)
        final = stream.get_final_message()

    assert "content_block_delta" in [getattr(e, "type", None) for e in events]
    assert final.id == "final"

    # Consuming the stream did not reopen, extend, or emit another request span.
    req_after = spans_by_name(exporter, "damper.request")
    assert len(req_after) == 1
    assert req_after[0].end_time == end_time_at_enter
    assert len(exporter.get_finished_spans()) == spans_at_enter


def test_streaming_abandoned_stream_emits_no_additional_request_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = install_local_tracer(monkeypatch)
    client = FakeAnthropic([Ok(FakeMessage(id="final"))])
    wrapped = resilient(client, policy=FAST)

    with wrapped.messages.stream(**REQ) as stream:
        assert stream is not None
        spans_at_enter = len(exporter.get_finished_spans())
        # Abandon the stream: never iterate it, never close it explicitly.

    req = spans_by_name(exporter, "damper.request")
    assert len(req) == 1
    assert req[0].attributes["damper.outcome"] == "ok"
    assert len(exporter.get_finished_spans()) == spans_at_enter


def test_streaming_mid_stream_failure_does_not_rewrite_ended_request_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = install_local_tracer(monkeypatch)
    client = FakeAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)

    with pytest.raises(anthropic.APIConnectionError):
        with wrapped.messages.stream(**REQ) as stream:
            req = spans_by_name(exporter, "damper.request")
            assert req[0].attributes["damper.outcome"] == "ok"
            end_time_at_enter = req[0].end_time
            for _event in stream:
                pass

    # The failure landed after the first token, so it was never replayed ...
    assert len(client.stream_calls) == 1
    # ... it is carried by the stream proxy's own damper metadata ...
    assert stream.damper.outcome == "stream_started_failure"

    # ... and the already-ended request span keeps its establishment outcome.
    req_after = spans_by_name(exporter, "damper.request")
    assert len(req_after) == 1
    assert req_after[0].attributes["damper.outcome"] == "ok"
    assert req_after[0].end_time == end_time_at_enter
    assert all(
        (s.attributes or {}).get("damper.outcome") != "stream_started_failure"
        for s in exporter.get_finished_spans()
    )


async def test_async_streaming_request_span_closes_before_proxy_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = install_local_tracer(monkeypatch)
    client = FakeAsyncAnthropic([Ok(FakeMessage(id="final"))])
    wrapped = resilient(client, policy=FAST)

    async with wrapped.messages.stream(**REQ) as stream:
        req = spans_by_name(exporter, "damper.request")
        assert len(req) == 1
        assert req[0].end_time is not None
        assert req[0].attributes["damper.outcome"] == "ok"
        spans_at_enter = len(exporter.get_finished_spans())

        events = [event async for event in stream]

    assert "content_block_delta" in [getattr(e, "type", None) for e in events]
    assert len(spans_by_name(exporter, "damper.request")) == 1
    assert len(exporter.get_finished_spans()) == spans_at_enter


async def test_async_mid_stream_failure_does_not_rewrite_ended_request_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = install_local_tracer(monkeypatch)
    client = FakeAsyncAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)

    seen: list[Any] = []
    with pytest.raises(anthropic.APIConnectionError):
        async with wrapped.messages.stream(**REQ) as stream:
            # Establishment succeeded, so the request span is already closed.
            req = spans_by_name(exporter, "damper.request")
            assert len(req) == 1
            assert req[0].attributes["damper.outcome"] == "ok"
            end_time_at_enter = req[0].end_time
            spans_at_enter = len(exporter.get_finished_spans())
            async for event in stream:
                seen.append(event)

    # Content had already streamed, so the failure was surfaced, never replayed.
    assert len(client.stream_calls) == 1
    assert [getattr(e, "type", None) for e in seen].count("content_block_delta") == 3

    # The async proxy's own damper metadata carries the mid-stream failure ...
    assert stream.damper.outcome == "stream_started_failure"

    # ... while the already-ended request span is left exactly as it closed, and
    # no second request span was emitted.
    req_after = spans_by_name(exporter, "damper.request")
    assert len(req_after) == 1
    assert req_after[0].attributes["damper.outcome"] == "ok"
    assert req_after[0].end_time == end_time_at_enter
    assert len(exporter.get_finished_spans()) == spans_at_enter
    assert all(
        (s.attributes or {}).get("damper.outcome") != "stream_started_failure"
        for s in exporter.get_finished_spans()
    )
