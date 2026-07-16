"""OpenTelemetry tracing for Damper (``SPEC.md`` section 18).

Runtime dependency is ``opentelemetry-api`` only. When no OTel SDK/exporter is
configured, ``trace.get_tracer`` returns a no-op tracer whose spans are
non-recording, so telemetry is a safe no-op with no hidden work: every aggregate
computation (token estimation, cost, budget snapshot) is gated behind
``span.is_recording()``.

Damper owns a narrow recorder abstraction (:data:`TraceFactory` ->
:class:`RequestRecorder` -> :class:`AttemptRecorder`) so the reliability-critical
executor never imports OpenTelemetry directly. The default factory
(:func:`default_trace_factory`) is OTel-backed; tests inject no-op, local-
provider, or deliberately-failing recorders.

Failure isolation
-----------------

Telemetry must never change retry behavior. Every span operation here is wrapped
so that a failure to create a tracer/span, set an attribute, or end a span
disables recording for that request and is swallowed -- it never propagates,
never replaces the provider/Damper exception, and ``__exit__`` never suppresses
the application exception. Only ``Exception`` is caught, never ``BaseException``.

Streaming span boundary (v0.1)
------------------------------

For non-streaming calls, ``damper.request`` covers the whole executor-owned
request lifecycle: authorization, every provider attempt, retry decisions,
backoff sleeps, and the successful completion.

In the v0.1 wrapper path, the ``damper.request`` span ends after retry-controlled
stream establishment succeeds and before caller-owned stream consumption
continues. It is ended before the stream proxy is returned because a caller may
abandon a stream without consuming or closing it, and a span held open across
caller-owned consumption would leak.

That boundary has consequences worth stating exactly:

- ``damper.outcome="ok"`` on a streaming request span means establishment
  succeeded. It does not mean the full stream was consumed successfully.
- Failures that occur later, while the caller consumes the stream, are
  represented in the stream proxy's ``resp.damper`` metadata, whose outcome
  becomes ``stream_started_failure``. They do not rewrite the already-ended
  request span.
- An ended span cannot be rewritten, so a request span that closed ``ok`` stays
  ``ok`` even when the stream later fails mid-consumption.
- A separate stream-lifecycle span covering caller consumption is outside v0.1.

Attribute names
---------------

``damper.*`` attributes are the stable project contract. ``gen_ai.*`` attributes
are best-effort compatibility mappings, not the stable API.

TODO(amit): verify ``gen_ai.*`` names against the current OpenTelemetry GenAI
semantic-convention release before tagging v0.1; only a small stable subset is
emitted here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer

from damper.cost import (
    estimate_input_tokens,
    estimate_output_tokens,
    estimate_retry_cost_usd,
)

if TYPE_CHECKING:
    from damper import Policy
    from damper.budget import RetryBudget

# --------------------------------------------------------------------------- #
# Outcome values (SPEC section 18, "Outcomes"). Stable telemetry contract.
# --------------------------------------------------------------------------- #

Outcome = Literal[
    "ok",
    "retries_exhausted",
    "budget_exhausted",
    "cost_ceiling",
    "not_retryable",
    "stream_started_failure",
]

OUTCOME_OK: Outcome = "ok"
OUTCOME_RETRIES_EXHAUSTED: Outcome = "retries_exhausted"
OUTCOME_BUDGET_EXHAUSTED: Outcome = "budget_exhausted"
OUTCOME_COST_CEILING: Outcome = "cost_ceiling"
OUTCOME_NOT_RETRYABLE: Outcome = "not_retryable"
OUTCOME_STREAM_STARTED_FAILURE: Outcome = "stream_started_failure"

_REQUEST_SPAN = "damper.request"
_ATTEMPT_SPAN = "damper.attempt"


# --------------------------------------------------------------------------- #
# Recorder protocols (Damper-owned; the executor depends only on these).
# --------------------------------------------------------------------------- #


class AttemptRecorder(Protocol):
    """Per-attempt span recorder used as a context manager around the call."""

    def __enter__(self) -> AttemptRecorder: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]: ...
    def record_failure(
        self,
        *,
        error_class: str,
        status_code: int | None,
        latency_s: float,
        backoff_s: float | None,
    ) -> None: ...
    def record_success(self, *, latency_s: float) -> None: ...


class RequestRecorder(Protocol):
    """Per-request span recorder used as a context manager around the loop."""

    def __enter__(self) -> RequestRecorder: ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]: ...
    def attempt(self, n: int) -> AttemptRecorder: ...
    def set_outcome(self, outcome: str) -> None: ...
    def set_retry_cost(self, retry_cost_usd: float | None) -> None: ...


TraceFactory = Callable[..., RequestRecorder]


# --------------------------------------------------------------------------- #
# No-op recorder (explicit "telemetry disabled" path for tests).
# --------------------------------------------------------------------------- #


class _NoOpAttemptRecorder:
    def __enter__(self) -> _NoOpAttemptRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        return False

    def record_failure(
        self,
        *,
        error_class: str,
        status_code: int | None,
        latency_s: float,
        backoff_s: float | None,
    ) -> None:
        pass

    def record_success(self, *, latency_s: float) -> None:
        pass


class _NoOpRequestRecorder:
    def __enter__(self) -> _NoOpRequestRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        return False

    def attempt(self, n: int) -> AttemptRecorder:
        return _NoOpAttemptRecorder()

    def set_outcome(self, outcome: str) -> None:
        pass

    def set_retry_cost(self, retry_cost_usd: float | None) -> None:
        pass


def noop_trace_factory(**kwargs: Any) -> RequestRecorder:
    """A factory that records nothing (telemetry explicitly disabled)."""
    return _NoOpRequestRecorder()


# --------------------------------------------------------------------------- #
# OpenTelemetry-backed recorder (defensive: telemetry failure never propagates).
# --------------------------------------------------------------------------- #


class _OTelAttemptRecorder:
    def __init__(self, span: Any, parent: _OTelRequestRecorder) -> None:
        self._span = span
        self._parent = parent

    def __enter__(self) -> _OTelAttemptRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        try:
            if self._span is not None:
                self._span.end()
        except Exception:
            self._parent._disable()
        return False  # never suppress the application exception

    def record_failure(
        self,
        *,
        error_class: str,
        status_code: int | None,
        latency_s: float,
        backoff_s: float | None,
    ) -> None:
        span = self._span
        if span is None:
            return
        try:
            if span.is_recording():
                span.set_attribute("damper.attempt.error_class", error_class)
                span.set_attribute("damper.attempt.latency_s", latency_s)
                if status_code is not None:
                    span.set_attribute("damper.attempt.status_code", status_code)
                if backoff_s is not None:
                    span.set_attribute("damper.attempt.backoff_s", backoff_s)
        except Exception:
            self._parent._disable()

    def record_success(self, *, latency_s: float) -> None:
        span = self._span
        if span is None:
            return
        try:
            if span.is_recording():
                span.set_attribute("damper.attempt.latency_s", latency_s)
        except Exception:
            self._parent._disable()


class _OTelRequestRecorder:
    """OTel-backed request recorder with defensive failure isolation."""

    def __init__(
        self,
        tracer: Tracer,
        *,
        provider: str | None,
        request: Mapping[str, Any],
        policy: Policy,
        budget: RetryBudget,
    ) -> None:
        self._tracer = tracer
        self._provider = provider
        self._request = request
        self._policy = policy
        self._budget = budget
        self._outcome: str | None = None
        self._retry_cost_usd: float | None = 0.0
        self._attempts = 0
        self._enabled = True
        self._span: Any = None
        self._ctx: Any = None

    def _disable(self) -> None:
        self._enabled = False

    def __enter__(self) -> _OTelRequestRecorder:
        try:
            self._span = self._tracer.start_span(_REQUEST_SPAN)
            self._ctx = trace.set_span_in_context(self._span)
            if self._span.is_recording():
                # Provider/model set at creation, not only at the end (SPEC 18.8).
                if self._provider:
                    self._span.set_attribute("damper.provider", self._provider)
                    self._span.set_attribute("gen_ai.provider.name", self._provider)
                self._span.set_attribute("gen_ai.operation.name", "chat")
                model = self._request.get("model")
                if isinstance(model, str):
                    self._span.set_attribute("damper.model", model)
                    self._span.set_attribute("gen_ai.request.model", model)
        except Exception:
            self._disable()
        return self

    def attempt(self, n: int) -> AttemptRecorder:
        self._attempts = n
        span: Any = None
        if self._enabled and self._span is not None:
            try:
                span = self._tracer.start_span(_ATTEMPT_SPAN, context=self._ctx)
                if span.is_recording():
                    span.set_attribute("damper.attempt.n", n)
            except Exception:
                self._disable()
                span = None
        return _OTelAttemptRecorder(span, self)

    def set_outcome(self, outcome: str) -> None:
        self._outcome = outcome

    def set_retry_cost(self, retry_cost_usd: float | None) -> None:
        self._retry_cost_usd = retry_cost_usd

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        span = self._span
        try:
            if span is not None:
                if self._enabled and span.is_recording():
                    self._finalize(span)
                span.end()
        except Exception:
            self._disable()
        return False  # never suppress the application exception

    def _finalize(self, span: Any) -> None:
        span.set_attribute("damper.attempts", self._attempts)
        if self._outcome is not None:
            span.set_attribute("damper.outcome", self._outcome)

        snapshot = self._budget.snapshot()
        span.set_attribute("damper.retry_budget.balance", snapshot.balance)
        span.set_attribute("damper.retry_budget.ratio", snapshot.ratio)

        input_tokens = estimate_input_tokens(self._request)
        output_tokens = estimate_output_tokens(self._request)
        if input_tokens is not None:
            span.set_attribute("damper.tokens.input", input_tokens)
        if output_tokens is not None:
            span.set_attribute("damper.tokens.output", output_tokens)

        model = self._request.get("model")
        if isinstance(model, str):
            estimate = estimate_retry_cost_usd(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                price_table=self._policy.price_table,
            )
            if estimate is not None:
                span.set_attribute("damper.cost.estimate_usd", estimate)
        if self._retry_cost_usd is not None:
            span.set_attribute("damper.cost.retry_usd", self._retry_cost_usd)

        if self._outcome is not None and self._outcome != OUTCOME_OK:
            span.set_status(Status(StatusCode.ERROR))


def _get_tracer() -> Tracer:
    """Acquire the global tracer lazily (picks up a provider set after import)."""
    return trace.get_tracer("damper")


def make_trace_factory(tracer: Tracer) -> TraceFactory:
    """Build a factory bound to a specific ``tracer`` (used by tests/examples)."""

    def factory(
        *,
        provider: str | None,
        request: Mapping[str, Any],
        policy: Policy,
        budget: RetryBudget,
    ) -> RequestRecorder:
        return _OTelRequestRecorder(
            tracer, provider=provider, request=request, policy=policy, budget=budget
        )

    return factory


def default_trace_factory(
    *,
    provider: str | None,
    request: Mapping[str, Any],
    policy: Policy,
    budget: RetryBudget,
) -> RequestRecorder:
    """Production factory: OTel-backed, no-op when no SDK/exporter is configured."""
    return _OTelRequestRecorder(
        _get_tracer(), provider=provider, request=request, policy=policy, budget=budget
    )
