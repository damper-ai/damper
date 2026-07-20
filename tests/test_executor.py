"""Tests for :mod:`damper._executor`.

Covers the retry-decision matrix:

* the core matrix runs against both ``execute`` and ``execute_async`` (async is
  the primary path), driven by the shared decision helper;
* cost-ceiling denial happens before budget acquisition and never touches the
  budget, while budget denial goes through ``try_acquire_retry`` and increments
  ``denied_retry_attempts``;
* an unknown retry cost stays ``None`` (never ``0.0``);
* the executor never parses Retry-After from the error itself -- a normalized
  value only reaches it through the injected extractor;
* ``AMBIGUOUS`` is never retried and the streaming probe is read once per
  failure;
* async cancellation propagates with no retry, sleep, or budget side effect;
* the final allowed attempt produces no cost, budget, RNG, or sleep side effect;
* every Damper exception carries its metadata and ``__cause__``.

No network access, no real sleeps (sleep is injected and captured), no provider
API calls.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from damper import (
    Policy,
    RetriesExhausted,
    RetryBudgetExhausted,
    RetryCostCeilingHit,
)
from damper._executor import (
    ExecutionResult,
    _accumulate_retry_cost,
    execute,
    execute_async,
)
from damper.budget import RetryBudget
from damper.classify import ErrorClass
from damper.cost import estimate_retry_cost_usd
from damper.prices import ModelPrice
from tests.fakes import Err, FakeProviderError, Ok, ScriptedFake

MODES = ["sync", "async"]

DEFAULT_REQUEST: dict[str, Any] = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "hi"}],
}
NO_MODEL_REQUEST: dict[str, Any] = {
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "hi"}],
}
# A model name deliberately absent from the default price table, used to
# exercise unknown-model pricing with an explicit (not missing) model argument.
UNPRICED_MODEL = "claude-unknown-test-model"
UNPRICED_MODEL_REQUEST: dict[str, Any] = {
    "model": UNPRICED_MODEL,
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "hi"}],
}


class FakeClock:
    """Monotonic-looking clock that advances 1.0s per read (deterministic)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class RngCounter:
    """Deterministic RNG that records how many times it was consumed."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.value


class Probe:
    """Streaming probe that records how many times it was read."""

    def __init__(self, started: bool = False) -> None:
        self.started = started
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.started


def read_retry_after(error: BaseException) -> float | None:
    """Test extractor: surface an already-normalized numeric retry-after."""
    value = getattr(error, "retry_after", None)
    return None if value is None else float(value)


def make_budget(
    *, ratio: float = 0.1, min_tokens: int = 10, window: float = 60.0
) -> RetryBudget:
    return RetryBudget(
        ratio=ratio, min_tokens=min_tokens, window=window, clock=lambda: 0.0
    )


def run(
    mode: str,
    sequence: list[Any],
    *,
    policy: Policy,
    budget: RetryBudget,
    sleeps: list[float],
    rng: RngCounter,
    request: dict[str, Any] | None = None,
    retry_after_extractor: Any = None,
    stream_started_probe: Any = None,
) -> ExecutionResult:
    """Drive one logical request through the chosen executor path.

    ``sleeps`` and ``rng`` are supplied by the caller so their side effects can
    be inspected even when the call raises.
    """
    fake = ScriptedFake.from_sequence(list(sequence))
    kw: dict[str, Any] = {
        "policy": policy,
        "budget": budget,
        "request": DEFAULT_REQUEST if request is None else request,
        "clock": FakeClock(),
        "rng": rng,
    }
    if retry_after_extractor is not None:
        kw["retry_after_extractor"] = retry_after_extractor
    if stream_started_probe is not None:
        kw["stream_started_probe"] = stream_started_probe

    if mode == "sync":
        kw["sleep"] = sleeps.append
        return execute(lambda: fake.next_outcome(), **kw)

    async def _attempt() -> Any:
        return fake.next_outcome()

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    kw["sleep"] = _sleep
    return asyncio.run(execute_async(_attempt, **kw))


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_first_attempt_success(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    result = run(mode, [Ok("resp")], policy=Policy(), budget=budget, sleeps=sleeps, rng=rng)

    assert result.response == "resp"
    assert result.metadata.attempts == 1
    assert result.metadata.retried is False
    assert result.metadata.outcome == "ok"
    assert result.metadata.retry_cost_usd == 0.0
    assert sleeps == []
    assert rng.calls == 0
    snap = budget.snapshot()
    # First-attempt success deposits ratio tokens; no retries consumed.
    assert snap.successful_first_attempts == 1
    assert snap.balance == pytest.approx(10.1)
    assert snap.retry_attempts == 0


@pytest.mark.parametrize("mode", MODES)
def test_two_retries_then_success(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    result = run(
        mode,
        [Err(529), Err(529), Ok("resp")],
        policy=Policy(),
        budget=budget,
        sleeps=sleeps,
        rng=rng,
    )

    assert result.metadata.attempts == 3
    assert result.metadata.retried is True
    assert len(sleeps) == 2
    assert rng.calls == 2
    snap = budget.snapshot()
    # Success arrived on attempt 3, not the first attempt: no deposit.
    assert snap.successful_first_attempts == 0
    assert snap.retry_attempts == 2
    assert snap.balance == pytest.approx(8.0)


# --------------------------------------------------------------------------- #
# Retries exhausted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_retries_exhausted(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    with pytest.raises(RetriesExhausted) as excinfo:
        run(
            mode,
            [Err(529), Err(529), Err(529)],
            policy=Policy(),
            budget=budget,
            sleeps=sleeps,
            rng=rng,
        )

    exc = excinfo.value
    assert exc.attempts == 3
    assert len(exc.attempt_outcomes) == 3
    assert isinstance(exc.last_provider_error, FakeProviderError)
    assert exc.__cause__ is exc.last_provider_error
    # Two retries slept and were withdrawn; the third (final) attempt did not.
    assert len(sleeps) == 2
    assert budget.snapshot().retry_attempts == 2


@pytest.mark.parametrize("mode", MODES)
def test_final_attempt_has_no_side_effects(mode: str) -> None:
    # max_attempts=1: the first failure is the final failure. No cost check, no
    # budget acquisition, no RNG draw, no sleep.
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    policy = Policy(max_attempts=1, max_retry_cost_usd=1.0)
    with pytest.raises(RetriesExhausted):
        run(mode, [Err(529)], policy=policy, budget=budget, sleeps=sleeps, rng=rng)

    assert sleeps == []
    assert rng.calls == 0
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


# --------------------------------------------------------------------------- #
# Classification: not-retryable and ambiguous
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_not_retryable_surfaces_original(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    with pytest.raises(FakeProviderError):
        run(mode, [Err(400)], policy=Policy(), budget=budget, sleeps=sleeps, rng=rng)

    assert sleeps == []
    assert rng.calls == 0
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


@pytest.mark.parametrize("mode", MODES)
def test_ambiguous_is_never_retried(mode: str) -> None:
    # A custom classifier forces AMBIGUOUS even for a normally-retryable 529 and
    # with stream_started False; the executor must still surface, never retry.
    def classifier(error: BaseException, *, stream_started: bool) -> ErrorClass:
        return ErrorClass.AMBIGUOUS

    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    with pytest.raises(FakeProviderError):
        run(
            mode,
            [Err(529), Ok()],
            policy=Policy(classifier=classifier),
            budget=budget,
            sleeps=sleeps,
            rng=rng,
        )

    assert sleeps == []
    assert budget.snapshot().retry_attempts == 0


# --------------------------------------------------------------------------- #
# Retry-After normalization seam
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_retry_after_used_exactly_and_not_capped(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    # backoff_max=5 but the normalized retry-after is 7: it must be used exactly,
    # not capped, and the RNG must not be consumed.
    policy = Policy(backoff_max=5.0)
    result = run(
        mode,
        [Err(429, retry_after=7.0), Ok()],
        policy=policy,
        budget=budget,
        sleeps=sleeps,
        rng=rng,
        retry_after_extractor=read_retry_after,
    )

    assert result.metadata.attempts == 2
    assert sleeps == [7.0]
    assert rng.calls == 0


@pytest.mark.parametrize("mode", MODES)
def test_retry_after_none_uses_full_jitter(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter(value=0.5)
    run(
        mode,
        [Err(429), Ok()],
        policy=Policy(),
        budget=budget,
        sleeps=sleeps,
        rng=rng,
        retry_after_extractor=read_retry_after,
    )

    # retry_number=1 -> upper bound min(30, 1*2**0)=1; draw 0.5 -> 0.5s.
    assert sleeps == [0.5]
    assert rng.calls == 1


@pytest.mark.parametrize("mode", MODES)
def test_executor_does_not_read_retry_after_from_error(mode: str) -> None:
    # Default extractor ignores the error entirely: even though the error carries
    # retry_after=7.0, the executor falls back to full-jitter backoff.
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter(value=0.5)
    run(
        mode,
        [Err(429, retry_after=7.0), Ok()],
        policy=Policy(),
        budget=budget,
        sleeps=sleeps,
        rng=rng,
    )

    assert sleeps == [0.5]
    assert rng.calls == 1


# --------------------------------------------------------------------------- #
# Budget exhaustion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_budget_exhausted_raises(mode: str) -> None:
    # Empty budget (min_tokens=0, ratio=0): the retry is denied through
    # try_acquire_retry, which increments denied_retry_attempts.
    budget = make_budget(ratio=0.0, min_tokens=0)
    sleeps: list[float] = []
    rng = RngCounter()
    with pytest.raises(RetryBudgetExhausted) as excinfo:
        run(mode, [Err(529), Ok()], policy=Policy(), budget=budget, sleeps=sleeps, rng=rng)

    exc = excinfo.value
    assert exc.attempts == 1
    assert exc.retry_budget_balance == pytest.approx(0.0)
    assert exc.retry_budget_ratio == pytest.approx(0.0)
    assert isinstance(exc.last_provider_error, FakeProviderError)
    assert exc.__cause__ is exc.last_provider_error
    assert sleeps == []
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 1


@pytest.mark.parametrize("mode", MODES)
def test_budget_passthrough_surfaces_provider_error(mode: str) -> None:
    budget = make_budget(ratio=0.0, min_tokens=0)
    sleeps: list[float] = []
    rng = RngCounter()
    policy = Policy(on_budget_exhausted="passthrough")
    with pytest.raises(FakeProviderError) as excinfo:
        run(mode, [Err(529), Ok()], policy=policy, budget=budget, sleeps=sleeps, rng=rng)

    # Passthrough surfaces the original error unwrapped (bare raise): Damper did
    # not chain it as the cause of some other exception.
    assert excinfo.value.__cause__ is None
    assert sleeps == []
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 1


# --------------------------------------------------------------------------- #
# Cost ceiling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_cost_ceiling_denies_before_budget(mode: str) -> None:
    # Estimated next-retry cost (~0.0077 USD) exceeds a 0.001 ceiling. The retry
    # is denied before any budget acquisition, so the budget is untouched.
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    policy = Policy(max_retry_cost_usd=0.001)
    with pytest.raises(RetryCostCeilingHit) as excinfo:
        run(mode, [Err(529), Ok()], policy=policy, budget=budget, sleeps=sleeps, rng=rng)

    expected = estimate_retry_cost_usd(
        model="claude-sonnet-4-6", input_tokens=1, output_tokens=512
    )
    assert expected is not None
    exc = excinfo.value
    assert exc.attempts == 1
    assert exc.estimated_retry_cost_usd == pytest.approx(expected)
    assert exc.max_retry_cost_usd == 0.001
    assert isinstance(exc.last_provider_error, FakeProviderError)
    assert exc.__cause__ is exc.last_provider_error
    assert sleeps == []
    assert rng.calls == 0
    # Cost denial must not consume or deny budget.
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


@pytest.mark.parametrize("mode", MODES)
def test_cost_below_ceiling_reports_float(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    policy = Policy(max_retry_cost_usd=1.0)
    result = run(
        mode, [Err(529), Ok()], policy=policy, budget=budget, sleeps=sleeps, rng=rng
    )

    expected = estimate_retry_cost_usd(
        model="claude-sonnet-4-6", input_tokens=1, output_tokens=512
    )
    assert expected is not None
    assert result.metadata.attempts == 2
    assert result.metadata.retry_cost_usd is not None
    assert result.metadata.retry_cost_usd == pytest.approx(expected)


@pytest.mark.parametrize("mode", MODES)
def test_unknown_cost_without_ceiling_reports_none(mode: str) -> None:
    # No model in the request -> unknown next-retry cost. With no ceiling the
    # retry proceeds, but the cumulative cost stays None (unknown, not free).
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    result = run(
        mode,
        [Err(529), Ok()],
        policy=Policy(),
        budget=budget,
        sleeps=sleeps,
        rng=rng,
        request=NO_MODEL_REQUEST,
    )

    assert result.metadata.attempts == 2
    assert result.metadata.retried is True
    assert result.metadata.retry_cost_usd is None


@pytest.mark.parametrize("mode", MODES)
def test_unknown_cost_with_ceiling_is_fail_closed(mode: str) -> None:
    # An unestimable request (no model -> unknown cost) is fail-closed only when
    # a ceiling is configured. The retry is denied before budget acquisition,
    # and the projected cost is reported as unknown (None).
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    policy = Policy(max_retry_cost_usd=1.0)
    with pytest.raises(RetryCostCeilingHit) as excinfo:
        run(
            mode,
            [Err(529), Ok()],
            policy=policy,
            budget=budget,
            sleeps=sleeps,
            rng=rng,
            request=NO_MODEL_REQUEST,
        )

    exc = excinfo.value
    assert exc.attempts == 1
    assert exc.estimated_retry_cost_usd is None
    assert exc.max_retry_cost_usd == 1.0
    assert exc.__cause__ is exc.last_provider_error
    assert sleeps == []
    assert rng.calls == 0
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


@pytest.mark.parametrize("mode", MODES)
def test_unpriced_model_with_ceiling_is_fail_closed(mode: str) -> None:
    # A model present on the request but absent from the effective price table
    # yields an unknown next-retry cost, so a configured ceiling is fail-closed
    # exactly like a missing model: the retry is denied before budget
    # acquisition and the projected cost is reported as unknown (None). This
    # exercises an explicit unknown model name, not a missing model.
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    policy = Policy(max_retry_cost_usd=0.05)
    with pytest.raises(RetryCostCeilingHit) as excinfo:
        run(
            mode,
            [Err(529), Ok()],
            policy=policy,
            budget=budget,
            sleeps=sleeps,
            rng=rng,
            request=UNPRICED_MODEL_REQUEST,
        )

    exc = excinfo.value
    assert exc.attempts == 1
    assert exc.estimated_retry_cost_usd is None
    assert exc.max_retry_cost_usd == 0.05
    assert isinstance(exc.last_provider_error, FakeProviderError)
    assert exc.__cause__ is exc.last_provider_error
    assert sleeps == []
    assert rng.calls == 0
    # Cost denial must not consume or deny budget.
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


@pytest.mark.parametrize("mode", MODES)
def test_custom_price_table_prices_unknown_model(mode: str) -> None:
    # A custom price_table entry makes the otherwise-unpriced model estimable, so
    # the retry is authorized on cost and the request proceeds normally. The
    # custom table replaces the built-in table rather than merging with it; that
    # replacement is what supplies the price for this model.
    table = {UNPRICED_MODEL: ModelPrice(UNPRICED_MODEL, 3.0, 15.0, "2026-07-12")}
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    policy = Policy(max_retry_cost_usd=1.0, price_table=table)
    result = run(
        mode,
        [Err(529), Ok("resp")],
        policy=policy,
        budget=budget,
        sleeps=sleeps,
        rng=rng,
        request=UNPRICED_MODEL_REQUEST,
    )

    # Retry cost derives from the custom entry: input text "hi" -> ceil(2/4) = 1
    # token, output reservation = max_tokens = 512.
    expected = estimate_retry_cost_usd(
        model=UNPRICED_MODEL,
        input_tokens=1,
        output_tokens=512,
        price_table=table,
    )
    assert expected is not None
    assert result.response == "resp"
    assert result.metadata.attempts == 2
    assert result.metadata.retried is True
    assert result.metadata.retry_cost_usd is not None
    assert result.metadata.retry_cost_usd == pytest.approx(expected)
    assert len(sleeps) == 1
    assert rng.calls == 1
    # The retry was authorized, not rejected for unknown pricing.
    snap = budget.snapshot()
    assert snap.retry_attempts == 1
    assert snap.denied_retry_attempts == 0


# --------------------------------------------------------------------------- #
# Cumulative retry-cost accumulation helper
# --------------------------------------------------------------------------- #


def test_accumulate_known_plus_known_is_float_total() -> None:
    total = _accumulate_retry_cost(0.01, 0.02)
    assert total == pytest.approx(0.03)


def test_accumulate_known_plus_unknown_is_none() -> None:
    # An accepted unknown-cost retry (only possible with no ceiling configured)
    # turns the cumulative total unknown, never $0.00.
    assert _accumulate_retry_cost(0.01, None) is None


def test_accumulate_none_plus_known_stays_none() -> None:
    # Once unknown, the cumulative cost stays unknown for the logical request.
    assert _accumulate_retry_cost(None, 0.02) is None


def test_accumulate_none_plus_none_stays_none() -> None:
    assert _accumulate_retry_cost(None, None) is None


# --------------------------------------------------------------------------- #
# Streaming boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_stream_started_surfaces_and_probe_read_once(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    probe = Probe(started=True)
    with pytest.raises(FakeProviderError):
        run(
            mode,
            [Err(529), Ok()],
            policy=Policy(),
            budget=budget,
            sleeps=sleeps,
            rng=rng,
            stream_started_probe=probe,
        )

    # A retryable error is surfaced, never replayed, once content has streamed.
    assert budget.snapshot().retry_attempts == 0
    assert probe.calls == 1


@pytest.mark.parametrize("mode", MODES)
def test_probe_read_once_per_failure_on_retry(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    probe = Probe(started=False)
    run(
        mode,
        [Err(529), Ok()],
        policy=Policy(),
        budget=budget,
        sleeps=sleeps,
        rng=rng,
        stream_started_probe=probe,
    )

    # Exactly one failed attempt -> probe read exactly once.
    assert probe.calls == 1


# --------------------------------------------------------------------------- #
# Cancellation (async only)
# --------------------------------------------------------------------------- #


def test_async_cancellation_propagates_without_side_effects() -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()

    async def attempt() -> Any:
        raise asyncio.CancelledError()

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            execute_async(
                attempt,
                policy=Policy(),
                budget=budget,
                request=DEFAULT_REQUEST,
                sleep=_sleep,
                clock=FakeClock(),
                rng=rng,
            )
        )

    assert sleeps == []
    assert rng.calls == 0
    snap = budget.snapshot()
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


# --------------------------------------------------------------------------- #
# Attempt-outcome record shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_attempt_outcomes_record_shape(mode: str) -> None:
    budget = make_budget()
    sleeps: list[float] = []
    rng = RngCounter()
    with pytest.raises(RetriesExhausted) as excinfo:
        run(
            mode,
            [Err(500), Err(500), Err(500)],
            policy=Policy(),
            budget=budget,
            sleeps=sleeps,
            rng=rng,
        )

    outcomes = excinfo.value.attempt_outcomes
    assert [o.attempt for o in outcomes] == [1, 2, 3]
    assert all(o.error_class == ErrorClass.RETRYABLE.value for o in outcomes)
    assert all(o.status_code == 500 for o in outcomes)
    assert all(o.succeeded is False for o in outcomes)
    assert all(math.isfinite(o.latency_s) for o in outcomes)
