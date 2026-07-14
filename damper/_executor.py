"""Retry loop and decision state machine.

Drives a single logical request through the owned retry pipeline described in
``SPEC.md`` section 17: attempt, classify, streaming boundary, budget, cost
ceiling, backoff, and response metadata. Sync and async entry points share one
synchronous decision helper (:func:`_plan_after_failure`) so both paths make
byte-for-byte identical retry decisions; only "call the attempt" and "sleep"
differ.

Reliability-critical. Kept small and explicit per ``CLAUDE.md``. This module
owns no provider or HTTP knowledge: the caller injects an ``attempt`` callable,
a ``sleep`` function, a monotonic ``clock``, an ``rng``, a
``retry_after_extractor`` that returns an already-normalized ``float | None``
duration (the executor never parses headers -- SESSION 6 owns that), and a
``stream_started_probe``.

Decision order (``SPEC.md`` section 17, with the corrections approved for
SESSION 5)::

    call attempt
    on success:
        if attempt 1: deposit retry budget
        return response + metadata
    on failure (Exception only; never BaseException):
        stream_started = probe()            # read exactly once
        classify(error, stream_started)
        if not RETRYABLE (NOT_RETRYABLE or AMBIGUOUS): surface original
        if RETRYABLE and stream_started:     surface original
        if attempt is the last allowed:      raise RetriesExhausted
        estimate next retry cost
        if cost ceiling denies:              raise RetryCostCeilingHit
        budget.try_acquire_retry()           # the ONLY atomic budget op
        if denied:                           on_budget_exhausted
        accumulate retry cost, compute backoff, sleep, loop

Cost-ceiling denial is checked before budget acquisition, so when both would
deny the same retry the cost ceiling wins and the budget is never touched.
``try_acquire_retry`` is the single source of truth for budget accounting; there
is deliberately no snapshot-based pre-check, which could go stale under
concurrency and would skip the ``denied_retry_attempts`` bookkeeping.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from damper import (
    DamperError,
    Policy,
    RetriesExhausted,
    RetryBudgetExhausted,
    RetryCostCeilingHit,
)
from damper.backoff import compute_backoff
from damper.budget import RetryBudget
from damper.classify import ErrorClass, classify
from damper.cost import (
    estimate_input_tokens,
    estimate_output_tokens,
    estimate_retry_cost_usd,
    would_exceed_retry_cost_ceiling,
)


@dataclass(frozen=True)
class AttemptOutcome:
    """Bounded record of a single provider attempt.

    Retains only summary data (``SPEC.md`` section 9.2 / SESSION 5 correction 8):
    never response bodies, prompts, or streamed content. The number of records
    for a logical request is bounded by ``max_attempts``.
    """

    attempt: int
    error_class: str  # ErrorClass.value on failure, "ok" on success
    status_code: int | None
    latency_s: float
    succeeded: bool


@dataclass(frozen=True)
class DamperMetadata:
    """Per-request result metadata (``SPEC.md`` section 10).

    ``retry_cost_usd`` is ``float | None``: ``None`` means the cumulative retry
    cost could not be estimated (unknown), never that a retry was free.
    """

    attempts: int
    retried: bool
    total_latency_s: float
    retry_cost_usd: float | None
    outcome: str
    retry_budget_balance: float


@dataclass(frozen=True)
class ExecutionResult:
    """A successful provider response plus its Damper metadata.

    SESSION 6's wrapper attaches :attr:`metadata` to the response as
    ``resp.damper`` without mutating SDK response typing.
    """

    response: Any
    metadata: DamperMetadata


class _Action(Enum):
    """Result of the shared failure-decision helper."""

    SURFACE = "surface"  # re-raise the original provider error (bare raise)
    RETRY = "retry"  # sleep then attempt again


@dataclass(frozen=True)
class _Decision:
    action: _Action
    sleep_s: float = 0.0
    retry_cost_so_far_usd: float | None = None


def _no_retry_after(error: BaseException) -> float | None:
    """Default extractor: no provider Retry-After is known to the executor."""
    return None


def _never_started() -> bool:
    """Default streaming probe: no content has streamed."""
    return False


def _status_code_of(error: BaseException) -> int | None:
    """Return a genuine integer ``status_code`` from an error, else ``None``.

    ``bool`` is a subclass of ``int``; a stray ``True``/``False`` must not be
    read as HTTP status ``1``/``0``.
    """
    code = getattr(error, "status_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return code


def _estimate_next_retry_cost(
    request: Mapping[str, Any], policy: Policy
) -> float | None:
    """Estimate the USD cost of the next retry attempt, or ``None`` if unknown.

    Uses request-local token estimation only (no usage from a failed attempt, no
    network). An absent or non-string ``model`` yields ``None`` (unknown), which
    a configured ceiling then treats as fail-closed via
    :func:`would_exceed_retry_cost_ceiling`.
    """
    model = request.get("model")
    if not isinstance(model, str):
        return None
    input_tokens = estimate_input_tokens(request)
    output_tokens = estimate_output_tokens(request)
    return estimate_retry_cost_usd(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        price_table=policy.price_table,
    )


def _accumulate_retry_cost(
    retry_cost_so_far_usd: float | None,
    next_retry_cost_usd: float | None,
) -> float | None:
    """Fold an accepted retry's cost into the cumulative total.

    Once the cumulative cost is unknown it stays unknown for the logical request
    (``SPEC.md`` SESSION 5 correction 3). An accepted retry can only have an
    unknown ``next_retry_cost_usd`` when no ceiling is configured; a configured
    ceiling denies unknown-cost retries before this point.
    """
    if retry_cost_so_far_usd is None or next_retry_cost_usd is None:
        return None
    return retry_cost_so_far_usd + next_retry_cost_usd


def _plan_after_failure(
    error: Exception,
    *,
    attempt_number: int,
    attempt_latency_s: float,
    retry_cost_so_far_usd: float | None,
    outcomes: list[AttemptOutcome],
    policy: Policy,
    budget: RetryBudget,
    request: Mapping[str, Any],
    rng: Callable[[], float],
    retry_after_extractor: Callable[[BaseException], float | None],
    stream_started_probe: Callable[[], bool],
) -> _Decision:
    """Decide what to do after a provider attempt failed.

    Synchronous and side-effecting: it records the attempt outcome, and for an
    accepted retry it withdraws one budget token (the only atomic budget op) and
    computes the backoff. Returns :data:`_Action.SURFACE` (the caller re-raises
    the original error with a bare ``raise``) or :data:`_Action.RETRY`. Raises a
    terminal Damper exception ``from error`` for the exhausted / ceiling paths.

    The streaming probe is read exactly once and the same value feeds
    classification and the streaming-boundary check.
    """
    stream_started = stream_started_probe()
    error_class = classify(
        error,
        stream_started=stream_started,
        classifier=policy.classifier,
    )
    outcomes.append(
        AttemptOutcome(
            attempt=attempt_number,
            error_class=error_class.value,
            status_code=_status_code_of(error),
            latency_s=attempt_latency_s,
            succeeded=False,
        )
    )

    # NOT_RETRYABLE and AMBIGUOUS both surface the original error, never retry --
    # even if a custom classifier returns AMBIGUOUS with stream_started False.
    if error_class is not ErrorClass.RETRYABLE:
        return _Decision(_Action.SURFACE)

    # Retryable, but content already streamed to the caller: surface, never
    # replay (SPEC.md section 15).
    if stream_started:
        return _Decision(_Action.SURFACE)

    # Final allowed attempt failed retryably: no cost check, no budget
    # acquisition, no RNG, no sleep -- just surface exhaustion.
    if attempt_number >= policy.max_attempts:
        raise RetriesExhausted(
            attempts=attempt_number,
            attempt_outcomes=tuple(outcomes),
            last_provider_error=error,
        ) from error

    # Cost ceiling (non-mutating) before budget acquisition. When a ceiling is
    # configured the cumulative cost so far is always a known float, because an
    # unknown-cost retry would have been denied here on an earlier iteration.
    next_retry_cost_usd = _estimate_next_retry_cost(request, policy)
    if policy.max_retry_cost_usd is not None:
        if retry_cost_so_far_usd is None:
            # Invariant: with a ceiling configured, an unknown-cost retry is
            # denied here on an earlier iteration and never accumulates to None,
            # so the cumulative cost is always a known float at this point. Guard
            # explicitly rather than with `assert`, which `python -O` strips.
            raise DamperError(
                "internal invariant violated: cumulative retry cost is unknown "
                "while a retry cost ceiling is configured"
            )
        if would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=retry_cost_so_far_usd,
            next_retry_cost_usd=next_retry_cost_usd,
            max_retry_cost_usd=policy.max_retry_cost_usd,
        ):
            projected = (
                None
                if next_retry_cost_usd is None
                else retry_cost_so_far_usd + next_retry_cost_usd
            )
            raise RetryCostCeilingHit(
                attempts=attempt_number,
                estimated_retry_cost_usd=projected,
                max_retry_cost_usd=policy.max_retry_cost_usd,
                last_provider_error=error,
            ) from error

    # Budget: the single atomic authorization + accounting operation.
    if not budget.try_acquire_retry():
        if policy.on_budget_exhausted == "passthrough":
            return _Decision(_Action.SURFACE)
        snapshot = budget.snapshot()
        raise RetryBudgetExhausted(
            attempts=attempt_number,
            retry_budget_balance=snapshot.balance,
            retry_budget_ratio=snapshot.ratio,
            last_provider_error=error,
        ) from error

    # Retry is authorized: update cumulative cost once, compute backoff once.
    new_retry_cost = _accumulate_retry_cost(
        retry_cost_so_far_usd, next_retry_cost_usd
    )
    retry_after = retry_after_extractor(error)
    sleep_s = compute_backoff(
        attempt_number,
        base=policy.backoff_base,
        cap=policy.backoff_max,
        retry_after=retry_after,
        respect_retry_after=policy.respect_retry_after,
        rng=rng,
    )
    return _Decision(
        _Action.RETRY,
        sleep_s=sleep_s,
        retry_cost_so_far_usd=new_retry_cost,
    )


def _success_result(
    response: Any,
    *,
    attempt_number: int,
    attempt_latency_s: float,
    total_latency_s: float,
    retry_cost_so_far_usd: float | None,
    outcomes: list[AttemptOutcome],
    budget: RetryBudget,
) -> ExecutionResult:
    """Deposit budget on a first-attempt success and build the result metadata."""
    if attempt_number == 1:
        budget.record_first_attempt_success()
    outcomes.append(
        AttemptOutcome(
            attempt=attempt_number,
            error_class="ok",
            status_code=None,
            latency_s=attempt_latency_s,
            succeeded=True,
        )
    )
    metadata = DamperMetadata(
        attempts=attempt_number,
        retried=attempt_number > 1,
        total_latency_s=total_latency_s,
        retry_cost_usd=retry_cost_so_far_usd,
        outcome="ok",
        retry_budget_balance=budget.snapshot().balance,
    )
    return ExecutionResult(response=response, metadata=metadata)


def execute(
    attempt: Callable[[], Any],
    *,
    policy: Policy,
    budget: RetryBudget,
    request: Mapping[str, Any],
    retry_after_extractor: Callable[[BaseException], float | None] = _no_retry_after,
    stream_started_probe: Callable[[], bool] = _never_started,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] = random.random,
) -> ExecutionResult:
    """Run a synchronous logical request through the owned retry loop.

    ``attempt`` performs one provider attempt (SDK retries already disabled by
    the caller in SESSION 6) and either returns the response or raises. Only
    ``Exception`` subclasses are treated as provider failures; ``KeyboardInterrupt``,
    ``SystemExit``, and other ``BaseException``\\s propagate immediately with no
    retry, budget withdrawal, cost accounting, or sleep.
    """
    start = clock()
    retry_cost_so_far_usd: float | None = 0.0
    outcomes: list[AttemptOutcome] = []

    for attempt_number in range(1, policy.max_attempts + 1):
        attempt_start = clock()
        try:
            response = attempt()
        except Exception as error:
            attempt_latency_s = clock() - attempt_start
            decision = _plan_after_failure(
                error,
                attempt_number=attempt_number,
                attempt_latency_s=attempt_latency_s,
                retry_cost_so_far_usd=retry_cost_so_far_usd,
                outcomes=outcomes,
                policy=policy,
                budget=budget,
                request=request,
                rng=rng,
                retry_after_extractor=retry_after_extractor,
                stream_started_probe=stream_started_probe,
            )
            if decision.action is _Action.SURFACE:
                raise  # bare: preserve the original provider traceback
            retry_cost_so_far_usd = decision.retry_cost_so_far_usd
            sleep(decision.sleep_s)
            continue
        else:
            attempt_latency_s = clock() - attempt_start
            return _success_result(
                response,
                attempt_number=attempt_number,
                attempt_latency_s=attempt_latency_s,
                total_latency_s=clock() - start,
                retry_cost_so_far_usd=retry_cost_so_far_usd,
                outcomes=outcomes,
                budget=budget,
            )

    # Unreachable: the last iteration either returns on success or raises
    # RetriesExhausted on a retryable failure (a RETRY decision can only occur
    # when attempt_number < max_attempts).
    raise AssertionError("executor loop terminated without a result")


async def execute_async(
    attempt: Callable[[], Awaitable[Any]],
    *,
    policy: Policy,
    budget: RetryBudget,
    request: Mapping[str, Any],
    retry_after_extractor: Callable[[BaseException], float | None] = _no_retry_after,
    stream_started_probe: Callable[[], bool] = _never_started,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] = random.random,
) -> ExecutionResult:
    """Async counterpart of :func:`execute` sharing its decision logic.

    ``asyncio.CancelledError`` is a ``BaseException`` (not ``Exception``), so a
    cancellation during either the provider attempt or the backoff sleep
    propagates immediately -- no further attempt, budget withdrawal, cost
    accounting, or sleep. An acquired budget token is deliberately not refunded
    on cancellation: it represents reserved, conservatively-shed retry capacity.
    """
    start = clock()
    retry_cost_so_far_usd: float | None = 0.0
    outcomes: list[AttemptOutcome] = []

    for attempt_number in range(1, policy.max_attempts + 1):
        attempt_start = clock()
        try:
            response = await attempt()
        except Exception as error:
            attempt_latency_s = clock() - attempt_start
            decision = _plan_after_failure(
                error,
                attempt_number=attempt_number,
                attempt_latency_s=attempt_latency_s,
                retry_cost_so_far_usd=retry_cost_so_far_usd,
                outcomes=outcomes,
                policy=policy,
                budget=budget,
                request=request,
                rng=rng,
                retry_after_extractor=retry_after_extractor,
                stream_started_probe=stream_started_probe,
            )
            if decision.action is _Action.SURFACE:
                raise  # bare: preserve the original provider traceback
            retry_cost_so_far_usd = decision.retry_cost_so_far_usd
            await sleep(decision.sleep_s)
            continue
        else:
            attempt_latency_s = clock() - attempt_start
            return _success_result(
                response,
                attempt_number=attempt_number,
                attempt_latency_s=attempt_latency_s,
                total_latency_s=clock() - start,
                retry_cost_so_far_usd=retry_cost_so_far_usd,
                outcomes=outcomes,
                budget=budget,
            )

    raise AssertionError("executor loop terminated without a result")


__all__: Sequence[str] = (
    "AttemptOutcome",
    "DamperMetadata",
    "ExecutionResult",
    "execute",
    "execute_async",
)
