"""Tests for :mod:`damper.budget`.

Pins the per-window invariant described in SPEC.md section 12.5:

    retry_attempts <= min_tokens + ratio * successful_first_attempts

Weakening any assertion here requires updating the spec first. All tests
use a fake or shared clock so wall-clock ``sleep`` is never necessary.
"""

from __future__ import annotations

import threading

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from damper.budget import BudgetSnapshot, RetryBudget

# Floating-point tolerance for the property test to absorb accumulated drift
# from many additions of a fractional ratio.
_EPS = 1e-9


class FakeClock:
    """Deterministic monotonic clock for single-threaded tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, delta: float) -> None:
        assert delta >= 0
        self._t += delta


class SharedClock:
    """Thread-safe deterministic clock for concurrent tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._t

    def advance(self, delta: float) -> None:
        with self._lock:
            self._t += delta


# --------------------------- validation ---------------------------


def test_rejects_negative_ratio() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=-0.1, min_tokens=0, window=60.0)


def test_rejects_nan_ratio() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=float("nan"), min_tokens=0, window=60.0)


def test_rejects_inf_ratio() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=float("inf"), min_tokens=0, window=60.0)


def test_rejects_negative_min_tokens() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=0.1, min_tokens=-1, window=60.0)


def test_rejects_zero_window() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=0.1, min_tokens=0, window=0.0)


def test_rejects_negative_window() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=0.1, min_tokens=0, window=-1.0)


def test_rejects_nan_window() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=0.1, min_tokens=0, window=float("nan"))


def test_rejects_inf_window() -> None:
    with pytest.raises(ValueError):
        RetryBudget(ratio=0.1, min_tokens=0, window=float("inf"))


# ----------------------- default clock smoke -----------------------


def test_default_clock_is_time_monotonic_and_functional() -> None:
    # No explicit clock supplied. The default is time.monotonic and the
    # budget should be immediately usable.
    budget = RetryBudget(ratio=0.1, min_tokens=1, window=60.0)
    snap = budget.snapshot()
    assert snap.balance == 1.0
    assert snap.retry_attempts == 0
    assert budget.try_acquire_retry() is True


# --------------------------- basic accounting ---------------------------


def test_default_initial_capacity_equals_min_tokens() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, min_tokens=10, window=60.0, clock=clock)
    snap = budget.snapshot()
    assert isinstance(snap, BudgetSnapshot)
    assert snap.balance == 10.0
    assert snap.min_tokens == 10
    assert snap.ratio == 0.1
    assert snap.window == 60.0
    assert snap.successful_first_attempts == 0
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


def test_zero_traffic_and_zero_min_tokens_denies_first_retry() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, min_tokens=0, window=60.0, clock=clock)
    assert budget.try_acquire_retry() is False
    snap = budget.snapshot()
    assert snap.balance == 0.0
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 1


def test_initial_capacity_allows_bootstrap_retries() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.0, min_tokens=3, window=60.0, clock=clock)
    assert budget.try_acquire_retry() is True
    assert budget.try_acquire_retry() is True
    assert budget.try_acquire_retry() is True
    assert budget.try_acquire_retry() is False
    snap = budget.snapshot()
    assert snap.retry_attempts == 3
    assert snap.denied_retry_attempts == 1
    assert snap.balance == 0.0


def test_successful_first_attempts_refill_budget() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.25, min_tokens=0, window=60.0, clock=clock)
    for _ in range(8):
        budget.record_first_attempt_success()
    snap = budget.snapshot()
    assert snap.successful_first_attempts == 8
    assert snap.balance == pytest.approx(2.0)


def test_retry_consumes_one_token() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.0, min_tokens=5, window=60.0, clock=clock)
    assert budget.try_acquire_retry() is True
    snap = budget.snapshot()
    assert snap.balance == 4.0
    assert snap.retry_attempts == 1
    assert snap.denied_retry_attempts == 0


def test_denied_retry_does_not_consume_budget() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, min_tokens=0, window=60.0, clock=clock)
    assert budget.try_acquire_retry() is False
    snap = budget.snapshot()
    assert snap.balance == 0.0
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 1


def test_balance_never_goes_negative_under_many_denied_attempts() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.0, min_tokens=1, window=60.0, clock=clock)
    assert budget.try_acquire_retry() is True
    for _ in range(100):
        assert budget.try_acquire_retry() is False
    snap = budget.snapshot()
    assert snap.balance == 0.0
    assert snap.retry_attempts == 1
    assert snap.denied_retry_attempts == 100


def test_no_shared_state_between_instances() -> None:
    clock_a = FakeClock()
    clock_b = FakeClock()
    budget_a = RetryBudget(ratio=0.1, min_tokens=5, window=60.0, clock=clock_a)
    budget_b = RetryBudget(ratio=0.5, min_tokens=1, window=60.0, clock=clock_b)

    for _ in range(3):
        assert budget_a.try_acquire_retry() is True
    budget_b.record_first_attempt_success()

    snap_a = budget_a.snapshot()
    snap_b = budget_b.snapshot()

    assert snap_a.retry_attempts == 3
    assert snap_a.balance == 2.0
    assert snap_b.retry_attempts == 0
    assert snap_b.balance == pytest.approx(1.5)
    assert snap_b.successful_first_attempts == 1


# ----------------------- fixed-window rollover -----------------------


def test_credits_do_not_carry_into_next_window() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.5, min_tokens=0, window=10.0, clock=clock)
    for _ in range(4):
        budget.record_first_attempt_success()
    assert budget.snapshot().balance == pytest.approx(2.0)

    clock.advance(10.0)

    snap = budget.snapshot()
    assert snap.successful_first_attempts == 0
    assert snap.balance == 0.0
    assert budget.try_acquire_retry() is False


def test_rollover_restores_balance_to_min_tokens() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.0, min_tokens=3, window=10.0, clock=clock)
    assert budget.try_acquire_retry() is True
    assert budget.try_acquire_retry() is True
    assert budget.try_acquire_retry() is True
    assert budget.snapshot().balance == 0.0

    clock.advance(10.0)

    assert budget.snapshot().balance == 3.0


def test_rollover_resets_successful_first_attempts() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, min_tokens=1, window=10.0, clock=clock)
    for _ in range(7):
        budget.record_first_attempt_success()
    assert budget.snapshot().successful_first_attempts == 7

    clock.advance(10.0)

    assert budget.snapshot().successful_first_attempts == 0


def test_rollover_resets_retry_attempts() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.0, min_tokens=3, window=10.0, clock=clock)
    for _ in range(3):
        assert budget.try_acquire_retry() is True
    assert budget.snapshot().retry_attempts == 3

    clock.advance(10.0)

    assert budget.snapshot().retry_attempts == 0


def test_rollover_resets_denied_retry_attempts() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.0, min_tokens=0, window=10.0, clock=clock)
    for _ in range(4):
        assert budget.try_acquire_retry() is False
    assert budget.snapshot().denied_retry_attempts == 4

    clock.advance(10.0)

    assert budget.snapshot().denied_retry_attempts == 0


def test_elapsed_time_exactly_equal_to_window_triggers_rollover() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.5, min_tokens=2, window=10.0, clock=clock)
    for _ in range(4):
        budget.record_first_attempt_success()
    assert budget.snapshot().successful_first_attempts == 4

    clock.advance(10.0)

    snap = budget.snapshot()
    assert snap.successful_first_attempts == 0
    assert snap.balance == 2.0


def test_time_just_before_boundary_does_not_trigger_rollover() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.5, min_tokens=2, window=10.0, clock=clock)
    for _ in range(4):
        budget.record_first_attempt_success()

    clock.advance(9.999)

    snap = budget.snapshot()
    assert snap.successful_first_attempts == 4
    assert snap.balance == pytest.approx(4.0)


def test_snapshot_performs_lazy_rollover() -> None:
    clock = FakeClock()
    budget = RetryBudget(ratio=0.1, min_tokens=1, window=10.0, clock=clock)
    budget.record_first_attempt_success()
    assert budget.snapshot().successful_first_attempts == 1

    clock.advance(10.0)

    # snapshot() alone must trigger the rollover; no other op was called.
    snap = budget.snapshot()
    assert snap.successful_first_attempts == 0
    assert snap.balance == 1.0
    assert snap.retry_attempts == 0
    assert snap.denied_retry_attempts == 0


def test_invariant_holds_independently_in_each_window() -> None:
    clock = FakeClock()
    ratio = 0.25
    min_tokens = 2
    budget = RetryBudget(ratio=ratio, min_tokens=min_tokens, window=10.0, clock=clock)

    # Window 1: 8 successes gives ratio*8 + min_tokens = 4 retries allowed.
    for _ in range(8):
        budget.record_first_attempt_success()
    for _ in range(4):
        assert budget.try_acquire_retry() is True
    assert budget.try_acquire_retry() is False

    snap1 = budget.snapshot()
    assert snap1.retry_attempts == 4
    ceiling1 = min_tokens + ratio * snap1.successful_first_attempts
    assert snap1.retry_attempts <= ceiling1 + _EPS

    clock.advance(10.0)

    # Window 2: 4 successes gives ratio*4 + min_tokens = 3 retries allowed.
    for _ in range(4):
        budget.record_first_attempt_success()
    for _ in range(3):
        assert budget.try_acquire_retry() is True
    assert budget.try_acquire_retry() is False

    snap2 = budget.snapshot()
    assert snap2.retry_attempts == 3
    ceiling2 = min_tokens + ratio * snap2.successful_first_attempts
    assert snap2.retry_attempts <= ceiling2 + _EPS


# --------------------- Hypothesis property (single window) ---------------------


_op_strategy = st.sampled_from(["success", "retry"])


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_tokens=st.integers(min_value=0, max_value=100),
    ops=st.lists(_op_strategy, max_size=300),
)
def test_invariant_holds_under_arbitrary_interleavings_within_window(
    ratio: float, min_tokens: int, ops: list[str]
) -> None:
    clock = FakeClock()
    # Window large enough that no rollover occurs during the op sequence.
    budget = RetryBudget(
        ratio=ratio, min_tokens=min_tokens, window=1_000_000.0, clock=clock
    )

    for op in ops:
        if op == "success":
            budget.record_first_attempt_success()
        else:
            budget.try_acquire_retry()

    snap = budget.snapshot()

    ceiling = min_tokens + ratio * snap.successful_first_attempts
    assert snap.retry_attempts <= ceiling + _EPS
    assert snap.balance >= 0.0

    n_success = sum(1 for op in ops if op == "success")
    n_retry = sum(1 for op in ops if op == "retry")
    assert snap.successful_first_attempts == n_success
    assert snap.retry_attempts + snap.denied_retry_attempts == n_retry


# ------------------------------ concurrency ------------------------------


def test_concurrent_acquisition_bounded_by_initial_capacity() -> None:
    clock = SharedClock()
    initial = 25
    workers = 100
    budget = RetryBudget(ratio=0.0, min_tokens=initial, window=1_000.0, clock=clock)

    barrier = threading.Barrier(workers)
    results: list[bool] = [False] * workers

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = budget.try_acquire_retry()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    granted = sum(results)
    snap = budget.snapshot()

    assert granted == initial
    assert snap.retry_attempts == initial
    assert snap.denied_retry_attempts == workers - initial
    assert snap.balance == 0.0


def test_concurrent_mix_of_successes_and_retries_preserves_invariant() -> None:
    clock = SharedClock()
    ratio = 0.5
    min_tokens = 5
    successes_per_thread = 20
    retry_attempts_per_thread = 20
    threads_count = 16
    budget = RetryBudget(
        ratio=ratio, min_tokens=min_tokens, window=1_000.0, clock=clock
    )

    barrier = threading.Barrier(threads_count)

    def worker() -> None:
        barrier.wait()
        for _ in range(successes_per_thread):
            budget.record_first_attempt_success()
        for _ in range(retry_attempts_per_thread):
            budget.try_acquire_retry()

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = budget.snapshot()
    assert snap.successful_first_attempts == successes_per_thread * threads_count
    assert (
        snap.retry_attempts + snap.denied_retry_attempts
        == retry_attempts_per_thread * threads_count
    )
    ceiling = min_tokens + ratio * snap.successful_first_attempts
    assert snap.retry_attempts <= ceiling + _EPS
    assert snap.balance >= 0.0


def test_concurrent_operations_at_rollover_remain_safe() -> None:
    """Concurrent ops that span a window boundary must not corrupt state.

    The clock is advanced past the boundary immediately before releasing the
    worker threads. The first op in each thread to acquire the budget's lock
    triggers rollover atomically; subsequent ops see the fresh window. All
    ops must complete without observing negative balance or violating the
    per-window invariant.
    """
    clock = SharedClock()
    ratio = 0.5
    min_tokens = 5
    budget = RetryBudget(
        ratio=ratio, min_tokens=min_tokens, window=10.0, clock=clock
    )

    # Populate window 1 so rollover has meaningful state to discard.
    for _ in range(10):
        budget.record_first_attempt_success()
    for _ in range(3):
        assert budget.try_acquire_retry() is True

    threads_count = 32
    barrier = threading.Barrier(threads_count)

    def worker(idx: int) -> None:
        barrier.wait()
        if idx % 2 == 0:
            for _ in range(5):
                budget.record_first_attempt_success()
        else:
            for _ in range(5):
                budget.try_acquire_retry()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_count)]

    # Trip rollover: all post-release ops will observe a rolled window.
    clock.advance(10.0)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = budget.snapshot()
    ceiling = min_tokens + ratio * snap.successful_first_attempts
    assert snap.retry_attempts <= ceiling + _EPS
    assert snap.balance >= 0.0
    assert snap.successful_first_attempts >= 0
    assert snap.retry_attempts >= 0
    assert snap.denied_retry_attempts >= 0
