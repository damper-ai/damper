"""Client-local fixed-window retry budget.

Reliability-critical. Uses fixed-window rollover: each window is
``Policy.retry_budget_window`` seconds long. At the start of every public
operation, while holding the module's :class:`threading.Lock`, the budget
checks whether the current window has elapsed and, if so, resets its state
(balance to ``min_tokens``, all counters to zero, window anchor to the
current clock reading). No credit carries across windows.

Per-window invariant preserved by construction:

    retry_attempts <= min_tokens + ratio * successful_first_attempts

Balance never goes negative. Denied retries increment only
``denied_retry_attempts`` and never touch balance or ``retry_attempts``.

Concurrency
-----------

A single :class:`threading.Lock` guards every state mutation, including
the rollover check and the snapshot read. The lock is never held across an
``await``: operations here are O(1) and never suspend. This is why the
module intentionally does not use :class:`asyncio.Lock` -- the same budget
instance must be safe from sync callers, async callers, and multiple
threads simultaneously, and ``asyncio.Lock`` only serializes callers inside
a single event loop.

Clock
-----

The clock is injectable so tests can drive rollover deterministically
without wall-clock sleeps. The default is :func:`time.monotonic`, which is
what production callers should use.

Scope
-----

One budget per wrapped client instance. No process-global state, no
distributed coordination, no external dependencies.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSnapshot:
    """Point-in-time view of a :class:`RetryBudget`.

    Reading a snapshot does not mutate the underlying budget beyond a lazy
    rollover if the current window has elapsed. Snapshots are safe to log
    or emit as telemetry attributes.
    """

    balance: float
    ratio: float
    min_tokens: int
    window: float
    successful_first_attempts: int
    retry_attempts: int
    denied_retry_attempts: int


class RetryBudget:
    """Client-local fixed-window retry budget.

    One instance per wrapped client. Not process-global. Two independent
    instances share no state.
    """

    __slots__ = (
        "_balance",
        "_clock",
        "_denied_retry_attempts",
        "_lock",
        "_min_tokens",
        "_ratio",
        "_retry_attempts",
        "_successful_first_attempts",
        "_window",
        "_window_started_at",
    )

    def __init__(
        self,
        *,
        ratio: float,
        min_tokens: int,
        window: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError(f"ratio must be finite and >= 0, got {ratio!r}")
        if min_tokens < 0:
            raise ValueError(f"min_tokens must be >= 0, got {min_tokens!r}")
        if not math.isfinite(window) or window <= 0:
            raise ValueError(f"window must be finite and > 0, got {window!r}")

        self._ratio: float = float(ratio)
        self._min_tokens: int = int(min_tokens)
        self._window: float = float(window)
        self._clock: Callable[[], float] = clock
        self._lock: threading.Lock = threading.Lock()

        self._window_started_at: float = clock()
        self._balance: float = float(min_tokens)
        self._successful_first_attempts: int = 0
        self._retry_attempts: int = 0
        self._denied_retry_attempts: int = 0

    def _maybe_rollover_locked(self) -> None:
        """Reset window state if the current window has elapsed.

        Must be called with ``self._lock`` held. Uses ``>=`` at the
        boundary so an elapsed time exactly equal to ``window`` triggers
        rollover.
        """
        now = self._clock()
        if now - self._window_started_at >= self._window:
            self._window_started_at = now
            self._balance = float(self._min_tokens)
            self._successful_first_attempts = 0
            self._retry_attempts = 0
            self._denied_retry_attempts = 0

    def record_first_attempt_success(self) -> None:
        """Deposit ``ratio`` tokens for a successful first attempt.

        Only successful first attempts refill the budget. Failed first
        attempts and successful retries never refill; that asymmetry is
        what makes the budget drain during a provider brownout. Credit
        never carries across windows.
        """
        with self._lock:
            self._maybe_rollover_locked()
            self._successful_first_attempts += 1
            self._balance += self._ratio

    def try_acquire_retry(self) -> bool:
        """Try to withdraw one token for a retry attempt.

        Returns ``True`` if the retry is allowed and one token was
        withdrawn. Returns ``False`` if the balance is below ``1.0``; in
        that case the balance is unchanged and only
        ``denied_retry_attempts`` is incremented.
        """
        with self._lock:
            self._maybe_rollover_locked()
            if self._balance < 1.0:
                self._denied_retry_attempts += 1
                return False
            self._balance -= 1.0
            self._retry_attempts += 1
            return True

    def snapshot(self) -> BudgetSnapshot:
        """Return a consistent point-in-time snapshot of this budget.

        Performs a lazy rollover if the current window has elapsed, so
        callers reading state after a quiet period observe a
        freshly-reset window rather than stale counters.
        """
        with self._lock:
            self._maybe_rollover_locked()
            return BudgetSnapshot(
                balance=self._balance,
                ratio=self._ratio,
                min_tokens=self._min_tokens,
                window=self._window,
                successful_first_attempts=self._successful_first_attempts,
                retry_attempts=self._retry_attempts,
                denied_retry_attempts=self._denied_retry_attempts,
            )
