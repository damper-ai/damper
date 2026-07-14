"""Damper: budgeted, cost-aware, streaming-safe retries for Anthropic LLM calls.

This module exposes the public API surface of Damper v0.1. Only the symbols
re-exported in :data:`__all__` are considered stable; everything else is an
implementation detail and may change without notice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from damper.classify import ErrorClassifier
    from damper.prices import ModelPrice

__all__ = [
    "Policy",
    "DamperError",
    "RetryBudgetExhausted",
    "RetriesExhausted",
    "RetryCostCeilingHit",
    "RetryOwnershipError",
    "resilient",
]


class DamperError(Exception):
    """Base class for all Damper-raised exceptions."""


class RetryBudgetExhausted(DamperError):
    """Raised when the client-local retry budget denies a retry.

    Carries the metadata required by ``SPEC.md`` section 9.2. The final provider
    error is preserved both in :attr:`last_provider_error` and, when raised via
    ``raise ... from``, as ``__cause__``.
    """

    def __init__(
        self,
        *,
        attempts: int,
        retry_budget_balance: float,
        retry_budget_ratio: float,
        last_provider_error: BaseException,
    ) -> None:
        super().__init__(
            f"retry budget exhausted after {attempts} attempt(s): "
            f"balance {retry_budget_balance:.4f} < 1.0 token "
            f"(ratio {retry_budget_ratio})"
        )
        self.attempts = attempts
        self.retry_budget_balance = retry_budget_balance
        self.retry_budget_ratio = retry_budget_ratio
        self.last_provider_error = last_provider_error


class RetriesExhausted(DamperError):
    """Raised when ``max_attempts`` has been reached without success.

    Carries the metadata required by ``SPEC.md`` section 9.2, including a bounded
    per-attempt outcome record (:attr:`attempt_outcomes`).
    """

    def __init__(
        self,
        *,
        attempts: int,
        attempt_outcomes: Sequence[Any],
        last_provider_error: BaseException,
    ) -> None:
        super().__init__(f"retries exhausted after {attempts} attempt(s)")
        self.attempts = attempts
        self.attempt_outcomes: tuple[Any, ...] = tuple(attempt_outcomes)
        self.last_provider_error = last_provider_error


class RetryCostCeilingHit(DamperError):
    """Raised when the next retry would exceed ``max_retry_cost_usd``.

    Carries the metadata required by ``SPEC.md`` section 9.2. The projected
    cumulative retry cost is ``None`` when it cannot be estimated (SPEC section
    13.4); the configured ceiling is always known when this is raised.
    """

    def __init__(
        self,
        *,
        attempts: int,
        estimated_retry_cost_usd: float | None,
        max_retry_cost_usd: float,
        last_provider_error: BaseException,
    ) -> None:
        estimate = (
            "unknown"
            if estimated_retry_cost_usd is None
            else f"${estimated_retry_cost_usd:.6f}"
        )
        super().__init__(
            f"retry cost ceiling hit after {attempts} attempt(s): "
            f"projected {estimate} exceeds ceiling ${max_retry_cost_usd:.6f}"
        )
        self.attempts = attempts
        self.estimated_retry_cost_usd = estimated_retry_cost_usd
        self.max_retry_cost_usd = max_retry_cost_usd
        self.last_provider_error = last_provider_error


class RetryOwnershipError(DamperError):
    """Raised when Damper cannot take exclusive ownership of retries.

    Carries the metadata required by ``SPEC.md`` section 9.2. Raised at wrap time
    by the Anthropic wrapper (SESSION 6), not by the executor.
    """

    def __init__(self, *, reason: str, client_type: str) -> None:
        super().__init__(f"cannot take retry ownership of {client_type}: {reason}")
        self.reason = reason
        self.client_type = client_type


@dataclass(frozen=True)
class Policy:
    """Configuration for a Damper-wrapped client.

    All defaults are drawn from ``SPEC.md`` and are chosen to be production-sane
    without further tuning. Fields are frozen after construction; build a new
    :class:`Policy` to change behavior.
    """

    # Per-request limits
    max_attempts: int = 3
    per_attempt_timeout: float = 60.0
    max_retry_cost_usd: float | None = None

    # Client-local retry budget
    retry_budget_ratio: float = 0.1
    retry_budget_window: float = 60.0
    retry_budget_min_tokens: int = 10

    # Backoff
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    respect_retry_after: bool = True

    # Behavior on budget exhaustion
    on_budget_exhausted: Literal["raise", "passthrough"] = "raise"

    # Cost estimation
    price_table: Mapping[str, ModelPrice] | None = None

    # Error classification override
    classifier: ErrorClassifier | None = None


def resilient(client: Any, *, policy: Policy | None = None) -> Any:
    """Wrap an Anthropic client with Damper's owned retry executor.

    Not implemented in the SESSION 1 scaffold. Later sessions will return a
    proxy exposing the same interface as ``client`` while intercepting
    ``messages.create`` and ``messages.stream`` (sync and async).
    """

    raise NotImplementedError(
        "damper.resilient() is not implemented in the SESSION 1 scaffold."
    )
