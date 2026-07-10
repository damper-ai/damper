"""Damper: budgeted, cost-aware, streaming-safe retries for Anthropic LLM calls.

This module exposes the public API surface of Damper v0.1. Only the symbols
re-exported in :data:`__all__` are considered stable; everything else is an
implementation detail and may change without notice.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    """Raised when the client-local retry budget denies a retry."""


class RetriesExhausted(DamperError):
    """Raised when ``max_attempts`` has been reached without success."""


class RetryCostCeilingHit(DamperError):
    """Raised when the next retry would exceed ``max_retry_cost_usd``."""


class RetryOwnershipError(DamperError):
    """Raised when Damper cannot take exclusive ownership of retries."""


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
