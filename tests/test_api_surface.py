"""Public API surface smoke tests.

These tests exist so accidental renames, missing exports, or changed defaults
are caught immediately. The default values checked here mirror the documented
:class:`Policy` defaults verbatim.
"""

from __future__ import annotations

import dataclasses

import pytest

import damper
from damper import (
    DamperError,
    Policy,
    RetriesExhausted,
    RetryBudgetExhausted,
    RetryCostCeilingHit,
    RetryOwnershipError,
    resilient,
)


def test_public_symbols_are_exported() -> None:
    for name in [
        "Policy",
        "DamperError",
        "RetryBudgetExhausted",
        "RetriesExhausted",
        "RetryCostCeilingHit",
        "RetryOwnershipError",
        "resilient",
    ]:
        assert hasattr(damper, name), f"damper.{name} is not exported"
        assert name in damper.__all__, f"damper.{name} missing from __all__"


def test_exception_hierarchy() -> None:
    assert issubclass(DamperError, Exception)
    for exc in (
        RetryBudgetExhausted,
        RetriesExhausted,
        RetryCostCeilingHit,
        RetryOwnershipError,
    ):
        assert issubclass(exc, DamperError)


def test_policy_defaults_match_spec() -> None:
    policy = Policy()
    assert policy.max_attempts == 3
    assert policy.per_attempt_timeout == 60.0
    assert policy.max_retry_cost_usd is None
    assert policy.retry_budget_ratio == 0.1
    assert policy.retry_budget_window == 60.0
    assert policy.retry_budget_min_tokens == 10
    assert policy.backoff_base == 1.0
    assert policy.backoff_max == 30.0
    assert policy.respect_retry_after is True
    assert policy.on_budget_exhausted == "raise"
    assert policy.price_table is None
    assert policy.classifier is None


def test_policy_is_frozen() -> None:
    policy = Policy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_attempts = 5  # type: ignore[misc]


def test_resilient_rejects_unsupported_client() -> None:
    # A plain object exposes no with_options(), so Damper cannot take retry
    # ownership and refuses to wrap.
    with pytest.raises(RetryOwnershipError):
        resilient(object())

    with pytest.raises(RetryOwnershipError):
        resilient(object(), policy=Policy())
