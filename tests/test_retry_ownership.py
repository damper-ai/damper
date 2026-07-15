"""Retry-ownership tests (SESSION 6, SPEC section 7).

Damper must own retries for intercepted calls and never stack on top of the
Anthropic SDK's own retries. With a fake client the ``[529, ok] -> 2 attempts``
count holds even if ``max_retries=0`` were forgotten (the fake has no SDK retry
loop to disable), so ownership is asserted two ways: (a) ``with_options`` was
called with ``max_retries=0`` (the supported disable mechanism), and (b) the
provider was invoked exactly twice.

This is wrap-time ownership -- Damper produced a ``max_retries=0`` client via the
supported mechanism -- not a runtime proof the real SDK never retries.

No network access, no real API key.
"""

from __future__ import annotations

import pytest

from damper import Policy, RetryOwnershipError, resilient
from tests.fakes import (
    Err,
    FakeAnthropic,
    FakeAsyncAnthropic,
    FakeClientNoWithOptions,
    FakeClientWithOptionsRaises,
    Ok,
)

# Zero backoff so retrying paths do not sleep on the real clock.
FAST = Policy(backoff_base=0.0, backoff_max=0.0)

_REQUEST = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


def test_wrap_disables_sdk_retries_via_with_options() -> None:
    client = FakeAnthropic([Ok()])
    resilient(client, policy=FAST)
    assert client.with_options_calls, "with_options was never called"
    assert client.with_options_calls[0]["max_retries"] == 0
    assert client.with_options_calls[0]["timeout"] == FAST.per_attempt_timeout


def test_529_then_ok_is_exactly_two_provider_attempts() -> None:
    client = FakeAnthropic([Err(529), Ok()])
    wrapped = resilient(client, policy=FAST)
    resp = wrapped.messages.create(**_REQUEST)

    # (a) SDK retries disabled at wrap time ...
    assert client.with_options_calls[0]["max_retries"] == 0
    # (b) ... and the provider saw exactly two attempts (never SDK x Damper).
    assert len(client.create_calls) == 2
    assert resp.damper.attempts == 2


async def test_async_529_then_ok_is_exactly_two_provider_attempts() -> None:
    client = FakeAsyncAnthropic([Err(529), Ok()])
    wrapped = resilient(client, policy=FAST)
    resp = await wrapped.messages.create(**_REQUEST)

    assert client.with_options_calls[0]["max_retries"] == 0
    assert len(client.create_calls) == 2
    assert resp.damper.attempts == 2


def test_per_call_max_retries_override_is_rejected() -> None:
    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(RetryOwnershipError):
        wrapped.messages.create(max_retries=5, **_REQUEST)
    # Rejected before the provider was ever called.
    assert client.create_calls == []


async def test_async_per_call_max_retries_override_is_rejected() -> None:
    client = FakeAsyncAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(RetryOwnershipError):
        await wrapped.messages.create(max_retries=5, **_REQUEST)
    assert client.create_calls == []


def test_cannot_disable_retries_without_with_options() -> None:
    with pytest.raises(RetryOwnershipError) as excinfo:
        resilient(FakeClientNoWithOptions(), policy=FAST)
    assert excinfo.value.client_type == "FakeClientNoWithOptions"


def test_cannot_disable_retries_when_with_options_raises() -> None:
    with pytest.raises(RetryOwnershipError) as excinfo:
        resilient(FakeClientWithOptionsRaises(), policy=FAST)
    assert excinfo.value.client_type == "FakeClientWithOptionsRaises"
    assert isinstance(excinfo.value.__cause__, RuntimeError)
