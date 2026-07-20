"""Smoke tests for :mod:`tests.fakes`.

These verify the ``ScriptedFake`` basics: that scripted outcomes are consumed
in order, ``Ok`` returns its payload, and ``Err`` raises a fake provider error.
"""

from __future__ import annotations

import pytest

from tests.fakes import (
    ConnectionResetBeforeFirstToken,
    Err,
    FakeConnectionResetError,
    FakeProviderError,
    FakeTimeoutError,
    Ok,
    ScriptedFake,
    StreamThenReset,
    Timeout,
)


def test_ok_returns_payload() -> None:
    fake = ScriptedFake.from_sequence([Ok(payload="hello")])
    assert fake.next_outcome() == "hello"
    assert fake.calls == 1
    assert fake.remaining == 0


def test_err_raises_fake_provider_error() -> None:
    fake = ScriptedFake.from_sequence([Err(status_code=529, retry_after=1.5)])
    with pytest.raises(FakeProviderError) as excinfo:
        fake.next_outcome()
    assert excinfo.value.status_code == 529
    assert excinfo.value.retry_after == 1.5


def test_timeout_raises_fake_timeout() -> None:
    fake = ScriptedFake.from_sequence([Timeout()])
    with pytest.raises(FakeTimeoutError):
        fake.next_outcome()


def test_scripted_outcomes_are_consumed_in_order() -> None:
    fake = ScriptedFake.from_sequence(
        [Err(status_code=529), Err(status_code=500), Ok(payload="finally")]
    )

    with pytest.raises(FakeProviderError) as first:
        fake.next_outcome()
    assert first.value.status_code == 529

    with pytest.raises(FakeProviderError) as second:
        fake.next_outcome()
    assert second.value.status_code == 500

    assert fake.next_outcome() == "finally"
    assert fake.calls == 3
    assert fake.remaining == 0


def test_stream_outcomes_raise_fake_connection_reset() -> None:
    fake = ScriptedFake.from_sequence(
        [ConnectionResetBeforeFirstToken(), StreamThenReset(tokens=2)]
    )
    with pytest.raises(FakeConnectionResetError):
        fake.next_outcome()
    with pytest.raises(FakeConnectionResetError):
        fake.next_outcome()


def test_exhausted_scripted_fake_raises_assertion_error() -> None:
    fake = ScriptedFake.from_sequence([Ok()])
    fake.next_outcome()
    with pytest.raises(AssertionError):
        fake.next_outcome()
