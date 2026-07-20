"""Wrapper tests: proxy behavior, create paths, policy validation, and
Retry-After header normalization.

The executor tests already prove that a normalized Retry-After value is used
exactly and uncapped, and that ``None`` falls back to full jitter; these tests
prove the wrapper produces the correct normalized ``float | None`` from a
provider ``Retry-After`` header (numeric, HTTP-date, past, negative, invalid,
missing, case-insensitive).

No network access, no real API key.
"""

from __future__ import annotations

from datetime import datetime, timezone

import anthropic
import pytest

from damper import Policy, RetriesExhausted, resilient
from damper._wrapper import (
    _get_header_ci,
    _make_retry_after_extractor,
    _parse_retry_after_header,
    _validate_policy,
    _WrappedClient,
)
from tests.fakes import (
    Err,
    FakeAnthropic,
    FakeAsyncAnthropic,
    FakeMessage,
    Ok,
    anthropic_connection_error,
    anthropic_status_error,
)

UTC = timezone.utc
FAST = Policy(backoff_base=0.0, backoff_max=0.0)
_REQUEST = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


# ---------------------------- proxy shape ---------------------------------- #


def test_resilient_returns_wrapper() -> None:
    wrapped = resilient(FakeAnthropic([]), policy=FAST)
    assert isinstance(wrapped, _WrappedClient)
    assert wrapped.messages is not None


def test_passthrough_client_attribute() -> None:
    # A non-intercepted client attribute delegates to the underlying client.
    client = FakeAnthropic([])
    wrapped = resilient(client, policy=FAST)
    assert wrapped.with_options_calls is client.with_options_calls


def test_passthrough_messages_attribute() -> None:
    # A non-intercepted messages method (count_tokens) delegates through.
    client = FakeAnthropic([])
    wrapped = resilient(client, policy=FAST)
    assert wrapped.messages.count_tokens() == 4242


# ---------------------------- create paths --------------------------------- #


def test_sync_create_attaches_metadata() -> None:
    client = FakeAnthropic([Ok(FakeMessage(id="m1"))])
    wrapped = resilient(client, policy=FAST)
    resp = wrapped.messages.create(**_REQUEST)

    assert resp.id == "m1"
    assert resp.damper.attempts == 1
    assert resp.damper.retried is False
    assert resp.damper.outcome == "ok"
    assert len(client.create_calls) == 1


async def test_async_create_attaches_metadata() -> None:
    client = FakeAsyncAnthropic([Ok(FakeMessage(id="a1"))])
    wrapped = resilient(client, policy=FAST)
    resp = await wrapped.messages.create(**_REQUEST)

    assert resp.id == "a1"
    assert resp.damper.attempts == 1
    assert resp.damper.retried is False


def test_not_retryable_surfaces_original_anthropic_error() -> None:
    client = FakeAnthropic([Err(400)])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(anthropic.APIStatusError) as excinfo:
        wrapped.messages.create(**_REQUEST)
    assert excinfo.value.status_code == 400
    assert len(client.create_calls) == 1


def test_retries_exhausted_preserves_provider_error_as_cause() -> None:
    client = FakeAnthropic([Err(529), Err(529), Err(529)])
    wrapped = resilient(client, policy=FAST)  # default max_attempts=3
    with pytest.raises(RetriesExhausted) as excinfo:
        wrapped.messages.create(**_REQUEST)
    assert isinstance(excinfo.value.__cause__, anthropic.APIStatusError)
    assert len(client.create_calls) == 3


# ---------------------- Retry-After normalization -------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("7", 7.0),
        ("0", 0.0),
        ("7.5", 7.5),
        ("-3", None),
        ("", None),
        ("   ", None),
        ("garbage", None),
        ("nan", None),
        ("inf", None),
    ],
)
def test_parse_retry_after_numeric(raw: str, expected: float | None) -> None:
    now = datetime(2020, 1, 1, tzinfo=UTC)
    assert _parse_retry_after_header(raw, now=now) == expected


def test_parse_retry_after_http_date_future() -> None:
    now = datetime(2015, 10, 21, 7, 27, 0, tzinfo=UTC)
    got = _parse_retry_after_header("Wed, 21 Oct 2015 07:28:00 GMT", now=now)
    assert got == 60.0


def test_parse_retry_after_http_date_past_is_zero() -> None:
    now = datetime(2015, 10, 21, 7, 29, 0, tzinfo=UTC)
    got = _parse_retry_after_header("Wed, 21 Oct 2015 07:28:00 GMT", now=now)
    assert got == 0.0


def test_get_header_ci_is_case_insensitive() -> None:
    assert _get_header_ci({"Retry-After": "7"}, "retry-after") == "7"
    assert _get_header_ci({"RETRY-AFTER": "7"}, "Retry-After") == "7"
    assert _get_header_ci({}, "retry-after") is None
    assert _get_header_ci(None, "retry-after") is None


def test_extractor_reads_numeric_header() -> None:
    extract = _make_retry_after_extractor(now=lambda: datetime(2020, 1, 1, tzinfo=UTC))
    assert extract(anthropic_status_error(429, retry_after=7)) == 7.0


def test_extractor_reads_http_date_header() -> None:
    extract = _make_retry_after_extractor(
        now=lambda: datetime(2015, 10, 21, 7, 27, 0, tzinfo=UTC)
    )
    err = anthropic_status_error(429, retry_after="Wed, 21 Oct 2015 07:28:00 GMT")
    assert extract(err) == 60.0


def test_extractor_missing_header_is_none() -> None:
    extract = _make_retry_after_extractor()
    assert extract(anthropic_status_error(429)) is None


def test_extractor_negative_header_is_none() -> None:
    extract = _make_retry_after_extractor()
    assert extract(anthropic_status_error(429, retry_after=-5)) is None


def test_extractor_without_response_is_none() -> None:
    extract = _make_retry_after_extractor()
    assert extract(anthropic_connection_error()) is None


# -------------------------- policy validation ------------------------------ #


@pytest.mark.parametrize(
    "policy",
    [
        Policy(max_attempts=0),
        Policy(max_attempts=True),  # bool is not a valid int count
        Policy(per_attempt_timeout=0.0),
        Policy(per_attempt_timeout=-1.0),
        Policy(per_attempt_timeout=float("nan")),
        Policy(max_retry_cost_usd=-1.0),
        Policy(retry_budget_ratio=-0.1),
        Policy(retry_budget_window=0.0),
        Policy(retry_budget_min_tokens=-1),
        Policy(retry_budget_min_tokens=True),
        Policy(backoff_base=-1.0),
        Policy(backoff_max=-1.0),
        Policy(backoff_base=10.0, backoff_max=1.0),
        Policy(respect_retry_after="yes"),  # type: ignore[arg-type]
        Policy(on_budget_exhausted="nope"),  # type: ignore[arg-type]
    ],
)
def test_invalid_policy_rejected(policy: Policy) -> None:
    with pytest.raises(ValueError):
        _validate_policy(policy)


def test_default_policy_is_valid() -> None:
    _validate_policy(Policy())  # must not raise


def test_invalid_policy_raises_before_with_options_or_state() -> None:
    client = FakeAnthropic([Ok()])
    with pytest.raises(ValueError, match="max_attempts"):
        resilient(client, policy=Policy(max_attempts=0))
    # Validation happens before retry-ownership configuration or wrapper state.
    assert client.with_options_calls == []


def test_backoff_ordering_message_is_field_specific() -> None:
    with pytest.raises(ValueError, match="backoff_base"):
        _validate_policy(Policy(backoff_base=5.0, backoff_max=1.0))
