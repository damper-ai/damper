"""Tests for :mod:`damper.classify`.

Covers the classification matrix. Uses real Anthropic SDK exception types where
they construct cleanly network-free, plus lightweight fakes for the broader
status matrix and edge cases. No network access.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest

from damper.classify import ErrorClass, classify

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class _StatusFake(Exception):
    """Network-free stand-in exposing an integer ``status_code``."""

    def __init__(self, status_code: object) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _anthropic_status_error(status_code: int) -> anthropic.APIStatusError:
    """Build a real ``anthropic.APIStatusError`` without any network call."""
    response = httpx.Response(status_code=status_code, request=_REQUEST)
    return anthropic.APIStatusError("boom", response=response, body=None)


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message="connection reset", request=_REQUEST)


def _timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_REQUEST)


# --------------------------- enum ---------------------------


def test_error_class_values_match_spec() -> None:
    assert ErrorClass.RETRYABLE.value == "retryable"
    assert ErrorClass.NOT_RETRYABLE.value == "not_retryable"
    assert ErrorClass.AMBIGUOUS.value == "ambiguous"


# --------------------------- transport errors ---------------------------


def test_api_connection_error_retryable_before_stream() -> None:
    assert classify(_connection_error()) is ErrorClass.RETRYABLE


def test_api_timeout_error_retryable_before_stream() -> None:
    assert classify(_timeout_error()) is ErrorClass.RETRYABLE


def test_connection_error_after_stream_is_ambiguous() -> None:
    assert classify(_connection_error(), stream_started=True) is ErrorClass.AMBIGUOUS


def test_timeout_error_after_stream_is_ambiguous() -> None:
    assert classify(_timeout_error(), stream_started=True) is ErrorClass.AMBIGUOUS


# --------------------------- status matrix ---------------------------


def test_real_anthropic_status_error_429_retryable() -> None:
    # At least one case exercises a genuine anthropic.APIStatusError end to end.
    assert classify(_anthropic_status_error(429)) is ErrorClass.RETRYABLE


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 529, 599])
def test_retryable_status_codes(code: int) -> None:
    assert classify(_StatusFake(code)) is ErrorClass.RETRYABLE


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 413])
def test_listed_non_retryable_4xx(code: int) -> None:
    assert classify(_StatusFake(code)) is ErrorClass.NOT_RETRYABLE


def test_409_conflict_not_retryable() -> None:
    assert classify(_StatusFake(409)) is ErrorClass.NOT_RETRYABLE


@pytest.mark.parametrize("code", [405, 418, 422, 451, 499])
def test_unknown_4xx_not_retryable(code: int) -> None:
    assert classify(_StatusFake(code)) is ErrorClass.NOT_RETRYABLE


@pytest.mark.parametrize("code", [499, 600, 700])
def test_range_boundaries_not_retryable(code: int) -> None:
    # 5xx is retryable; anything just outside the band is not.
    assert classify(_StatusFake(code)) is ErrorClass.NOT_RETRYABLE


# --------------------------- degenerate inputs ---------------------------


def test_unknown_exception_not_retryable() -> None:
    assert classify(ValueError("nope")) is ErrorClass.NOT_RETRYABLE


def test_boolean_status_code_falls_through_to_not_retryable() -> None:
    # bool is an int subclass, but True must not be read as HTTP status 1.
    assert classify(_StatusFake(True)) is ErrorClass.NOT_RETRYABLE


def test_non_integer_status_code_falls_through_to_not_retryable() -> None:
    assert classify(_StatusFake("429")) is ErrorClass.NOT_RETRYABLE


# --------------------------- custom classifier hook ---------------------------


def test_custom_classifier_override_wins() -> None:
    def always_retry(error: BaseException, *, stream_started: bool) -> ErrorClass:
        return ErrorClass.RETRYABLE

    # 400 would default to NOT_RETRYABLE; the hook overrides it.
    assert classify(_StatusFake(400), classifier=always_retry) is ErrorClass.RETRYABLE


def test_custom_classifier_none_falls_back_to_default() -> None:
    def defer(error: BaseException, *, stream_started: bool) -> None:
        return None

    assert classify(_StatusFake(400), classifier=defer) is ErrorClass.NOT_RETRYABLE
    assert classify(_connection_error(), classifier=defer) is ErrorClass.RETRYABLE


def test_custom_classifier_receives_stream_context() -> None:
    seen: list[bool] = []

    def record(error: BaseException, *, stream_started: bool) -> None:
        seen.append(stream_started)
        return None

    classify(_connection_error(), stream_started=True, classifier=record)
    classify(_connection_error(), stream_started=False, classifier=record)
    assert seen == [True, False]


def test_custom_classifier_invalid_return_raises_typeerror() -> None:
    def bogus(error: BaseException, *, stream_started: bool) -> object:
        return "retryable"

    with pytest.raises(TypeError):
        classify(_StatusFake(500), classifier=bogus)  # type: ignore[arg-type]


def test_custom_classifier_exception_propagates_unchanged() -> None:
    class Boom(RuntimeError):
        pass

    def explode(error: BaseException, *, stream_started: bool) -> ErrorClass:
        raise Boom("hook failed")

    with pytest.raises(Boom):
        classify(_StatusFake(500), classifier=explode)
