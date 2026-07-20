"""Tests for :mod:`damper.backoff`.

Covers the backoff matrix: deterministic injected RNG, overflow-safe upper
bound, retry-after semantics, and strict input/RNG validation. No sleeping, no
network.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from damper.backoff import compute_backoff


def _const_rng(value: float):
    def rng() -> float:
        return value

    return rng


# --------------------------- full jitter bounds ---------------------------


def test_first_retry_upper_bound_is_base_when_below_cap() -> None:
    # retry_number == 1 -> upper bound = min(cap, base * 2**0) = base.
    assert compute_backoff(1, base=2.0, cap=30.0, rng=_const_rng(1.0)) == 2.0
    assert compute_backoff(1, base=2.0, cap=30.0, rng=_const_rng(0.5)) == 1.0


def test_first_retry_lower_bound_is_zero() -> None:
    assert compute_backoff(1, base=2.0, cap=30.0, rng=_const_rng(0.0)) == 0.0


def test_later_retry_exponential_growth() -> None:
    # retry_number == 3 -> base * 2**2 = 4.0 (below cap).
    assert compute_backoff(3, base=1.0, cap=100.0, rng=_const_rng(1.0)) == 4.0
    assert compute_backoff(3, base=1.0, cap=100.0, rng=_const_rng(0.5)) == 2.0


def test_cap_clamps_exponential_growth() -> None:
    # retry_number == 10 -> base * 2**9 = 512, clamped to cap == 30.
    assert compute_backoff(10, base=1.0, cap=30.0, rng=_const_rng(1.0)) == 30.0


def test_very_large_retry_number_is_cap_without_overflow() -> None:
    # Must not build a huge 2**(n-1) integer or raise; upper bound is just cap.
    assert compute_backoff(10**18, base=1.0, cap=30.0, rng=_const_rng(1.0)) == 30.0


def test_zero_base_yields_zero() -> None:
    assert compute_backoff(5, base=0.0, cap=30.0, rng=_const_rng(1.0)) == 0.0


# --------------------------- retry-after ---------------------------


def test_retry_after_returned_exactly_without_jitter() -> None:
    # RNG would return a non-zero jitter; retry-after must win verbatim.
    assert (
        compute_backoff(2, base=1.0, cap=30.0, retry_after=7.0, rng=_const_rng(1.0)) == 7.0
    )


def test_retry_after_not_capped() -> None:
    # Provider instruction is not clamped to cap.
    assert (
        compute_backoff(1, base=1.0, cap=30.0, retry_after=120.0, rng=_const_rng(1.0))
        == 120.0
    )


def test_retry_after_zero_is_honored() -> None:
    assert compute_backoff(1, base=1.0, cap=30.0, retry_after=0.0, rng=_const_rng(1.0)) == 0.0


def test_retry_after_ignored_when_disabled() -> None:
    # respect_retry_after False -> jitter path (rng drives the result).
    result = compute_backoff(
        1,
        base=4.0,
        cap=30.0,
        retry_after=7.0,
        respect_retry_after=False,
        rng=_const_rng(0.5),
    )
    assert result == 2.0


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_retry_after_falls_back_to_jitter(bad: float) -> None:
    result = compute_backoff(1, base=4.0, cap=30.0, retry_after=bad, rng=_const_rng(0.25))
    assert result == 1.0


# --------------------------- input validation ---------------------------


def test_retry_number_zero_raises() -> None:
    with pytest.raises(ValueError):
        compute_backoff(0, base=1.0, cap=30.0, rng=_const_rng(0.5))


def test_retry_number_non_integer_raises() -> None:
    with pytest.raises(ValueError):
        compute_backoff(1.5, base=1.0, cap=30.0, rng=_const_rng(0.5))  # type: ignore[arg-type]


def test_retry_number_bool_raises() -> None:
    with pytest.raises(ValueError):
        compute_backoff(True, base=1.0, cap=30.0, rng=_const_rng(0.5))


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_base_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        compute_backoff(1, base=bad, cap=30.0, rng=_const_rng(0.5))


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_cap_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        compute_backoff(1, base=1.0, cap=bad, rng=_const_rng(0.5))


# --------------------------- rng validation ---------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan, math.inf, -math.inf])
def test_invalid_rng_output_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        compute_backoff(1, base=1.0, cap=30.0, rng=_const_rng(bad))


def test_rng_not_called_when_retry_after_returned() -> None:
    calls = 0

    def rng() -> float:
        nonlocal calls
        calls += 1
        return 0.5

    compute_backoff(1, base=1.0, cap=30.0, retry_after=7.0, rng=rng)
    assert calls == 0


def test_rng_called_exactly_once_in_jitter_path() -> None:
    calls = 0

    def rng() -> float:
        nonlocal calls
        calls += 1
        return 0.5

    compute_backoff(1, base=1.0, cap=30.0, rng=rng)
    assert calls == 1


def test_default_rng_stays_within_bounds() -> None:
    for _ in range(1000):
        delay = compute_backoff(3, base=1.0, cap=30.0)
        assert 0.0 <= delay <= 4.0


# --------------------------- property ---------------------------


@given(
    retry_number=st.integers(min_value=1, max_value=64),
    base=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    cap=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    draw=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_jitter_within_bounds_property(
    retry_number: int, base: float, cap: float, draw: float
) -> None:
    delay = compute_backoff(retry_number, base=base, cap=cap, rng=_const_rng(draw))
    upper = min(cap, base * 2 ** (retry_number - 1))
    assert 0.0 <= delay <= upper + 1e-9
