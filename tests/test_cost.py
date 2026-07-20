"""Tests for :mod:`damper.cost` and :mod:`damper.prices`.

Covers the cost matrix: unknown estimates are ``None``, a configured ceiling
denies an unknown cost, the ceiling is cumulative across retries with a strict
greater-than boundary, ``max_tokens`` drives the output reservation, non-text
content yields unknown, and strict numeric validation. No network, no provider
token-counting API.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from damper.cost import (
    estimate_input_tokens,
    estimate_output_tokens,
    estimate_retry_cost_usd,
    would_exceed_retry_cost_ceiling,
)
from damper.prices import DEFAULT_PRICE_TABLE, ModelPrice

# ------------------------------- prices --------------------------------------


def test_model_price_is_frozen() -> None:
    price = ModelPrice("claude-opus-4-8", 5.0, 25.0, "2026-07-12")
    with pytest.raises(dataclasses.FrozenInstanceError):
        price.input_price_per_million_tokens = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, -math.inf, True])
def test_model_price_rejects_invalid_input_price(bad: object) -> None:
    with pytest.raises(ValueError):
        ModelPrice("m", bad, 25.0, "2026-07-12")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_model_price_rejects_invalid_output_price(bad: float) -> None:
    with pytest.raises(ValueError):
        ModelPrice("m", 5.0, bad, "2026-07-12")


def test_default_table_is_read_only() -> None:
    # The shared default table must not be mutable at runtime; overrides go
    # through Policy.price_table, not by writing to the shared object.
    with pytest.raises(TypeError):
        DEFAULT_PRICE_TABLE["claude-opus-4-8"] = ModelPrice(  # type: ignore[index]
            "claude-opus-4-8", 0.0, 0.0, "2026-07-12"
        )
    with pytest.raises(TypeError):
        del DEFAULT_PRICE_TABLE["claude-opus-4-8"]  # type: ignore[attr-defined]


def test_default_table_excludes_retired_model_ids() -> None:
    # Retired / no-longer-available model IDs must never appear as if current.
    retired = {
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
        "claude-2.1",
        "claude-2.0",
    }
    assert retired.isdisjoint(DEFAULT_PRICE_TABLE)


def test_default_table_entries_are_well_formed() -> None:
    assert DEFAULT_PRICE_TABLE  # non-empty
    for model, price in DEFAULT_PRICE_TABLE.items():
        assert price.model == model
        assert price.last_verified
        assert math.isfinite(price.input_price_per_million_tokens)
        assert math.isfinite(price.output_price_per_million_tokens)
        assert price.input_price_per_million_tokens >= 0
        assert price.output_price_per_million_tokens >= 0


# --------------------------- input token estimation --------------------------


def test_input_tokens_chars_over_four_fallback() -> None:
    request = {
        "system": "abcd",  # 4 chars
        "messages": [{"role": "user", "content": "abcdefgh"}],  # 8 chars
    }
    # ceil((4 + 8) / 4) == 3
    assert estimate_input_tokens(request) == 3


def test_input_tokens_ceils() -> None:
    request = {"messages": [{"role": "user", "content": "abcde"}]}  # 5 chars
    # ceil(5 / 4) == 2
    assert estimate_input_tokens(request) == 2


def test_input_tokens_prefers_valid_usage() -> None:
    request = {"messages": [{"role": "user", "content": "a" * 400}]}

    class Usage:
        input_tokens = 42

    assert estimate_input_tokens(request, usage=Usage()) == 42


def test_input_tokens_usage_mapping_supported() -> None:
    request = {"messages": [{"role": "user", "content": "a" * 400}]}
    assert estimate_input_tokens(request, usage={"input_tokens": 7}) == 7


@pytest.mark.parametrize("bad", [True, -3])
def test_input_tokens_ignores_invalid_usage(bad: object) -> None:
    request = {"messages": [{"role": "user", "content": "abcd"}]}  # 4 chars -> 1
    assert estimate_input_tokens(request, usage={"input_tokens": bad}) == 1


def test_input_tokens_none_for_non_text_content_without_usage() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "source": {"type": "base64", "data": "...."}},
                ],
            }
        ]
    }
    assert estimate_input_tokens(request) is None


def test_input_tokens_uses_usage_even_with_non_text_content() -> None:
    request = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "source": {}}]},
        ]
    }
    assert estimate_input_tokens(request, usage={"input_tokens": 99}) == 99


def test_input_tokens_handles_text_block_lists() -> None:
    request = {
        "system": [{"type": "text", "text": "abcd"}],  # 4
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "efgh"}]},  # 4
        ],
    }
    assert estimate_input_tokens(request) == 2  # ceil(8 / 4)


# --------------------------- output token estimation -------------------------


def test_output_tokens_uses_max_tokens() -> None:
    assert estimate_output_tokens({"max_tokens": 512}) == 512


def test_output_tokens_none_when_absent() -> None:
    assert estimate_output_tokens({"messages": []}) is None


@pytest.mark.parametrize("bad", [True, -1, 1.5, "512", None])
def test_output_tokens_none_for_invalid_max_tokens(bad: object) -> None:
    assert estimate_output_tokens({"max_tokens": bad}) is None


def test_output_tokens_ignores_prior_output_usage() -> None:
    # No usage parameter exists by design: a prior smaller output must not
    # reduce the next attempt's reservation. max_tokens is the reservation.
    assert estimate_output_tokens({"max_tokens": 4096}) == 4096


# ----------------------------- retry cost estimate ---------------------------


def test_retry_cost_known_model_hand_computed() -> None:
    # opus-4-8: $5 / 1M input, $25 / 1M output.
    # 1_000_000 input -> $5; 100_000 output -> $2.5; total $7.5.
    cost = estimate_retry_cost_usd(
        model="claude-opus-4-8",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )
    assert cost == pytest.approx(7.5)


def test_retry_cost_unknown_model_is_none() -> None:
    assert (
        estimate_retry_cost_usd(
            model="no-such-model",
            input_tokens=100,
            output_tokens=100,
        )
        is None
    )


def test_retry_cost_none_when_input_unknown() -> None:
    assert (
        estimate_retry_cost_usd(
            model="claude-opus-4-8",
            input_tokens=None,
            output_tokens=100,
        )
        is None
    )


def test_retry_cost_none_when_output_unknown() -> None:
    assert (
        estimate_retry_cost_usd(
            model="claude-opus-4-8",
            input_tokens=100,
            output_tokens=None,
        )
        is None
    )


def test_retry_cost_price_table_override_makes_model_estimable() -> None:
    table = {"custom-model": ModelPrice("custom-model", 2.0, 6.0, "2026-07-12")}
    cost = estimate_retry_cost_usd(
        model="custom-model",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        price_table=table,
    )
    assert cost == pytest.approx(8.0)


def test_retry_cost_empty_table_prices_nothing() -> None:
    assert (
        estimate_retry_cost_usd(
            model="claude-opus-4-8",
            input_tokens=10,
            output_tokens=10,
            price_table={},
        )
        is None
    )


@pytest.mark.parametrize("bad", [True, -1])
def test_retry_cost_rejects_invalid_input_tokens(bad: object) -> None:
    with pytest.raises(ValueError):
        estimate_retry_cost_usd(
            model="claude-opus-4-8",
            input_tokens=bad,  # type: ignore[arg-type]
            output_tokens=10,
        )


@pytest.mark.parametrize("bad", [True, -1])
def test_retry_cost_rejects_invalid_output_tokens(bad: object) -> None:
    with pytest.raises(ValueError):
        estimate_retry_cost_usd(
            model="claude-opus-4-8",
            input_tokens=10,
            output_tokens=bad,  # type: ignore[arg-type]
        )


# ----------------------------- cost ceiling ----------------------------------


def test_ceiling_none_never_blocks() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=100.0,
            next_retry_cost_usd=100.0,
            max_retry_cost_usd=None,
        )
        is False
    )


def test_ceiling_none_never_blocks_even_when_next_unknown() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=None,
            max_retry_cost_usd=None,
        )
        is False
    )


def test_ceiling_unknown_next_cost_blocks_when_configured() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=None,
            max_retry_cost_usd=0.05,
        )
        is True
    )


def test_ceiling_cumulative_denies_two_affordable_retries() -> None:
    # Each retry is $0.03; ceiling is $0.05. The first is fine; the second
    # pushes the cumulative total to $0.06 > $0.05 and is denied.
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=0.03,
            max_retry_cost_usd=0.05,
        )
        is False
    )
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.03,
            next_retry_cost_usd=0.03,
            max_retry_cost_usd=0.05,
        )
        is True
    )


def test_ceiling_equality_is_allowed() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.02,
            next_retry_cost_usd=0.03,
            max_retry_cost_usd=0.05,
        )
        is False
    )


def test_ceiling_strictly_above_is_denied() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.02,
            next_retry_cost_usd=0.031,
            max_retry_cost_usd=0.05,
        )
        is True
    )


def test_ceiling_float_noise_at_boundary_is_allowed() -> None:
    # 0.1 + 0.2 == 0.30000000000000004 in IEEE-754; within tolerance of a
    # 0.3 ceiling, so treated as equality and allowed.
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.1,
            next_retry_cost_usd=0.2,
            max_retry_cost_usd=0.3,
        )
        is False
    )


def test_ceiling_material_overage_is_denied() -> None:
    # 0.300001 is materially (not float-noise) above the 0.3 ceiling.
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=0.300001,
            max_retry_cost_usd=0.3,
        )
        is True
    )


def test_ceiling_exact_equality_is_allowed() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=0.3,
            max_retry_cost_usd=0.3,
        )
        is False
    )


def test_ceiling_ordinary_above_remains_denied() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=0.5,
            max_retry_cost_usd=0.3,
        )
        is True
    )


def test_ceiling_below_is_allowed() -> None:
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=0.01,
            max_retry_cost_usd=0.05,
        )
        is False
    )


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, True])
def test_ceiling_rejects_invalid_so_far(bad: object) -> None:
    with pytest.raises(ValueError):
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=bad,  # type: ignore[arg-type]
            next_retry_cost_usd=0.01,
            max_retry_cost_usd=0.05,
        )


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, True])
def test_ceiling_rejects_invalid_max(bad: object) -> None:
    with pytest.raises(ValueError):
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=0.01,
            max_retry_cost_usd=bad,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, True])
def test_ceiling_rejects_invalid_next_cost(bad: object) -> None:
    with pytest.raises(ValueError):
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=bad,  # type: ignore[arg-type]
            max_retry_cost_usd=0.05,
        )


# ------------------------ cross-checks (decisions 1 & 3) ---------------------


def test_unknown_model_no_ceiling_does_not_block() -> None:
    cost = estimate_retry_cost_usd(
        model="no-such-model", input_tokens=100, output_tokens=100
    )
    assert cost is None
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=cost,
            max_retry_cost_usd=None,
        )
        is False
    )


def test_unknown_model_with_ceiling_blocks() -> None:
    cost = estimate_retry_cost_usd(
        model="no-such-model", input_tokens=100, output_tokens=100
    )
    assert cost is None
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=cost,
            max_retry_cost_usd=0.05,
        )
        is True
    )


def test_price_table_override_unblocks_estimable_model() -> None:
    table = {"custom-model": ModelPrice("custom-model", 1.0, 1.0, "2026-07-12")}
    cost = estimate_retry_cost_usd(
        model="custom-model",
        input_tokens=1,
        output_tokens=1,
        price_table=table,
    )
    assert cost is not None
    assert (
        would_exceed_retry_cost_ceiling(
            retry_cost_so_far_usd=0.0,
            next_retry_cost_usd=cost,
            max_retry_cost_usd=0.05,
        )
        is False
    )
