"""Token estimation and retry cost ceiling.

Local-only helpers for the per-request retry cost ceiling described in SPEC.md
section 13. Cost estimation and ceiling logic are reliability-critical: they
gate whether a retry is allowed to fire, so this module stays small and
explicit and is hand-reviewed by the maintainer before release.

Design stance
-------------

* **Safety guard, not billing truth.** The estimate exists to bound retry cost
  during an outage, not to reconcile an invoice.
* **Local only.** No network access, and in particular no call to Anthropic's
  token-counting endpoint. Token counts are estimated from request fields or
  prior usage metadata, never fetched.
* **"Unknown" is ``None``.** Whenever a value cannot be estimated safely, the
  helper returns ``None`` rather than guessing. A configured cost ceiling then
  denies the retry (see :func:`would_exceed_retry_cost_ceiling`): enabling a
  ceiling means "do not exceed a known limit", so an unknown cost must not be
  allowed to bypass it.
* **Cumulative ceiling.** The ceiling applies to the *total* additional retry
  cost accrued across one logical request, not independently to each attempt.

Estimation priority
--------------------

Input tokens: valid ``usage.input_tokens`` -> local request text inspection
(``ceil(chars / 4)``) -> unknown when non-text content prevents a defensible
estimate and no usage count exists.

Output tokens: the request's ``max_tokens`` as a conservative reservation for
the next attempt -> unknown when it is absent or invalid. A prior attempt's
smaller ``usage.output_tokens`` is intentionally *not* used to reduce the next
retry's reservation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from damper.prices import DEFAULT_PRICE_TABLE, ModelPrice

# Relative tolerance for the cumulative cost-ceiling comparison. Cost estimates
# accumulate through float addition (``so_far + next_cost``), so a projected
# total that is mathematically equal to the ceiling can land a few ULPs above it
# (e.g. ``0.1 + 0.2`` -> ``0.30000000000000004``). We treat a projected cost
# within this relative tolerance of the ceiling as "at the ceiling" (allowed),
# so float representation error alone never denies an on-budget retry. A
# materially larger overage (well beyond rounding noise) is still denied.
_CEILING_REL_TOL = 1e-9


def _valid_token_count(value: object) -> int | None:
    """Return ``value`` if it is a real, non-negative ``int``, else ``None``.

    ``bool`` is a subclass of ``int``; a stray ``True``/``False`` must not be
    read as a token count of ``1``/``0``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _require_real_nonneg_int(value: object, name: str) -> int:
    """Validate a caller-supplied integer token count.

    Rejects ``bool``, non-``int`` types, and negatives with :class:`ValueError`.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a real int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def _require_finite_nonneg(value: object, name: str) -> float:
    """Validate a caller-supplied USD amount.

    Rejects ``bool``, non-numeric types, NaN, infinity, and negatives with
    :class:`ValueError`.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return float(value)


def _usage_field(usage: Any, field: str) -> int | None:
    """Read a non-negative integer field from a usage object or mapping."""
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        raw = usage.get(field)
    else:
        raw = getattr(usage, field, None)
    return _valid_token_count(raw)


def _iter_text_from_content(content: Any) -> list[str] | None:
    """Collect estimable text from a message ``content`` value.

    Returns a list of text fragments, or ``None`` if the content contains a
    part that cannot be estimated safely as text (an image, document, or other
    non-text block). Byte/base64 payload length is never counted as text.
    """
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        # Unknown content shape; refuse to guess.
        return None
    fragments: list[str] = []
    for part in content:
        if isinstance(part, str):
            fragments.append(part)
            continue
        if isinstance(part, Mapping):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                fragments.append(part["text"])
                continue
            # A non-text block (image, document, tool payload, ...) cannot be
            # estimated as text; no defensible count is available.
            return None
        # Unrecognized part shape.
        return None
    return fragments


def _estimate_request_text_tokens(request: Mapping[str, Any]) -> int | None:
    """Estimate input tokens from request text via ``ceil(chars / 4)``.

    Includes system text and message text. Returns ``None`` when the request
    contains content that cannot be estimated safely as text.
    """
    total_chars = 0

    system = request.get("system")
    if system is not None:
        system_fragments = _iter_text_from_content(system)
        if system_fragments is None:
            return None
        total_chars += sum(len(fragment) for fragment in system_fragments)

    messages = request.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            return None
        for message in messages:
            if not isinstance(message, Mapping):
                return None
            fragments = _iter_text_from_content(message.get("content"))
            if fragments is None:
                return None
            total_chars += sum(len(fragment) for fragment in fragments)

    return math.ceil(total_chars / 4)


def estimate_input_tokens(
    request: Mapping[str, Any],
    *,
    usage: Any | None = None,
) -> int | None:
    """Estimate input tokens for the next attempt.

    Priority: a valid ``usage.input_tokens`` (the prompt is identical across
    attempts, so a completed attempt's input count bounds the next one), then
    local request-text inspection (``ceil(total_text_chars / 4)`` over system
    and message text). Returns ``None`` when neither is available — in
    particular when the request carries non-text content (images, documents,
    binary/base64 payloads) and no usage count exists.
    """
    from_usage = _usage_field(usage, "input_tokens")
    if from_usage is not None:
        return from_usage
    return _estimate_request_text_tokens(request)


def estimate_output_tokens(request: Mapping[str, Any]) -> int | None:
    """Estimate output tokens reserved for the next attempt.

    Uses the request's ``max_tokens`` as a conservative worst-case reservation.
    Returns ``None`` when ``max_tokens`` is absent or invalid. A prior attempt's
    ``usage.output_tokens`` is deliberately not consulted: a smaller completed
    output does not bound what the next attempt may generate, so it must not
    reduce the reservation. Output size is never invented from prompt characters.
    """
    return _valid_token_count(request.get("max_tokens"))


def estimate_retry_cost_usd(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    price_table: Mapping[str, ModelPrice] | None = None,
) -> float | None:
    """Estimate the USD cost of one next retry attempt.

    Returns ``None`` (unknown) when the model is not priced in the table, or
    when either token estimate is ``None``. Provided token counts must be real,
    non-negative ints (``bool`` and negatives raise :class:`ValueError`); pass
    ``None`` to signal an unknown estimate instead.

    ``price_table`` defaults to :data:`damper.prices.DEFAULT_PRICE_TABLE`. Pass
    a mapping (e.g. ``Policy.price_table``) to override; an empty mapping prices
    no models and yields ``None`` for every model.
    """
    table = DEFAULT_PRICE_TABLE if price_table is None else price_table
    price = table.get(model)
    if price is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None

    input_count = _require_real_nonneg_int(input_tokens, "input_tokens")
    output_count = _require_real_nonneg_int(output_tokens, "output_tokens")

    cost = (
        input_count / 1_000_000 * price.input_price_per_million_tokens
        + output_count / 1_000_000 * price.output_price_per_million_tokens
    )
    if not math.isfinite(cost):
        return None
    return cost


def would_exceed_retry_cost_ceiling(
    *,
    retry_cost_so_far_usd: float,
    next_retry_cost_usd: float | None,
    max_retry_cost_usd: float | None,
) -> bool:
    """Decide whether the next retry would breach the cumulative cost ceiling.

    The ceiling bounds the *total* additional retry cost for one logical
    request:

        projected = retry_cost_so_far_usd + next_retry_cost_usd

    Semantics:

    * ``max_retry_cost_usd is None`` -> cost control is disabled; never blocks
      (returns ``False``), even when ``next_retry_cost_usd`` is unknown.
    * a configured ceiling with an unknown ``next_retry_cost_usd`` (``None``)
      -> blocks (returns ``True``). Enabling a ceiling asks Damper not to exceed
      a known limit; an unknown cost cannot be allowed to bypass it.
    * otherwise the retry is denied only when ``projected`` exceeds the ceiling.
      A projected cost at the ceiling is allowed, and a projected cost within a
      small relative tolerance (:data:`_CEILING_REL_TOL`) of the ceiling is
      treated as being at it — so float rounding error alone (e.g.
      ``0.1 + 0.2`` vs a ``0.3`` ceiling) never denies an on-budget retry.

    Numeric inputs must be finite and non-negative (NaN, infinity, negatives,
    and ``bool`` raise :class:`ValueError`).
    """
    so_far = _require_finite_nonneg(retry_cost_so_far_usd, "retry_cost_so_far_usd")

    if max_retry_cost_usd is None:
        return False
    ceiling = _require_finite_nonneg(max_retry_cost_usd, "max_retry_cost_usd")

    if next_retry_cost_usd is None:
        return True
    next_cost = _require_finite_nonneg(next_retry_cost_usd, "next_retry_cost_usd")

    projected = so_far + next_cost
    if math.isclose(projected, ceiling, rel_tol=_CEILING_REL_TOL):
        return False
    return projected > ceiling
