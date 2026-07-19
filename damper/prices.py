"""Anthropic model price table for retry cost estimation.

This module hosts the default price table used by :mod:`damper.cost` for the
per-request retry cost ceiling (SPEC.md section 13). Each entry carries input
and output prices per million tokens along with a ``last_verified`` date. Users
can override the table via :attr:`damper.Policy.price_table`.

Prices are a **safety guard** for the retry cost ceiling, **not billing truth**.
The estimate exists to bound retry cost during outages, not to reconcile an
invoice. Lookups are by exact model-string match; there is no runtime pricing
lookup and no network access.

Unsupported pricing modifiers (v0.1)
------------------------------------

A single input/output price pair per model deliberately does **not** model:

* prompt caching (cache-write / cache-read multipliers)
* fast mode
* Batch API pricing
* data-residency multipliers
* server-side tool charges

None of these are represented here, and Damper does not silently assume they are
covered. When such a modifier can increase cost and Damper cannot estimate it
safely, the cost estimate should be treated as unknown (``None``) rather than
guessed.

For any time-limited promotional price, this table uses the conservative
non-promotional rate so a temporary lower rate cannot silently become stale.

.. note::

   ``last_verified`` reflects the date these values were last checked against
   Anthropic's public pricing. Maintainers should re-verify against official
   pricing before each release; prices change and this table is not
   authoritative.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelPrice:
    """Price entry for a single model.

    Prices are expressed in US dollars per one million tokens. Instances are
    immutable; both prices are validated at construction and must be finite and
    non-negative.
    """

    model: str
    input_price_per_million_tokens: float
    output_price_per_million_tokens: float
    last_verified: str  # ISO date, e.g. "2026-07-12"

    def __post_init__(self) -> None:
        for name in (
            "input_price_per_million_tokens",
            "output_price_per_million_tokens",
        ):
            value = getattr(self, name)
            # bool is a subclass of int/float; reject it explicitly so a stray
            # True/False cannot masquerade as a price of 1.0/0.0.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


# Default Anthropic price table. Conservative non-promotional rates only,
# verified against Anthropic's models overview on the date below. Only current
# and still-active legacy models appear here. Retired / no-longer-available
# model IDs (e.g. the Claude 3.x line) are intentionally excluded so a retired
# ID is never presented as if it were current. Anthropic maintains a separate
# model-deprecation schedule; maintainers should re-verify prices and prune
# newly-retired entries before each release.
_LAST_VERIFIED = "2026-07-18"

_PRICES: dict[str, ModelPrice] = {
    # --- Current models ---
    "claude-fable-5": ModelPrice("claude-fable-5", 10.0, 50.0, _LAST_VERIFIED),
    # Mythos 5: Project Glasswing only; same specs and pricing as Fable 5.
    "claude-mythos-5": ModelPrice("claude-mythos-5", 10.0, 50.0, _LAST_VERIFIED),
    "claude-opus-4-8": ModelPrice("claude-opus-4-8", 5.0, 25.0, _LAST_VERIFIED),
    # Sonnet 5: non-promotional sticker rate ($3/$15); the $2/$10 introductory
    # rate is intentionally not embedded so it cannot silently become stale.
    "claude-sonnet-5": ModelPrice("claude-sonnet-5", 3.0, 15.0, _LAST_VERIFIED),
    "claude-haiku-4-5": ModelPrice("claude-haiku-4-5", 1.0, 5.0, _LAST_VERIFIED),
    # --- Legacy models (still active on the API, on a separate deprecation
    #     schedule; not "current"). ---
    "claude-opus-4-7": ModelPrice("claude-opus-4-7", 5.0, 25.0, _LAST_VERIFIED),
    "claude-opus-4-6": ModelPrice("claude-opus-4-6", 5.0, 25.0, _LAST_VERIFIED),
    "claude-opus-4-5": ModelPrice("claude-opus-4-5", 5.0, 25.0, _LAST_VERIFIED),
    "claude-sonnet-4-6": ModelPrice("claude-sonnet-4-6", 3.0, 15.0, _LAST_VERIFIED),
    "claude-sonnet-4-5": ModelPrice("claude-sonnet-4-5", 3.0, 15.0, _LAST_VERIFIED),
}

# Exposed read-only: the shared default table must not be mutated at runtime.
# ``MappingProxyType`` makes accidental writes raise ``TypeError`` (the entries
# are themselves frozen), so callers can only override via ``Policy.price_table``.
DEFAULT_PRICE_TABLE: Mapping[str, ModelPrice] = MappingProxyType(_PRICES)
