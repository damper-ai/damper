"""Exponential backoff with full jitter and retry-after handling.

Implements the backoff computation described in ``SPEC.md`` section 16 and the
SESSION 3 design in ``.local/PLAYBOOK.md``: exponential growth with full jitter,
capped at ``backoff_max``, yielding to a provider ``retry-after`` when present
and enabled.

Attempt numbering
-----------------

``retry_number`` counts retries, not total attempts: ``retry_number == 1`` is
the first retry (fired after the first attempt failed), ``retry_number == 2`` is
the second retry, and so on.

Retry-after
-----------

When ``respect_retry_after`` is true and ``retry_after`` is a finite value
``>= 0``, it is returned exactly: no jitter is applied and it is deliberately
*not* clamped to ``cap``, because it is a provider instruction rather than a
client-side backoff choice (``SPEC.md`` section 14.3 lists what still constrains
it -- ``max_attempts``, the retry budget, the cost ceiling, and the per-attempt
timeout -- and ``backoff_max`` is intentionally not among them). An invalid
``retry_after`` (non-finite or negative) falls back to computed full jitter.

TODO(amit): v0.1 only reads a numeric ``retry_after``; parsing an HTTP-date
``Retry-After`` header is deferred.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable


def _upper_bound(base: float, retry_number: int, cap: float) -> float:
    """Compute ``min(cap, base * 2 ** (retry_number - 1))`` overflow-safely.

    ``math.ldexp`` avoids materializing a huge ``2 ** (retry_number - 1)``
    integer for an unexpectedly large ``retry_number``; on overflow it yields
    ``inf`` (and ``OverflowError`` is caught as a belt-and-suspenders fallback),
    which the cap then clamps down to ``cap``.
    """
    try:
        scaled = math.ldexp(base, retry_number - 1)
    except OverflowError:
        scaled = math.inf
    return cap if scaled >= cap else scaled


def compute_backoff(
    retry_number: int,
    *,
    base: float,
    cap: float,
    retry_after: float | None = None,
    respect_retry_after: bool = True,
    rng: Callable[[], float] = random.random,
) -> float:
    """Return the number of seconds to sleep before ``retry_number``.

    See the module docstring for attempt numbering and retry-after semantics.

    Raises
    ------
    ValueError
        If ``retry_number`` is not an ``int`` (``bool`` excluded) ``>= 1``, if
        ``base`` or ``cap`` is not finite and ``>= 0``, or if ``rng()`` returns
        a value that is not finite and within ``[0.0, 1.0]``.
    """
    if isinstance(retry_number, bool) or not isinstance(retry_number, int):
        raise ValueError(f"retry_number must be an int, got {type(retry_number).__name__!r}")
    if retry_number < 1:
        raise ValueError(f"retry_number must be >= 1, got {retry_number!r}")
    if not math.isfinite(base) or base < 0:
        raise ValueError(f"base must be finite and >= 0, got {base!r}")
    if not math.isfinite(cap) or cap < 0:
        raise ValueError(f"cap must be finite and >= 0, got {cap!r}")

    if (
        respect_retry_after
        and retry_after is not None
        and math.isfinite(retry_after)
        and retry_after >= 0
    ):
        # Provider instruction wins verbatim; do not consume the RNG.
        return float(retry_after)

    # Full jitter. Draw exactly once, then validate.
    draw = rng()
    if not math.isfinite(draw) or draw < 0.0 or draw > 1.0:
        raise ValueError(f"rng() must return a finite value in [0.0, 1.0], got {draw!r}")

    return draw * _upper_bound(base, retry_number, cap)
