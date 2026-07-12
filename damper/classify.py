"""Provider error classification.

Data-driven mapping from provider errors and low-level transport failures to
:class:`ErrorClass` values, per ``SPEC.md`` section 14 and the SESSION 3 design
in ``.local/PLAYBOOK.md``.

Design notes
------------

* Classification is data-driven: HTTP status codes are looked up in an explicit
  table with a documented range fallback, not a giant ``if`` ladder.
* Transport failures (connection errors and timeouts) are recognized through
  the official Anthropic SDK exception types (:class:`anthropic.APIConnectionError`,
  which :class:`anthropic.APITimeoutError` subclasses), never by matching
  exception class names as strings.
* The same transport failure classifies differently depending on streaming
  state: :data:`ErrorClass.RETRYABLE` before the first token, and
  :data:`ErrorClass.AMBIGUOUS` once content has started streaming. The executor
  treats :data:`ErrorClass.AMBIGUOUS` as not retryable and separately owns the
  hard streaming boundary (once any content has streamed to the caller, the
  failure is surfaced, never replayed).

A user-supplied classifier hook (:class:`ErrorClassifier`, wired through
``Policy.classifier``) receives the same streaming context the default
classifier sees, so it can make an equally-informed decision. Returning ``None``
defers to Damper's default classification.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, TypeGuard, runtime_checkable

import anthropic


class ErrorClass(Enum):
    """Retry disposition assigned to a provider failure."""

    RETRYABLE = "retryable"
    NOT_RETRYABLE = "not_retryable"
    AMBIGUOUS = "ambiguous"


@runtime_checkable
class ErrorClassifier(Protocol):
    """User-supplied classification hook (``Policy.classifier``).

    Called with the provider error and the current streaming context. Return an
    :class:`ErrorClass` to override Damper's decision, or ``None`` to defer to
    the default classification. Any other return value is a programming error
    and is rejected with :class:`TypeError` by :func:`classify` so that only
    well-typed dispositions ever reach the executor. Exceptions raised inside
    the hook propagate unchanged.
    """

    def __call__(
        self,
        error: BaseException,
        *,
        stream_started: bool,
    ) -> ErrorClass | None: ...


# Explicit per-code dispositions. Any code absent from this table is resolved by
# the range fallback in :func:`_classify_status`. Keeping the table explicit
# keeps the mapping auditable rather than hidden inside branching logic.
_STATUS_TABLE: dict[int, ErrorClass] = {
    429: ErrorClass.RETRYABLE,
    400: ErrorClass.NOT_RETRYABLE,
    401: ErrorClass.NOT_RETRYABLE,
    402: ErrorClass.NOT_RETRYABLE,
    403: ErrorClass.NOT_RETRYABLE,
    # 409 Conflict is intentionally NOT retried in v0.1: it can require the
    # caller to resolve a state conflict before another attempt is safe.
    409: ErrorClass.NOT_RETRYABLE,
    404: ErrorClass.NOT_RETRYABLE,
    413: ErrorClass.NOT_RETRYABLE,
}


def _is_real_int(value: object) -> TypeGuard[int]:
    """Return ``True`` only for a genuine ``int`` status code.

    ``bool`` is a subclass of ``int``; a stray ``True``/``False`` on an error's
    ``status_code`` must not be mistaken for HTTP status ``1``/``0``.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _classify_status(status_code: int) -> ErrorClass:
    """Classify a genuine integer HTTP status code.

    Explicit table first, then a documented range fallback: all ``5xx`` are
    retryable (including 504 and Anthropic's 529 overloaded); every other code,
    including all remaining ``4xx``, is not retryable in v0.1.
    """
    disposition = _STATUS_TABLE.get(status_code)
    if disposition is not None:
        return disposition
    if 500 <= status_code <= 599:
        return ErrorClass.RETRYABLE
    return ErrorClass.NOT_RETRYABLE


def _default_classify(error: BaseException, *, stream_started: bool) -> ErrorClass:
    """Damper's built-in classification, ignoring any user hook."""
    status_code = getattr(error, "status_code", None)
    if _is_real_int(status_code):
        return _classify_status(status_code)

    # No usable status code: fall back to transport-type inspection.
    # APITimeoutError subclasses APIConnectionError, so this single check
    # covers both connection resets and read timeouts.
    if isinstance(error, anthropic.APIConnectionError):
        return ErrorClass.AMBIGUOUS if stream_started else ErrorClass.RETRYABLE

    return ErrorClass.NOT_RETRYABLE


def classify(
    error: BaseException,
    *,
    stream_started: bool = False,
    classifier: ErrorClassifier | None = None,
) -> ErrorClass:
    """Classify a provider error into an :class:`ErrorClass`.

    If ``classifier`` is provided it is consulted first. It must return an
    :class:`ErrorClass` (used directly) or ``None`` (defer to the default). Any
    other return value raises :class:`TypeError`. Exceptions raised by the hook
    itself propagate unchanged.
    """
    if classifier is not None:
        result = classifier(error, stream_started=stream_started)
        if result is not None and not isinstance(result, ErrorClass):
            raise TypeError(
                "custom classifier must return an ErrorClass or None, "
                f"got {type(result).__name__!r}"
            )
        if result is not None:
            return result
    return _default_classify(error, stream_started=stream_started)
