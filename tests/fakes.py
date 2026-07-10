"""Scripted fake provider outcomes for Damper tests.

This module defines outcome dataclasses that later sessions will use to script
sequences like ``[Err(529), Err(529), Ok()]`` against a fake Anthropic client.

SESSION 1 provides only enough machinery for a smoke test that proves scripted
outcomes are consumed in order. Full sync/async ``messages.create`` and
``messages.stream`` fakes land in later sessions.

No network access. No ``anthropic`` imports.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


class FakeProviderError(Exception):
    """Baseline exception raised by :class:`Err` outcomes.

    Later sessions will introduce richer fake exception hierarchies that mimic
    the ``anthropic`` SDK's error taxonomy. SESSION 1 uses a single class so
    the smoke tests do not depend on the SDK being importable.
    """

    def __init__(
        self,
        status_code: int | None = None,
        message: str = "fake provider error",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class FakeTimeoutError(Exception):
    """Baseline exception raised by :class:`Timeout` outcomes."""


class FakeConnectionResetError(Exception):
    """Baseline exception raised by :class:`ConnectionResetBeforeFirstToken`."""


@dataclass(frozen=True)
class Ok:
    """Scripted successful outcome.

    ``payload`` stands in for a provider response object; later sessions will
    replace it with a fake ``anthropic.types.Message``-shaped value.
    """

    payload: Any = None


@dataclass(frozen=True)
class Err:
    """Scripted provider error outcome."""

    status_code: int
    message: str = "fake provider error"
    retry_after: float | None = None


@dataclass(frozen=True)
class Timeout:
    """Scripted per-attempt timeout outcome."""

    message: str = "fake attempt timeout"


@dataclass(frozen=True)
class StreamThenReset:
    """Scripted streaming outcome that emits N tokens then resets.

    Used in SESSION 6 tests to verify that Damper does not retry once any
    content has been streamed to the caller.
    """

    tokens: int = 3
    message: str = "connection reset mid-stream"


@dataclass(frozen=True)
class ConnectionResetBeforeFirstToken:
    """Scripted streaming outcome that resets before any content arrives.

    Used in SESSION 6 tests to verify that Damper may retry when the stream
    has not yet produced content.
    """

    message: str = "connection reset before first token"


Outcome = Ok | Err | Timeout | StreamThenReset | ConnectionResetBeforeFirstToken


@dataclass
class ScriptedFake:
    """Consumes a scripted sequence of outcomes in order.

    This is a deliberately minimal fake for SESSION 1. It exposes a single
    :meth:`next_outcome` method that pops the next scripted outcome and either
    returns the ``Ok`` payload or raises a fake exception mirroring the
    scripted failure. Later sessions extend this into full sync/async
    ``messages.create`` and ``messages.stream`` fakes.
    """

    outcomes: deque[Outcome] = field(default_factory=deque)
    calls: int = 0

    @classmethod
    def from_sequence(cls, sequence: list[Outcome]) -> ScriptedFake:
        return cls(outcomes=deque(sequence))

    def next_outcome(self) -> Any:
        if not self.outcomes:
            raise AssertionError("ScriptedFake exhausted: no more outcomes queued")
        self.calls += 1
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Ok):
            return outcome.payload
        if isinstance(outcome, Err):
            raise FakeProviderError(
                status_code=outcome.status_code,
                message=outcome.message,
                retry_after=outcome.retry_after,
            )
        if isinstance(outcome, Timeout):
            raise FakeTimeoutError(outcome.message)
        if isinstance(outcome, (StreamThenReset, ConnectionResetBeforeFirstToken)):
            raise FakeConnectionResetError(outcome.message)
        raise AssertionError(f"unrecognized scripted outcome: {outcome!r}")

    @property
    def remaining(self) -> int:
        return len(self.outcomes)
