"""Scripted fake provider outcomes for Damper tests.

This module defines outcome dataclasses used to script sequences like
``[Err(529), Err(529), Ok()]`` against a fake Anthropic client.

``ScriptedFake`` consumes scripted outcomes in order. ``FakeAnthropic`` and
``FakeAsyncAnthropic`` are fake sync/async Anthropic clients that mimic the
``messages.create`` / ``messages.stream`` surface and ``with_options`` retry
ownership.

No network access. The fakes raise **real** ``anthropic`` error types
(``APIStatusError`` / ``APIConnectionError`` / ``APITimeoutError``) constructed
network-free from an ``httpx`` request/response, so classification and
``Retry-After`` header reading exercise the same shapes the wrapper sees in
production.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
import httpx


class FakeProviderError(Exception):
    """Baseline exception raised by :class:`Err` outcomes.

    A single class is used, rather than a richer fake exception hierarchy that
    mimics the ``anthropic`` SDK's error taxonomy, so tests that rely on it do
    not depend on the SDK being importable.
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

    ``payload`` stands in for a provider response object.
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

    Used to verify that Damper does not retry once any content has been
    streamed to the caller.
    """

    tokens: int = 3
    message: str = "connection reset mid-stream"


@dataclass(frozen=True)
class ConnectionResetBeforeFirstToken:
    """Scripted streaming outcome that resets before any content arrives.

    Used to verify that Damper may retry when the stream has not yet produced
    content.
    """

    message: str = "connection reset before first token"


Outcome = Ok | Err | Timeout | StreamThenReset | ConnectionResetBeforeFirstToken


@dataclass
class ScriptedFake:
    """Consumes a scripted sequence of outcomes in order.

    A deliberately minimal fake. It exposes a single :meth:`next_outcome`
    method that pops the next scripted outcome and either returns the ``Ok``
    payload or raises a fake exception mirroring the scripted failure.
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


# --------------------------------------------------------------------------- #
# Fake Anthropic clients.
# --------------------------------------------------------------------------- #

_FAKE_REQUEST = httpx.Request("POST", "https://api.anthropic.test/v1/messages")


def anthropic_status_error(
    status_code: int,
    *,
    retry_after: float | int | str | None = None,
    message: str = "fake status error",
) -> anthropic.APIStatusError:
    """Build a real ``anthropic.APIStatusError`` network-free.

    When ``retry_after`` is given it is placed in the response ``Retry-After``
    header (verbatim if a ``str``, else stringified), so the wrapper's header
    normalization is exercised through the real ``error.response.headers``.
    """
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = (
            retry_after if isinstance(retry_after, str) else str(retry_after)
        )
    response = httpx.Response(
        status_code=status_code, request=_FAKE_REQUEST, headers=headers
    )
    return anthropic.APIStatusError(message, response=response, body=None)


def anthropic_connection_error(
    message: str = "connection reset",
) -> anthropic.APIConnectionError:
    """Build a real ``anthropic.APIConnectionError`` network-free."""
    return anthropic.APIConnectionError(message=message, request=_FAKE_REQUEST)


class FakeMessage:
    """Minimal stand-in for ``anthropic.types.Message``.

    Accepts arbitrary attributes and allows attribute assignment, so the wrapper
    can attach ``resp.damper`` metadata exactly as it does to the real (extra-
    allowing) ``Message`` model.
    """

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _resolve_create(outcome: Outcome) -> Any:
    """Return an ``Ok`` payload or raise the real error for a failure outcome."""
    if isinstance(outcome, Ok):
        return outcome.payload if outcome.payload is not None else FakeMessage()
    if isinstance(outcome, Err):
        raise anthropic_status_error(
            outcome.status_code,
            retry_after=outcome.retry_after,
            message=outcome.message,
        )
    if isinstance(outcome, Timeout):
        raise anthropic.APITimeoutError(request=_FAKE_REQUEST)
    if isinstance(outcome, ConnectionResetBeforeFirstToken):
        raise anthropic_connection_error(outcome.message)
    if isinstance(outcome, StreamThenReset):
        raise AssertionError("StreamThenReset is a streaming-only outcome")
    raise AssertionError(f"unrecognized scripted outcome: {outcome!r}")


class _FakeMessagesBase:
    """Shared script/spy state for the sync and async messages fakes."""

    def __init__(self, client: _FakeClientBase) -> None:
        self._client = client


@dataclass(frozen=True)
class FakeStreamEvent:
    """Minimal stand-in for an Anthropic stream event with a ``type`` tag."""

    type: str
    text: str = ""


@dataclass(frozen=True)
class _RaiseEvent:
    """Sentinel in a stream plan: raise ``error`` when reached."""

    error: BaseException


def _plan_stream(outcome: Outcome) -> tuple[list[Any], Any]:
    """Return ``(plan, final_message)`` for a streaming outcome.

    A plan is a list of :class:`FakeStreamEvent` (a metadata event followed by
    ``content_block_delta`` content tokens) with optional :class:`_RaiseEvent`
    markers where the stream should fail. ``message_start`` is a metadata-only
    event that precedes the first content token, so the wrapper must treat a
    failure before the first ``content_block_delta`` as "before the first token".
    """
    if isinstance(outcome, Ok):
        final = outcome.payload if outcome.payload is not None else FakeMessage()
        plan: list[Any] = [
            FakeStreamEvent("message_start"),
            FakeStreamEvent("content_block_delta", "a"),
            FakeStreamEvent("content_block_delta", "b"),
            FakeStreamEvent("message_stop"),
        ]
        return plan, final
    if isinstance(outcome, ConnectionResetBeforeFirstToken):
        # Metadata event, then a reset before any content token -> retryable.
        return [
            FakeStreamEvent("message_start"),
            _RaiseEvent(anthropic_connection_error(outcome.message)),
        ], FakeMessage()
    if isinstance(outcome, StreamThenReset):
        plan = [FakeStreamEvent("message_start")]
        plan += [
            FakeStreamEvent("content_block_delta", f"t{i}")
            for i in range(outcome.tokens)
        ]
        plan.append(_RaiseEvent(anthropic_connection_error(outcome.message)))
        return plan, FakeMessage()
    if isinstance(outcome, Err):
        # Status failure at establishment, before any content token.
        return [
            _RaiseEvent(
                anthropic_status_error(
                    outcome.status_code,
                    retry_after=outcome.retry_after,
                    message=outcome.message,
                )
            )
        ], FakeMessage()
    raise AssertionError(f"unrecognized streaming outcome: {outcome!r}")


class FakeMessageStream:
    """Sync stream object yielded by ``FakeStreamManager.__enter__``.

    A single-pass iterator (``__iter__`` returns ``self``), like a real
    ``MessageStream`` backed by one HTTP response: consuming the first token and
    then continuing resumes the same cursor rather than restarting.
    """

    def __init__(self, plan: list[Any], final: Any) -> None:
        self._it = iter(plan)
        self._final = final
        self.closed = False
        self._consumed_text: list[str] = []

    def __iter__(self) -> FakeMessageStream:
        return self

    def __next__(self) -> Any:
        item = next(self._it)
        if isinstance(item, _RaiseEvent):
            raise item.error
        if getattr(item, "type", None) == "content_block_delta" and item.text:
            self._consumed_text.append(item.text)
        return item

    def get_final_message(self) -> Any:
        # Reflect fully-consumed content, like the SDK's accumulated snapshot.
        self._final.text = "".join(self._consumed_text)
        return self._final

    def get_final_text(self) -> str:
        return "".join(self._consumed_text)

    def close(self) -> None:
        self.closed = True


class FakeStreamManager:
    """Sync context manager returned by ``FakeMessages.stream``."""

    def __init__(self, plan: list[Any], final: Any) -> None:
        self._plan = plan
        self._final = final
        self.entered = False
        self.exited = False

    def __enter__(self) -> FakeMessageStream:
        self.entered = True
        return FakeMessageStream(self._plan, self._final)

    def __exit__(self, *exc_info: Any) -> Literal[False]:
        self.exited = True
        return False


class FakeAsyncMessageStream:
    """Async stream object yielded by ``FakeAsyncStreamManager.__aenter__``.

    A single-pass async iterator (``__aiter__`` returns ``self``), like a real
    ``AsyncMessageStream``.
    """

    def __init__(self, plan: list[Any], final: Any) -> None:
        self._it = iter(plan)
        self._final = final
        self.closed = False
        self._consumed_text: list[str] = []

    def __aiter__(self) -> FakeAsyncMessageStream:
        return self

    async def __anext__(self) -> Any:
        try:
            item = next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None
        if isinstance(item, _RaiseEvent):
            raise item.error
        if getattr(item, "type", None) == "content_block_delta" and item.text:
            self._consumed_text.append(item.text)
        return item

    async def get_final_message(self) -> Any:
        self._final.text = "".join(self._consumed_text)
        return self._final

    async def get_final_text(self) -> str:
        return "".join(self._consumed_text)

    async def close(self) -> None:
        self.closed = True


class FakeAsyncStreamManager:
    """Async context manager returned by ``FakeAsyncMessages.stream``."""

    def __init__(self, plan: list[Any], final: Any) -> None:
        self._plan = plan
        self._final = final

    async def __aenter__(self) -> FakeAsyncMessageStream:
        return FakeAsyncMessageStream(self._plan, self._final)

    async def __aexit__(self, *exc_info: Any) -> Literal[False]:
        return False


class FakeMessages(_FakeMessagesBase):
    """Sync ``messages`` fake exposing ``create`` and ``stream``."""

    def create(self, **kwargs: Any) -> Any:
        self._client.create_calls.append(kwargs)
        return _resolve_create(self._client.next_outcome())

    def stream(self, **kwargs: Any) -> FakeStreamManager:
        self._client.stream_calls.append(kwargs)
        plan, final = _plan_stream(self._client.next_outcome())
        return FakeStreamManager(plan, final)

    def count_tokens(self, **kwargs: Any) -> int:
        """Non-intercepted method; used to prove passthrough delegation."""
        return 4242


class FakeAsyncMessages(_FakeMessagesBase):
    """Async ``messages`` fake exposing ``create`` and ``stream``."""

    async def create(self, **kwargs: Any) -> Any:
        self._client.create_calls.append(kwargs)
        return _resolve_create(self._client.next_outcome())

    def stream(self, **kwargs: Any) -> FakeAsyncStreamManager:
        # Real AsyncAnthropic.stream() is a sync method returning an async CM.
        self._client.stream_calls.append(kwargs)
        plan, final = _plan_stream(self._client.next_outcome())
        return FakeAsyncStreamManager(plan, final)

    def count_tokens(self, **kwargs: Any) -> int:
        """Non-intercepted method; used to prove passthrough delegation."""
        return 4242


class _FakeClientBase:
    """Scripted fake client with a ``with_options`` retry-ownership spy."""

    def __init__(self, script: list[Outcome]) -> None:
        self._script: deque[Outcome] = deque(script)
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.with_options_calls: list[dict[str, Any]] = []

    def with_options(self, **kwargs: Any) -> Any:
        # Real clients return a configured copy; the fake shares the same script
        # so a single instance both records the call and serves the attempts.
        self.with_options_calls.append(kwargs)
        return self

    def next_outcome(self) -> Outcome:
        if not self._script:
            raise AssertionError("fake client script exhausted")
        return self._script.popleft()


class FakeAnthropic(_FakeClientBase):
    """Sync fake mirroring ``anthropic.Anthropic`` (create + ownership)."""

    def __init__(self, script: list[Outcome]) -> None:
        super().__init__(script)
        self.messages = FakeMessages(self)


class FakeAsyncAnthropic(_FakeClientBase):
    """Async fake mirroring ``anthropic.AsyncAnthropic`` (create + ownership)."""

    def __init__(self, script: list[Outcome]) -> None:
        super().__init__(script)
        self.messages = FakeAsyncMessages(self)


class FakeClientNoWithOptions:
    """Client lacking ``with_options`` -> retry ownership cannot be taken."""

    def __init__(self) -> None:
        self.messages = FakeMessages(_FakeClientBase([]))


class FakeClientWithOptionsRaises:
    """Client whose ``with_options`` raises -> retry ownership cannot be taken."""

    def __init__(self) -> None:
        self.messages = FakeMessages(_FakeClientBase([]))

    def with_options(self, **kwargs: Any) -> Any:
        raise RuntimeError("cannot disable SDK retries")
