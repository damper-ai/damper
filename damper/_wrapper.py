"""Public wrapper around Anthropic sync/async clients.

``resilient(client, policy)`` returns a proxy that exposes the same interface as
the wrapped client but intercepts ``messages.create`` and ``messages.stream``
(sync and async), routing them through Damper's owned retry executor. Everything
else passes through unchanged.

Reliability-critical (``CLAUDE.md``). This module owns two things the executor
deliberately does not:

* **Retry ownership** (``SPEC.md`` section 7). At wrap time Damper builds a
  configured client with ``with_options(max_retries=0, timeout=...)`` so the
  Anthropic SDK never retries underneath Damper. If SDK retries cannot be
  disabled via the supported mechanism, wrapping fails with
  :class:`RetryOwnershipError`. Per-call ``max_retries`` overrides on intercepted
  calls are rejected; per-call ``timeout`` is allowed. This is wrap-time
  ownership -- "we produced a ``max_retries=0`` client via the supported
  mechanism" -- not a runtime proof the SDK never retries.

* **Retry-After normalization** (``SPEC.md`` sections 14.4 / 14.5, SESSION 6).
  Reading and normalizing the provider ``Retry-After`` response header lives here
  and never in :mod:`damper.backoff`. The executor and ``compute_backoff`` only
  ever receive an already-normalized ``float | None`` duration.

The streaming boundary (``SPEC.md`` section 15) is enforced structurally by the
stream proxy: the first content token is pulled *inside* the retry loop, so a
failure before it can retry; once the first token is handed to the caller, any
later failure is raised during the caller's own iteration, outside the retry
loop, and is never replayed.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal

from damper import Policy, RetryOwnershipError
from damper._executor import execute, execute_async
from damper.budget import RetryBudget

# Telemetry provider label for wrapped Anthropic calls (SPEC section 18.2).
_PROVIDER = "anthropic"

# --------------------------------------------------------------------------- #
# Policy validation (SPEC section 8.2), enforced at wrap time.
# --------------------------------------------------------------------------- #


def _require_real_number(name: str, value: object) -> float:
    """Return ``value`` as a finite float, rejecting ``bool`` and non-numbers.

    ``bool`` is a subclass of ``int``; a stray ``True``/``False`` must not pass
    as a numeric policy value.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _validate_policy(policy: Policy) -> None:
    """Reject invalid :class:`Policy` field values per ``SPEC.md`` section 8.2.

    Raises a field-specific :class:`ValueError`. Does not mutate the policy and
    does not normalize invalid values. Called at wrap time in :func:`resilient`
    before any client configuration or wrapper state is created.
    """
    max_attempts = policy.max_attempts
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError(f"max_attempts must be an int, got {max_attempts!r}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")

    per_attempt_timeout = _require_real_number(
        "per_attempt_timeout", policy.per_attempt_timeout
    )
    if per_attempt_timeout <= 0:
        raise ValueError(
            f"per_attempt_timeout must be > 0, got {policy.per_attempt_timeout!r}"
        )

    if policy.max_retry_cost_usd is not None:
        max_retry_cost_usd = _require_real_number(
            "max_retry_cost_usd", policy.max_retry_cost_usd
        )
        if max_retry_cost_usd < 0:
            raise ValueError(
                f"max_retry_cost_usd must be >= 0, got {policy.max_retry_cost_usd!r}"
            )

    ratio = _require_real_number("retry_budget_ratio", policy.retry_budget_ratio)
    if ratio < 0:
        raise ValueError(
            f"retry_budget_ratio must be >= 0, got {policy.retry_budget_ratio!r}"
        )

    window = _require_real_number("retry_budget_window", policy.retry_budget_window)
    if window <= 0:
        raise ValueError(
            f"retry_budget_window must be > 0, got {policy.retry_budget_window!r}"
        )

    min_tokens = policy.retry_budget_min_tokens
    if isinstance(min_tokens, bool) or not isinstance(min_tokens, int):
        raise ValueError(f"retry_budget_min_tokens must be an int, got {min_tokens!r}")
    if min_tokens < 0:
        raise ValueError(f"retry_budget_min_tokens must be >= 0, got {min_tokens!r}")

    backoff_base = _require_real_number("backoff_base", policy.backoff_base)
    if backoff_base < 0:
        raise ValueError(f"backoff_base must be >= 0, got {policy.backoff_base!r}")
    backoff_max = _require_real_number("backoff_max", policy.backoff_max)
    if backoff_max < 0:
        raise ValueError(f"backoff_max must be >= 0, got {policy.backoff_max!r}")
    if backoff_base > backoff_max:
        raise ValueError(
            f"backoff_base ({policy.backoff_base!r}) must be <= "
            f"backoff_max ({policy.backoff_max!r})"
        )

    if not isinstance(policy.respect_retry_after, bool):
        raise ValueError(
            f"respect_retry_after must be a bool, got {policy.respect_retry_after!r}"
        )

    if policy.on_budget_exhausted not in ("raise", "passthrough"):
        raise ValueError(
            "on_budget_exhausted must be 'raise' or 'passthrough', got "
            f"{policy.on_budget_exhausted!r}"
        )


# --------------------------------------------------------------------------- #
# Retry-After header normalization (provider-specific; never in backoff.py).
# --------------------------------------------------------------------------- #


def _get_header_ci(headers: Any, name: str) -> str | None:
    """Case-insensitively read a header value, or ``None`` if absent.

    Works for ``httpx.Headers`` (already case-insensitive) and plain mappings.
    """
    if headers is None:
        return None
    items = getattr(headers, "items", None)
    if not callable(items):
        return None
    lowered = name.lower()
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == lowered:
            return value if isinstance(value, str) else str(value)
    return None


def _try_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _parse_retry_after_header(raw: str, *, now: datetime) -> float | None:
    """Normalize a ``Retry-After`` header value to seconds, or ``None``.

    Delta-seconds: a finite, non-negative number normalizes to itself; a negative
    number normalizes to ``None``. HTTP-date: a future date normalizes to the
    seconds from ``now`` (assumed UTC); a past date normalizes to ``0.0``.
    Empty / unparseable input normalizes to ``None``. ``now`` is injected so
    HTTP-date tests are deterministic.
    """
    text = raw.strip()
    if not text:
        return None

    numeric = _try_float(text)
    if numeric is not None:
        # Delta-seconds. A negative value is invalid (distinct from a past date).
        if not math.isfinite(numeric) or numeric < 0:
            return None
        return numeric

    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = (parsed - now).total_seconds()
    # A past HTTP-date means "retry now": 0.0, not None.
    return 0.0 if delta < 0 else delta


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_retry_after_extractor(
    now: Callable[[], datetime] = _utcnow,
) -> Callable[[BaseException], float | None]:
    """Build the extractor the executor calls per failed attempt.

    Reads the provider error's ``Retry-After`` response header and normalizes it.
    Returns ``None`` when there is no readable header. Whether the value is then
    honored is decided downstream by ``compute_backoff`` from
    ``Policy.respect_retry_after``; this extractor only normalizes.
    """

    def extract(error: BaseException) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        raw = _get_header_ci(headers, "retry-after")
        if raw is None:
            return None
        return _parse_retry_after_header(raw, now=now())

    return extract


# --------------------------------------------------------------------------- #
# Retry ownership + response metadata.
# --------------------------------------------------------------------------- #


def _configure_retry_ownership(client: Any, policy: Policy) -> Any:
    """Return a client copy with SDK retries disabled, or raise.

    Uses the supported ``with_options(max_retries=0, timeout=...)`` mechanism and
    never mutates private SDK internals. Raises :class:`RetryOwnershipError` when
    the mechanism is unavailable or fails.
    """
    client_type = type(client).__name__
    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        raise RetryOwnershipError(
            reason=(
                "client has no with_options(); cannot disable SDK retries for "
                "owned calls"
            ),
            client_type=client_type,
        )
    try:
        configured = with_options(max_retries=0, timeout=policy.per_attempt_timeout)
    except Exception as exc:
        raise RetryOwnershipError(
            reason=f"with_options(max_retries=0) failed: {exc}",
            client_type=client_type,
        ) from exc
    if configured is None:
        raise RetryOwnershipError(
            reason="with_options(max_retries=0) returned None",
            client_type=client_type,
        )
    return configured


def _reject_max_retries(kwargs: Mapping[str, Any], client_type: str) -> None:
    """Reject a per-call ``max_retries`` override on an intercepted call."""
    if "max_retries" in kwargs:
        raise RetryOwnershipError(
            reason=(
                "per-call max_retries override is not allowed on intercepted "
                "calls; Damper owns retries for wrapped calls"
            ),
            client_type=client_type,
        )


def _attach_metadata(response: Any, metadata: Any) -> None:
    """Attach ``resp.damper`` metadata non-invasively.

    The Anthropic ``Message`` model allows extra attributes, so a plain
    ``setattr`` does not break SDK response typing or usage. If a particular
    response object refuses attribute assignment, the response is returned
    unchanged rather than raising.
    """
    try:
        response.damper = metadata
    except Exception:
        pass


def _is_async_client(client: Any) -> bool:
    """Detect an async client by whether ``messages.create`` is a coroutine fn.

    Duck-typed so both ``anthropic.AsyncAnthropic`` and test fakes are handled
    without coupling to concrete SDK classes.
    """
    messages = getattr(client, "messages", None)
    create = getattr(messages, "create", None)
    return inspect.iscoroutinefunction(create)


# --------------------------------------------------------------------------- #
# Streaming boundary (SPEC section 15).
#
# The first content token is pulled *inside* the retry loop, so a failure before
# it can retry (a fresh stream is opened). Once the first token is handed back,
# the proxy relays the rest of the stream live; a later failure is raised during
# the caller's own iteration -- outside the retry loop -- and is never replayed.
# "stream_started" is the first *content* delta, not a metadata-only event
# (message_start / content_block_start), which stays "before the first token".
# --------------------------------------------------------------------------- #

_CONTENT_EVENT_TYPES = frozenset({"content_block_delta", "text"})


def _is_first_token(event: Any) -> bool:
    return getattr(event, "type", None) in _CONTENT_EVENT_TYPES


def _event_text(event: Any) -> str | None:
    """Extract streamed text from a content event, or ``None`` for non-text.

    Covers the SDK's high-level ``text`` event (``event.text``) and a raw
    ``content_block_delta`` (``event.delta.text``). Empty strings and non-content
    events yield ``None`` so ``text_stream`` ignores them.
    """
    if getattr(event, "type", None) not in _CONTENT_EVENT_TYPES:
        return None
    text = getattr(event, "text", None)
    if isinstance(text, str) and text != "":
        return text
    delta_text = getattr(getattr(event, "delta", None), "text", None)
    if isinstance(delta_text, str) and delta_text != "":
        return delta_text
    return None


@dataclass
class _StreamState:
    """Per-attempt flag: has a content token been seen on this attempt yet?"""

    started: bool = False


@dataclass
class _PrimedStream:
    """A live stream whose first content token (if any) has been buffered.

    ``iterator`` is the single (a)iterator captured once at prime time and shared
    by the relay, so the wrapper never calls ``iter()`` / ``__aiter__`` on the
    underlying stream a second time -- correctness does not depend on the SDK's
    ``__iter__`` resuming a shared cursor. ``stream`` is retained only for
    ``get_final_message`` and ``close``.
    """

    manager: Any
    stream: Any
    iterator: Any
    buffered: list[Any]
    exhausted: bool


def _safe_close(manager: Any) -> None:
    try:
        manager.__exit__(None, None, None)
    except Exception:
        pass


async def _safe_aclose(manager: Any) -> None:
    try:
        await manager.__aexit__(None, None, None)
    except Exception:
        pass


def _prime_sync_stream(manager: Any, state: _StreamState) -> _PrimedStream:
    """Open a stream and pull up to and including the first content token.

    Raises before the first token propagate to the executor (with
    ``stream_started`` still ``False``), so a fresh stream can be retried. The
    manager is closed on such a failure.
    """
    state.started = False
    stream = manager.__enter__()
    iterator = iter(stream)
    buffered: list[Any] = []
    try:
        for event in iterator:
            buffered.append(event)
            if _is_first_token(event):
                state.started = True
                return _PrimedStream(manager, stream, iterator, buffered, exhausted=False)
    except BaseException:
        _safe_close(manager)
        raise
    # Stream ended with no content token: a complete (content-less) stream.
    return _PrimedStream(manager, stream, iterator, buffered, exhausted=True)


async def _prime_async_stream(manager: Any, state: _StreamState) -> _PrimedStream:
    """Async counterpart of :func:`_prime_sync_stream`."""
    state.started = False
    stream = await manager.__aenter__()
    iterator = stream.__aiter__()
    buffered: list[Any] = []
    try:
        async for event in iterator:
            buffered.append(event)
            if _is_first_token(event):
                state.started = True
                return _PrimedStream(manager, stream, iterator, buffered, exhausted=False)
    except BaseException:
        await _safe_aclose(manager)
        raise
    return _PrimedStream(manager, stream, iterator, buffered, exhausted=True)


class _SyncDamperStream:
    """Iterable stream proxy with a single owned consumption path.

    Every consuming API -- event iteration, :attr:`text_stream`,
    :meth:`until_done`, :meth:`get_final_text`, :meth:`get_final_message` --
    draws from one shared event generator (:attr:`_damper_events`) that yields
    the buffered first token(s) exactly once and then the live iterator. Nothing
    stream-consuming is delegated through ``__getattr__``, so the buffered first
    token can never be skipped or duplicated.

    ``damper`` starts at the executor's ``ok`` metadata; a failure after the
    first token flips its outcome to ``stream_started_failure`` before the
    original error is re-raised, regardless of which consumption API triggered
    it. Because the proxy is only returned once the first token is in hand, all
    of these failures occur outside the retry loop and are never replayed.
    """

    def __init__(self, primed: _PrimedStream, metadata: Any) -> None:
        self._damper_manager = primed.manager
        self._damper_stream = primed.stream
        self._damper_iterator = primed.iterator
        self._damper_exhausted = primed.exhausted
        self.damper = metadata
        self._damper_events = self._consume(primed.buffered)

    def _consume(self, buffered: list[Any]) -> Iterator[Any]:
        """The one shared event stream: buffered first, then live."""
        try:
            yield from buffered
            if not self._damper_exhausted:
                yield from self._damper_iterator
        except Exception:
            self.damper = replace(self.damper, outcome="stream_started_failure")
            raise

    def __iter__(self) -> Iterator[Any]:
        return self._damper_events

    @property
    def text_stream(self) -> Iterator[str]:
        for event in self._damper_events:
            text = _event_text(event)
            if text is not None:
                yield text

    def until_done(self) -> None:
        for _event in self._damper_events:
            pass

    def get_final_message(self) -> Any:
        self.until_done()
        return self._damper_stream.get_final_message()

    def get_final_text(self) -> Any:
        self.until_done()
        return self._damper_stream.get_final_text()

    def close(self) -> None:
        _safe_close(self._damper_manager)

    def __getattr__(self, name: str) -> Any:
        # Non-consuming attributes (e.g. the raw response) may delegate; every
        # stream-consuming API above is implemented explicitly.
        return getattr(self._damper_stream, name)


class _AsyncDamperStream:
    """Async counterpart of :class:`_SyncDamperStream` with one owned path."""

    def __init__(self, primed: _PrimedStream, metadata: Any) -> None:
        self._damper_manager = primed.manager
        self._damper_stream = primed.stream
        self._damper_iterator = primed.iterator
        self._damper_exhausted = primed.exhausted
        self.damper = metadata
        self._damper_events = self._consume(primed.buffered)

    async def _consume(self, buffered: list[Any]) -> AsyncIterator[Any]:
        """The one shared async event stream: buffered first, then live."""
        try:
            for event in buffered:
                yield event
            if not self._damper_exhausted:
                async for event in self._damper_iterator:
                    yield event
        except Exception:
            self.damper = replace(self.damper, outcome="stream_started_failure")
            raise

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._damper_events

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._atext_stream()

    async def _atext_stream(self) -> AsyncIterator[str]:
        async for event in self._damper_events:
            text = _event_text(event)
            if text is not None:
                yield text

    async def until_done(self) -> None:
        async for _event in self._damper_events:
            pass

    async def get_final_message(self) -> Any:
        await self.until_done()
        return await self._damper_stream.get_final_message()

    async def get_final_text(self) -> Any:
        await self.until_done()
        return await self._damper_stream.get_final_text()

    async def aclose(self) -> None:
        await _safe_aclose(self._damper_manager)

    def __getattr__(self, name: str) -> Any:
        # Non-consuming attributes may delegate; every stream-consuming API
        # above is implemented explicitly.
        return getattr(self._damper_stream, name)


class _SyncDamperStreamManager:
    """Context manager whose ``__enter__`` establishes the stream with retries."""

    def __init__(
        self,
        *,
        open_stream: Callable[[], Any],
        policy: Policy,
        budget: RetryBudget,
        request: Mapping[str, Any],
        retry_after_extractor: Callable[[BaseException], float | None],
    ) -> None:
        self._open_stream = open_stream
        self._policy = policy
        self._budget = budget
        self._request = request
        self._extractor = retry_after_extractor
        self._proxy: _SyncDamperStream | None = None

    def __enter__(self) -> _SyncDamperStream:
        state = _StreamState()

        def attempt() -> _PrimedStream:
            return _prime_sync_stream(self._open_stream(), state)

        result = execute(
            attempt,
            policy=self._policy,
            budget=self._budget,
            request=self._request,
            retry_after_extractor=self._extractor,
            stream_started_probe=lambda: state.started,
            provider=_PROVIDER,
        )
        self._proxy = _SyncDamperStream(result.response, result.metadata)
        return self._proxy

    def __exit__(self, *exc_info: Any) -> Literal[False]:
        if self._proxy is not None:
            self._proxy.close()
        return False


class _AsyncDamperStreamManager:
    """Async context manager whose ``__aenter__`` establishes with retries."""

    def __init__(
        self,
        *,
        open_stream: Callable[[], Any],
        policy: Policy,
        budget: RetryBudget,
        request: Mapping[str, Any],
        retry_after_extractor: Callable[[BaseException], float | None],
    ) -> None:
        self._open_stream = open_stream
        self._policy = policy
        self._budget = budget
        self._request = request
        self._extractor = retry_after_extractor
        self._proxy: _AsyncDamperStream | None = None

    async def __aenter__(self) -> _AsyncDamperStream:
        state = _StreamState()

        async def attempt() -> _PrimedStream:
            return await _prime_async_stream(self._open_stream(), state)

        result = await execute_async(
            attempt,
            policy=self._policy,
            budget=self._budget,
            request=self._request,
            retry_after_extractor=self._extractor,
            stream_started_probe=lambda: state.started,
            provider=_PROVIDER,
        )
        self._proxy = _AsyncDamperStream(result.response, result.metadata)
        return self._proxy

    async def __aexit__(self, *exc_info: Any) -> Literal[False]:
        if self._proxy is not None:
            await self._proxy.aclose()
        return False


# --------------------------------------------------------------------------- #
# Proxies.
# --------------------------------------------------------------------------- #


class _SyncWrappedMessages:
    """Sync ``messages`` proxy: intercepts create/stream, delegates the rest."""

    def __init__(
        self,
        real_messages: Any,
        configured_messages: Any,
        *,
        policy: Policy,
        budget: RetryBudget,
        client_type: str,
        retry_after_extractor: Callable[[BaseException], float | None],
    ) -> None:
        self._damper_real = real_messages
        self._damper_configured = configured_messages
        self._damper_policy = policy
        self._damper_budget = budget
        self._damper_client_type = client_type
        self._damper_extractor = retry_after_extractor

    def create(self, **kwargs: Any) -> Any:
        _reject_max_retries(kwargs, self._damper_client_type)

        def attempt() -> Any:
            return self._damper_configured.create(**kwargs)

        result = execute(
            attempt,
            policy=self._damper_policy,
            budget=self._damper_budget,
            request=kwargs,
            retry_after_extractor=self._damper_extractor,
            provider=_PROVIDER,
        )
        _attach_metadata(result.response, result.metadata)
        return result.response

    def stream(self, **kwargs: Any) -> _SyncDamperStreamManager:
        _reject_max_retries(kwargs, self._damper_client_type)

        def open_stream() -> Any:
            return self._damper_configured.stream(**kwargs)

        return _SyncDamperStreamManager(
            open_stream=open_stream,
            policy=self._damper_policy,
            budget=self._damper_budget,
            request=kwargs,
            retry_after_extractor=self._damper_extractor,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._damper_real, name)


class _AsyncWrappedMessages:
    """Async ``messages`` proxy: intercepts create/stream, delegates the rest."""

    def __init__(
        self,
        real_messages: Any,
        configured_messages: Any,
        *,
        policy: Policy,
        budget: RetryBudget,
        client_type: str,
        retry_after_extractor: Callable[[BaseException], float | None],
    ) -> None:
        self._damper_real = real_messages
        self._damper_configured = configured_messages
        self._damper_policy = policy
        self._damper_budget = budget
        self._damper_client_type = client_type
        self._damper_extractor = retry_after_extractor

    async def create(self, **kwargs: Any) -> Any:
        _reject_max_retries(kwargs, self._damper_client_type)

        async def attempt() -> Any:
            return await self._damper_configured.create(**kwargs)

        result = await execute_async(
            attempt,
            policy=self._damper_policy,
            budget=self._damper_budget,
            request=kwargs,
            retry_after_extractor=self._damper_extractor,
            provider=_PROVIDER,
        )
        _attach_metadata(result.response, result.metadata)
        return result.response

    def stream(self, **kwargs: Any) -> _AsyncDamperStreamManager:
        _reject_max_retries(kwargs, self._damper_client_type)

        def open_stream() -> Any:
            return self._damper_configured.stream(**kwargs)

        return _AsyncDamperStreamManager(
            open_stream=open_stream,
            policy=self._damper_policy,
            budget=self._damper_budget,
            request=kwargs,
            retry_after_extractor=self._damper_extractor,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._damper_real, name)


class _WrappedClient:
    """Client proxy exposing the wrapped client's interface.

    ``messages`` is replaced by the Damper messages proxy; every other attribute
    delegates to the underlying client unchanged.
    """

    def __init__(self, client: Any, messages: Any) -> None:
        self._damper_client = client
        self._damper_messages = messages

    @property
    def messages(self) -> Any:
        return self._damper_messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._damper_client, name)


def resilient(client: Any, *, policy: Policy | None = None) -> Any:
    """Wrap an Anthropic client with Damper's owned retry executor.

    Returns a proxy exposing the same interface as ``client`` while intercepting
    ``messages.create`` and ``messages.stream`` (sync and async). All other
    attributes pass through unchanged.

    Raises
    ------
    ValueError
        If ``policy`` (or the default policy) has an invalid field value
        (``SPEC.md`` section 8.2).
    RetryOwnershipError
        If Damper cannot disable the SDK's own retries via the supported
        ``with_options`` mechanism (``SPEC.md`` section 7).
    """
    resolved = Policy() if policy is None else policy
    _validate_policy(resolved)

    configured = _configure_retry_ownership(client, resolved)

    budget = RetryBudget(
        ratio=resolved.retry_budget_ratio,
        min_tokens=resolved.retry_budget_min_tokens,
        window=resolved.retry_budget_window,
    )
    extractor = _make_retry_after_extractor()
    client_type = type(client).__name__

    messages: Any
    if _is_async_client(client):
        messages = _AsyncWrappedMessages(
            client.messages,
            configured.messages,
            policy=resolved,
            budget=budget,
            client_type=client_type,
            retry_after_extractor=extractor,
        )
    else:
        messages = _SyncWrappedMessages(
            client.messages,
            configured.messages,
            policy=resolved,
            budget=budget,
            client_type=client_type,
            retry_after_extractor=extractor,
        )
    return _WrappedClient(client, messages)
