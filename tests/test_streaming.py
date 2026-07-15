"""Streaming boundary tests (SESSION 6, SPEC section 15).

Damper retries a streaming call only before the first content token. Once any
content has streamed to the caller, a failure is surfaced during the caller's
own iteration -- never replayed -- and the stream proxy's ``damper.outcome``
becomes ``stream_started_failure``.

No network access, no real API key.
"""

from __future__ import annotations

from typing import Any

import anthropic
import pytest

from damper import Policy, resilient
from tests.fakes import (
    ConnectionResetBeforeFirstToken,
    FakeAnthropic,
    FakeAsyncAnthropic,
    FakeMessage,
    Ok,
    StreamThenReset,
)

FAST = Policy(backoff_base=0.0, backoff_max=0.0)
_REQUEST = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


def _types(events: list[Any]) -> list[Any]:
    return [getattr(e, "type", None) for e in events]


# --------------------------------- sync ------------------------------------ #


def test_stream_success_relays_events_and_metadata() -> None:
    client = FakeAnthropic([Ok(FakeMessage(id="final"))])
    wrapped = resilient(client, policy=FAST)

    with wrapped.messages.stream(**_REQUEST) as stream:
        events = list(stream)
        final = stream.get_final_message()

    assert len(client.stream_calls) == 1
    assert "content_block_delta" in _types(events)
    assert final.id == "final"
    assert stream.damper.outcome == "ok"
    assert stream.damper.attempts == 1


def test_stream_reset_before_first_token_retries() -> None:
    client = FakeAnthropic(
        [ConnectionResetBeforeFirstToken(), Ok(FakeMessage(id="ok"))]
    )
    wrapped = resilient(client, policy=FAST)

    with wrapped.messages.stream(**_REQUEST) as stream:
        events = list(stream)
        final = stream.get_final_message()

    # Failure arrived before any content token: a fresh stream was opened.
    assert len(client.stream_calls) == 2
    assert final.id == "ok"
    assert "content_block_delta" in _types(events)
    assert stream.damper.outcome == "ok"
    assert stream.damper.attempts == 2


def test_stream_reset_after_tokens_does_not_retry() -> None:
    client = FakeAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)

    seen: list[Any] = []
    with pytest.raises(anthropic.APIConnectionError):
        with wrapped.messages.stream(**_REQUEST) as stream:
            for event in stream:
                seen.append(event)

    # No retry once content has streamed, and the original error surfaced.
    assert len(client.stream_calls) == 1
    assert _types(seen).count("content_block_delta") == 3
    assert stream.damper.outcome == "stream_started_failure"


def test_stream_rejects_per_call_max_retries() -> None:
    from damper import RetryOwnershipError

    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(RetryOwnershipError):
        wrapped.messages.stream(max_retries=2, **_REQUEST)
    assert client.stream_calls == []


# --------------------------------- async ----------------------------------- #


async def test_async_stream_reset_before_first_token_retries() -> None:
    client = FakeAsyncAnthropic(
        [ConnectionResetBeforeFirstToken(), Ok(FakeMessage(id="ok"))]
    )
    wrapped = resilient(client, policy=FAST)

    events: list[Any] = []
    async with wrapped.messages.stream(**_REQUEST) as stream:
        async for event in stream:
            events.append(event)
        final = await stream.get_final_message()

    assert len(client.stream_calls) == 2
    assert final.id == "ok"
    assert "content_block_delta" in _types(events)
    assert stream.damper.attempts == 2


async def test_async_stream_reset_after_tokens_does_not_retry() -> None:
    client = FakeAsyncAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)

    seen: list[Any] = []
    with pytest.raises(anthropic.APIConnectionError):
        async with wrapped.messages.stream(**_REQUEST) as stream:
            async for event in stream:
                seen.append(event)

    assert len(client.stream_calls) == 1
    assert _types(seen).count("content_block_delta") == 3
    assert stream.damper.outcome == "stream_started_failure"


# ------------------- single owned consumption path (sync) ------------------ #
#
# The Ok stream plan is: message_start, content_block_delta("a"),
# content_block_delta("b"), message_stop. The first content token ("a") is
# pulled during priming and buffered; every consumption API below must include
# it exactly once and never duplicate it.


def test_iteration_includes_buffered_first_event_exactly_once() -> None:
    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with wrapped.messages.stream(**_REQUEST) as stream:
        events = list(stream)
    assert _types(events) == [
        "message_start",
        "content_block_delta",
        "content_block_delta",
        "message_stop",
    ]


def test_text_stream_includes_first_token_without_duplication() -> None:
    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with wrapped.messages.stream(**_REQUEST) as stream:
        texts = list(stream.text_stream)
    assert texts == ["a", "b"]


def test_get_final_text_returns_complete_text() -> None:
    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with wrapped.messages.stream(**_REQUEST) as stream:
        text = stream.get_final_text()
    assert text == "ab"


def test_get_final_message_reflects_full_content() -> None:
    client = FakeAnthropic([Ok(FakeMessage(id="done"))])
    wrapped = resilient(client, policy=FAST)
    with wrapped.messages.stream(**_REQUEST) as stream:
        message = stream.get_final_message()
    assert message.id == "done"
    assert message.text == "ab"  # buffered first token included


def test_until_done_drains_the_whole_stream() -> None:
    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with wrapped.messages.stream(**_REQUEST) as stream:
        stream.until_done()
        text = stream.get_final_text()
    assert text == "ab"


def test_partial_iteration_then_get_final_text_no_loss() -> None:
    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with wrapped.messages.stream(**_REQUEST) as stream:
        for event in stream:
            assert event.type == "message_start"
            break
        text = stream.get_final_text()
    # Continues from the current position: no replay, no dropped token.
    assert text == "ab"


def test_partial_iteration_then_until_done() -> None:
    client = FakeAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    with wrapped.messages.stream(**_REQUEST) as stream:
        consumed = 0
        for _event in stream:
            consumed += 1
            if consumed == 2:  # message_start, then first content token
                break
        stream.until_done()
        text = stream.get_final_text()
    assert text == "ab"


def test_text_stream_midstream_failure_sets_outcome() -> None:
    client = FakeAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(anthropic.APIConnectionError):
        with wrapped.messages.stream(**_REQUEST) as stream:
            for _text in stream.text_stream:
                pass
    assert stream.damper.outcome == "stream_started_failure"
    assert len(client.stream_calls) == 1  # no retry after first content


def test_get_final_text_midstream_failure_sets_outcome() -> None:
    client = FakeAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(anthropic.APIConnectionError):
        with wrapped.messages.stream(**_REQUEST) as stream:
            stream.get_final_text()
    assert stream.damper.outcome == "stream_started_failure"
    assert len(client.stream_calls) == 1


def test_until_done_midstream_failure_sets_outcome() -> None:
    client = FakeAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(anthropic.APIConnectionError):
        with wrapped.messages.stream(**_REQUEST) as stream:
            stream.until_done()
    assert stream.damper.outcome == "stream_started_failure"
    assert len(client.stream_calls) == 1


# ------------------- single owned consumption path (async) ----------------- #


async def test_async_text_stream_includes_first_token_without_duplication() -> None:
    client = FakeAsyncAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    texts: list[Any] = []
    async with wrapped.messages.stream(**_REQUEST) as stream:
        async for text in stream.text_stream:
            texts.append(text)
    assert texts == ["a", "b"]


async def test_async_get_final_text_returns_complete_text() -> None:
    client = FakeAsyncAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    async with wrapped.messages.stream(**_REQUEST) as stream:
        text = await stream.get_final_text()
    assert text == "ab"


async def test_async_until_done_drains_the_whole_stream() -> None:
    client = FakeAsyncAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    async with wrapped.messages.stream(**_REQUEST) as stream:
        await stream.until_done()
        text = await stream.get_final_text()
    assert text == "ab"


async def test_async_partial_iteration_then_get_final_text() -> None:
    client = FakeAsyncAnthropic([Ok()])
    wrapped = resilient(client, policy=FAST)
    async with wrapped.messages.stream(**_REQUEST) as stream:
        async for _event in stream:
            break  # consume only the first event
        text = await stream.get_final_text()
    assert text == "ab"


async def test_async_text_stream_midstream_failure_sets_outcome() -> None:
    client = FakeAsyncAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(anthropic.APIConnectionError):
        async with wrapped.messages.stream(**_REQUEST) as stream:
            async for _text in stream.text_stream:
                pass
    assert stream.damper.outcome == "stream_started_failure"
    assert len(client.stream_calls) == 1


async def test_async_get_final_text_midstream_failure_sets_outcome() -> None:
    client = FakeAsyncAnthropic([StreamThenReset(tokens=3)])
    wrapped = resilient(client, policy=FAST)
    with pytest.raises(anthropic.APIConnectionError):
        async with wrapped.messages.stream(**_REQUEST) as stream:
            await stream.get_final_text()
    assert stream.damper.outcome == "stream_started_failure"
    assert len(client.stream_calls) == 1
