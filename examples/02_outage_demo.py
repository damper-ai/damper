"""Reproducible provider-brownout demo: naive per-request retries vs Damper.

Run the deterministic, network-free comparison used in CI::

    python examples/02_outage_demo.py --fake

``--fake`` (the default) simulates a total provider brownout across 1000 logical
requests and compares two clients against the *same* failure model:

* **Naive per-request retries** -- a hand-written per-request loop that retries
  each request up to three times. This is a plain illustration of independent
  per-request retrying; it is not a reproduction of any particular Anthropic SDK
  version's retry loop.
* **Damper** -- one wrapped client whose client-local retry budget sheds retry
  load once the budget drains, instead of amplifying the outage.

The fake provider raises real ``anthropic.APIStatusError`` values built without
any network (an ``httpx`` request/response pair), so the demo exercises the same
error shapes the wrapper sees in production. Nothing here is public Damper API:
the ``_Brownout*`` / ``_FakeResponse`` classes are private to this example and
exist only to drive ``resilient()`` offline.

``--live`` is an optional, guarded sanity check against a real provider. It does
*not* create failures or run amplification assertions; the reproducible outage
comparison is ``--fake`` only.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import anthropic
import httpx

from damper import Policy, RetryBudgetExhausted, resilient

_LOGICAL_REQUESTS = 1000
_AMPLIFICATION_BOUND = 1.1  # Damper provider attempts must stay within 1.1x.

# One reused httpx.Request is enough to construct provider errors offline; the
# fake never sends it anywhere.
_FAKE_REQUEST = httpx.Request("POST", "https://api.anthropic.test/v1/messages")


# --------------------------------------------------------------------------- #
# Private example-only fake provider (NOT public API, NOT damper.testing).
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Minimal stand-in for an Anthropic ``Message`` that accepts ``.damper``.

    Attribute assignment is allowed so the wrapper can attach ``resp.damper``
    metadata exactly as it does to the real (extra-allowing) SDK model.
    """

    def __init__(self, request_id: str) -> None:
        self.id = request_id


def _overloaded_529() -> anthropic.APIStatusError:
    """Build a real ``anthropic.APIStatusError`` (HTTP 529) with no network.

    No ``Retry-After`` header is set, so backoff falls to Damper's full-jitter
    path -- which the demo pins to zero delay through the public ``Policy``.
    """
    response = httpx.Response(status_code=529, request=_FAKE_REQUEST)
    return anthropic.APIStatusError(
        "overloaded_error", response=response, body=None
    )


def _request_id_of(kwargs: dict[str, Any]) -> str:
    """Extract the stable per-logical-request id carried in the message content."""
    messages = kwargs["messages"]
    content = messages[0]["content"]
    if not isinstance(content, str):
        raise AssertionError("demo request content must be a string id")
    return content


class _BrownoutMessages:
    """``messages`` surface: per-request-id brownout, shared provider state."""

    def __init__(self, provider: _BrownoutAnthropic) -> None:
        self._provider = provider

    def create(self, **kwargs: Any) -> _FakeResponse:
        request_id = _request_id_of(kwargs)
        self._provider.total_attempts += 1
        attempt = self._provider.attempts_by_id.get(request_id, 0) + 1
        self._provider.attempts_by_id[request_id] = attempt
        # Deterministic model: fail the first two provider attempts for every
        # logical request, then succeed on the third.
        if attempt < 3:
            raise _overloaded_529()
        return _FakeResponse(request_id)


class _BrownoutAnthropic:
    """Fake sync Anthropic client with shared brownout state across copies.

    ``with_options`` mirrors the SDK's configured-copy contract, but returns
    ``self`` so the ``max_retries=0`` client Damper builds shares one provider's
    attempt counters -- the fake both records retry ownership and serves calls.
    """

    def __init__(self) -> None:
        self.attempts_by_id: dict[str, int] = {}
        self.total_attempts = 0
        self.with_options_calls: list[dict[str, Any]] = []
        self.messages = _BrownoutMessages(self)

    def with_options(self, **kwargs: Any) -> _BrownoutAnthropic:
        self.with_options_calls.append(kwargs)
        return self


# --------------------------------------------------------------------------- #
# Runs.
# --------------------------------------------------------------------------- #


def _request_kwargs(request_id: str) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": request_id}],
    }


def _run_naive(provider: _BrownoutAnthropic, *, max_attempts: int = 3) -> None:
    """Naive per-request retries: retry each request independently up to N times."""
    for i in range(_LOGICAL_REQUESTS):
        kwargs = _request_kwargs(f"req-{i}")
        for _attempt in range(max_attempts):
            try:
                provider.messages.create(**kwargs)
                break
            except anthropic.APIStatusError:
                continue


def _run_damper(policy: Policy) -> tuple[_BrownoutAnthropic, int]:
    """Damper run: one wrapped client, shared budget, counts shed events."""
    provider = _BrownoutAnthropic()
    client = resilient(provider, policy=policy)
    budget_exhausted_events = 0
    for i in range(_LOGICAL_REQUESTS):
        try:
            client.messages.create(**_request_kwargs(f"req-{i}"))
        except RetryBudgetExhausted:
            budget_exhausted_events += 1
        except anthropic.APIStatusError:
            # A retryable failure that Damper chose not to (or could not) retry.
            pass
    return provider, budget_exhausted_events


# --------------------------------------------------------------------------- #
# Fake mode (deterministic, network-free, CI gate).
# --------------------------------------------------------------------------- #


def _damper_policy() -> Policy:
    return Policy(
        max_attempts=3,
        retry_budget_ratio=0.1,
        retry_budget_window=3600.0,
        retry_budget_min_tokens=10,
        backoff_base=0.0,
        backoff_max=0.0,
        max_retry_cost_usd=None,
        on_budget_exhausted="raise",
    )


def _print_report(
    *,
    naive_attempts: int,
    damper_attempts: int,
    budget_exhausted_events: int,
) -> None:
    naive_amp = naive_attempts / _LOGICAL_REQUESTS
    damper_amp = damper_attempts / _LOGICAL_REQUESTS
    print("Provider brownout simulation (deterministic --fake mode)")
    print()
    print("Naive per-request retries:")
    print(f"  logical requests:        {_LOGICAL_REQUESTS}")
    print(f"  provider attempts:       {naive_attempts}")
    print(f"  amplification:           {naive_amp:.2f}x")
    print()
    print("Damper:")
    print(f"  logical requests:        {_LOGICAL_REQUESTS}")
    print(f"  provider attempts:       {damper_attempts}")
    print(f"  amplification:           {damper_amp:.2f}x")
    print(f"  budget exhausted events: {budget_exhausted_events}")


def run_fake() -> int:
    """Run the deterministic comparison; return a process exit code."""
    naive_provider = _BrownoutAnthropic()
    _run_naive(naive_provider)
    naive_attempts = naive_provider.total_attempts

    damper_provider, budget_exhausted_events = _run_damper(_damper_policy())
    damper_attempts = damper_provider.total_attempts

    _print_report(
        naive_attempts=naive_attempts,
        damper_attempts=damper_attempts,
        budget_exhausted_events=budget_exhausted_events,
    )

    # Public invariants -- computed from recorded provider calls, never hardcoded.
    failures: list[str] = []
    if naive_attempts != _LOGICAL_REQUESTS * 3:
        failures.append(
            f"naive attempts {naive_attempts} != {_LOGICAL_REQUESTS * 3} "
            "(expected exactly 3.0x for this deterministic fake)"
        )
    if damper_attempts > _LOGICAL_REQUESTS * _AMPLIFICATION_BOUND:
        failures.append(
            f"damper attempts {damper_attempts} exceed the "
            f"{_AMPLIFICATION_BOUND:.1f}x bound "
            f"({int(_LOGICAL_REQUESTS * _AMPLIFICATION_BOUND)})"
        )
    if damper_attempts >= naive_attempts:
        failures.append(
            f"damper attempts {damper_attempts} not fewer than naive "
            f"{naive_attempts}"
        )
    if budget_exhausted_events <= 0:
        failures.append("expected at least one budget-exhausted event")

    print()
    if failures:
        for reason in failures:
            print(f"FAIL: {reason}")
        return 1
    print("PASS: Damper bounded retry amplification during the brownout.")
    return 0


# --------------------------------------------------------------------------- #
# Live mode (optional, guarded, no failure injection, no assertions).
# --------------------------------------------------------------------------- #


def run_live() -> int:
    """Optional real-provider sanity check. Not the amplification comparison."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Live mode needs ANTHROPIC_API_KEY. The reproducible outage "
            "comparison is the deterministic --fake mode; run:\n"
            "    python examples/02_outage_demo.py --fake"
        )
        return 0

    client = resilient(anthropic.Anthropic(), policy=Policy())
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16,
        messages=[{"role": "user", "content": "Say hi in one word."}],
    )
    print("Live request succeeded (this is a sanity check, not a brownout).")
    print(f"  attempts: {resp.damper.attempts}")
    print(f"  outcome:  {resp.damper.outcome}")
    print(
        "\nLive mode never injects failures or runs amplification assertions. "
        "Use --fake for the reproducible outage comparison."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fake",
        action="store_true",
        help="deterministic, network-free brownout comparison (default)",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="optional real-provider sanity check (needs ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args(argv)
    if args.live:
        return run_live()
    return run_fake()


if __name__ == "__main__":
    sys.exit(main())
