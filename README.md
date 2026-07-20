# Damper

[![PyPI version](https://img.shields.io/pypi/v/damper.svg)](https://pypi.org/project/damper/)
[![Python versions](https://img.shields.io/pypi/pyversions/damper.svg)](https://pypi.org/project/damper/)
[![CI](https://github.com/damper-ai/damper/actions/workflows/ci.yml/badge.svg)](https://github.com/damper-ai/damper/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/damper-ai/damper/blob/main/LICENSE)

**Reliability control for the Anthropic Python SDK.**

> This README documents the current main branch. The latest released version is
> v0.1.0.

Damper is an LLM reliability library. v0.1 starts with retry control for the
Anthropic Python SDK.

It supports `Anthropic` and `AsyncAnthropic`. It intercepts
`messages.create()` and `messages.stream()`. Other client methods pass through
unchanged.

Damper is an independent open source project. It is not affiliated with,
sponsored by, or endorsed by Anthropic.

```python
from anthropic import Anthropic
from damper import Policy, resilient

client = resilient(Anthropic(), policy=Policy())

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Explain retry storms"}],
)

print(response.damper.attempts)
print(response.damper.retry_cost_usd)
```

> Damper v0.1.0 supports wrapped Anthropic clients only. The API may continue
> to evolve while the project is below 1.0.

---

## Why Damper exists

An LLM call has a different failure profile from a typical HTTP request. It may
take a long time to produce output, fail partway through a stream, or become
slow across many requests at once. Retrying also repeats some amount of token
cost.

The Anthropic SDK already handles retries for an individual request. That is
useful for isolated failures. The problem appears when many requests fail
together and each one starts its own retry sequence.

```text
10,000 logical requests
x 3 attempts
= 30,000 provider attempts
```

During a provider brownout, that extra traffic can make the situation worse.
Damper adds retry admission, cost limits, streaming rules, and telemetry around
calls made through the same wrapped client.

---

## What v0.1 handles

Damper v0.1 provides:

- a client-local retry budget
- a cumulative retry cost ceiling
- Anthropic-specific error classification
- exponential backoff with full jitter
- Anthropic `Retry-After` handling
- safe retry behavior for streaming calls
- OpenTelemetry request and attempt spans
- response metadata
- a deterministic outage demo

The scope is intentionally narrow. v0.1 is about retry discipline for
Anthropic calls.

---

## Retry budget

Each wrapped client has its own fixed-window retry budget.

A new window starts with `retry_budget_min_tokens` units of retry capacity.
Successful first attempts add capacity according to `retry_budget_ratio`.
An authorized retry consumes one unit. Unused capacity does not carry into the
next window.

Within one window, the number of authorized retries is bounded by:

```text
retry_budget_min_tokens
+ retry_budget_ratio * successful first-attempt successes
```

With the default policy, each window starts with 10 units. The ratio is `0.1`,
so 100 successful first attempts in the same window add 10 more units.

When the provider degrades, first attempts stop succeeding. The budget then
stops growing, existing capacity drains, and further retries are denied. This
limits retry amplification without requiring a separate gateway or service.

Budgets are not coordinated across client instances or processes. Each wrapped
client has its own retry budget, so if an application creates multiple wrapped
clients or runs multiple replicas, each one starts with its own configured
retry capacity.

---

## Error classification

Damper classifies each provider failure before deciding whether to retry. The
classification is Anthropic-specific. It uses the error's HTTP status code, or
the transport failure type when there is no status code.

| Class | What Damper puts here | Damper action |
| --- | --- | --- |
| Retryable | HTTP 429 and any 5xx, including 500, 502, 503, 504, and Anthropic's 529 overloaded. Also connection resets and timeouts that occur before any content has streamed. | Retried, subject to the retry budget, cost ceiling, and `max_attempts`. |
| Not retryable | HTTP 400, 401, 402, 403, 404, 409, and 413, every other 4xx, and any failure that is neither a retryable status code nor an Anthropic transport error. | Surfaced to the caller. Not retried. |
| Ambiguous | A connection reset or timeout that occurs after content has started streaming. | Surfaced to the caller. Not retried, because part of the output may already have reached the caller. |

A custom classifier supplied through `Policy.classifier` can override these
decisions.

---

## Streaming behavior

Damper retries a streaming call only while no output content delta has been
received.

Once the first output content delta arrives, later failures are surfaced and
the stream is not replayed. The caller may already have consumed partial
output, so replaying the request could duplicate work or produce inconsistent
results.

---

## Retry cost ceiling

A retry repeats part or all of the request cost. This matters for large prompts
and long output reservations.

Damper can block another retry when the cumulative estimated retry cost for one
logical request would cross a configured limit:

```python
from damper import Policy

policy = Policy(max_retry_cost_usd=0.05)
```

Damper's built-in model prices are a versioned snapshot of Anthropic's published
pricing. They are used only for retry-cost estimates and are not billing data.

Anthropic may change model pricing between Damper releases. Applications that
need stricter control can provide an updated `price_table` through `Policy`
without waiting for a new Damper release.

If a retry cost ceiling is configured and Damper cannot determine the model
price, the retry is denied and `RetryCostCeilingHit` is raised. Applications
can provide a `price_table` through `Policy` for models that are not included
in Damper's built-in table.

A custom `price_table` replaces the built-in table rather than merging with it.

---

## Configuration

`Policy` contains the public configuration for v0.1:

```python
from damper import Policy

policy = Policy(
    max_attempts=3,
    per_attempt_timeout=60.0,
    max_retry_cost_usd=None,
    retry_budget_ratio=0.1,
    retry_budget_window=60.0,
    retry_budget_min_tokens=10,
    backoff_base=1.0,
    backoff_max=30.0,
    respect_retry_after=True,
    on_budget_exhausted="raise",
    price_table=None,
    classifier=None,
)
```

When the retry budget blocks a retry:

- `on_budget_exhausted="raise"` raises `RetryBudgetExhausted`
- `on_budget_exhausted="passthrough"` surfaces the last provider error

`Policy` is frozen. Create a new instance when you need different settings.

---

## Async clients

`resilient()` also supports `AsyncAnthropic`:

```python
import asyncio

from anthropic import AsyncAnthropic
from damper import Policy, resilient


async def main() -> None:
    client = resilient(AsyncAnthropic(), policy=Policy())

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": "Explain retry storms"}],
    )

    print(response.damper.attempts)


asyncio.run(main())
```

---

## Telemetry

Damper emits one `damper.request` span for the logical request and one
`damper.attempt` span for each provider attempt.

Stable Damper attributes include:

```text
damper.attempts
damper.outcome
damper.retry_budget.balance
damper.retry_budget.ratio
damper.cost.estimate_usd
damper.cost.retry_usd
damper.attempt.error_class
damper.attempt.backoff_s
damper.provider
damper.model
```

Some attributes are conditional. Provider and model attributes require those
values to be present on the request. Cost estimates require a priced model.
`damper.attempt.backoff_s` is added only when another retry follows the
attempt.

If no OpenTelemetry SDK or exporter is configured, these calls become no-ops.

---

## How is Damper different from Tenacity?

Tenacity is a mature general-purpose retry library. It gives developers
flexible primitives for retrying Python functions.

Damper is focused on reliability around LLM calls. In v0.1 it provides
Anthropic-specific retry control directly:

- fixed-window retry budgets shared across calls on one wrapped client
- cumulative retry cost ceilings
- one Damper-owned retry loop for intercepted Anthropic calls
- no retry after the first output content delta
- Anthropic-specific error classification and `Retry-After` normalization
- response metadata and OpenTelemetry spans for retry decisions

Use Tenacity when you need general-purpose retry primitives or want to build
your own policy.

Use Damper when you want these controls around Anthropic calls without
assembling the policy and SDK integration yourself.

---

## How is Damper different from the Anthropic SDK retries?

The Anthropic SDK handles retries inside an individual request.

Damper adds admission control across calls made through the same wrapped client.
It also adds:

- client-local retry budgeting
- retry cost estimation
- response metadata
- `damper.request` and `damper.attempt` spans

For intercepted calls, Damper disables the SDK retry loop and runs one
Damper-owned loop. The two retry layers do not stack.

---

## Install

```bash
pip install damper
```

Requirements:

- Python 3.10 or newer
- `anthropic` 0.30 or newer
- `opentelemetry-api` 1.20 or newer

The OpenTelemetry SDK and exporter are optional.

For local development:

```bash
git clone https://github.com/damper-ai/damper.git
cd damper
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
git clone https://github.com/damper-ai/damper.git
cd damper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

---

## Examples

```text
examples/01_basic.py        # basic client wrapping and response metadata
examples/02_outage_demo.py  # deterministic retry amplification demo
examples/03_telemetry.py    # console output for Damper spans
```

The outage demo is deterministic and does not use the network:

```bash
python examples/02_outage_demo.py --fake
```

In the demo, every request fails twice before it succeeds. A plain
three-attempt retry loop therefore makes 3,000 provider attempts for 1,000
logical requests. With the configured Damper policy, the retry budget drains
and the run makes 1,010 provider attempts.

```text
Naive per-request retries:
  logical requests:        1000
  provider attempts:       3000
  amplification:           3.00x

Damper:
  logical requests:        1000
  provider attempts:       1010
  amplification:           1.01x
  budget exhausted events: 995
```

These numbers belong to this deterministic simulation. They are not a claim
about every outage or every policy. The demo uses `1.1x` as its pass or fail
threshold.

---

## Non-goals for v0.1

Damper v0.1 does not include:

- caching
- routing
- prompt management
- evals
- guardrails
- a standalone proxy
- multi-provider support
- distributed retry budgets
- circuit breakers
- hedging
- adaptive timeouts
- fallback chains

The current release focuses on budgeted, cost-aware, streaming-safe retries for
Anthropic calls.

---

## Roadmap

```text
v0.1: retry discipline for Anthropic
v0.2: request hedging
```

Each version should be useful on its own before the next milestone starts.

---

## Development

Run the same checks used by CI:

```bash
ruff check .
mypy .
pytest
python examples/02_outage_demo.py --fake
```

Tests must not require network access, provider API keys, or external services.

See `CONTRIBUTING.md` for setup and contribution guidelines.
`CLAUDE.md` contains additional guidance for coding agents.

---

## Repository layout

```text
damper/
  __init__.py
  _wrapper.py
  _executor.py
  budget.py
  cost.py
  classify.py
  backoff.py
  telemetry.py
  prices.py
  py.typed

tests/
examples/
```

---

## License

Apache-2.0
