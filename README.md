# Damper

**Reliability control for Anthropic LLM clients.**

Damper is a reliability library for LLM applications. v0.1 focuses on budgeted,
cost-aware, streaming-safe retries for the Anthropic Python SDK.

v0.1 supports `Anthropic` and `AsyncAnthropic`. It intercepts `messages.create()` and
`messages.stream()`. Other client methods pass through unchanged.

```python
from anthropic import Anthropic
from damper import resilient, Policy

client = resilient(Anthropic(), policy=Policy())

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Explain retry storms"}],
)

print(resp.damper.attempts)
print(resp.damper.retry_cost_usd)
```

> Damper v0.1.0 supports wrapped Anthropic clients only. As a pre-1.0 library,
> the API may continue to evolve in future minor releases.

---

## Why Damper exists

LLM provider calls are not normal HTTP calls.

They have:

- long-tail latency
- correlated provider failures
- streaming edge cases
- token-based cost
- retries that can multiply load during outages
- partial-output failure modes
- incident behavior that is often invisible to operators

Most SDKs provide competent per-request retry behavior. That is useful, but it is not enough during a provider brownout.

If every request retries independently, your application can amplify the outage:

```text
10,000 logical requests
x 3 attempts each
= 30,000 provider attempts during the brownout
```

Damper gives applications using Anthropic explicit control over retry admission, retry
cost, streaming failures, and retry telemetry.

---

## What v0.1 does

v0.1 focuses on one problem:

> LLM retries should not amplify provider outages, burn unbounded token cost, retry unsafe streaming failures, or disappear from telemetry.

Damper v0.1 provides:

- client-local retry budgets
- cost-aware retry ceilings
- Anthropic-specific error classification
- exponential backoff with full jitter
- Anthropic `Retry-After` handling
- streaming-safe retry semantics
- OpenTelemetry traces
- response metadata
- a fake outage demo

---

## Retry budget

Damper uses a client-local, fixed-window retry budget. One budget belongs to one wrapped
client instance.

Each window starts with the configured initial capacity. Successful first attempts add
capacity. Retries consume it. Nothing carries across a window boundary.

Within a window, retries are bounded by:

```text
retry_budget_min_tokens
+ retry_budget_ratio * successful first-attempt successes
```

With the default policy, each fixed window starts with 10 units of retry capacity. One
hundred successful first attempts in that window add 10 more units.

During a provider brownout, first attempts stop succeeding, so the budget stops being
replenished, drains, and Damper stops retrying.

That is the core behavior:

```text
shed retry load instead of amplifying the outage
```

---

## Streaming rule

Damper retries a streaming call only while no output content delta has been received.
Once the first output content delta arrives, later failures are surfaced and the stream is
not replayed.

This is intentional.

After streaming starts, the caller may already have consumed partial output. In agentic or tool-use flows, blindly replaying can duplicate work or create inconsistent behavior.

---

## Cost ceiling

Retries are not free.

A failed retry against a large prompt can burn real money. Damper can enforce a per-request retry cost ceiling:

```python
from damper import Policy

policy = Policy(max_retry_cost_usd=0.05)
```

The retry cost is an estimate used for safety. It is not billing truth.

---

## Configuration

`Policy` holds every knob. These are the complete defaults:

```python
from damper import Policy

Policy(
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

When the retry budget is exhausted, `on_budget_exhausted="raise"` raises
`RetryBudgetExhausted`. `on_budget_exhausted="passthrough"` surfaces the last provider
error directly.

`Policy` is frozen. Build a new one to change behavior.

---

## Async clients

`resilient()` accepts `AsyncAnthropic` and returns an async proxy:

```python
import asyncio

from anthropic import AsyncAnthropic
from damper import resilient, Policy


async def main() -> None:
    client = resilient(AsyncAnthropic(), policy=Policy())

    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": "Explain retry storms"}],
    )

    print(resp.damper.attempts)


asyncio.run(main())
```

---

## Telemetry

Damper emits OpenTelemetry spans for logical requests and provider attempts.

Spans:

```text
damper.request
damper.attempt
```

Stable attributes include:

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

Attributes are emitted where their value is known. `damper.provider` and `damper.model`
require the request to carry them, `damper.cost.estimate_usd` requires a priced model, and
`damper.attempt.backoff_s` is set on an attempt only when a retry follows it.

If no OpenTelemetry SDK/exporter is configured, telemetry is a no-op.

---

## Why not Tenacity?

Tenacity is a strong generic retry library.

Damper is narrower and more opinionated for LLM provider calls.

Damper understands:

- token cost
- retry budgets
- streaming retry boundaries
- Anthropic-specific error classification
- Anthropic `Retry-After` handling
- retry-decision telemetry

---

## Why not the SDK's built-in retries?

The Anthropic SDK provides per-request retries.

Damper controls retry admission across calls made through the same wrapped client
instance.

On top of the SDK's per-request behavior, Damper adds:

- client-local retry budgeting
- retry-cost estimation
- response metadata
- `damper.request` and `damper.attempt` spans

For intercepted calls, Damper disables the SDK's retries and replaces them with one owned
retry loop, so the two do not stack.

---

## Install

```bash
pip install damper
```

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
examples/01_basic.py        # wrap a client, read resp.damper metadata (needs a key)
examples/02_outage_demo.py  # deterministic brownout: naive per-request retries vs Damper
examples/03_telemetry.py    # view damper.request / damper.attempt spans via a console exporter
```

`examples/02_outage_demo.py --fake` runs in CI and exits non-zero if Damper fails
to bound retry amplification. It is deterministic and network-free.

In that fake brownout every request is failed twice and then succeeds, so a
naive per-request retry loop makes three attempts per request while Damper's
retry budget drains and sheds the rest:

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

The exact numbers are properties of this deterministic fake, not a claim about
any specific Anthropic SDK version or real outage. The baseline is a plain
per-request retry loop, not a reproduction of the SDK's retry behavior.

---

## Non-goals for v0.1

Damper v0.1 does not build:

- caching
- routing
- prompt management
- evals
- guardrails
- standalone proxy
- multi-provider support
- distributed retry budgets
- circuit breakers
- hedging
- adaptive timeouts
- fallback chains

This is intentional.

v0.1 ships one complete thing:

```text
budgeted, cost-aware, streaming-safe retries for Anthropic LLM calls
```

---

## Roadmap

Near-term:

```text
v0.1: budgeted retries for Anthropic
v0.2: request hedging
```

The rule: every version must be independently useful before the next milestone starts.

---

## Development

Quality gates:

```bash
ruff check .
mypy .
pytest
python examples/02_outage_demo.py --fake
```

Tests must not require:

- network access
- real provider API keys
- external services

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

See `CONTRIBUTING.md` for development setup and contribution guidelines.
`CLAUDE.md` contains additional guidance for coding agents.

---

## License

Apache-2.0
