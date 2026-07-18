# Damper

**Reliability control for LLM clients.**

Damper adds budgeted, cost-aware, streaming-safe retry behavior around LLM provider SDKs.

v0.1 starts with Anthropic.

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

> Status: pre-v0.1 implementation. The public API may change until `v0.1.0` is tagged.

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

Damper exists to make LLM clients fail with control.

---

## What v0.1 does

v0.1 focuses on one problem:

> LLM retries should not amplify provider outages, burn unbounded token cost, retry unsafe streaming failures, or disappear from telemetry.

Damper v0.1 provides:

- client-local retry budgets
- cost-aware retry ceilings
- LLM-specific error classification
- exponential backoff with full jitter
- provider retry-after handling
- streaming-safe retry semantics
- OpenTelemetry traces
- response metadata
- a fake outage demo

---

## Retry budget

Damper uses a client-local retry budget.

Successful first attempts add retry budget. Retries consume it.

Example:

```text
retry_budget_ratio = 0.1

100 successful first attempts
=> 10 retry tokens
=> retries stay bounded to roughly 10% of recent successful traffic
```

During a provider brownout, first attempts stop succeeding, the retry budget drains, and Damper stops retrying.

That is the core behavior:

```text
shed retry load instead of amplifying the outage
```

---

## Streaming rule

Damper retries streaming calls only before the first token arrives.

Once content has streamed to the caller, failures are surfaced instead of replayed.

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

If no OpenTelemetry SDK/exporter is configured, telemetry is a no-op.

---

## Why not Tenacity?

Tenacity is a strong generic retry library.

Damper is narrower and more opinionated for LLM provider calls.

Damper understands:

- token cost
- retry budgets
- streaming retry boundaries
- LLM-provider error classes
- provider retry-after
- LLM-specific telemetry

---

## Why not the SDK's built-in retries?

Provider SDK retries are per-request retry hygiene.

Damper controls retry behavior at the application/client level.

The difference:

- SDK retries are per-request only.
- Per-request retries can amplify provider brownouts.
- SDK retries are cost-blind.
- SDK retries are mostly invisible during incidents.
- Damper disables SDK retries for wrapped calls and replaces them with one owned retry loop.

Damper does not stack on top of SDK retries.

---

## Install

Damper is not published yet.

After `v0.1.0` is released:

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

See `CLAUDE.md` for coding-agent and contributor guidance.

---

## License

Apache-2.0
