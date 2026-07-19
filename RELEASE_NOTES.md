# v0.1.0: Budgeted, cost-aware, streaming-safe retries for Anthropic

Damper v0.1.0 is the first public release.

Damper owns the retry loop for wrapped Anthropic `messages.create` and
`messages.stream` calls, for both sync and async clients.

It disables SDK retries for those intercepted calls and replaces them with one
controlled retry loop. This prevents SDK retries and Damper retries from
stacking, while bounding retry amplification through a client-local retry
budget.

## Highlights

- **Client-local retry budget**

  Fixed-window accounting limits retries to a configurable fraction of
  successful first attempts. During a provider outage, first-attempt successes
  stop replenishing the budget, available retry capacity drains, and additional
  retry load is shed.

- **Cost-aware retry ceiling**

  An optional per-request USD ceiling limits the cumulative estimated cost of
  retry attempts. The estimate is a safety guard, not billing truth.

- **Streaming-safe retry boundary**

  Streaming calls are retried only before the first content event. Once content
  has been exposed to the caller, later failures are surfaced and the request is
  not replayed.

- **Anthropic-specific error classification**

  Retryable, non-retryable, and ambiguous failures are classified using
  provider-aware rules. Applications can supply their own classifier.

- **Full-jitter exponential backoff**

  Damper supports capped full-jitter backoff and normalized provider
  `Retry-After` values.

- **OpenTelemetry traces**

  Damper emits `damper.request` and `damper.attempt` spans. Telemetry safely
  becomes a no-op when no OpenTelemetry SDK is configured.

- **Response metadata**

  Successful responses expose metadata including:

  - `resp.damper.attempts`
  - `resp.damper.retried`
  - `resp.damper.total_latency_s`
  - `resp.damper.retry_cost_usd`
  - `resp.damper.outcome`
  - `resp.damper.retry_budget_balance`

- **Deterministic outage demo**

  In the included fake brownout simulation, 1,000 logical requests produce
  3,000 provider attempts with a naive three-attempt retry loop and 1,010
  attempts (about 1.01x) with the demo's configured Damper policy.

  These numbers belong to the deterministic simulation. The demo asserts the
  broader configured bound of at most 1.1 provider attempts per logical request,
  so the exact 1.01x figure is the observed result and 1.1x is the enforced
  ceiling.

## Install

```bash
pip install damper
```

Requirements:

- Python >= 3.10
- `anthropic` >= 0.30
- `opentelemetry-api` >= 1.20

The OpenTelemetry SDK/exporter is optional. Without it, Damper's telemetry is a
safe no-op.

## Scope

v0.1 ships one complete thing:

```text
budgeted, cost-aware, streaming-safe retries for Anthropic LLM calls
```

Not in v0.1:

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

Roadmap: v0.2 adds request hedging. Every version must be independently useful
before the next milestone starts.

## License

Apache-2.0
