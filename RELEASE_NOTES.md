# v0.1.0: Retry control for Anthropic

Damper v0.1.0 is the first public release.

Damper is an LLM reliability library. This release focuses on retry discipline
for the Anthropic Python SDK.

It supports wrapped `Anthropic` and `AsyncAnthropic` clients. Damper intercepts
`messages.create()` and `messages.stream()`. Other client methods continue to
work as they did before wrapping.

For intercepted calls, Damper disables the Anthropic SDK retry loop and runs
one retry loop of its own. This avoids stacked retries and lets calls made
through the same wrapped client share one retry budget.

## What is included

### Retry budget

Each wrapped client has a fixed window retry budget. A new window starts with
the configured capacity. Successful first attempts add capacity, and each
authorized retry consumes one unit.

During a provider outage, first attempts stop succeeding. The budget stops
growing, the remaining capacity drains, and Damper denies further retries.

### Retry cost ceiling

Applications can set a USD limit for the cumulative estimated cost of retries
for one logical request. The estimate is used only for retry decisions. It is
not billing data.

When a ceiling is configured and Damper cannot determine the model price, the
retry is denied with `RetryCostCeilingHit`. Applications can provide their own
price table through `Policy`.

### Streaming behavior

Damper retries a streaming call only while no output content delta has been
received.

Once the first output content delta arrives, a later failure is returned to the
caller and the stream is not replayed. This avoids repeating a request after
the caller may already have consumed part of the output.

### Error handling and backoff

Damper classifies Anthropic failures as retryable, not retryable, or ambiguous.
Applications can replace the built in classifier when they need different
rules.

Retries use exponential backoff with full jitter. Damper also parses and honors
valid Anthropic `Retry-After` values.

### Telemetry

Damper emits `damper.request` and `damper.attempt` spans through OpenTelemetry.
When no OpenTelemetry SDK or exporter is configured, telemetry does nothing and
does not affect the request.

Successful responses include metadata such as:

```text
resp.damper.attempts
resp.damper.retried
resp.damper.total_latency_s
resp.damper.retry_cost_usd
resp.damper.outcome
resp.damper.retry_budget_balance
```

### Outage demo

The repository includes a deterministic fake brownout simulation.

For 1,000 logical requests, a plain retry loop that allows three attempts makes
3,000 provider attempts. With the demo policy, Damper makes 1,010 provider
attempts.

```text
Naive retry loop
logical requests:        1000
provider attempts:       3000
amplification:           3.00x

Damper
logical requests:        1000
provider attempts:       1010
amplification:           1.01x
budget exhausted events: 995
```

The 1.01x result belongs to this deterministic simulation. It is not a general
product guarantee. The demo uses 1.1x as its pass or fail threshold.

## Install

```bash
pip install damper
```

Damper requires Python 3.10 or newer, `anthropic` 0.30 or newer, and
`opentelemetry-api` 1.20 or newer.

The OpenTelemetry SDK and exporter are optional.

## Not included in v0.1

This release does not include:

```text
caching
routing
prompt management
evals
guardrails
a standalone proxy
support for multiple providers
distributed retry budgets
circuit breakers
request hedging
adaptive timeouts
fallback chains
```

v0.1 focuses on budgeted retries, retry cost control, and safe streaming
behavior for Anthropic calls.

## Roadmap

The next planned milestone is request hedging in v0.2. Later milestones will be
decided after real use and contributor feedback.

## License

Apache-2.0
