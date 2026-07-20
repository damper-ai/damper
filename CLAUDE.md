# CLAUDE.md

This file contains development rules for Damper.

Damper is a Python reliability library for LLM clients. v0.1 focuses on retry
discipline for the Anthropic Python SDK.

This file is public. Keep it useful for contributors and forks. Do not refer to
private local files, personal directories, or machine specific setup.

## Project identity

Package name:

```text
damper
```

Public import:

```python
from damper import Policy, resilient
```

## Before changing code

For changes that affect behavior, public API, retry decisions, telemetry,
exceptions, packaging, or more than one file:

1. summarize the task
2. identify the files involved
3. explain any effect on v0.1 behavior
4. propose a small plan
5. wait for approval

For an exact and limited documentation or test edit, make the requested change
and show the diff unless the user asks for a plan first.

## Hard rules

Do not add work outside the current release scope.

Do not add a runtime dependency without approval.

Do not use the network, live provider calls, real API keys, or external
services in tests.

Do not silently change public API, exception behavior, telemetry names, retry
semantics, or model pricing behavior.

Do not run any of these commands:

```text
git add
git commit
git push
git tag
twine upload
```

Do not create a GitHub release or publish a package. The maintainer performs all
source control and release actions manually.

When a requested change would break one of these rules, stop and explain why.

## v0.1 scope

Damper v0.1 supports:

```text
Anthropic sync clients
Anthropic async clients
messages.create()
messages.stream()
one Damper owned retry loop for intercepted calls
client local fixed window retry budgets
cumulative retry cost ceilings
Anthropic specific error classification
exponential backoff with full jitter
Anthropic Retry-After handling
streaming retry boundaries
OpenTelemetry spans
response metadata
deterministic fake provider tests
the outage demo
documentation
CI
packaging
```

The following work is outside the scope of v0.1:

```text
request hedging
adaptive timeouts
circuit breakers
bulkheads
adaptive concurrency
priority shedding
fallback chains
support for multiple providers
proxy mode
Grafana dashboards
distributed retry budgets
prompt management
caching
routing
guardrails
evals
```

Open an issue or ask the maintainer before starting work near these boundaries.

## Files that need extra review

Changes to these files can alter retry safety or public behavior:

```text
damper/budget.py
damper/cost.py
damper/prices.py
damper/classify.py
damper/backoff.py
damper/_executor.py
damper/_wrapper.py
damper/telemetry.py
```

Relevant tests include:

```text
tests/test_budget.py
tests/test_cost.py
tests/test_executor.py
tests/test_retry_ownership.py
tests/test_streaming.py
tests/test_concurrency.py
tests/test_telemetry.py
```

When working in these areas:

```text
keep the code small and explicit
test the relevant invariant or failure mode
avoid hidden background behavior
avoid network calls in the decision path
avoid stacked retry loops
preserve provider exceptions as causes
cover synchronous and asynchronous paths where both exist
```

## Retry ownership

Damper owns retries for intercepted calls:

```text
client.messages.create(...)
client.messages.stream(...)
async_client.messages.create(...)
async_client.messages.stream(...)
```

The Anthropic SDK retry loop must be disabled through a supported SDK mechanism
before Damper runs its own loop.

This sequence:

```text
529, success
```

must result in exactly two provider attempts.

SDK retries and Damper retries must never stack. When Damper cannot safely take
ownership of retries, raise `RetryOwnershipError`.

## Retry budget

Each wrapped client has one local fixed window retry budget.

Within one window:

```text
authorized retries
<= retry_budget_min_tokens
   + retry_budget_ratio * successful first attempts
```

A new window starts with `retry_budget_min_tokens` units. Successful first
attempts add capacity. Each authorized retry consumes one unit. Unused capacity
does not carry into the next window.

Do not add Redis, database storage, process wide shared state, or any other
distributed coordination in v0.1.

## Retry cost ceiling

The retry cost ceiling applies to the cumulative estimated cost of retries for
one logical request.

When `max_retry_cost_usd` is configured and the model price cannot be
determined, Damper must fail closed:

```text
unknown retry cost
=> no retry
=> RetryCostCeilingHit
```

An unknown model must not be treated as free.

A custom `Policy.price_table` replaces the built in table. It does not merge
with the default table. Preserve this behavior unless a public API change is
reviewed and approved.

The built in price table is a versioned snapshot. Do not describe it as live
pricing or billing data.

## Streaming behavior

A streaming call may be retried only while no output content delta has been
received.

After the first output content delta, a later failure must be surfaced to the
caller. Damper must not replay the stream.

Prelude events such as stream or message start events do not close the retry
boundary unless they contain output content.

## Error handling

Damper must not swallow provider errors.

When Damper raises one of its own exceptions, preserve the provider exception
as `__cause__` when one exists.

Public exception types include:

```text
DamperError
RetryBudgetExhausted
RetriesExhausted
RetryCostCeilingHit
RetryOwnershipError
```

Do not add or rename public exceptions without approval.

## Telemetry

Damper emits these spans:

```text
damper.request
damper.attempt
```

Damper must continue to work when no OpenTelemetry SDK or exporter is
configured.

The `damper.*` attributes are the stable Damper telemetry contract.
Compatibility attributes outside that namespace may evolve independently.

Telemetry must not add network calls, blocking exporters, or hidden work to the
retry decision path.

Do not rename telemetry attributes without approval.

## Tests

Tests must not require:

```text
network access
provider API keys
external services
real delays
```

Use fake providers, deterministic clocks, injected sleep functions, fixed
random number generators, and scripted failures.

A behavior change must include focused tests for the affected path. Cover both
synchronous and asynchronous clients where both use the changed behavior.

Run:

```bash
ruff check .
mypy .
pytest
python examples/02_outage_demo.py --fake
```

When a command cannot run in the current environment, say so and list the
command that still needs to run locally.

## Quality bar

Before a change is ready for maintainer review:

```text
Ruff passes
mypy passes
pytest passes
tests use no network or real provider keys
intercepted Anthropic calls have one retry loop
provider exceptions remain available as causes
README claims match executable behavior
public API changes are deliberate, tested, and documented
the package name is damper everywhere
no later milestone feature has entered v0.1
```

## Working style

Prefer small changes that are easy to review.

After editing, report:

1. files changed
2. behavior changed
3. tests added or updated
4. commands run
5. command results
6. remaining risks or questions
7. whether production code changed

Do not hide uncertainty. When something is not known, say so.
