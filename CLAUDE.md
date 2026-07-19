# CLAUDE.md - Damper Development Guidelines

You are helping implement Damper, a Python reliability library for LLM clients.

Damper v0.1 focuses on budgeted, cost-aware, streaming-safe retries for Anthropic LLM calls.

This file is public project guidance for Claude Code and other coding agents. It must stay useful for contributors who fork the repository. Do not reference private local files in this document.

---

## Project identity

Package name:

```text
damper
```

Import path:

```python
from damper import resilient, Policy
```

Do not use old names such as:

```text
steadfast
llm-resilience
bulwark-llm
```

If old names appear in generated code, docs, comments, tests, CI, or examples, stop and ask before changing broadly.

---

## First response rule

Before editing files:

1. summarize the task
2. summarize the relevant v0.1 scope
3. identify reliability-critical files touched by the task
4. propose a small plan
5. wait for approval

Do not write code until Amit approves the plan.

---

## Hard rules

- Do not build features outside v0.1 scope.
- Do not implement hedging, adaptive timeouts, circuit breakers, bulkheads, fallbacks, multi-provider support, proxy mode, caching, guardrails, or evals.
- Do not use network access in tests.
- Do not make live provider API calls in tests.
- Do not require real API keys for CI.
- Do not silently change public API.
- Do not add new runtime dependencies without approval.
- Do not tag releases.
- Do not publish to PyPI.
- Do not push to remote unless Amit explicitly asks.
- If implementation requires changing public API, exception behavior, telemetry names, or retry semantics, stop and explain the tradeoff first.

---

## v0.1 scope

Allowed in v0.1:

```text
Anthropic sync wrapper
Anthropic async wrapper
SDK retry ownership
client-local retry budget
cost ceiling
error classification
backoff with full jitter
retry-after handling
streaming retry boundary
OpenTelemetry traces
response metadata
fake Anthropic client
outage demo
README
CI
packaging
```

Not allowed in v0.1:

```text
hedging
adaptive timeouts
circuit breakers
bulkheads
adaptive concurrency
priority shedding
fallback chains
multi-provider adapters
proxy mode
Grafana dashboard
distributed retry budget
prompt management
caching
guardrails
evals
```

---

## Core implementation rule

The agent may generate an initial implementation for all modules.

Amit will manually review, rewrite, or directly control reliability-critical parts before release.

Reliability-critical files:

```text
damper/budget.py
damper/cost.py
damper/_executor.py
damper/_wrapper.py
tests/test_budget.py
tests/test_retry_ownership.py
tests/test_concurrency.py
```

Extra care is required for:

```text
client-local retry budget accounting
cost estimation and retry cost ceiling
core retry-decision logic
SDK retry ownership
streaming retry boundary
concurrency safety
budget invariant tests
```

Generated code in these areas is not final until Amit approves the diff.

When working on reliability-critical files:

- keep the code small and explicit
- add tests for the relevant invariant or failure mode
- avoid clever abstractions
- avoid hidden background behavior
- avoid network calls in the decision path
- avoid SDK retry stacking
- preserve original provider exceptions as causes

---

## Retry ownership rule

Damper owns retries for wrapped calls.

Do not allow SDK retries and Damper retries to stack.

For intercepted Anthropic calls:

```text
client.messages.create(...)
client.messages.stream(...)
async_client.messages.create(...)
async_client.messages.stream(...)
```

the implementation must disable Anthropic SDK retries using the supported SDK mechanism.

Required behavior:

```text
[529, ok] -> exactly 2 provider attempts total
```

Forbidden behavior:

```text
SDK retries x Damper retries
```

If SDK retries cannot be disabled safely, raise `RetryOwnershipError`.

---

## Retry budget rule

The v0.1 retry budget is client-local.

Do not describe it as a distributed fleet-wide budget.

The implementation must use one budget per wrapped client instance.

Required invariant:

```text
retries <= retry_budget_ratio * successful_first_attempts + initial_capacity
```

Initial capacity comes from `retry_budget_min_tokens`.

Do not implement external coordination, Redis, database storage, shared process state, or distributed budget accounting in v0.1.

---

## Streaming rule

Damper retries a streaming call only while no output content delta has been received.

Once the first output content delta arrives, Damper must surface the failure and not replay the stream.

This is a correctness rule, not a missing feature.

---

## Error handling rule

Damper must not swallow provider errors.

When Damper raises its own exception, preserve the original provider exception as `__cause__` when one exists.

Required exception types:

```text
DamperError
RetryBudgetExhausted
RetriesExhausted
RetryCostCeilingHit
RetryOwnershipError
```

---

## Telemetry rule

Damper emits OpenTelemetry spans, but must work when no OTel SDK or exporter is configured.

Expected span names:

```text
damper.request
damper.attempt
```

Telemetry must not add network calls, blocking exporters, or hidden work to the hot path.

Do not change telemetry attribute names without approval.

---

## Testing rules

No test may require:

```text
network access
real Anthropic API key
real OpenAI API key
real Gemini API key
external services
wall-clock sleeps longer than necessary
```

Use fakes, deterministic clocks, and scripted failure sequences.

Every implementation change should include or update tests.

Required quality gates:

```text
ruff check .
mypy .
pytest
python examples/02_outage_demo.py --fake
```

If a command cannot run in the current environment, say so explicitly and explain what should be run locally.

---

## Quality bar

- ruff green
- mypy green
- pytest green
- no network access in tests
- no hidden retries under wrapped Anthropic calls
- no SDK x Damper retry multiplication
- fake outage demo proves retry amplification is bounded
- README claims match executable examples
- public API is stable and documented
- package name is `damper` everywhere
- original provider exceptions are preserved
- no future-scope features are implemented in v0.1

---

## Git and release rules

You may suggest commit messages.

Do not run:

```text
git add
git commit
git push
git tag
twine upload
```

unless Amit explicitly asks.

Never publish to PyPI.

Release prep may generate:

```text
CHANGELOG.md
dist build instructions
twine check instructions
manual publish commands
```

Amit performs commits, pushes, tags, and uploads manually.

---

## Stop conditions

Stop and ask Amit when:

- Anthropic SDK retry disabling is unclear or unsupported
- public API needs to change
- reliability-critical interfaces are insufficient
- tests require network access
- implementation requires a new runtime dependency
- v0.1 scope starts drifting into later milestones
- current repo state is unclear
- old names like `steadfast` are mixed with `damper`
- a design choice affects public API, telemetry attribute names, or exception behavior

---

## Working style

Prefer small, reviewable changes.

For sessions touching more than one file:

1. read the relevant README sections and existing tests
2. propose a plan
3. wait for approval
4. edit
5. summarize changed files
6. list commands run
7. list remaining risks

Do not hide uncertainty. If something is unknown, say it.

---

## Suggested first session

Use this for the first agent session:

```text
Read README.md and CLAUDE.md.

Summarize the v0.1 scope in two sentences.
List the reliability-critical files that need extra maintainer review.
Then propose a plan for scaffolding only.

Do not implement retry budget, cost estimation, executor behavior, wrapper behavior, telemetry, or provider logic yet.

Create only the package skeleton, pyproject.toml, test skeleton, fake client skeleton, CI, and placeholder modules.

Stop after presenting the plan for approval.
```
