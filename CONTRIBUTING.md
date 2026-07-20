# Contributing to Damper

Thanks for taking the time to contribute.

This file explains how to set up the project, run the checks, and prepare a
change for review.

## Development setup

Damper requires Python 3.10 or newer.

Create a virtual environment and install the development dependencies.

### Linux and macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Checks to run

Run these commands before opening a pull request:

```bash
ruff check .
mypy .
pytest
python examples/02_outage_demo.py --fake
```

They check linting, import order, types, tests, and the deterministic outage
demo.

The outage demo exits with a nonzero status when Damper does not keep retry
amplification within the limit used by the demo.

## Tests

A reliability library needs tests that behave the same way on every run.

Tests must not use the network, real provider keys, or external services. Do
not add secrets or credentials to tests, examples, fixtures, or CI.

Use the existing fake providers and injected clocks, sleep functions, and
random number generators. Do not use real delays when the same behavior can be
tested with an injected clock or captured sleep call.

Any change that affects a retry decision must include focused tests for that
behavior. This includes retry budget accounting, retry cost ceilings, error
classification, backoff, streaming behavior, and ownership of the retry loop.

Where both synchronous and asynchronous paths exist, cover both unless the
change is limited to one path.

## Scope of v0.1

Damper v0.1 supports wrapped Anthropic clients and provides:

```text
client local fixed window retry budgets
cumulative retry cost ceilings
Anthropic specific error classification
exponential backoff with full jitter
Anthropic Retry-After handling
safe retry behavior for streaming calls
one Damper owned retry loop for intercepted calls
OpenTelemetry spans and response metadata
```

The following work is outside the scope of v0.1:

```text
request hedging
adaptive timeouts
circuit breakers
bulkheads
adaptive concurrency
fallback chains
support for multiple providers
proxy mode
caching
routing
prompt management
evals
guardrails
distributed retry budgets
```

Open an issue before starting work that changes these boundaries. This gives us
a chance to agree on the behavior and release scope before code is written.

## Pull requests

Keep each pull request focused on one change.

Include tests with behavior changes. Update documentation when a public API,
configuration option, exception, metadata field, or user visible behavior
changes.

Make sure all four checks pass locally. In the pull request description,
explain what changed, why it changed, and any compatibility or operational
impact reviewers should know about.

`CLAUDE.md` contains additional instructions for coding agents. Contributors do
not need to read it to follow the setup and contribution rules in this file.
