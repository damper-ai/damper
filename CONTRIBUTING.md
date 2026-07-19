# Contributing to Damper

Thanks for your interest in improving Damper. This guide covers everything you
need to develop and validate a change locally.

## Development environment

Damper requires **Python 3.10 or newer**.

Create and activate a virtual environment, then install the package with its
development extras:

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Quality gates

Every change must pass all four gates before it is submitted:

```bash
ruff check .
mypy .
pytest
python examples/02_outage_demo.py --fake
```

- `ruff check .`: linting and import ordering
- `mypy .`: static type checking
- `pytest`: the full test suite
- `python examples/02_outage_demo.py --fake`: the deterministic outage demo,
  which exits non-zero if Damper fails to bound retry amplification

## Testing rules

Tests are the credibility of a reliability library. They must be deterministic
and self-contained:

- Tests must **not** require network access.
- Tests must **not** require a real provider API key.
- Do not put secrets or credentials in tests, examples, fixtures, or CI.
- Use the scripted fakes and injected clocks/sleeps rather than live calls or
  wall-clock delays.

Any change to retry decisions must come with focused tests that cover the
relevant invariant or failure mode. This applies to budget accounting, cost
ceilings, error classification, backoff, the streaming boundary, and retry
ownership.

## Scope (v0.1)

Damper v0.1 is intentionally narrow. It provides:

- client-local retry budgets
- cost-aware retry ceilings
- LLM-specific error classification
- full-jitter exponential backoff with Retry-After handling
- streaming-safe retry semantics
- exclusive retry ownership for wrapped Anthropic calls
- OpenTelemetry traces and response metadata

The following are explicitly **out of scope** for v0.1 and should not be added:

- hedging
- adaptive timeouts
- circuit breakers
- bulkheads or adaptive concurrency
- fallback chains
- multi-provider adapters
- proxy mode
- caching, routing, prompt management, evals, or guardrails
- distributed / fleet-wide retry budgets

Please open an issue to discuss any feature near or beyond these boundaries
before writing code.

## Pull requests

- Keep pull requests focused on a single change.
- Include or update tests alongside behavior changes.
- Make sure all four quality gates pass locally.

`CLAUDE.md` contains additional repository guidance for coding agents and
contributors, but you do not need to read it to follow the rules above.
