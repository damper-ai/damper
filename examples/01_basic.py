"""Basic usage: wrap an Anthropic client and read Damper response metadata.

Run::

    python examples/01_basic.py

The real request is guarded behind ``ANTHROPIC_API_KEY``. Without a key this
example prints guidance and exits successfully, so it is safe to run in CI and
offline. It makes no network call unless a key is present.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "This example makes one real Anthropic request.\n"
            "Set ANTHROPIC_API_KEY to run it, e.g.:\n\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "    python examples/01_basic.py\n\n"
            "For a fully offline demonstration, run the deterministic outage "
            "demo instead:\n\n"
            "    python examples/02_outage_demo.py --fake"
        )
        return 0

    import anthropic

    from damper import Policy, resilient

    client = resilient(anthropic.Anthropic(), policy=Policy())

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": "Explain retry storms in one line."}],
    )

    print(resp.content[0].text)
    print()
    print("Damper metadata:")
    print(f"  attempts:             {resp.damper.attempts}")
    print(f"  retried:              {resp.damper.retried}")
    print(f"  total_latency_s:      {resp.damper.total_latency_s:.3f}")
    print(f"  retry_cost_usd:       {resp.damper.retry_cost_usd}")
    print(f"  outcome:              {resp.damper.outcome}")
    print(f"  retry_budget_balance: {resp.damper.retry_budget_balance:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
