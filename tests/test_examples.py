"""Example smoke tests.

The deterministic outage demo (``examples/02_outage_demo.py --fake``) is the CI
gate: it must exit ``0`` and exits non-zero if any amplification invariant fails.
The other examples are run in their offline / no-key paths.

Examples run as subprocesses (never imported), so this exercises the real
``python examples/...`` entry points. No network access and no API key: the
subprocess environment has ``ANTHROPIC_API_KEY`` removed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run_example(*args: str) -> subprocess.CompletedProcess[str]:
    import os

    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # force offline / no-key paths
    return subprocess.run(
        [sys.executable, str(_EXAMPLES / args[0]), *args[1:]],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_outage_demo_fake_passes_and_shows_bounded_amplification() -> None:
    result = _run_example("02_outage_demo.py", "--fake")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Naive per-request retries:" in result.stdout
    assert "Damper:" in result.stdout
    assert "PASS:" in result.stdout
    # Deterministic fake: exactly 3.0x naive, and Damper within the 1.1x bound.
    assert "provider attempts:       3000" in result.stdout
    assert "amplification:           3.00x" in result.stdout
    assert "1.01x" in result.stdout


def test_outage_demo_defaults_to_fake_mode() -> None:
    # No flag behaves as --fake and still passes its invariants.
    result = _run_example("02_outage_demo.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_basic_example_exits_cleanly_without_api_key() -> None:
    result = _run_example("01_basic.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ANTHROPIC_API_KEY" in result.stdout


def test_telemetry_example_runs_offline() -> None:
    # opentelemetry-sdk is present via the dev extra, so spans are exported to
    # the console; the run must complete cleanly with no network.
    result = _run_example("03_telemetry.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "damper.request" in result.stdout
