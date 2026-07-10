"""Retry loop and decision state machine.

This module will host the executor that drives a single logical request through
the retry pipeline: classification, budget accounting, cost ceiling checks,
backoff, and telemetry span lifecycle.

The core retry-decision logic here is reliability-critical and must be
hand-reviewed by the maintainer before release. SESSION 1 leaves this module
intentionally empty. Executor behavior lands in SESSION 5 per
``.local/PLAYBOOK.md``.
"""

from __future__ import annotations
