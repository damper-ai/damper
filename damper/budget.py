"""Client-local retry budget (token bucket).

This module will host the per-client-instance token bucket that bounds retries
to a configurable fraction of recent successful traffic, per SPEC.md section
12.

Budget accounting is reliability-critical and must be hand-reviewed by the
maintainer before release. SESSION 1 leaves this module intentionally empty.
Budget behavior lands in SESSION 2 per ``.local/PLAYBOOK.md``.
"""

from __future__ import annotations
