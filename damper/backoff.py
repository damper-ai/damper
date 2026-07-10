"""Exponential backoff with full jitter and retry-after handling.

This module will host the backoff computation described in SPEC.md section 16:
exponential growth with full jitter, capped at ``backoff_max``, and yielding to
provider ``retry-after`` when present and enabled.

SESSION 1 leaves this module intentionally empty. Backoff logic lands in
SESSION 3 per ``.local/PLAYBOOK.md``.
"""

from __future__ import annotations
