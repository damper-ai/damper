"""Provider error classification.

This module will host the data-driven classification table that maps provider
errors and low-level transport failures to :class:`ErrorClass` values per
SPEC.md section 14. SESSION 1 leaves the classifier itself unimplemented;
classification lands in SESSION 3 per ``.local/PLAYBOOK.md``.

The :data:`ErrorClassifier` alias exported here is a minimal placeholder used
only so that :class:`damper.Policy` can accept a user-supplied classifier hook
without importing runtime logic in this session.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ErrorClassifier = Callable[[BaseException], Any]
"""Placeholder callable type for a user-supplied error classifier.

The full :class:`ErrorClass` enum and default classification table will be
defined in SESSION 3. This alias exists in SESSION 1 only to satisfy the type
annotation of :attr:`damper.Policy.classifier`.
"""
