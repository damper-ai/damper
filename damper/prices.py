"""Anthropic model price table.

This module will host the default price table used for retry cost estimation
per SPEC.md section 13. Each entry carries input and output prices per million
tokens along with a ``last_verified`` date. Users can override the table via
:attr:`damper.Policy.price_table`.

Prices are a safety guard for the retry cost ceiling, not a billing system.

SESSION 1 leaves the table unpopulated. Price entries and estimation helpers
land in SESSION 4 per ``.local/PLAYBOOK.md``.

The :class:`ModelPrice` type exported here is a minimal placeholder used only
so that :class:`damper.Policy` can accept a price table without importing
runtime logic in this session.
"""

from __future__ import annotations


class ModelPrice:
    """Placeholder for a model price entry.

    The final field set (model name, input/output price per million tokens,
    ``last_verified`` date) is defined in SESSION 4. This class exists in
    SESSION 1 only to satisfy the type annotation of
    :attr:`damper.Policy.price_table`.
    """
