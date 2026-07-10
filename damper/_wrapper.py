"""Client proxy for Anthropic sync/async clients.

This module will host the ``resilient()`` proxy implementation that intercepts
``messages.create`` and ``messages.stream`` on both :class:`anthropic.Anthropic`
and :class:`anthropic.AsyncAnthropic`, disables SDK-level retries for those
calls, attaches Damper response metadata, and delegates the retry loop to the
executor.

SESSION 1 leaves this module intentionally empty. Wrapper behavior lands in
SESSION 6 per ``.local/PLAYBOOK.md``.
"""

from __future__ import annotations
