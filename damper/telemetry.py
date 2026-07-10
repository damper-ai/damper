"""OpenTelemetry spans for Damper requests and attempts.

This module will host span creation for ``damper.request`` and
``damper.attempt`` per SPEC.md section 18. Telemetry depends only on
``opentelemetry-api`` and no-ops safely when no SDK/exporter is configured.

SESSION 1 leaves this module intentionally empty. Telemetry lands in SESSION 7
per ``.local/PLAYBOOK.md``.
"""

from __future__ import annotations
