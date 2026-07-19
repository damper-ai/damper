# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0] - 2026-07-18

### Added

- Client-local fixed-window retry budgets
- Cumulative per-request retry cost ceilings
- Anthropic-specific error classification
- Full-jitter exponential backoff with Retry-After support
- Exclusive retry ownership for wrapped Anthropic calls
- Streaming retries only before the first content event
- Sync and async Anthropic wrappers
- Response-level Damper metadata
- OpenTelemetry request and attempt spans
- Deterministic outage demonstration
