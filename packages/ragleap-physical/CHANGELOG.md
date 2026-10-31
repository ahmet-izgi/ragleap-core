# Changelog

All notable changes are listed here. Format: Keep a Changelog.

## [Unreleased]

### Added
- Observation and proposal formats v0, capability levels, safety envelope, bridge with approvals,
  dry run, outcome check and stop.
- Built-in deterministic one-joint adapter for tests.
- Four tools (`PhysicalTools`): list sources, get observation, get envelope, propose action. None can approve, apply,
  stop or change limits.
- Small stdlib MCP server (JSON-RPC over HTTP): loopback by default, Host and Origin checks, size cap, optional bearer token.
- Interop test against the official MCP Python SDK (optional extra `interop`, `mcp>=1.28,<2`).
