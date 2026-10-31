"""Model-facing tools (transport-agnostic; an MCP server wraps these in Phase 2).

Exactly four tools. None of them can approve, apply, stop or change the envelope, so a model that
uses them can at most read state and ask for a proposal. Arguments are checked strictly: unknown keys,
wrong types and over-long text are refused. Every result is marked untrusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .bridge import Bridge
from .capability import Capability
from .errors import PhysicalError
from .proposal import MAX_REASON, ExpectedOutcome

_OUTCOME_PROPS = {"entity": "string", "quantity": "string", "lo": "number", "hi": "number", "unit": "string",
                  "by_sim_time_s": "number"}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    level: Capability
    properties: Mapping[str, Any]
    required: tuple
    handler: Callable[[Mapping[str, Any]], dict]

    def schema(self) -> dict:
        """MCP-style tool description."""
        return {"name": self.name, "description": self.description,
                "inputSchema": {"type": "object", "properties": dict(self.properties),
                                "required": list(self.required), "additionalProperties": False}}


def _is_type(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    return False


class PhysicalTools:
    """Wraps one bridge. ``proposer`` is fixed here: the caller cannot choose who is asking."""

    def __init__(self, bridge: Bridge, *, proposer: str = "employee:agent") -> None:
        self._b, self._proposer = bridge, proposer
        self._tools = {t.name: t for t in (
            Tool("physical_list_sources", "Say what this bridge offers (read only).", Capability.OBSERVE, {}, (),
                 self._sources),
            Tool("physical_get_observation", "Read the current simulated state (read only; result is untrusted data).",
                 Capability.OBSERVE, {}, (), self._observation),
            Tool("physical_get_envelope", "View the safety limits (read only; you cannot change them).",
                 Capability.OBSERVE, {}, (), self._envelope),
            Tool("physical_propose_action",
                 "Ask for a proposal. It is only checked and described. Nothing moves until a person approves.",
                 Capability.RECOMMEND,
                 {"target": {"type": "string"}, "command": {"type": "string"}, "value": {"type": "number"},
                  "unit": {"type": "string"}, "reason": {"type": "string", "maxLength": MAX_REASON},
                  "expected_outcome": {"type": "object", "properties": {k: {"type": v} for k, v in _OUTCOME_PROPS.items()},
                                       "required": list(_OUTCOME_PROPS), "additionalProperties": False}},
                 ("target", "command", "value", "unit", "expected_outcome"), self._propose),
        ) if t.level <= bridge.max_capability}

    def list_tools(self) -> list:
        """Tools this bridge's capability level allows."""
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: Any = None) -> dict:
        """Run one tool. Always returns a JSON-safe dict; never raises for bad input."""
        try:
            tool = self._tools.get(name) if isinstance(name, str) else None
            if tool is None:
                return _err("unknown_tool", "no such tool at this capability level")
            args = {} if arguments is None else arguments
            _check_args(args, tool.properties, tool.required)
            data = tool.handler(args)
            out = {"ok": True, "trust": "untrusted", "data": data}
            json.dumps(out, allow_nan=False)
            return out
        except PhysicalError as exc:
            return _err(getattr(exc, "code", None) or _CODES.get(type(exc).__name__, "error"), str(exc))
        except _BadArgs as exc:
            return _err("bad_arguments", str(exc))
        except Exception:  # never leak internals to the model
            return _err("internal_error", "the tool failed; see the operator log")

    # ---- handlers
    def _sources(self, _: Mapping[str, Any]) -> dict:
        return {"org_id": self._b.org_id, "simulated": True, "max_capability": self._b.max_capability.name.lower(),
                "tools": sorted(self._tools)}

    def _observation(self, _: Mapping[str, Any]) -> dict:
        return self._b.observe().to_dict()

    def _envelope(self, _: Mapping[str, Any]) -> dict:
        return self._b.envelope_view()

    def _propose(self, a: Mapping[str, Any]) -> dict:
        o = a["expected_outcome"]
        outcome = ExpectedOutcome(o["entity"], o["quantity"], o["lo"], o["hi"], o["unit"], o["by_sim_time_s"])
        p = self._b.propose(requested_by=self._proposer, target=a["target"], command=a["command"],
                            value=a["value"], unit=a["unit"], expected_outcome=outcome, reason=a.get("reason", ""))
        return {"proposal": p.to_dict(), "digest": p.digest(), "status": "validated",
                "next": "A person must approve this exact proposal before anything happens."}


_CODES = {"ValidationError": "invalid_value", "ApprovalError": "approval", "CapabilityError": "capability",
          "SafetyStop": "stopped", "ConfigError": "config"}


class _BadArgs(Exception):
    pass


def _err(code: str, message: str) -> dict:
    return {"ok": False, "trust": "untrusted", "error": {"code": code, "message": message}}


def _check_args(args: Any, props: Mapping[str, Any], required: tuple) -> None:
    """Strict check against the tool's own schema: no unknown keys, right types, text length."""
    if not isinstance(args, dict):
        raise _BadArgs("arguments must be an object")
    extra = set(args) - set(props)
    if extra:
        raise _BadArgs("unknown arguments are not allowed")
    for key in required:
        if key not in args:
            raise _BadArgs(f"missing argument: {key}")
    for key, spec in props.items():
        if key not in args:
            continue
        if not _is_type(args[key], spec["type"]):
            raise _BadArgs(f"argument {key} has the wrong type")
        if spec["type"] == "string" and len(args[key]) > spec.get("maxLength", 200):
            raise _BadArgs(f"argument {key} is too long")
        if spec["type"] == "object":
            _check_args(args[key], spec["properties"], tuple(spec.get("required", ())))
