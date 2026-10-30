"""ToolRegistry: ONE protocol, TWO kinds of members (perception | construction).

Same registry / same describe()/run() for Layer-1 tools and Layer-2 operators (D2).
Every run() is traced into the session's EpisodeTrace (tool, params, latency, ids).
The registry never interprets results — composition semantics live in the tools.
"""
from __future__ import annotations

import time
import math
from numbers import Integral, Real
from typing import Any, Callable

from .spec import ToolSpec
from .session import EvidenceSession
from .units import EvidenceUnit, ToolRefusal


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}

    def register(self, tool: Any) -> None:
        spec = tool.describe()
        if not isinstance(spec, ToolSpec):
            raise TypeError(f"{tool!r}.describe() must return ToolSpec")
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = tool

    def describe_all(self) -> list[dict]:
        """The single menu any caller (MLLM / rule / human) sees."""
        return [self._tools[n].describe().to_dict() for n in sorted(self._tools)]

    def run(self, name: str, session: EvidenceSession, **params) -> list[EvidenceUnit] | ToolRefusal:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        t0 = time.perf_counter()
        try:
            out = self._tools[name].run(session=session, **params)
        except Exception as exc:  # the protocol returns units or a refusal, never raises
            out = ToolRefusal(tool=name, reason=f"{type(exc).__name__}: {exc}",
                              conditions=["unexpected tool exception; see trace result 'error'"])
            session.trace_call(tool=name, params=_digest(params),
                               latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                               result="error", detail=out.to_dict())
            return out
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if isinstance(out, ToolRefusal):
            session.trace_call(tool=name, params=_digest(params), latency_ms=latency_ms,
                               result="refusal", detail=out.to_dict())
        else:
            session.trace_call(tool=name, params=_digest(params), latency_ms=latency_ms,
                               result="ok", detail={"ids": [u.id for u in out]})
        return out


def _digest(params: dict) -> dict:
    """Preserve small tool parameters so coordinate/ROI calls remain reproducible."""
    def summarize(value):
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value) if math.isfinite(value) else str(value)
        if isinstance(value, (tuple, list)):
            if len(value) <= 32:
                return [summarize(item) for item in value]
            return {'length': len(value), 'preview': [summarize(item) for item in value[:32]]}
        if isinstance(value, dict):
            items = list(value.items())
            result = {str(key): summarize(item) for key, item in items[:32]}
            if len(items) > 32:
                result['_truncated_length'] = len(items)
            return result
        return str(value)[:512]
    return {key: summarize(value) for key, value in params.items()}
