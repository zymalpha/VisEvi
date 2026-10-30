"""Deterministic dependency planning and closed-catalogue function routing.

The planner operates on ``ToolStep`` objects without executing a tool.  Required
steps include the complete dependency closure, even when an upstream step was
declared optional.  The router accepts explicit catalogue IDs or literal
keywords; it never resolves an ambiguous natural-language request by guessing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import re
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from ..core.registry import ToolRegistry
    from .specs import FunctionSpec, ToolStep


class DependencyGraph:
    """Validate and order a function's tool steps, retaining catalogue order.

    Invalid plans raise ``ValueError`` at construction, before any tool runs.
    Both ordering methods return step objects, rather than tool names, because
    the same tool can occur multiple times with different parameters.
    """

    def __init__(self, steps: Iterable[ToolStep]) -> None:
        self.steps = tuple(steps)
        self._by_id: dict[str, ToolStep] = {}
        self._index: dict[str, int] = {}
        for index, step in enumerate(self.steps):
            if not isinstance(step.step_id, str) or not step.step_id:
                raise ValueError("step_id must be a non-empty string")
            if step.step_id in self._by_id:
                raise ValueError(f"duplicate step_id: {step.step_id}")
            if not isinstance(step.tool, str) or not step.tool:
                raise ValueError(f"step {step.step_id}: tool must be non-empty")
            self._by_id[step.step_id] = step
            self._index[step.step_id] = index
        for step in self.steps:
            dependencies = tuple(step.depends_on)
            if len(set(dependencies)) != len(dependencies):
                raise ValueError(f"step {step.step_id}: duplicate dependency")
            missing = [dependency for dependency in dependencies
                       if dependency not in self._by_id]
            if missing:
                raise ValueError(f"step {step.step_id}: missing dependency {missing}")
        self._ordered = self._sort()

    def _sort(self) -> tuple[ToolStep, ...]:
        indegree = {step.step_id: len(step.depends_on) for step in self.steps}
        outgoing: dict[str, list[str]] = {step.step_id: [] for step in self.steps}
        for step in self.steps:
            for dependency in step.depends_on:
                outgoing[dependency].append(step.step_id)
        ready = [self._index[step_id] for step_id, degree in indegree.items()
                 if degree == 0]
        heapq.heapify(ready)
        ordered = []
        while ready:
            step = self.steps[heapq.heappop(ready)]
            ordered.append(step)
            for downstream in outgoing[step.step_id]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    heapq.heappush(ready, self._index[downstream])
        if len(ordered) != len(self.steps):
            cycle_nodes = [step_id for step_id, degree in indegree.items() if degree]
            raise ValueError(f"dependency graph contains a cycle: {cycle_nodes}")
        return tuple(ordered)

    def topological_order(self) -> tuple[ToolStep, ...]:
        """Return all steps in a stable, dependency-respecting order."""
        return self._ordered

    def required_closure(self) -> tuple[ToolStep, ...]:
        """Return required steps and every upstream dependency, in order."""
        required = {step.step_id for step in self.steps if step.required}
        pending = list(required)
        while pending:
            for upstream in self._by_id[pending.pop()].depends_on:
                if upstream not in required:
                    required.add(upstream)
                    pending.append(upstream)
        return tuple(step for step in self._ordered if step.step_id in required)

    def validate_registry(self, registry: ToolRegistry,
                          *, required_only: bool = False) -> None:
        """Raise for unregistered tool names without executing any tool.

        ``required_only`` permits a caller to run a minimal plan while recording
        absent optional tools separately.  The default checks the full plan.
        """
        available = {spec["name"] for spec in registry.describe_all()}
        steps = self.required_closure() if required_only else self._ordered
        unknown = list(dict.fromkeys(step.tool for step in steps
                                     if step.tool not in available))
        if unknown:
            raise ValueError(f"unknown tool(s) in dependency graph: {', '.join(unknown)}")


@dataclass(frozen=True)
class RouteResult:
    function_ids: tuple[str, ...] = ()
    status: str = "unknown"
    reason: str = ""
    matched_by: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["function_ids"] = list(self.function_ids)
        return result


# ASCII boundaries deliberately allow Chinese text directly beside an ID.
_FUNCTION_ID = re.compile(r"(?<![A-Za-z0-9_])([QPTSG]\d+)(?![A-Za-z0-9_])", re.I)


def _keyword_matches(keyword: str, query: str) -> bool:
    if not isinstance(keyword, str) or not keyword.strip():
        return False
    keyword = keyword.strip()
    expression = re.escape(keyword)
    if re.match(r"[A-Za-z0-9_]", keyword):
        expression = r"(?<![A-Za-z0-9_])" + expression
    if re.search(r"[A-Za-z0-9_]$", keyword):
        expression += r"(?![A-Za-z0-9_])"
    return re.search(expression, query, flags=re.I) is not None


class RuleRouter:
    """Resolve requests within a versioned, closed function catalogue.

    ``resolve(function_id="P9")`` is the preferred reproducible entry point.
    ``route(query)`` first recognizes explicit IDs (possibly several), then
    literal catalogue keywords.  Multiple keyword candidates are returned with
    ``status="ambiguous"`` and must not be executed automatically.  Unknown
    explicit IDs do not silently fall back to a keyword match.
    """

    def __init__(self, specs: Mapping[str, FunctionSpec] | None = None) -> None:
        if specs is None:
            from .specs import load_function_specs
            specs = load_function_specs()
        self.specs = dict(specs)
        for function_id, spec in self.specs.items():
            if function_id != spec.function_id:
                raise ValueError(f"catalogue key does not match function_id: {function_id}")

    def _explicit(self, function_ids: Iterable[str], matched_by: str) -> RouteResult:
        ids = tuple(dict.fromkeys(function_ids))
        unknown = [function_id for function_id in ids if function_id not in self.specs]
        if unknown:
            return RouteResult(ids, "unknown", f"unknown function ID(s): {', '.join(unknown)}",
                               matched_by)
        if not ids:
            return RouteResult(reason="no function ID supplied", matched_by=matched_by)
        return RouteResult(ids, "matched", "explicit catalogue function ID(s)", matched_by)

    def resolve(self, function_id: str | Sequence[str] | None = None,
                query: str | None = None) -> RouteResult:
        if function_id is not None:
            values = (function_id,) if isinstance(function_id, str) else tuple(function_id)
            if not all(isinstance(value, str) and value.strip() for value in values):
                return RouteResult(reason="function IDs must be non-empty strings",
                                   matched_by="function_id")
            return self._explicit((value.strip().upper() for value in values), "function_id")
        return self.route(query or "")

    def route(self, query: str) -> RouteResult:
        if not isinstance(query, str) or not query.strip():
            return RouteResult(reason="request is empty")
        explicit = tuple(match.group(1).upper() for match in _FUNCTION_ID.finditer(query))
        if explicit:
            return self._explicit(explicit, "explicit_id")
        candidates = tuple(function_id for function_id, spec in self.specs.items()
                           if any(_keyword_matches(keyword, query) for keyword in spec.keywords))
        if not candidates:
            return RouteResult(reason="no catalogue ID or keyword matched", matched_by="keyword")
        if len(candidates) > 1:
            return RouteResult(candidates, "ambiguous",
                               "multiple keyword candidates; specify function IDs explicitly",
                               "keyword")
        return RouteResult(candidates, "matched", "one catalogue keyword candidate", "keyword")
