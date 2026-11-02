"""Opt-in bridge from Q/P/T/S/G gates to the existing function VQA contract."""
from __future__ import annotations

from .category_gate import CategoryGate
from .gate import RuleFunctionGate
from .schema import RouteDecision
from ..functions.specs import FunctionSpec, load_function_specs


class CategoryFunctionGate:
    """Resolve functions only within the selected category.

    Category routing supports several experts. The current VQA consumer handles
    one function, so this bridge refuses multi-category plans instead of silently
    discarding experts. A-tool dependency selection remains in FunctionSpec.
    """

    def __init__(self, category_gate: CategoryGate,
                 specs: dict[str, FunctionSpec] | None = None) -> None:
        self.category_gate = category_gate
        self.specs = dict(specs) if specs is not None else load_function_specs()

    def route(self, question: str) -> RouteDecision:
        category = self.category_gate.route_categories(question)
        common = {"gate_version": category.gate_version, "category_route": category.to_dict()}
        if category.abstain:
            status = "unsupported" if category.status == "untrained" else category.status
            return RouteDecision(status, reason=category.reason, matched_by="category_gate", **common)
        if len(category.selected_categories) != 1:
            return RouteDecision("ambiguous", reason="single-function VQA cannot execute multiple category experts",
                                 matched_by="category_gate", **common)
        selected = category.selected_categories[0]
        subset = {fid: spec for fid, spec in self.specs.items() if spec.category == selected}
        result = RuleFunctionGate(subset).route(question)
        return RouteDecision(result.status, result.selected_function_ids, result.candidates,
                             result.reason, "category_gate+" + (result.matched_by or "class_rule"), **common)
