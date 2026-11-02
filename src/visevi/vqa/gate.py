"""Closed-catalogue function gate used as the first VQA routing baseline."""
from __future__ import annotations

import re
from typing import Protocol

from ..functions.planning import RuleRouter
from ..functions.specs import FunctionSpec, load_function_specs
from .schema import RouteCandidate, RouteDecision


class FunctionGate(Protocol):
    def route(self, question: str) -> RouteDecision: ...


_ALIASES: dict[str, tuple[str, ...]] = {
    "Q4": (
        "投照", "投照方式", "胸片体位", "正位还是侧位", "pa还是ap", "ap还是pa",
        "posteroanterior", "anteroposterior", "view position", "projection",
    ),
    "P9": (
        "心胸比", "心胸比例", "ctr", "cardiothoracic ratio", "心影大小",
    ),
}


def _alias_matches(question: str, alias: str) -> bool:
    """Match Chinese phrases and English aliases without substring hits."""
    pieces: list[str] = []
    for index, char in enumerate(alias):
        if char.isspace():
            pieces.append(r"\s+")
            continue
        pieces.append(re.escape(char))
        if index + 1 < len(alias) and (not char.isascii() or not alias[index + 1].isascii()):
            pieces.append(r"\s*")
    expression = "".join(pieces)
    if alias[0].isascii() and alias[0].isalnum():
        expression = r"(?<![A-Za-z0-9_])" + expression
    if alias[-1].isascii() and alias[-1].isalnum():
        expression += r"(?![A-Za-z0-9_])"
    return re.search(expression, question, flags=re.I) is not None


class RuleFunctionGate:
    """Map an image question to one registered function or abstain.

    Alias phrases address the Q4/P9 wording used in the first VQA slice. All
    selected IDs are still checked against the versioned FunctionSpec catalog.
    Scores are omitted because this deterministic baseline is not calibrated.
    """

    version = "rules-1.0"

    def __init__(self, specs: dict[str, FunctionSpec] | None = None) -> None:
        self.specs = dict(specs) if specs is not None else load_function_specs()
        self.router = RuleRouter(self.specs)

    def route(self, question: str) -> RouteDecision:
        if not isinstance(question, str) or not question.strip():
            return RouteDecision("unknown", reason="question is empty", gate_version=self.version)

        result = self.router.route(question)
        if result.matched_by == "explicit_id":
            candidates = tuple(RouteCandidate(function_id, "explicit_id")
                               for function_id in result.function_ids)
            if result.status != "matched":
                return RouteDecision("unknown", candidates=candidates, reason=result.reason,
                                     matched_by="explicit_id", gate_version=self.version)
            if len(result.function_ids) != 1:
                return RouteDecision("ambiguous", candidates=candidates,
                                     reason="multiple explicit function IDs were supplied",
                                     matched_by="explicit_id", gate_version=self.version)
            return RouteDecision("matched", result.function_ids, candidates,
                                 result.reason, "explicit_id", self.version)

        alias_hits = [function_id for function_id, aliases in _ALIASES.items()
                      if function_id in self.specs and any(
                          _alias_matches(question, alias) for alias in aliases)]
        candidate_reasons: dict[str, str] = {function_id: "curated_alias"
                                             for function_id in alias_hits}
        for function_id in result.function_ids:
            candidate_reasons.setdefault(function_id, result.matched_by or "catalogue_keyword")
        candidates = tuple(RouteCandidate(function_id, reason)
                           for function_id, reason in candidate_reasons.items())
        if len(candidates) > 1:
            return RouteDecision("ambiguous", candidates=candidates,
                                 reason="question matches multiple function candidates",
                                 matched_by="alias+catalogue", gate_version=self.version)
        if candidates:
            function_id = candidates[0].function_id
            return RouteDecision("matched", (function_id,), candidates,
                                 "one curated or catalogue candidate matched",
                                 candidates[0].reason, self.version)
        return RouteDecision("unknown", reason=result.reason,
                             matched_by=result.matched_by, gate_version=self.version)
