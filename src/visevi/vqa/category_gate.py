"""Category selection contracts, independent of PyTorch and training objectives.

Scores select capabilities in Q/P/T/S/G; they are not disease probabilities or
weights for mixing heterogeneous tool evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Mapping, Protocol


CATEGORY_IDS = ("Q", "P", "T", "S", "G")


def _scores(values: Mapping[str, float], label: str, *, bounded: bool) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(CATEGORY_IDS):
        raise ValueError(f"{label} must contain exactly Q, P, T, S and G")
    result = {}
    for category in CATEGORY_IDS:
        value = values[category]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{label}.{category} must be finite")
        if bounded and not 0 <= value <= 1:
            raise ValueError(f"{label}.{category} must be in [0, 1]")
        result[category] = float(value)
    return result


@dataclass(frozen=True)
class CategoryRouteDecision:
    status: str
    selected_categories: tuple[str, ...] = ()
    activation_scores: Mapping[str, float] = field(default_factory=dict)
    logits: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    gate_version: str = "five-category-gates-1.0"
    model_id: str | None = None
    selection_thresholds: Mapping[str, float] = field(default_factory=dict)
    max_categories: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {"matched", "unknown", "ambiguous", "untrained", "invalid_input"}:
            raise ValueError("invalid category route status")
        if isinstance(self.selected_categories, str):
            raise ValueError("selected_categories must be a sequence of category IDs")
        categories = tuple(self.selected_categories)
        if len(set(categories)) != len(categories) or any(c not in CATEGORY_IDS for c in categories):
            raise ValueError("selected_categories contains unknown or duplicate categories")
        if (self.status == "matched") != bool(categories):
            raise ValueError("only matched category routes may select one or more categories")
        object.__setattr__(self, "selected_categories", categories)
        if self.activation_scores:
            object.__setattr__(self, "activation_scores", _scores(
                self.activation_scores, "activation_scores", bounded=True))
        if self.logits:
            object.__setattr__(self, "logits", _scores(self.logits, "logits", bounded=False))
        if self.selection_thresholds:
            object.__setattr__(self, "selection_thresholds", _scores(
                self.selection_thresholds, "selection_thresholds", bounded=True))
        if self.max_categories is not None and (
                type(self.max_categories) is not int or not 1 <= self.max_categories <= len(CATEGORY_IDS)):
            raise ValueError("max_categories must be None or an integer from 1 to 5")

    @property
    def abstain(self) -> bool:
        return self.status != "matched"

    def to_dict(self) -> dict:
        return {"status": self.status, "selected_categories": list(self.selected_categories),
                "activation_scores": dict(self.activation_scores), "logits": dict(self.logits),
                "score_calibrated": False, "reason": self.reason,
                "gate_version": self.gate_version, "model_id": self.model_id,
                "selection_policy": {"thresholds": dict(self.selection_thresholds),
                                     "max_categories": self.max_categories},
                "abstain": self.abstain}


class CategoryGate(Protocol):
    def route_categories(self, question: str) -> CategoryRouteDecision: ...


@dataclass(frozen=True)
class CategorySelectionPolicy:
    """Independent thresholds; never force an argmax or silently drop a category.

    Thresholds are deployment configuration, not clinically calibrated defaults.
    A category budget, if set, refuses an over-budget selection instead of
    truncating it. No availability mask may change the user's intended task.
    """

    thresholds: Mapping[str, float] = field(default_factory=lambda: dict.fromkeys(CATEGORY_IDS, 0.5))
    max_categories: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "thresholds", _scores(self.thresholds, "thresholds", bounded=True))
        if self.max_categories is not None and (
                type(self.max_categories) is not int or not 1 <= self.max_categories <= len(CATEGORY_IDS)):
            raise ValueError("max_categories must be None or an integer from 1 to 5")

    def select(self, activation_scores: Mapping[str, float], *,
               logits: Mapping[str, float] | None = None,
               model_id: str | None = None) -> CategoryRouteDecision:
        scores = _scores(activation_scores, "activation_scores", bounded=True)
        selected = tuple(c for c in CATEGORY_IDS if scores[c] >= self.thresholds[c])
        status, reason = "matched", "categories met their independent activation thresholds"
        if not selected:
            status, reason = "unknown", "no category met its activation threshold"
        elif self.max_categories is not None and len(selected) > self.max_categories:
            status, reason = "ambiguous", "selected categories exceed the configured budget; none were dropped"
            selected = ()
        return CategoryRouteDecision(status, selected, scores, logits or {}, reason,
                                     model_id=model_id, selection_thresholds=self.thresholds,
                                     max_categories=self.max_categories)
