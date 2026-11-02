"""Typed request, routing, and response contracts for single-image VQA."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class VQARequest:
    image_ref: str
    question: str
    dataset_id: str | None = None
    sample_id: str | None = None
    slots: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image_ref, str) or not self.image_ref.strip():
            raise ValueError("image_ref must be a non-empty string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(self.slots, Mapping):
            raise TypeError("slots must be a mapping")
        if not all(isinstance(name, str) and name.strip() for name in self.slots):
            raise ValueError("slot names must be non-empty strings")
        try:
            slots = json.loads(json.dumps(dict(self.slots), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("slots must contain finite JSON values") from exc
        object.__setattr__(self, "slots", slots)
        for name, value in (("dataset_id", self.dataset_id), ("sample_id", self.sample_id)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")


@dataclass(frozen=True)
class RouteCandidate:
    function_id: str
    reason: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteDecision:
    status: str
    selected_function_ids: tuple[str, ...] = ()
    candidates: tuple[RouteCandidate, ...] = ()
    reason: str = ""
    matched_by: str | None = None
    gate_version: str = "rules-1.0"
    category_route: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"matched", "ambiguous", "unknown", "unsupported", "invalid_input"}:
            raise ValueError(f"invalid route status: {self.status}")
        if self.status == "matched" and len(self.selected_function_ids) != 1:
            raise ValueError("a matched VQA route must select exactly one function")
        if self.status != "matched" and self.selected_function_ids:
            raise ValueError("only a matched route may select a function")

    @property
    def abstain(self) -> bool:
        return self.status != "matched"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "selected_function_ids": list(self.selected_function_ids),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason": self.reason,
            "matched_by": self.matched_by,
            "gate_version": self.gate_version,
            "abstain": self.abstain,
        }
        if self.category_route is not None:
            result["category_route"] = self.category_route
        return result


@dataclass(frozen=True)
class VQAResponse:
    status: str
    answer: str | None = None
    citations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"answered", "abstained"}:
            raise ValueError(f"invalid response status: {self.status}")
        if self.status == "answered" and (not self.answer or self.refusal_reason):
            raise ValueError("an answered response needs answer text and no refusal reason")
        if self.status == "abstained" and not self.refusal_reason:
            raise ValueError("an abstained response needs a refusal reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "citations": list(self.citations),
            "limitations": list(self.limitations),
            "refusal_reason": self.refusal_reason,
        }
