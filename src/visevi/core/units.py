"""Core evidence contracts (v0).

Element  = Layer-1 primitive observation, the typed input algebra element
           e = (z, τ, b, u, π) — here carried by EvidenceUnit fields.
Claim    = Layer-2 composite evidence with DS [Bel, Pl] interval.
Refusal  = two levels: ToolRefusal (tool) / AnswerVerdict (system).

All dataclasses serialize via `to_dict()`; nothing here touches pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class EvidenceUnit:
    """A node of the evidence graph. category: observation | relation | claim."""
    id: str
    category: str                    # observation | relation | claim
    evidence_type: str               # τ: region|landmark|measurement|attribute|appearance|distribution|global_condition|claim
    finding_or_anatomy: str          # what this unit is about (organ / finding / proposition id)
    value: dict                      # typed payload (mask stats, ratio, mass interval, ...)
    region: Optional[dict] = None    # bbox in ORIGINAL pixel frame {x0,y0,x1,y1} (None for global elements)
    confidence: Optional[float] = None
    source_tool: str = ""            # registered tool name that produced it
    provenance: list[str] = field(default_factory=list)      # π: upstream unit ids / tool call ids
    validity_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolRefusal:
    """Tool-level refusal — a legal, structured output (D5)."""
    tool: str
    reason: str
    missing: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnswerVerdict:
    """System-level verdict: answer OR abstain, always with evidence refs."""
    answerable: bool
    answer: Optional[str] = None
    abstain_reason: Optional[str] = None
    evidence_refs: list[str] = field(default_factory=list)
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Demand:
    """What the reasoner currently needs (written by caller / demand parser)."""
    proposition: str
    requirements: list[dict] = field(default_factory=list)   # [{evidence_type, target, rationale}]
    priority: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def fov_refusal(tool: str, truncated: dict[str, list[str]]) -> ToolRefusal:
    """Measurements over a partially observed region are refused, not biased."""
    return ToolRefusal(
        tool=tool,
        reason="input mask touches the model field-of-view edge inside the image; "
               "the region may continue into unobserved pixels",
        missing=[f"fov_complete:{name}" for name in truncated],
        conditions=[f"{name} truncated at {', '.join(sides)}" for name, sides in truncated.items()])


# --- claim helpers (DS interval, see construction/ds.py) -------------------

def make_claim(unit_id: str, proposition: str, bel: float, pl: float, conflict_k: float,
               support_ids: list[str], counter_ids: list[str], rule_ids: list[str],
               validity_conditions: list[str], source_tool: str,
               extras: Optional[dict] = None) -> EvidenceUnit:
    """EvidenceClaim as an EvidenceUnit of category 'claim' (组合≠诊断: proposition
    stays at finding-evidence level; diagnosis belongs to the reasoner)."""
    return EvidenceUnit(
        id=unit_id, category="claim", evidence_type="claim",
        finding_or_anatomy=proposition,
        value={"proposition": proposition, "bel": round(bel, 4), "pl": round(pl, 4),
               "interval_width": round(pl - bel, 4), "conflict_k": round(conflict_k, 4),
               **(extras or {})},
        confidence=round(bel, 4), source_tool=source_tool,
        provenance=support_ids + counter_ids,
        validity_conditions=validity_conditions,
    )


def result_ids(units: list[EvidenceUnit]) -> list[str]:
    return [u.id for u in units]
