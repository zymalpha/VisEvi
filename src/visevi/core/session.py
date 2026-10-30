"""EvidenceSession: the serializable center both layers read/write.

Holds Demand, EvidenceGraph (nodes+edges), EpisodeTrace, program traces
(teacher-program supervision samples) and the system-level AnswerVerdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .units import EvidenceUnit, Demand, AnswerVerdict


@dataclass
class EvidenceGraph:
    nodes: dict[str, EvidenceUnit] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)   # {src,dst,relation,basis}

    def add(self, unit: EvidenceUnit) -> EvidenceUnit:
        if unit.id in self.nodes:
            raise ValueError(f"duplicate node id: {unit.id}")
        self.nodes[unit.id] = unit
        return unit

    def add_edge(self, src: str, dst: str, relation: str, basis: str) -> None:
        # 红线: 关系必须有依据 (basis = tool/rule/program-step id)
        self.edges.append({"src": src, "dst": dst, "relation": relation, "basis": basis})

    def by_category(self, category: str) -> list[EvidenceUnit]:
        return [u for u in self.nodes.values() if u.category == category]

    def to_dict(self) -> dict:
        return {"nodes": [u.to_dict() for u in self.nodes.values()], "edges": self.edges}


@dataclass
class EvidenceSession:
    image_ref: str
    demand: Optional[Demand] = None
    graph: EvidenceGraph = field(default_factory=EvidenceGraph)
    trace: list[dict] = field(default_factory=list)                # EpisodeTrace
    program_traces: list[dict] = field(default_factory=list)       # teacher-program GT (L2 supervision)
    verdict: Optional[AnswerVerdict] = None

    def __post_init__(self) -> None:
        # runtime-only payload caches (masks / landmarks) — never serialized
        self._masks: dict[str, object] = {}
        self._mask_metadata: dict[str, dict] = {}
        self._landmarks: dict = {}

    def attach_masks(self, masks: dict, metadata: dict | None = None) -> None:
        self._masks.update(masks)
        self._mask_metadata.update({name: (metadata or {}).get(name, {}) for name in masks})

    @property
    def masks(self) -> dict:
        return self._masks

    @property
    def mask_metadata(self) -> dict:
        return self._mask_metadata

    def mask_provenance(self, names) -> list[str]:
        """Observation ids that produced the masks currently attached under ``names``."""
        ids = (self._mask_metadata.get(name, {}).get("observation_id") for name in names)
        return list(dict.fromkeys(i for i in ids if i))

    def fov_truncated(self, names) -> dict[str, list[str]]:
        """Masks touching a model field-of-view edge that lies inside the image."""
        sides = {name: self._mask_metadata.get(name, {}).get("fov_truncated_sides") for name in names}
        return {name: value for name, value in sides.items() if value}

    def attach_landmarks(self, payload: dict) -> None:
        self._landmarks.update(payload)

    @property
    def landmarks(self) -> dict:
        return self._landmarks

    # -- graph helpers -------------------------------------------------
    def add_observation(self, unit: EvidenceUnit) -> EvidenceUnit:
        assert unit.category != "claim"
        return self.graph.add(unit)

    def add_claim(self, unit: EvidenceUnit) -> EvidenceUnit:
        assert unit.category == "claim"
        return self.graph.add(unit)

    def trace_call(self, tool: str, params: dict, latency_ms: float,
                   result: str, detail: dict) -> None:
        self.trace.append({"tool": tool, "params": params, "latency_ms": latency_ms,
                           "result": result, **({"detail": detail} if detail else {})})

    def trace_program(self, program: dict) -> None:
        self.program_traces.append(program)

    def next_id(self, prefix: str) -> str:
        n = sum(1 for nid in self.graph.nodes if nid.startswith(prefix))
        return f"{prefix}-{n + 1:04d}"

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "image_ref": self.image_ref,
            "demand": self.demand.to_dict() if self.demand else None,
            "graph": self.graph.to_dict(),
            "trace": self.trace,
            "program_traces": self.program_traces,
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceSession":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        s = cls(image_ref=d["image_ref"])
        if d.get("demand"):
            s.demand = Demand(**d["demand"])
        for n in d["graph"]["nodes"]:
            s.graph.add(EvidenceUnit(**n))
        s.graph.edges = d["graph"]["edges"]
        s.trace = d["trace"]
        s.program_traces = d["program_traces"]
        if d.get("verdict"):
            s.verdict = AnswerVerdict(**d["verdict"])
        return s
