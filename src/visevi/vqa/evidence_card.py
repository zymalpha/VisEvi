"""Deterministic, lossless rendering of function evidence for a VQA consumer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from ..functions.runner import FunctionEpisodeResult, FunctionRun


def _json_value(value: Any) -> Any:
    """Convert common numeric/container values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    raise TypeError(f"EvidenceCard contains non-JSON value {type(value).__name__}")


@dataclass(frozen=True)
class EvidenceCard:
    card_version: str
    function_id: str
    question: str
    status: str
    route: dict[str, Any]
    evidence_nodes: tuple[dict[str, Any], ...]
    claim_ids: tuple[str, ...]
    function_output: dict[str, Any]
    tool_steps: tuple[dict[str, Any], ...]
    refusal: dict[str, Any] | None
    provenance: dict[str, Any]

    @classmethod
    def from_episode(cls, question: str, episode: FunctionEpisodeResult,
                     run: FunctionRun) -> "EvidenceCard":
        graph_nodes = episode.session.graph.nodes
        node_ids = set(run.evidence_refs.observation_ids)
        node_ids.update(run.evidence_refs.relation_ids)
        node_ids.update(run.evidence_refs.claim_ids)
        pending = list(node_ids)
        while pending:
            current = pending.pop()
            node = graph_nodes.get(current)
            if node is None:
                raise ValueError(f"function references missing evidence node: {current}")
            for upstream in node.provenance:
                if upstream in graph_nodes and upstream not in node_ids:
                    node_ids.add(upstream)
                    pending.append(upstream)
        ordered_nodes = tuple(_json_value(node.to_dict()) for node_id, node in graph_nodes.items()
                              if node_id in node_ids)
        image_binding = episode.session.__dict__.get("_function_image_binding")
        image_sha256 = image_binding[1] if image_binding else None
        traces = [trace for trace in episode.session.program_traces
                  if trace.get("program_id") == "run_function_episode"]
        spec_hash = traces[-1].get("spec_hashes", {}).get(run.function_id) if traces else None
        refusal = run.refusal.to_dict() if run.refusal else None
        status = "answerable" if run.status == "ok" and run.evidence_refs.claim_ids else "refused"
        return cls(
            card_version="1.0",
            function_id=run.function_id,
            question=question,
            status=status,
            route=_json_value(episode.route.to_dict()),
            evidence_nodes=ordered_nodes,
            claim_ids=tuple(run.evidence_refs.claim_ids),
            function_output=_json_value(run.outputs),
            tool_steps=tuple(_json_value(run.steps)),
            refusal=_json_value(refusal) if refusal else None,
            provenance={"image_sha256": image_sha256, "spec_hash": spec_hash,
                        "function_version": run.version},
        )

    @classmethod
    def routing_refusal(cls, question: str, route: dict[str, Any]) -> "EvidenceCard":
        return cls("1.0", "unselected", question, "refused", route, (), (), {}, (),
                   {"reason": route.get("reason", "function routing did not produce one candidate")},
                   {"image_sha256": None, "spec_hash": None, "function_version": None})

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "card_version": self.card_version,
            "function_id": self.function_id,
            "question": self.question,
            "status": self.status,
            "route": self.route,
            "evidence_nodes": list(self.evidence_nodes),
            "claim_ids": list(self.claim_ids),
            "function_output": self.function_output,
            "tool_steps": list(self.tool_steps),
            "refusal": self.refusal,
            "provenance": self.provenance,
        }
        return _json_value(payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                           indent=2, allow_nan=False)

    def to_text(self) -> str:
        return "MedVisEvi evidence card (JSON; cite claim IDs only):\n```json\n" + self.to_json() + "\n```"

    @property
    def claims(self) -> dict[str, dict[str, Any]]:
        return {node["id"]: node for node in self.evidence_nodes
                if node.get("category") == "claim" and node.get("id") in self.claim_ids}


def image_sha256(card: EvidenceCard) -> str | None:
    value = card.provenance.get("image_sha256")
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("invalid image SHA-256 in evidence provenance")
    int(value, 16)
    return value
