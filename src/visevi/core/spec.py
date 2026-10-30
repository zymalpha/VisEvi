"""ToolSpec: vendor-neutral self-description (protocol `describe()`)."""
from __future__ import annotations

from dataclasses import dataclass, asdict


def parameter_schema(properties: dict | None = None, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


def evidence_result_schema() -> dict:
    """Serialized run result: evidence units or an explicit tool refusal."""
    return {"oneOf": [
        {"type": "array", "items": {"type": "object",
            "required": ["id", "category", "evidence_type", "finding_or_anatomy", "value"],
            "properties": {"id": {"type": "string"}, "category": {"type": "string"},
                "evidence_type": {"type": "string"}, "finding_or_anatomy": {"type": "string"},
                "value": {"type": "object"}}}},
        {"type": "object", "required": ["tool", "reason"],
         "properties": {"tool": {"type": "string"}, "reason": {"type": "string"},
             "missing": {"type": "array", "items": {"type": "string"}},
             "conditions": {"type": "array", "items": {"type": "string"}}}}]}


@dataclass
class ToolSpec:
    name: str
    category: str          # "perception" | "construction"
    description: str
    input_schema: dict     # JSON-Schema-ish parameter description
    output_schema: dict    # JSON-Schema-ish result description

    def to_dict(self) -> dict:
        return asdict(self)
