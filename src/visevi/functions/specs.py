"""Versioned function protocols, separate from A-tool implementations.

A specification describes a research task and its evidence dependencies.  Its
presence does not mean that a clinical interpretation, executor or independent
evaluation has been completed.  JSON protocols ship with the Python package.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping


FUNCTION_IDS = tuple(
    f"{category}{number}" for category, count in (("Q", 4), ("P", 12), ("T", 4), ("S", 3), ("G", 3))
    for number in range(1, count + 1)
)
SPEC_DIRECTORY = Path(__file__).with_name("specs")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings")
    result = tuple(_text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    # Reject non-JSON/NaN parameter payloads at the protocol boundary.
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc


@dataclass(frozen=True)
class ToolStep:
    step_id: str
    tool: str
    depends_on: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    required: bool = True
    input_role: str = "current"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", _text(self.step_id, "step_id")):
            raise ValueError("step_id must be a lower-case identifier")
        _text(self.tool, "tool")
        object.__setattr__(self, "depends_on", _strings(self.depends_on, "depends_on"))
        object.__setattr__(self, "parameters", _mapping(self.parameters, "parameters"))
        if type(self.required) is not bool:
            raise ValueError("required must be a boolean")
        if self.input_role not in {"current", "prior"}:
            raise ValueError("input_role must be 'current' or 'prior'")
        if self.step_id in self.depends_on:
            raise ValueError("a tool step cannot depend on itself")

    def to_dict(self) -> dict:
        return {**asdict(self), "depends_on": list(self.depends_on)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolStep":
        try:
            return cls(**dict(data))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid tool step: {exc}") from exc


@dataclass(frozen=True)
class FunctionSpec:
    function_id: str
    name: str
    category: str
    proposition: str
    input_type: str
    steps: tuple[ToolStep, ...]
    outputs: tuple[str, ...]
    refusal_rules: tuple[str, ...]
    evaluation_protocol: dict[str, Any]
    status: str = "specified"
    blockers: tuple[str, ...] = ()
    version: str = "1.0"
    keywords: tuple[str, ...] = ()
    slots: tuple[str, ...] = ()
    guards: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.function_id not in FUNCTION_IDS:
            raise ValueError(f"unknown function_id: {self.function_id!r}")
        if self.category != self.function_id[0]:
            raise ValueError("category must match function_id")
        for name in ("name", "proposition", "version"):
            _text(getattr(self, name), name)
        if self.input_type not in {"single_image", "paired_images", "image_sequence"}:
            raise ValueError("unsupported input_type")
        if self.status not in {"specified", "experimental", "validated"}:
            raise ValueError("unsupported specification status")
        if not isinstance(self.steps, (tuple, list)) or not self.steps:
            raise ValueError("steps must be a non-empty list of ToolStep")
        if not all(isinstance(step, ToolStep) for step in self.steps):
            raise ValueError("steps must contain ToolStep objects")
        object.__setattr__(self, "steps", tuple(self.steps))
        ids = [step.step_id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate step_id")
        for step in self.steps:
            unknown = set(step.depends_on) - set(ids)
            if unknown:
                raise ValueError(f"{step.step_id}: unknown dependencies: {sorted(unknown)}")
        if self.input_type == "single_image" and any(step.input_role != "current" for step in self.steps):
            raise ValueError("single-image functions can only use current inputs")
        if self.input_type == "paired_images" and {step.input_role for step in self.steps} != {"prior", "current"}:
            raise ValueError("paired-image functions must declare prior and current steps")
        # An invalid dependency cycle must fail during load, before tool calls.
        pending = {step.step_id: set(step.depends_on) for step in self.steps}
        while pending:
            ready = [key for key, dependencies in pending.items() if not dependencies]
            if not ready:
                raise ValueError("function dependency graph contains a cycle")
            for key in ready:
                del pending[key]
            for dependencies in pending.values():
                dependencies.difference_update(ready)
        for name in ("outputs", "refusal_rules", "blockers", "keywords", "slots", "guards"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if not self.outputs or not self.refusal_rules or not self.slots or not self.guards:
            raise ValueError("outputs, refusal_rules, slots and guards must be non-empty")
        protocol = _mapping(self.evaluation_protocol, "evaluation_protocol")
        for field_name in ("reference", "primary_metrics", "secondary_metrics", "split", "statistics", "status"):
            if field_name not in protocol or not protocol[field_name]:
                raise ValueError(f"evaluation_protocol missing {field_name}")
        object.__setattr__(self, "evaluation_protocol", protocol)

    @property
    def required_tools(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(step.tool for step in self.steps if step.required))

    @property
    def optional_tools(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(step.tool for step in self.steps if not step.required))

    @property
    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        return {step.step_id: step.depends_on for step in self.steps}

    def to_dict(self) -> dict:
        result = asdict(self)
        result["steps"] = [step.to_dict() for step in self.steps]
        for name in ("outputs", "refusal_rules", "blockers", "keywords", "slots", "guards"):
            result[name] = list(getattr(self, name))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FunctionSpec":
        try:
            fields = dict(data)
            fields["steps"] = tuple(ToolStep.from_dict(step) for step in fields["steps"])
            return cls(**fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid FunctionSpec: {exc}") from exc


def load_function_specs(directory: str | Path | None = None) -> dict[str, FunctionSpec]:
    """Load a complete 26-task catalog, rejecting missing/duplicate IDs.

    No global mutable cache is used: editing or validating an independent copy
    of a protocol cannot silently mutate another caller's specifications.
    """
    root = Path(directory) if directory is not None else SPEC_DIRECTORY
    specs: dict[str, FunctionSpec] = {}
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            spec = FunctionSpec.from_dict(data)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid specification {path}: {exc}") from exc
        if spec.function_id in specs:
            raise ValueError(f"duplicate function_id: {spec.function_id}")
        if path.stem != spec.function_id:
            raise ValueError(f"filename {path.name} does not match {spec.function_id}")
        specs[spec.function_id] = spec
    missing = set(FUNCTION_IDS) - set(specs)
    if missing:
        raise ValueError(f"function catalog missing specifications: {sorted(missing)}")
    return {function_id: specs[function_id] for function_id in FUNCTION_IDS}


def get_function_spec(function_id: str, directory: str | Path | None = None) -> FunctionSpec:
    """Return one protocol by explicit ID; unknown IDs raise ``KeyError``."""
    if not isinstance(function_id, str) or function_id not in FUNCTION_IDS:
        raise KeyError(f"unknown function_id: {function_id!r}")
    return load_function_specs(directory)[function_id]
