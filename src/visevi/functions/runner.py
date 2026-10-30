"""Replayable function episodes over the existing A-tool registry.

The catalogue and execution coverage are deliberately separate.  All functions
can be planned; only programs with an explicit claim rule can be executed.
No clinical thresholds, calibrated probabilities or fusion weights are invented
here.  Q4, P9 and G1 are the first executable integration programs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..construction.evidence import (EvidenceRefs, add_claim, add_observation,
                                     add_relation, validate_graph)
from ..core.registry import ToolRegistry
from ..core.session import EvidenceSession
from ..core.units import AnswerVerdict, Demand, EvidenceUnit, ToolRefusal
from .planning import DependencyGraph, RouteResult, RuleRouter
from .specs import FunctionSpec, ToolStep, load_function_specs


IMPLEMENTED_FUNCTIONS = ("Q4", "P9", "G1")


@dataclass
class FunctionRun:
    function_id: str
    version: str
    status: str = "blocked"
    evidence_refs: EvidenceRefs = field(default_factory=EvidenceRefs)
    outputs: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)
    refusal: ToolRefusal | None = None
    missing_optional: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"function_id": self.function_id, "version": self.version,
                "status": self.status, "evidence_refs": self.evidence_refs.to_dict(),
                "outputs": self.outputs, "steps": self.steps,
                "refusal": self.refusal.to_dict() if self.refusal else None,
                "missing_optional": self.missing_optional}


@dataclass
class FunctionEpisodeResult:
    status: str
    session: EvidenceSession
    route: RouteResult
    functions: list[FunctionRun] = field(default_factory=list)

    @property
    def function_ids(self) -> tuple[str, ...]:
        return self.route.function_ids

    @property
    def function_id(self) -> str | None:
        return self.function_ids[0] if len(self.function_ids) == 1 else None

    @property
    def executed_tools(self) -> list[str]:
        return [step["tool"] for run in self.functions for step in run.steps
                if step["status"] in {"ok", "refusal", "invalid_output"}]

    @property
    def reused_tools(self) -> list[str]:
        return [step["tool"] for run in self.functions for step in run.steps
                if step["status"] == "reused"]

    @property
    def evidence_refs(self) -> list[str]:
        return [node_id for run in self.functions for node_id in run.evidence_refs.claim_ids]

    def to_dict(self) -> dict:
        return {"status": self.status, "route": self.route.to_dict(),
                "functions": [run.to_dict() for run in self.functions],
                "executed_tools": self.executed_tools, "reused_tools": self.reused_tools,
                "evidence_refs": self.evidence_refs, "session": self.session.to_dict()}


def _json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _resolve(value: Any, inputs: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$input."):
        path = value[len("$input."):].split(".")
        current = inputs
        for component in path:
            if not isinstance(current, Mapping) or component not in current:
                raise ValueError(f"missing function input: {value}")
            current = current[component]
        return current
    if isinstance(value, dict):
        return {key: _resolve(item, inputs) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve(item, inputs) for item in value]
    return value


def _mask_digest(session: EvidenceSession, name: str) -> str | None:
    mask = session.masks.get(name)
    if mask is None:
        return None
    array = np.ascontiguousarray(mask)
    return _json_hash({"shape": list(array.shape), "dtype": str(array.dtype),
                       "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                       "metadata": session.mask_metadata.get(name, {})})


def _input_mask_names(tool: str, params: dict) -> list[str]:
    names = [params[key] for key in ("mask_name", "mask_a", "mask_b", "lesion_mask_name",
                                    "anatomy_mask_name", "roi_mask_name", "region_mask_name")
             if key in params]
    for key in ("region_mask_names", "lesion_mask_names"):
        names.extend(params.get(key, []))
    if tool == "measure_ctr":
        names.append(params.get("heart_mask_name", "Heart"))
    if tool in {"assign_lung_zones", "measure_lung_symmetry", "compare_lung_geometry",
                "analyze_distribution"}:
        names.extend(("Right_Lung", "Left_Lung"))
    return list(dict.fromkeys(names))


def _call_key(registry: ToolRegistry, session: EvidenceSession, tool: str,
              params: dict, image_hash: str, auto_output: bool) -> str:
    semantic_params = {key: value for key, value in params.items()
                       if not (auto_output and key == "output_dir")}
    return _json_hash({"registry_instance": id(registry), "image_sha256": image_hash,
                       "tool": tool, "parameters": semantic_params,
                       "masks": {name: _mask_digest(session, name)
                                 for name in _input_mask_names(tool, params)}})


def _cache_valid(record: dict, session: EvidenceSession) -> bool:
    ids = record["ids"]
    if any(node_id not in session.graph.nodes for node_id in ids):
        return False
    if _json_hash([session.graph.nodes[node_id].to_dict() for node_id in ids]) != record["units_hash"]:
        return False
    if any(_mask_digest(session, name) != digest for name, digest in record["masks"].items()):
        return False
    return record["landmarks"] is None or _json_hash(session.landmarks) == record["landmarks"]


def _commit_tool_outputs(session: EvidenceSession, tool: str, output: Any) -> list[EvidenceUnit]:
    if not isinstance(output, list) or not output:
        raise ValueError("A tool must return non-empty observations or ToolRefusal")
    ids = []
    for unit in output:
        if not isinstance(unit, EvidenceUnit) or unit.category != "observation":
            raise ValueError("A tool returned a non-observation node")
        if not unit.id or unit.id in ids or unit.source_tool != tool:
            raise ValueError("A tool returned duplicate/empty IDs or an incorrect source_tool")
        _json_hash(unit.to_dict())
        ids.append(unit.id)
        existing = session.graph.nodes.get(unit.id)
        if existing is not None and existing.to_dict() != unit.to_dict():
            raise ValueError("A tool reused a node ID with different contents")
    for unit in output:
        if unit.id not in session.graph.nodes:
            session.add_observation(unit)
        for upstream in unit.provenance:
            if upstream in session.graph.nodes:
                edge = {"src": upstream, "dst": unit.id, "relation": "derived_from", "basis": tool}
                if edge not in session.graph.edges:
                    session.graph.add_edge(**edge)
    return output


def _runtime_snapshot(session: EvidenceSession) -> dict:
    """Capture mutable tool side effects while leaving the audit trace append-only."""
    masks = dict(session._masks)
    writeable = {}
    for name, mask in masks.items():
        if isinstance(mask, np.ndarray):
            writeable[name] = mask.flags.writeable
            mask.flags.writeable = False
    return {"nodes": dict(session.graph.nodes), "edges": list(session.graph.edges),
            "masks": masks, "mask_writeable": writeable,
            "mask_metadata": {name: dict(metadata) for name, metadata in session._mask_metadata.items()},
            "landmarks": dict(session._landmarks)}


def _release_runtime_snapshot(snapshot: dict) -> None:
    for name, writeable in snapshot["mask_writeable"].items():
        snapshot["masks"][name].flags.writeable = writeable


def _restore_runtime(session: EvidenceSession, snapshot: dict) -> None:
    session.graph.nodes = snapshot["nodes"]
    session.graph.edges = snapshot["edges"]
    session._masks = snapshot["masks"]
    session._mask_metadata = snapshot["mask_metadata"]
    session._landmarks = snapshot["landmarks"]


def _units(outputs: dict[str, list[EvidenceUnit]], tool: str) -> list[EvidenceUnit]:
    return [unit for values in outputs.values() for unit in values if unit.source_tool == tool]


def _view(outputs: dict[str, list[EvidenceUnit]]) -> EvidenceUnit:
    views = _units(outputs, "classify_view")
    if len(views) != 1 or views[0].value.get("status") != "ok":
        raise ValueError("projection evidence is missing or unparsed")
    if views[0].value.get("assumed") is not False:
        raise ValueError("projection must be observed, not assumed")
    if views[0].value.get("view") not in {"PA", "AP", "lateral"}:
        raise ValueError("projection is unknown")
    return views[0]


def _p9_guard(outputs: dict[str, list[EvidenceUnit]]) -> None:
    if _units(outputs, "classify_view") and _view(outputs).value["view"] != "PA":
        raise ValueError("P9 primary protocol requires PA projection; AP/lateral interpretation is not enabled")
    quality = _units(outputs, "assess_image_quality")
    if quality:
        by_name = {unit.finding_or_anatomy: unit for unit in quality}
        for name in ("rotation", "inspiration", "penetration", "field_of_view"):
            if name not in by_name or by_name[name].value.get("status") != "ok":
                raise ValueError(f"P9 quality evidence missing or unknown: {name}")
        fov = by_name["field_of_view"].value
        anatomy = {key: value for key, value in fov.items()
                   if isinstance(value, dict) and "segmented" in value}
        if not anatomy:
            raise ValueError("P9 field-of-view evidence has no structure coverage")
        for name, record in anatomy.items():
            if record.get("segmented") is not True or any(
                    record.get(f"touches_{side}") is not False
                    for side in ("top", "bottom", "left", "right")):
                raise ValueError(f"P9 anatomy missing or touching the image border: {name}")


def _q4_program(session: EvidenceSession, spec: FunctionSpec, outputs: dict, inputs: dict):
    view = _view(outputs)
    value = {"view": view.value["view"], "status": "observed",
             "raw_text": view.value.get("raw_text"),
             "option_raw_probabilities": view.value.get("option_raw_probabilities"),
             "probability_calibrated": False,
             "applicability": "projection evidence only; independent accuracy not accepted"}
    relation = add_relation(session, "projection", value, input_ids=[view.id],
                            rule_id=f"Q4:projection:{spec.version}")
    return value, relation, [view.id]


def _p9_program(session: EvidenceSession, spec: FunctionSpec, outputs: dict, inputs: dict):
    _p9_guard(outputs)
    view = _view(outputs)
    ctr_units = _units(outputs, "measure_ctr")
    if len(ctr_units) != 1:
        raise ValueError("P9 requires exactly one automatic CTR measurement")
    heart_units = _units(outputs, "segment_heart")
    if len(heart_units) != 1 or session.mask_metadata.get("Heart", {}).get("observation_id") != heart_units[0].id:
        raise ValueError("P9 Heart mask is not bound to this segment_heart observation")
    raw = ctr_units[0].value
    if raw.get("heart_mask_name") != "Heart" or heart_units[0].id not in ctr_units[0].provenance:
        raise ValueError("P9 CTR is not bound to the current Heart mask")
    ctr = raw.get("CTR")
    if isinstance(ctr, bool) or not isinstance(ctr, Real) or not math.isfinite(ctr) or not 0 < ctr <= 1:
        raise ValueError("P9 CTR must be finite and within (0, 1]")
    heart_width, thorax_width = raw.get("heart_width_px"), raw.get("thoracic_inner_width_px")
    if (isinstance(heart_width, bool) or not isinstance(heart_width, Real)
            or isinstance(thorax_width, bool) or not isinstance(thorax_width, Real)
            or not math.isfinite(heart_width) or not math.isfinite(thorax_width)
            or heart_width <= 0 or thorax_width <= 0):
        raise ValueError("P9 heart and thoracic widths must be finite and positive")
    if not math.isclose(float(ctr), float(heart_width) / float(thorax_width),
                        rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("P9 CTR disagrees with its component widths")
    if session.fov_truncated([raw.get("heart_mask_name", "Heart")]):
        raise ValueError("P9 heart mask is truncated by the model field of view")
    observations = [unit.id for values in outputs.values() for unit in values]
    value = {"CTR": float(ctr), "unit": "ratio", "projection": view.value["view"],
             "heart_width_px": raw.get("heart_width_px"),
             "thoracic_inner_width_px": raw.get("thoracic_inner_width_px"),
             "quality_conditions": {unit.finding_or_anatomy: unit.value
                                    for unit in _units(outputs, "assess_image_quality")},
             "status": "measured", "enlargement": None,
             "clinical_interpretation": "not_assessed",
             "limitations": ["Raw quality quantities have no frozen clinical grade thresholds",
                             "No calibrated enlargement rule or independent accuracy acceptance"]}
    relation = add_relation(session, "cardiothoracic_ratio", value,
                            input_ids=observations, rule_id=f"P9:ctr_with_conditions:{spec.version}")
    return value, relation, observations


def _g1_program(session: EvidenceSession, spec: FunctionSpec, outputs: dict, inputs: dict):
    candidates = _units(outputs, "detect_abnormalities")
    if len(candidates) != 1 or candidates[0].value.get("finding") != inputs["finding"]:
        raise ValueError("G1 requires candidate evidence for exactly the requested finding")
    candidate = candidates[0]
    expected_mask = "Candidate_" + inputs["finding"].replace(" ", "_")
    if inputs.get("lesion_mask_name") != expected_mask:
        raise ValueError("G1 lesion_mask_name must be the detector mask for the requested finding")
    observations = [unit.id for values in outputs.values() for unit in values]
    status = candidate.value.get("status")
    value = {"finding": inputs["finding"], "scope": "candidate_box_union_lung_overlap",
             "status": "localized",
             "image_sides": [], "patient_sides": None,
             "orientation_verified": inputs.get("orientation_verified", False),
             "orientation_source": inputs.get("orientation_source"),
             "disease_absence_established": False,
             "limitations": ["Candidate boxes are predictions, not lesion segmentations",
                             "Overlap of the box union does not assign each individual lesion"]}
    if status == "detected":
        right, left = session.masks.get("Right_Lung"), session.masks.get("Left_Lung")
        if right is None or left is None or not np.any(right) or not np.any(left):
            raise ValueError("G1 requires non-empty lung masks")
        if right.shape != left.shape or np.any(right & left):
            raise ValueError("G1 lung masks overlap or have different image frames")
        right_x, left_x = float(np.nonzero(right)[1].mean()), float(np.nonzero(left)[1].mean())
        if right_x >= left_x:
            raise ValueError("G1 lung-label/image-axis convention is inconsistent")
        relations = _units(outputs, "relate_lesion_to_anatomy")
        if session.mask_metadata.get(expected_mask, {}).get("observation_id") != candidate.id:
            raise ValueError("G1 candidate mask is not bound to this detector observation")
        by_mask = {unit.value.get("parameters", {}).get("anatomy_mask_name"): unit
                   for unit in relations}
        if set(by_mask) != {"Right_Lung", "Left_Lung"}:
            raise ValueError("G1 requires independent overlap measurements against both lung masks")
        if any(unit.value.get("parameters", {}).get("lesion_mask_name") != expected_mask
               or candidate.id not in unit.provenance for unit in by_mask.values()):
            raise ValueError("G1 spatial relations are not bound to the candidate detector mask")
        value["lung_overlap"] = {name: unit.value["overlap_fraction_of_a"]
                                  for name, unit in by_mask.items()}
        value["image_sides"] = [side for name, side in (("Right_Lung", "image_left"),
                                                        ("Left_Lung", "image_right"))
                                if by_mask[name].value.get("intersection_px", 0) > 0]
        if not value["image_sides"]:
            raise ValueError("G1 candidate does not overlap either lung; side is not assigned")
        if inputs.get("orientation_verified"):
            mapping = ({"image_left": "patient_right", "image_right": "patient_left"}
                       if inputs["patient_right_is_image_left"] else
                       {"image_left": "patient_left", "image_right": "patient_right"})
            value["patient_sides"] = [mapping[side] for side in value["image_sides"]]
            orientation = add_observation(
                session, "patient_image_orientation",
                {"patient_right_is_image_left": inputs["patient_right_is_image_left"],
                 "orientation_source": inputs["orientation_source"], "verified": True},
                evidence_type="global_condition", source_tool="verified_orientation_input",
                validity_conditions=["trusted caller-supplied orientation record"])
            observations.append(orientation.id)
        else:
            value["limitations"].append("Patient laterality withheld: image orientation is unverified")
    elif status == "not_detected":
        raise ValueError("G1 has no detected candidate to localize; disease absence is not established")
    else:
        raise ValueError("G1 candidate answer is unparsed")
    relation = add_relation(session, "candidate_laterality", value, input_ids=observations,
                            rule_id=f"G1:lung_overlap:{spec.version}")
    return value, relation, observations


_PROGRAMS = {"Q4": _q4_program, "P9": _p9_program, "G1": _g1_program}


def _prepare_inputs(inputs: Mapping[str, Any] | None) -> dict:
    result = dict(inputs or {})
    _json_hash(result)
    if "orientation_verified" in result and type(result["orientation_verified"]) is not bool:
        raise ValueError("orientation_verified must be boolean")
    if result.get("orientation_verified"):
        if type(result.get("patient_right_is_image_left")) is not bool:
            raise ValueError("verified orientation requires patient_right_is_image_left boolean")
        if not isinstance(result.get("orientation_source"), str) or not result["orientation_source"].strip():
            raise ValueError("verified orientation requires a recorded orientation_source")
    if isinstance(result.get("finding"), str):
        expected = "Candidate_" + result["finding"].replace(" ", "_")
        if "lesion_mask_name" in result and result["lesion_mask_name"] != expected:
            raise ValueError("lesion_mask_name must match the requested finding")
        result["lesion_mask_name"] = expected
    return result


def _parameters(function_id: str, step: ToolStep, inputs: dict, overrides: dict, schema: dict,
                out: Path | None, device: str | None) -> tuple[dict, bool]:
    raw = {**step.parameters, **overrides.get(step.tool, {}),
           **overrides.get(step.step_id, {}),
           **overrides.get(f"{function_id}.{step.step_id}", {})}
    params = _resolve(raw, inputs)
    properties = schema.get("properties", {})
    if device is not None and "device" in properties and "device" not in params and "masks_dir" not in params:
        params["device"] = device
    auto_output = False
    if "output_dir" in properties and not ({"output_dir", "masks_dir"} & params.keys()) and out is not None:
        params["output_dir"] = str(out / "artifacts" / step.tool / _json_hash(params)[:16])
        auto_output = True
    extra, missing = params.keys() - properties.keys(), set(schema.get("required", [])) - params.keys()
    if extra or missing:
        raise ValueError(f"{step.step_id}: invalid parameters; unknown={sorted(extra)}, missing={sorted(missing)}")
    _json_hash(params)
    return params, auto_output


def _run_spec(spec: FunctionSpec, session: EvidenceSession, registry: ToolRegistry,
              inputs: dict, overrides: dict, out: Path | None, device: str | None,
              image_hash: str, reuse: bool, include_optional: bool) -> FunctionRun:
    run = FunctionRun(spec.function_id, spec.version)
    if spec.function_id not in _PROGRAMS:
        run.refusal = ToolRefusal(spec.function_id, "function specification exists; claim program is not implemented",
                                  missing=list(spec.blockers) or ["function_program"])
        return run
    try:
        graph = DependencyGraph(spec.steps)
        graph.validate_registry(registry, required_only=True)
        steps = graph.topological_order() if include_optional else graph.required_closure()
        required_ids = {step.step_id for step in graph.required_closure()}
        menu = {entry["name"]: entry for entry in registry.describe_all()}
        prepared = {}
        for step in steps:
            try:
                if step.tool not in menu:
                    raise ValueError(f"unknown optional tool: {step.tool}")
                prepared[step.step_id] = _parameters(spec.function_id, step, inputs, overrides,
                                                     menu[step.tool]["input_schema"], out, device)
            except (TypeError, ValueError) as exc:
                if step.step_id in required_ids:
                    raise
                run.missing_optional.append(f"{step.step_id}: {exc}")
        run.missing_optional.extend(step.step_id for step in graph.topological_order()
                                    if step not in steps)
    except (TypeError, ValueError) as exc:
        run.status, run.refusal = "refused", ToolRefusal(spec.function_id, str(exc))
        return run

    outputs: dict[str, list[EvidenceUnit]] = {}
    cache = session.__dict__.setdefault("_function_call_cache", {})
    for step in steps:
        if step.step_id not in prepared:
            continue
        if any(dependency not in outputs for dependency in step.depends_on):
            run.steps.append({"step_id": step.step_id, "tool": step.tool, "status": "skipped",
                              "reason": "upstream optional step unavailable"})
            run.missing_optional.append(step.step_id)
            continue
        # A parsed negative candidate is not a failed measurement or disease-negative claim.
        if spec.function_id == "G1" and step.tool == "relate_lesion_to_anatomy" and any(
                unit.value.get("status") == "not_detected" for unit in _units(outputs, "detect_abnormalities")):
            run.steps.append({"step_id": step.step_id, "tool": step.tool, "status": "skipped",
                              "reason": "no candidate region to localize"})
            continue
        params, auto_output = prepared[step.step_id]
        key = _call_key(registry, session, step.tool, params, image_hash, auto_output)
        record = cache.get(key)
        if reuse and record is not None and _cache_valid(record, session):
            units = [session.graph.nodes[node_id] for node_id in record["ids"]]
            run.steps.append({"step_id": step.step_id, "tool": step.tool, "status": "reused",
                              "parameters": record["parameters"], "ids": record["ids"]})
        else:
            snapshot = _runtime_snapshot(session)
            result = registry.run(step.tool, session, **params)
            _release_runtime_snapshot(snapshot)
            record_step = {"step_id": step.step_id, "tool": step.tool, "parameters": params}
            if isinstance(result, ToolRefusal):
                _restore_runtime(session, snapshot)
                record_step.update(status="refusal", refusal=result.to_dict())
                run.steps.append(record_step)
                if step.step_id in required_ids:
                    run.status, run.refusal = "refused", result
                    break
                run.missing_optional.append(step.step_id)
                continue
            try:
                units = _commit_tool_outputs(session, step.tool, result)
                produced_masks = {name: _mask_digest(session, name)
                                  for name in set(snapshot["masks"]) | set(session.masks)
                                  if _mask_digest(session, name) != (
                                      None if name not in snapshot["masks"] else _json_hash({
                                          "shape": list(np.ascontiguousarray(snapshot["masks"][name]).shape),
                                          "dtype": str(np.ascontiguousarray(snapshot["masks"][name]).dtype),
                                          "sha256": hashlib.sha256(np.ascontiguousarray(
                                              snapshot["masks"][name]).tobytes()).hexdigest(),
                                          "metadata": snapshot["mask_metadata"].get(name, {})}))}
                bad_masks = [name for name in produced_masks
                             if session.mask_metadata.get(name, {}).get("observation_id")
                             not in {unit.id for unit in units}]
                if bad_masks:
                    raise ValueError(f"A tool produced masks without returned observation owners: {bad_masks}")
            except (TypeError, ValueError) as exc:
                _restore_runtime(session, snapshot)
                record_step.update(status="invalid_output", reason=str(exc))
                run.steps.append(record_step)
                run.status, run.refusal = "refused", ToolRefusal(step.tool, str(exc))
                break
            ids = [unit.id for unit in units]
            record_step.update(status="ok", ids=ids)
            run.steps.append(record_step)
            cache[key] = {"ids": ids, "units_hash": _json_hash([unit.to_dict() for unit in units]),
                          "parameters": params,
                          "masks": produced_masks,
                          "landmarks": _json_hash(session.landmarks)
                          if step.tool == "locate_lung_landmarks" else None}
        outputs[step.step_id] = units
        try:
            if spec.function_id == "P9":
                _p9_guard(outputs)
            if spec.function_id == "G1" and step.tool == "detect_abnormalities":
                if len(units) != 1 or units[0].value.get("status") not in {"detected", "not_detected"}:
                    raise ValueError("G1 candidate answer is missing, ambiguous or unparsed")
        except (TypeError, ValueError) as exc:
            run.status, run.refusal = "refused", ToolRefusal(spec.function_id, str(exc))
            break

    observations = tuple(dict.fromkeys(unit.id for values in outputs.values() for unit in values))
    run.evidence_refs = EvidenceRefs(observation_ids=observations)
    if run.refusal:
        return run
    try:
        errors = validate_graph(session)
        if errors:
            raise ValueError("invalid input evidence graph: " + "; ".join(errors))
        value, relation, _support = _PROGRAMS[spec.function_id](session, spec, outputs, inputs)
        missing_outputs = set(spec.outputs) - set(value)
        if missing_outputs:
            raise ValueError(f"function program omitted declared outputs: {sorted(missing_outputs)}")
        conditions = list(dict.fromkeys(condition for node_id in observations
                                        for condition in session.graph.nodes[node_id].validity_conditions))
        claim = add_claim(session, spec.proposition,
                          {**value, "function_id": spec.function_id, "spec_version": spec.version},
                          support_ids=[relation.id], rule_id=f"{spec.function_id}:claim:{spec.version}",
                          validity_conditions=conditions + list(spec.guards))
        run.status, run.outputs = "ok", value
        run.evidence_refs = EvidenceRefs(observations, (relation.id,), (claim.id,))
    except (TypeError, ValueError, KeyError) as exc:
        run.status, run.refusal = "refused", ToolRefusal(spec.function_id, str(exc))
    return run


def run_function_episode(
    function_id: str | Sequence[str] | None = None,
    *,
    image_ref: str | None = None,
    session: EvidenceSession | None = None,
    query: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    registry: ToolRegistry | None = None,
    tool_parameters: Mapping[str, dict] | None = None,
    out_dir: str | Path | None = None,
    device: str | None = None,
    reuse: bool = True,
    include_optional: bool = False,
    specs: Mapping[str, FunctionSpec] | None = None,
) -> FunctionEpisodeResult:
    """Route, plan and execute one or several closed-catalogue functions.

    ``out_dir`` must be new. Tool parameters may be overridden by tool name or
    step ID; step ID overrides take precedence.  Model tools receive their own
    artifact directories.  G1 needs ``inputs={"finding": "nodule"}``; patient
    laterality additionally requires explicit verified orientation and its
    source. P9 returns a conditioned CTR, not an enlargement diagnosis.

    Successful A calls are shared within a live session, bound to registry,
    image contents, parameters, input masks and current output artifacts in
    memory. Loaded sessions rerun tools because runtime masks are not saved.
    ToolRefusal/unknown/ambiguous/blocked are structured results; invalid caller
    configuration (e.g. an existing output directory) raises ``ValueError``.
    """
    if session is None:
        if not image_ref:
            raise ValueError("image_ref or an existing session is required")
        session = EvidenceSession(image_ref=str(Path(image_ref).resolve()))
    elif image_ref is not None and Path(image_ref).resolve() != Path(session.image_ref).resolve():
        raise ValueError("image_ref must match the existing session")
    catalog = dict(specs) if specs is not None else load_function_specs()
    route = RuleRouter(catalog).resolve(function_id=function_id, query=query)
    parsed_inputs = _prepare_inputs(inputs)
    overrides = dict(tool_parameters or {})
    if any(not isinstance(value, dict) for value in overrides.values()):
        raise ValueError("tool_parameters must map tool names/step IDs to parameter dictionaries")
    allowed = {name for fid in route.function_ids if fid in catalog
               for step in catalog[fid].steps
               for name in (step.step_id, step.tool, f"{fid}.{step.step_id}")}
    if overrides.keys() - allowed:
        raise ValueError(f"unknown tool/step overrides: {sorted(overrides.keys() - allowed)}")
    out = Path(out_dir).resolve() if out_dir is not None else None
    if out is not None:
        try:
            out.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("out_dir must be a new directory; existing results are never overwritten") from exc
    session.demand = Demand(query or "; ".join(catalog[fid].proposition for fid in route.function_ids if fid in catalog),
                            requirements=[{"function_id": fid} for fid in route.function_ids])
    episode = FunctionEpisodeResult(route.status, session, route)
    if route.status == "matched":
        if registry is None:
            registry = session.__dict__.get("_function_registry")
            if registry is None:
                from ..eval.runner import build_a_registry
                registry = build_a_registry()
                session.__dict__["_function_registry"] = registry
        try:
            image_hash = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
            prior_binding = session.__dict__.get("_function_image_binding")
            binding = (str(Path(session.image_ref).resolve()), image_hash)
            if prior_binding is not None and prior_binding != binding:
                raise ValueError("session input image changed; create a new session")
            session.__dict__["_function_image_binding"] = binding
            graph_errors = validate_graph(session)
            if graph_errors:
                raise ValueError("invalid existing evidence graph: " + "; ".join(graph_errors))
        except (OSError, ValueError) as exc:
            episode.functions = [FunctionRun(fid, catalog[fid].version, status="refused",
                                             refusal=ToolRefusal(fid, str(exc))) for fid in route.function_ids]
        else:
            episode.functions = [_run_spec(catalog[fid], session, registry, parsed_inputs,
                                            overrides, out, device, image_hash, reuse,
                                            include_optional)
                                 for fid in route.function_ids]
        statuses = {run.status for run in episode.functions}
        episode.status = ("ok" if statuses == {"ok"} else "partial" if "ok" in statuses
                          else "refused" if "refused" in statuses else "blocked")
    session.verdict = AnswerVerdict(answerable=episode.status == "ok",
                                    abstain_reason=None if episode.status == "ok" else
                                    "; ".join(run.refusal.reason for run in episode.functions if run.refusal) or route.reason,
                                    evidence_refs=episode.evidence_refs)
    session.trace_program({"program_id": "run_function_episode", "route": route.to_dict(),
                           "status": episode.status, "inputs": parsed_inputs,
                           "spec_hashes": {fid: _json_hash(catalog[fid].to_dict())
                                           for fid in route.function_ids if fid in catalog},
                           "functions": [run.to_dict() for run in episode.functions]})
    if out is not None:
        session.save(out / "session.json")
        (out / "episode.json").write_text(json.dumps(episode.to_dict(), ensure_ascii=False,
                                                    indent=2, allow_nan=False), encoding="utf-8")
    return episode
