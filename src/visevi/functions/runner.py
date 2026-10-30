"""Replayable function episodes over the existing A-tool registry.

The catalogue and execution coverage are deliberately separate.  All functions
can be planned; only programs with an explicit claim rule can be executed.
No clinical thresholds, calibrated probabilities or fusion weights are invented
here. Q4, P3, P6, P9, T1, G1 and S1 are executable integration pilots.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..construction.evidence import (EvidenceRefs, add_claim, add_observation,
                                     add_relation, validate_graph)
from ..backends import inspect_image
from ..core.registry import ToolRegistry
from ..core.session import EvidenceSession
from ..core.units import AnswerVerdict, Demand, EvidenceUnit, ToolRefusal
from ..perception.vlm import box_mask
from .planning import DependencyGraph, RouteResult, RuleRouter
from .specs import FunctionSpec, ToolStep, load_function_specs


IMPLEMENTED_FUNCTIONS = ("Q4", "P3", "P6", "P9", "T1", "G1", "S1")


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
    study_sessions: dict[str, EvidenceSession] = field(default_factory=dict)

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
                "evidence_refs": self.evidence_refs, "session": self.session.to_dict(),
                "study_sessions": {role: study.to_dict() for role, study in self.study_sessions.items()}}


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


def _single_tool_unit(outputs: dict[str, list[EvidenceUnit]], tool: str,
                      *, finding: str | None = None) -> EvidenceUnit:
    units = _units(outputs, tool)
    if finding is not None:
        units = [unit for unit in units if unit.finding_or_anatomy == finding]
    if len(units) != 1:
        suffix = f" for {finding}" if finding else ""
        raise ValueError(f"expected exactly one {tool} observation{suffix}")
    return units[0]


def _finite_unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite raw score in [0, 1]")
    return float(value)


def _candidate_binding(session: EvidenceSession, candidate: EvidenceUnit,
                       finding: str) -> tuple[str, list[list[float]]]:
    if candidate.value.get("finding") != finding:
        raise ValueError("candidate finding does not match the requested finding")
    status = candidate.value.get("status")
    if status != "detected":
        raise ValueError(f"candidate evidence is {status}; no region is available for this function")
    mask_name = "Candidate_" + finding.replace(" ", "_")
    if session.mask_metadata.get(mask_name, {}).get("observation_id") != candidate.id:
        raise ValueError("candidate mask is not bound to the detector observation")
    boxes = candidate.value.get("boxes_xyxy_px")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("detected candidate has no boxes")
    normalized = []
    frame = inspect_image(session.image_ref)
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in box)):
            raise ValueError("candidate boxes must contain four finite coordinates")
        x0, y0, x1, y1 = map(float, box)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("candidate boxes must have positive width and height")
        if x0 < 0 or y0 < 0 or x1 > frame["width"] or y1 > frame["height"]:
            raise ValueError("candidate box lies outside the original image frame")
        normalized.append([x0, y0, x1, y1])
    mask = np.asarray(session.masks.get(mask_name), dtype=bool)
    expected_shape = (frame["height"], frame["width"])
    if mask.shape != expected_shape or not np.array_equal(
            mask, box_mask(normalized, frame["width"], frame["height"])):
        raise ValueError("candidate mask disagrees with the union of detector boxes")
    return mask_name, normalized


def _p3_program(session: EvidenceSession, spec: FunctionSpec, outputs: dict, inputs: dict):
    finding = inputs["finding"]
    candidate = _single_tool_unit(outputs, "detect_abnormalities")
    mask_name, boxes = _candidate_binding(session, candidate, finding)
    shape = _single_tool_unit(outputs, "describe_region_shape")
    zones = _single_tool_unit(outputs, "assign_lung_zones")
    extent = _single_tool_unit(outputs, "measure_region_extent")
    lung_ids = session.mask_provenance(["Right_Lung", "Left_Lung"])
    for unit in (shape, zones, extent):
        if candidate.id not in unit.provenance:
            raise ValueError(f"{unit.source_tool} is not bound to the candidate mask")
    if shape.value.get("parameters", {}).get("mask_name") != mask_name:
        raise ValueError("shape result used the wrong candidate mask")
    if zones.value.get("parameters", {}).get("region_mask_name") != mask_name:
        raise ValueError("lung-zone result used the wrong candidate mask")
    if extent.value.get("parameters", {}).get("mask_name") != mask_name:
        raise ValueError("extent result used the wrong candidate mask")
    if not set(lung_ids) <= set(zones.provenance):
        raise ValueError("lung-zone result is not bound to both current lung observations")
    if zones.value.get("status") != "ok":
        raise ValueError("candidate does not overlap either lung")
    outside = zones.value.get("outside_lung_overlap_px")
    if isinstance(outside, bool) or not isinstance(outside, int) or outside < 0:
        raise ValueError("invalid outside-lung overlap count")
    centers = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes]
    box_extents = [{"width": box[2] - box[0], "height": box[3] - box[1]} for box in boxes]
    value = {
        "candidate_boxes": boxes,
        "center_xy_px": centers,
        "box_extent_px": box_extents,
        "box_shape": {**shape.value, "scope": "candidate_box_union_approximation"},
        "lung_zones": zones.value,
        "applicability": {
            "finding": finding, "mask_name": mask_name,
            "coordinate_unit": "original_image_pixels",
            "physical_size_available": False,
            "outside_lung_overlap_px": outside,
            "candidate_union_extent_check": {
                key: extent.value.get(key) for key in
                ("bbox_width_px", "bbox_height_px", "feret_max_px")},
            "status": "limited" if outside else "applicable",
            "clinical_interpretation": "not_assessed",
            "limitations": ["Candidate geometry is a union of model boxes, not a lesion contour",
                            "No physical spacing or true lesion diameter is available"],
        },
    }
    observations = [unit.id for values in outputs.values() for unit in values]
    relation = add_relation(session, "nodule_mass_candidate_geometry", value,
                            input_ids=observations, rule_id=f"P3:candidate_geometry:{spec.version}")
    return value, relation, observations


def _p6_program(session: EvidenceSession, spec: FunctionSpec, outputs: dict, inputs: dict):
    a8 = _single_tool_unit(outputs, "segment_pneumothorax")
    a6 = _single_tool_unit(outputs, "detect_abnormalities")
    symmetry = _single_tool_unit(outputs, "measure_lung_symmetry")
    if a6.value.get("finding") != "pneumothorax":
        raise ValueError("semantic candidate is not bound to pneumothorax")
    a6_status = a6.value.get("status")
    if a6_status not in {"detected", "not_detected"}:
        raise ValueError(f"semantic pneumothorax channel is {a6_status}")
    a8_status = a8.value.get("status")
    if a8_status not in {"region_above_threshold", "no_region_above_threshold"}:
        raise ValueError("specialized pneumothorax channel has an invalid status")
    case_score = _finite_unit_interval(a8.value.get("case_raw_score"), "case_raw_score")
    mask_threshold = _finite_unit_interval(a8.value.get("mask_threshold"), "mask_threshold")
    max_probability = _finite_unit_interval(a8.value.get("max_pixel_probability"), "max_pixel_probability")
    mirror = _finite_unit_interval(symmetry.value.get("mirror_dice"), "mirror_dice")
    lung_ids = session.mask_provenance(["Right_Lung", "Left_Lung"])
    if not set(lung_ids) <= set(symmetry.provenance):
        raise ValueError("lung-symmetry context is not bound to both lung observations")
    a8_positive, a6_positive = a8_status == "region_above_threshold", a6_status == "detected"
    if a8_positive:
        if session.mask_metadata.get("Pneumothorax_Candidate", {}).get("observation_id") != a8.id:
            raise ValueError("A8 mask is not bound to the specialized observation")
        mask = np.asarray(session.masks.get("Pneumothorax_Candidate"), dtype=bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError("A8 positive status requires a non-empty 2D candidate mask")
        frame = inspect_image(session.image_ref)
        if mask.shape != (frame["height"], frame["width"]):
            raise ValueError("A8 candidate mask is not in the original image frame")
        area = int(mask.sum())
        reported_area = a8.value.get("area_px")
        if isinstance(reported_area, bool) or not isinstance(reported_area, int) or reported_area != area:
            raise ValueError("A8 reported area disagrees with its candidate mask")
        ys, xs = np.nonzero(mask)
        expected_bbox = {"x0": int(xs.min()), "y0": int(ys.min()),
                         "x1": int(xs.max()), "y1": int(ys.max()),
                         "w": int(xs.max() - xs.min() + 1),
                         "h": int(ys.max() - ys.min() + 1)}
        if a8.value.get("bbox_inclusive_px") != expected_bbox:
            raise ValueError("A8 reported bounding box disagrees with its candidate mask")
        if max_probability < mask_threshold:
            raise ValueError("A8 positive mask contradicts its reported threshold and maximum score")
        mask_summary = {"mask_name": "Pneumothorax_Candidate", "status": a8_status,
                        "area_px": area, "bbox_inclusive_px": expected_bbox}
    else:
        reported_area = a8.value.get("area_px", 0)
        if isinstance(reported_area, bool) or not isinstance(reported_area, int) or reported_area != 0:
            raise ValueError("A8 no-region status requires zero reported mask area")
        if session.mask_metadata.get("Pneumothorax_Candidate", {}).get("observation_id") != a8.id:
            raise ValueError("A8 no-region observation is not bound to its empty current mask")
        mask = np.asarray(session.masks.get("Pneumothorax_Candidate"), dtype=bool)
        frame = inspect_image(session.image_ref)
        if mask.shape != (frame["height"], frame["width"]) or mask.any():
            raise ValueError("A8 no-region status requires an empty mask in the original image frame")
        mask_summary = {"mask_name": None, "status": a8_status,
                        "area_px": 0, "bbox_inclusive_px": None}
    if a6_positive:
        _, boxes = _candidate_binding(session, a6, "pneumothorax")
    else:
        boxes = []
    conflict_status = ({(True, True): "concordant_positive",
                        (False, False): "concordant_no_region",
                        (True, False): "discordant_a8_positive",
                        (False, True): "discordant_a6_positive"})[(a8_positive, a6_positive)]
    value = {
        "pneumothorax_mask": mask_summary,
        "case_raw_score": case_score,
        "score_calibrated": False,
        "candidate_boxes": boxes,
        "lung_mirror_similarity": {**symmetry.value, "mirror_dice": mirror,
                                    "disease_evidence": False},
        "conflict": {"status": conflict_status, "a8_region": a8_positive,
                     "a6_candidate": a6_positive},
        "clinical_interpretation": "not_assessed",
        "disease_absence_established": False,
    }
    observations = [unit.id for values in outputs.values() for unit in values]
    relation = add_relation(session, "pneumothorax_multichannel_state", value,
                            input_ids=observations, rule_id=f"P6:channel_state:{spec.version}")
    return value, relation, observations


def _point_xy(value: Any, name: str) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in value)):
        raise ValueError(f"{name} must contain two finite pixel coordinates")
    return [float(value[0]), float(value[1])]


def _t1_program(session: EvidenceSession, spec: FunctionSpec, outputs: dict, inputs: dict):
    devices = _units(outputs, "localize_devices")
    ett = [unit for unit in devices if unit.finding_or_anatomy == "endotracheal_tube"]
    carina = [unit for unit in devices if unit.finding_or_anatomy == "carina"]
    position = [unit for unit in devices if unit.finding_or_anatomy == "device_position_classification"]
    if len(ett) != 1 or len(carina) != 1 or len(position) != 1:
        raise ValueError("T1 requires exactly one ETT, carina and position observation from A26")
    score = _finite_unit_interval(ett[0].value.get("presence_raw_score"), "ETT presence_raw_score")
    threshold = _finite_unit_interval(ett[0].value.get("screening_threshold"), "ETT screening_threshold")
    present = score >= threshold
    if ett[0].value.get("status") != ("present" if present else "below_screening_threshold") or not present:
        raise ValueError("ETT presence is absent or inconsistent with the screening threshold")
    if carina[0].value.get("status") != "ok":
        raise ValueError("carina localization is invalid")
    tip = _point_xy(ett[0].value.get("distal_tip_xy"), "ETT distal tip")
    carina_xy = _point_xy(carina[0].value.get("point_xy"), "carina")
    frame = inspect_image(session.image_ref)
    for name, point in (("ETT distal tip", tip), ("carina", carina_xy)):
        if not (-0.5 <= point[0] <= frame["width"] - 0.5
                and -0.5 <= point[1] <= frame["height"] - 0.5):
            raise ValueError(f"{name} lies outside the original image frame")
    line_name = "Device_endotracheal_tube"
    if session.mask_metadata.get(line_name, {}).get("observation_id") != ett[0].id:
        raise ValueError("present ETT observation is not bound to its line mask")
    line = np.asarray(session.masks.get(line_name), dtype=bool)
    if line.shape != (frame["height"], frame["width"]) or not line.any():
        raise ValueError("ETT line mask is empty or outside the original image frame")
    _finite_unit_interval(ett[0].value.get("distal_tip_peak_probability"),
                          "ETT distal-tip peak score")
    _finite_unit_interval(carina[0].value.get("heatmap_peak"), "carina heatmap peak score")
    dx, dy = tip[0] - carina_xy[0], tip[1] - carina_xy[1]
    distance = math.hypot(dx, dy)
    all_raw_classes = position[0].value.get("raw_scores")
    if not isinstance(all_raw_classes, dict):
        raise ValueError("A26 position classification raw scores are missing")
    expected_classes = {"ETT - Abnormal", "ETT - Borderline", "ETT - Normal"}
    raw_classes = {key: _finite_unit_interval(all_raw_classes.get(key), key)
                   for key in sorted(expected_classes)}
    value = {
        "device_presence": {"status": "present", "presence_raw_score": score,
                            "screening_threshold": threshold, "probability_calibrated": False},
        "ett_distal_tip_xy_px": tip,
        "carina_xy_px": carina_xy,
        "tip_carina_distance_px": distance,
        "position_evidence": {
            "tip_minus_carina_vector_xy_px": [dx, dy],
            "carina_minus_tip_vertical_px": carina_xy[1] - tip[1],
            "status": "measurement_only", "clinical_position": None,
            "position_classification_raw_scores": raw_classes,
            "limitations": ["No trusted pixel spacing; distance is reported in pixels",
                            "No clinical position thresholds are frozen"],
        },
    }
    relation = add_relation(session, "ett_tip_to_carina", value,
                            input_ids=[ett[0].id, carina[0].id, position[0].id],
                            rule_id=f"T1:point_distance:{spec.version}")
    return value, relation, [ett[0].id, carina[0].id, position[0].id]


def _pair_metadata(inputs: dict) -> dict:
    metadata = inputs.get("pair_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("S1 pair_metadata is required")
    allowed = {"patient_id_hash", "identity_verified", "prior_study_id", "current_study_id",
               "prior_time", "current_time", "chronology_verified", "projection_comparable",
               "registration", "lesion_match"}
    if set(metadata) != allowed:
        raise ValueError(f"pair metadata fields must be exactly {sorted(allowed)}")
    for key in ("patient_id_hash", "prior_study_id", "current_study_id",
                "prior_time", "current_time"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"pair metadata {key} must be a non-empty string")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", metadata["patient_id_hash"]):
        raise ValueError("pair metadata patient_id_hash must be a lowercase SHA-256 digest")
    if metadata.get("identity_verified") is not True:
        raise ValueError("pair identity must be verified")
    if metadata["prior_study_id"] == metadata["current_study_id"]:
        raise ValueError("paired comparison requires distinct studies")
    if metadata.get("chronology_verified") is not True:
        raise ValueError("pair chronology must be verified")
    try:
        prior_time = datetime.fromisoformat(metadata["prior_time"].replace("Z", "+00:00"))
        current_time = datetime.fromisoformat(metadata["current_time"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("pair times must be ISO-8601 timestamps") from exc
    if current_time <= prior_time:
        raise ValueError("current study time must be strictly later than prior study time")
    if metadata.get("projection_comparable") is not True:
        raise ValueError("paired projections must be explicitly comparable")
    registration = metadata.get("registration")
    registration_fields = {"status", "method", "id", "frame_id",
                           "common_pixel_frame_verified", "applied_to_inputs"}
    if (not isinstance(registration, dict) or registration.get("status") != "verified"
            or set(registration) != registration_fields
            or not all(isinstance(registration.get(key), str) and registration[key].strip()
                       for key in ("method", "id", "frame_id"))
            or registration.get("common_pixel_frame_verified") is not True
            or registration.get("applied_to_inputs") is not True):
        raise ValueError("verified registration must be applied to both inputs in a named common pixel frame")
    lesion_match = metadata.get("lesion_match")
    lesion_fields = {"status", "id", "source"}
    if (not isinstance(lesion_match, dict) or lesion_match.get("status") != "verified"
            or set(lesion_match) != lesion_fields
            or not all(isinstance(lesion_match.get(key), str) and lesion_match[key].strip()
                       for key in ("id", "source"))):
        raise ValueError("verified lesion match metadata with id and source is required")
    return metadata


def _step_unit(outputs: dict[str, list[EvidenceUnit]], step_id: str) -> EvidenceUnit:
    units = outputs.get(step_id, [])
    if len(units) != 1:
        raise ValueError(f"S1 step {step_id} must produce exactly one observation")
    return units[0]


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _s1_program(session: EvidenceSession, spec: FunctionSpec, outputs: dict, inputs: dict):
    metadata = _pair_metadata(inputs)
    finding = inputs["finding"]
    candidates = {role: _step_unit(outputs, f"{role}_candidates")
                  for role in ("prior", "current")}
    for role, candidate in candidates.items():
        if candidate.value.get("finding") != finding or candidate.value.get("status") != "detected":
            raise ValueError(f"{role} candidate is missing or unparsed")
        boxes = candidate.value.get("boxes_xyxy_px")
        if not isinstance(boxes, list) or len(boxes) != 1:
            raise ValueError(f"{role} candidate requires exactly one matched box")
    extents = {role: _step_unit(outputs, f"{role}_extent") for role in ("prior", "current")}
    intensities = {role: _step_unit(outputs, f"{role}_intensity") for role in ("prior", "current")}
    for role in ("prior", "current"):
        if candidates[role].id not in extents[role].provenance:
            raise ValueError(f"{role} extent is not bound to its candidate")
        if candidates[role].id not in intensities[role].provenance:
            raise ValueError(f"{role} intensity is not bound to its candidate")
    width = {role: _numeric(extents[role].value.get("bbox_width_px"),
                            f"{role} bbox_width_px") for role in ("prior", "current")}
    height = {role: _numeric(extents[role].value.get("bbox_height_px"),
                             f"{role} bbox_height_px") for role in ("prior", "current")}
    mean = {role: _numeric(intensities[role].value.get("mean"),
                           f"{role} mean intensity") for role in ("prior", "current")}

    binding = {**metadata, "image_sha256": dict(inputs["_pair_image_sha256"])}
    pair = add_observation(
        session, "paired_study_binding", binding, evidence_type="global_condition",
        source_tool="verified_pair_metadata",
        validity_conditions=["trusted caller record; registration and lesion matching are not recomputed"])
    value = {
        "matched_lesions": {
            "finding": finding,
            "match_id": metadata["lesion_match"]["id"],
            "matching_source": metadata["lesion_match"]["source"],
            "prior_candidate_count": 1, "current_candidate_count": 1,
        },
        "change_magnitude": {
            "extent": {
                "bbox_width_px": {"prior": width["prior"], "current": width["current"],
                                  "delta": width["current"] - width["prior"]},
                "bbox_height_px": {"prior": height["prior"], "current": height["current"],
                                   "delta": height["current"] - height["prior"]},
                "scope": "matched_candidate_box_union_in_each_study",
            },
            "intensity": {
                "mean_display_gray": {"prior": mean["prior"], "current": mean["current"],
                                      "delta": mean["current"] - mean["prior"]},
                "scope": "raw input display gray; not normalized across acquisition systems",
            },
        },
        "change_direction": "not_assessed",
        "comparability": {
            "patient_id_hash": metadata["patient_id_hash"],
            "prior_study_id": metadata["prior_study_id"],
            "current_study_id": metadata["current_study_id"],
            "prior_time": metadata["prior_time"], "current_time": metadata["current_time"],
            "projection_comparable": True,
            "registration_id": metadata["registration"]["id"],
            "registration_method": metadata["registration"]["method"],
            "common_pixel_frame_id": metadata["registration"]["frame_id"],
            "clinical_direction_threshold_frozen": False,
            "automatic_registration_validated": False,
        },
    }
    observations = [unit.id for values in outputs.values() for unit in values] + [pair.id]
    relation = add_relation(session, "paired_candidate_change", value, input_ids=observations,
                            rule_id=f"S1:raw_paired_delta:{spec.version}")
    return value, relation, observations


_PROGRAMS = {"Q4": _q4_program, "P3": _p3_program, "P6": _p6_program,
             "P9": _p9_program, "T1": _t1_program, "G1": _g1_program,
             "S1": _s1_program}


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


def _trace_inputs(inputs: dict) -> dict:
    """Keep rejected pair metadata from persisting an unhashed patient ID."""
    result = json.loads(json.dumps(inputs, ensure_ascii=False, allow_nan=False))
    if "pair_metadata" in result:
        try:
            _pair_metadata(result)
        except (TypeError, ValueError):
            result["pair_metadata"] = "redacted_invalid_pair_metadata"
    return result


def _parameters(function_id: str, step: ToolStep, inputs: dict, overrides: dict, schema: dict,
                out: Path | None, device: str | None, input_role: str) -> tuple[dict, bool]:
    raw = {**step.parameters, **overrides.get(step.tool, {}),
           **overrides.get(step.step_id, {}),
           **overrides.get(f"{function_id}.{step.step_id}", {})}
    params = _resolve(raw, inputs)
    properties = schema.get("properties", {})
    if device is not None and "device" in properties and "device" not in params and "masks_dir" not in params:
        params["device"] = device
    auto_output = False
    if "output_dir" in properties and not ({"output_dir", "masks_dir"} & params.keys()) and out is not None:
        params["output_dir"] = str(out / "artifacts" / input_role / step.tool / _json_hash(params)[:16])
        auto_output = True
    extra, missing = params.keys() - properties.keys(), set(schema.get("required", [])) - params.keys()
    if extra or missing:
        raise ValueError(f"{step.step_id}: invalid parameters; unknown={sorted(extra)}, missing={sorted(missing)}")
    _json_hash(params)
    return params, auto_output


def _import_prior_outputs(session: EvidenceSession, spec: FunctionSpec,
                          outputs: dict[str, list[EvidenceUnit]], image_hash: str) -> None:
    prior_ids = [unit.id for step in spec.steps if step.input_role == "prior"
                 for unit in outputs.get(step.step_id, [])]
    prior_payload = [unit.to_dict() for step in spec.steps if step.input_role == "prior"
                     for unit in outputs.get(step.step_id, [])]
    namespace = _json_hash({"image_sha256": image_hash, "observations": prior_payload})
    id_map = {node_id: f"prior:{namespace}:{node_id}" for node_id in dict.fromkeys(prior_ids)}
    for step in spec.steps:
        if step.input_role != "prior" or step.step_id not in outputs:
            continue
        imported = []
        for unit in outputs[step.step_id]:
            provenance = [id_map.get(item, item) for item in unit.provenance]
            value = json.loads(json.dumps(unit.value, ensure_ascii=False, allow_nan=False))
            value["study_role"] = "prior"
            copy = replace(unit, id=id_map[unit.id], value=value, provenance=provenance)
            if copy.id in session.graph.nodes:
                if session.graph.nodes[copy.id].to_dict() != copy.to_dict():
                    raise ValueError("prior evidence namespace collision")
            else:
                session.add_observation(copy)
                for upstream in provenance:
                    if upstream in session.graph.nodes:
                        session.graph.add_edge(upstream, copy.id, "derived_from", copy.source_tool)
            imported.append(session.graph.nodes[copy.id])
        outputs[step.step_id] = imported


def _run_spec(spec: FunctionSpec, session: EvidenceSession, registry: ToolRegistry,
              inputs: dict, overrides: dict, out: Path | None, device: str | None,
              image_hash: str, reuse: bool, include_optional: bool,
              study_sessions: Mapping[str, EvidenceSession] | None = None,
              image_hashes: Mapping[str, str] | None = None) -> FunctionRun:
    run = FunctionRun(spec.function_id, spec.version)
    if spec.function_id not in _PROGRAMS:
        run.refusal = ToolRefusal(spec.function_id, "function specification exists; claim program is not implemented",
                                  missing=list(spec.blockers) or ["function_program"])
        return run
    try:
        if spec.function_id == "P3" and inputs.get("finding") not in {"nodule", "mass"}:
            raise ValueError("P3 finding must be nodule or mass")
        sessions = dict(study_sessions or {"current": session})
        hashes = dict(image_hashes or {"current": image_hash})
        if spec.function_id == "S1":
            _pair_metadata(inputs)
            if "prior" not in sessions or "prior" not in hashes:
                raise ValueError("S1 requires a separate prior image/session")
            if hashes["prior"] == hashes["current"]:
                raise ValueError("paired comparison requires distinct images")
            prior_frame = inspect_image(sessions["prior"].image_ref)
            current_frame = inspect_image(sessions["current"].image_ref)
            if (prior_frame["width"], prior_frame["height"]) != (
                    current_frame["width"], current_frame["height"]):
                raise ValueError("registered paired inputs must have the same common pixel-frame dimensions")
            inputs["_pair_image_sha256"] = {"prior": hashes["prior"], "current": hashes["current"]}
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
                if step.input_role not in sessions:
                    raise ValueError(f"missing {step.input_role} study session")
                prepared[step.step_id] = _parameters(spec.function_id, step, inputs, overrides,
                                                     menu[step.tool]["input_schema"], out, device,
                                                     step.input_role)
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
        role = step.input_role
        target_session = sessions[role]
        params, auto_output = prepared[step.step_id]
        key = _call_key(registry, target_session, step.tool, params, hashes[role], auto_output)
        target_cache = target_session.__dict__.setdefault("_function_call_cache", {})
        record = target_cache.get(key)
        if reuse and record is not None and _cache_valid(record, target_session):
            units = [target_session.graph.nodes[node_id] for node_id in record["ids"]]
            run.steps.append({"step_id": step.step_id, "tool": step.tool, "status": "reused",
                              "input_role": role, "parameters": record["parameters"],
                              "ids": record["ids"]})
        else:
            snapshot = _runtime_snapshot(target_session)
            result = registry.run(step.tool, target_session, **params)
            _release_runtime_snapshot(snapshot)
            record_step = {"step_id": step.step_id, "tool": step.tool,
                           "input_role": role, "parameters": params}
            if isinstance(result, ToolRefusal):
                _restore_runtime(target_session, snapshot)
                refusal = ToolRefusal(result.tool, f"{role} study: {result.reason}",
                                      missing=result.missing, conditions=result.conditions)
                record_step.update(status="refusal", refusal=refusal.to_dict())
                run.steps.append(record_step)
                if step.step_id in required_ids:
                    run.status, run.refusal = "refused", refusal
                    break
                run.missing_optional.append(step.step_id)
                continue
            try:
                units = _commit_tool_outputs(target_session, step.tool, result)
                produced_masks = {name: _mask_digest(target_session, name)
                                  for name in set(snapshot["masks"]) | set(target_session.masks)
                                  if _mask_digest(target_session, name) != (
                                      None if name not in snapshot["masks"] else _json_hash({
                                          "shape": list(np.ascontiguousarray(snapshot["masks"][name]).shape),
                                          "dtype": str(np.ascontiguousarray(snapshot["masks"][name]).dtype),
                                          "sha256": hashlib.sha256(np.ascontiguousarray(
                                              snapshot["masks"][name]).tobytes()).hexdigest(),
                                          "metadata": snapshot["mask_metadata"].get(name, {})}))}
                bad_masks = [name for name in produced_masks
                             if target_session.mask_metadata.get(name, {}).get("observation_id")
                             not in {unit.id for unit in units}]
                if bad_masks:
                    raise ValueError(f"A tool produced masks without returned observation owners: {bad_masks}")
            except (TypeError, ValueError) as exc:
                _restore_runtime(target_session, snapshot)
                record_step.update(status="invalid_output", reason=str(exc))
                run.steps.append(record_step)
                run.status, run.refusal = "refused", ToolRefusal(step.tool, str(exc))
                break
            ids = [unit.id for unit in units]
            record_step.update(status="ok", ids=ids)
            run.steps.append(record_step)
            target_cache[key] = {
                "ids": ids, "units_hash": _json_hash([unit.to_dict() for unit in units]),
                "parameters": params, "masks": produced_masks,
                "landmarks": _json_hash(target_session.landmarks)
                if step.tool == "locate_lung_landmarks" else None}
        outputs[step.step_id] = units
        try:
            if spec.function_id == "P9":
                _p9_guard(outputs)
            if spec.function_id == "G1" and step.tool == "detect_abnormalities":
                if len(units) != 1 or units[0].value.get("status") not in {"detected", "not_detected"}:
                    raise ValueError("G1 candidate answer is missing, ambiguous or unparsed")
            if spec.function_id == "S1" and step.step_id in {"prior_candidates", "current_candidates"}:
                if len(units) != 1 or units[0].value.get("status") != "detected":
                    raise ValueError(f"{role} candidate is missing or unparsed")
                _, boxes = _candidate_binding(target_session, units[0], inputs["finding"])
                if len(boxes) != 1:
                    raise ValueError(f"{role} candidate requires exactly one matched box")
        except (TypeError, ValueError) as exc:
            run.status, run.refusal = "refused", ToolRefusal(spec.function_id, str(exc))
            break

    if spec.input_type == "paired_images":
        _import_prior_outputs(session, spec, outputs, hashes["prior"])
    for role, study in sessions.items():
        errors = validate_graph(study)
        if errors:
            run.status = "refused"
            run.refusal = ToolRefusal(spec.function_id,
                                      f"invalid {role} evidence graph: {'; '.join(errors)}")
            break
    observations = tuple(dict.fromkeys(unit.id for values in outputs.values() for unit in values))
    run.evidence_refs = EvidenceRefs(observation_ids=observations)
    if run.refusal:
        return run
    try:
        errors = validate_graph(session)
        if errors:
            raise ValueError("invalid input evidence graph: " + "; ".join(errors))
        value, relation, support = _PROGRAMS[spec.function_id](session, spec, outputs, inputs)
        program_observations = tuple(dict.fromkeys(support))
        if any(node_id not in session.graph.nodes for node_id in program_observations):
            raise ValueError("function program returned an unknown observation reference")
        missing_outputs = set(spec.outputs) - set(value)
        if missing_outputs:
            raise ValueError(f"function program omitted declared outputs: {sorted(missing_outputs)}")
        conditions = list(dict.fromkeys(condition for node_id in program_observations
                                        for condition in session.graph.nodes[node_id].validity_conditions))
        claim_value = {**value, "function_id": spec.function_id, "spec_version": spec.version}
        if spec.function_id == "T1":
            claim_value["clinical_position"] = value["position_evidence"]["clinical_position"]
        claim = add_claim(session, spec.proposition, claim_value,
                          support_ids=[relation.id], rule_id=f"{spec.function_id}:claim:{spec.version}",
                          validity_conditions=conditions + list(spec.guards))
        run.status, run.outputs = "ok", value
        run.evidence_refs = EvidenceRefs(program_observations, (relation.id,), (claim.id,))
    except (TypeError, ValueError, KeyError) as exc:
        run.status, run.refusal = "refused", ToolRefusal(spec.function_id, str(exc))
    return run


def run_function_episode(
    function_id: str | Sequence[str] | None = None,
    *,
    image_ref: str | None = None,
    session: EvidenceSession | None = None,
    prior_image_ref: str | None = None,
    prior_session: EvidenceSession | None = None,
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
    source. P9 returns a conditioned CTR, not an enlargement diagnosis. Paired
    functions use ``prior_image_ref``/``prior_session`` for a distinct study.

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
    needs_pair = any(catalog[fid].input_type == "paired_images"
                     for fid in route.function_ids if fid in catalog)
    if prior_session is None and prior_image_ref is not None:
        prior_session = EvidenceSession(image_ref=str(Path(prior_image_ref).resolve()))
    elif prior_session is not None and prior_image_ref is not None:
        if Path(prior_image_ref).resolve() != Path(prior_session.image_ref).resolve():
            raise ValueError("prior_image_ref must match the existing prior_session")
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
    study_sessions = {"current": session}
    if prior_session is not None:
        study_sessions["prior"] = prior_session
    episode = FunctionEpisodeResult(route.status, session, route,
                                    study_sessions=study_sessions if needs_pair else {})
    if route.status == "matched":
        if registry is None:
            registry = session.__dict__.get("_function_registry")
            if registry is None:
                from ..eval.runner import build_a_registry
                registry = build_a_registry()
                session.__dict__["_function_registry"] = registry
        try:
            image_hashes = {}
            for role, study in study_sessions.items():
                digest = hashlib.sha256(Path(study.image_ref).read_bytes()).hexdigest()
                prior_binding = study.__dict__.get("_function_image_binding")
                binding = (str(Path(study.image_ref).resolve()), digest)
                if prior_binding is not None and prior_binding != binding:
                    raise ValueError(f"{role} session input image changed; create a new session")
                study.__dict__["_function_image_binding"] = binding
                graph_errors = validate_graph(study)
                if graph_errors:
                    raise ValueError(f"invalid existing {role} evidence graph: " + "; ".join(graph_errors))
                image_hashes[role] = digest
            image_hash = image_hashes["current"]
        except (OSError, ValueError) as exc:
            episode.functions = [FunctionRun(fid, catalog[fid].version, status="refused",
                                             refusal=ToolRefusal(fid, str(exc))) for fid in route.function_ids]
        else:
            episode.functions = [_run_spec(catalog[fid], session, registry, parsed_inputs,
                                            overrides, out, device, image_hash, reuse,
                                            include_optional, study_sessions, image_hashes)
                                 for fid in route.function_ids]
        statuses = {run.status for run in episode.functions}
        episode.status = ("ok" if statuses == {"ok"} else "partial" if "ok" in statuses
                          else "refused" if "refused" in statuses else "blocked")
    session.verdict = AnswerVerdict(answerable=episode.status == "ok",
                                    abstain_reason=None if episode.status == "ok" else
                                    "; ".join(run.refusal.reason for run in episode.functions if run.refusal) or route.reason,
                                    evidence_refs=episode.evidence_refs)
    session.trace_program({"program_id": "run_function_episode", "route": route.to_dict(),
                           "status": episode.status, "inputs": _trace_inputs(parsed_inputs),
                           "spec_hashes": {fid: _json_hash(catalog[fid].to_dict())
                                           for fid in route.function_ids if fid in catalog},
                           "functions": [run.to_dict() for run in episode.functions]})
    if out is not None:
        session.save(out / "session.json")
        if needs_pair and prior_session is not None:
            prior_session.save(out / "prior_session.json")
        (out / "episode.json").write_text(json.dumps(episode.to_dict(), ensure_ascii=False,
                                                    indent=2, allow_nan=False), encoding="utf-8")
    return episode
