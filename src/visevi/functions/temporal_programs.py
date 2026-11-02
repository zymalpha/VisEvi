"""Conservative temporal evidence programs for S2 and S3.

The programs compare detector observations across two explicitly bound studies.
They never interpret a detector negative as disease absence and never convert
raw box changes into a clinical stable/progressed label without a frozen
matching and tolerance protocol.
"""
from __future__ import annotations

import math
from numbers import Real

from ..backends import inspect_image
from ..construction.evidence import add_observation, add_relation
from ..core.units import EvidenceUnit


def _units(outputs: dict[str, list[EvidenceUnit]], step: str) -> list[EvidenceUnit]:
    return outputs.get(step, [])


def _one(outputs: dict[str, list[EvidenceUnit]], step: str) -> EvidenceUnit:
    values = _units(outputs, step)
    if len(values) != 1:
        raise ValueError(f"{step} requires exactly one observation")
    return values[0]


def _candidate(unit: EvidenceUnit, finding: str, width: int, height: int) -> list[list[float]]:
    if unit.value.get("finding") != finding:
        raise ValueError("candidate finding does not match the requested finding")
    status = unit.value.get("status")
    boxes = unit.value.get("boxes_xyxy_px")
    if not isinstance(boxes, list):
        raise ValueError("candidate boxes must be a list")
    if status == "not_detected":
        if boxes:
            raise ValueError("not_detected candidate must have no boxes")
        return []
    if status != "detected" or not boxes:
        raise ValueError(f"candidate status is {status!r}; temporal comparison is unavailable")
    normalized = []
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not math.isfinite(value) for value in box)):
            raise ValueError("candidate boxes must contain four finite coordinates")
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("candidate box is outside the common image frame")
        normalized.append([x0, y0, x1, y1])
    return normalized


def _frame(session) -> tuple[int, int]:
    info = inspect_image(session.image_ref)
    return int(info["width"]), int(info["height"])


def _pair_binding(session, inputs: dict, image_hashes: dict[str, str], *, require_match: bool) -> EvidenceUnit:
    from .runner import _pair_metadata
    metadata = _pair_metadata(inputs, require_lesion_match=require_match)
    value = {**metadata, "image_sha256": dict(image_hashes),
             "pair_role": "prior_current", "registration_recomputed": False}
    return add_observation(
        session, "paired_study_binding", value, evidence_type="global_condition",
        source_tool="verified_pair_metadata",
        validity_conditions=["trusted caller record; chronology and registration are not recomputed"])


def _bbox_features(boxes: list[list[float]]) -> list[dict]:
    return [{"box_index": index,
             "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
             "width_px": box[2] - box[0], "height_px": box[3] - box[1],
             "center_xy_px": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}
            for index, box in enumerate(boxes)]


def _iou(first: list[float], second: list[float]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union else 0.0


def _comparability(metadata: dict) -> dict:
    registration = metadata["registration"]
    return {
        "patient_id_hash": metadata["patient_id_hash"],
        "prior_study_id": metadata["prior_study_id"],
        "current_study_id": metadata["current_study_id"],
        "prior_time": metadata["prior_time"],
        "current_time": metadata["current_time"],
        "projection_comparable": True,
        "registration_id": registration["id"],
        "registration_method": registration["method"],
        "common_pixel_frame_id": registration["frame_id"],
        "registration_recomputed": False,
    }


def s2_program(session, spec, outputs, inputs):
    from .runner import _pair_metadata
    metadata = _pair_metadata(inputs, require_lesion_match=False)
    finding = inputs.get("finding")
    if not isinstance(finding, str) or not finding.strip():
        raise ValueError("S2 requires a finding")
    prior_width, prior_height = _frame(session.__dict__.get("_paired_prior_session", session))
    current_width, current_height = _frame(session)
    if (prior_width, prior_height) != (current_width, current_height):
        raise ValueError("paired images must have the same common pixel-frame dimensions")
    prior_unit, current_unit = _one(outputs, "prior_candidates"), _one(outputs, "current_candidates")
    prior_boxes = _candidate(prior_unit, finding, current_width, current_height)
    current_boxes = _candidate(current_unit, finding, current_width, current_height)
    pair = _pair_binding(session, inputs, inputs["_pair_image_sha256"], require_match=False)
    reference = inputs.get("prior_reference")
    if reference is not None:
        if (not isinstance(reference, dict)
                or set(reference) != {"status", "source", "id"}
                or reference.get("status") not in {"independently_absent", "independently_present", "not_assessed"}
                or not all(isinstance(reference.get(key), str) and reference[key].strip()
                           for key in ("status", "source", "id"))):
            raise ValueError("prior_reference must record an explicit independent status, source and id")
    prior_status, current_status = prior_unit.value.get("status"), current_unit.value.get("status")
    if current_status == "not_detected":
        new_status = "no_current_candidate_observed"
        interpretation = "not_assessed"
    elif prior_status == "not_detected" and reference and reference["status"] == "independently_absent":
        new_status = "candidate_new_with_independent_prior_absence"
        interpretation = "candidate_newness_only"
    elif prior_status == "not_detected":
        new_status = "candidate_current_but_prior_absence_unverified"
        interpretation = "not_assessed"
    else:
        new_status = "prior_candidate_also_observed"
        interpretation = "not_assessed"
    if prior_boxes and current_boxes:
        overlap = [[_iou(first, second) for second in current_boxes] for first in prior_boxes]
        match_status = "not_assessed_without_verified_correspondence"
    else:
        overlap, match_status = [], "not_applicable"
    value = {
        "new_candidate_regions": {
            "status": new_status, "interpretation": interpretation,
            "prior_detector_status": prior_status, "current_detector_status": current_status,
            "prior_boxes": prior_boxes, "current_boxes": current_boxes,
            "independent_prior_reference": reference,
            "disease_absence_or_new_lesion_established": False,
        },
        "temporal_matches": {"status": match_status, "prior_current_iou_matrix": overlap,
                              "matching_source": None},
        "chronology": {"prior_time": metadata["prior_time"], "current_time": metadata["current_time"],
                       "verified": True},
        "comparability": _comparability(metadata),
    }
    support = [unit.id for values in outputs.values() for unit in values] + [pair.id]
    relation = add_relation(session, "new_candidate_comparison", value, input_ids=support,
                            rule_id=f"{spec.function_id}:conservative_newness:{spec.version}")
    return value, relation, support


def s3_program(session, spec, outputs, inputs):
    from .runner import _pair_metadata
    metadata = _pair_metadata(inputs, require_lesion_match=True)
    finding = inputs.get("finding")
    if not isinstance(finding, str) or not finding.strip():
        raise ValueError("S3 requires a finding")
    width, height = _frame(session)
    prior_session = session.__dict__.get("_paired_prior_session", session)
    if _frame(prior_session) != (width, height):
        raise ValueError("paired images must have the same common pixel-frame dimensions")
    prior_unit, current_unit = _one(outputs, "prior_candidates"), _one(outputs, "current_candidates")
    prior_boxes = _candidate(prior_unit, finding, width, height)
    current_boxes = _candidate(current_unit, finding, width, height)
    if len(prior_boxes) != 1 or len(current_boxes) != 1:
        raise ValueError("S3 requires exactly one detected candidate per study for the verified match")
    pair = _pair_binding(session, inputs, inputs["_pair_image_sha256"], require_match=True)
    prior_features, current_features = _bbox_features(prior_boxes)[0], _bbox_features(current_boxes)[0]
    delta = {key: current_features[key] - prior_features[key]
             for key in ("width_px", "height_px")}
    delta["center_xy_px"] = [current_features["center_xy_px"][index]
                              - prior_features["center_xy_px"][index] for index in (0, 1)]
    value = {
        "matched_lesions": {"finding": finding, "match_id": metadata["lesion_match"]["id"],
                             "matching_source": metadata["lesion_match"]["source"],
                             "prior": prior_features, "current": current_features},
        "change_magnitude": {"bbox": {"prior": prior_features, "current": current_features,
                                        "delta": delta},
                              "iou": _iou(prior_boxes[0], current_boxes[0]),
                              "units": "original common-frame pixels"},
        "stability": "not_assessed",
        "comparability": {**_comparability(metadata),
                           "stability_tolerance_frozen": False,
                           "clinical_interpretation": "not_assessed"},
    }
    support = [unit.id for values in outputs.values() for unit in values] + [pair.id]
    relation = add_relation(session, "matched_candidate_change", value, input_ids=support,
                            rule_id=f"{spec.function_id}:raw_change_without_threshold:{spec.version}")
    return value, relation, support


TEMPORAL_PROGRAMS = {"S2": s2_program, "S3": s3_program}
