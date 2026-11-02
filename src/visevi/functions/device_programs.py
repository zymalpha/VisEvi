"""Conservative T2/T3 device-position evidence programs.

The A26 detector supplies a device line and a model endpoint. Anatomical
reference points/curves are caller-provided only when they carry an explicit
source, image hash and original-image coordinate frame. No clinical position
label is inferred from an uncalibrated model score.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from numbers import Real

import numpy as np

from ..backends import inspect_image
from ..construction.evidence import add_observation, add_relation
from ..core.units import EvidenceUnit


def _devices(outputs: dict[str, list[EvidenceUnit]]) -> list[EvidenceUnit]:
    return [unit for values in outputs.values() for unit in values
            if unit.source_tool == "localize_devices"]


def _device(outputs: dict, family: str) -> EvidenceUnit:
    units = [unit for unit in _devices(outputs)
             if unit.finding_or_anatomy == family]
    if len(units) != 1:
        raise ValueError(f"A26 must provide exactly one {family} observation")
    unit = units[0]
    value = unit.value
    if value.get("status") != "present":
        raise ValueError(f"{family} is not screened in by A26")
    tip = value.get("distal_tip_xy")
    if (not isinstance(tip, (list, tuple)) or len(tip) != 2
            or any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
                   for x in tip)):
        raise ValueError(f"{family} distal endpoint is unavailable")
    return unit


def _frame(session) -> tuple[int, int]:
    info = inspect_image(session.image_ref)
    return int(info["width"]), int(info["height"])


def _point(value, name: str, width: int, height: int) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
                   for x in value)):
        raise ValueError(f"{name} must be two finite coordinates")
    point = [float(value[0]), float(value[1])]
    if not (0 <= point[0] <= width - 1 and 0 <= point[1] <= height - 1):
        raise ValueError(f"{name} is outside the original image frame")
    return point


def _line_mask(session, unit: EvidenceUnit, name: str) -> np.ndarray:
    mask = session.masks.get(name)
    if mask is None:
        raise ValueError(f"{name} line mask is missing")
    frame = _frame(session)
    result = np.asarray(mask, dtype=bool)
    if result.shape != (frame[1], frame[0]) or not result.any():
        raise ValueError(f"{name} line mask is empty or out of frame")
    metadata = session.mask_metadata.get(name, {})
    if metadata.get("observation_id") != unit.id:
        raise ValueError(f"{name} line mask is not bound to the current A26 observation")
    if metadata.get("coordinate_frame") != "original_image_pixels":
        raise ValueError(f"{name} line mask is not in the original image frame")
    if not isinstance(metadata.get("fov_truncated_sides"), list) or metadata["fov_truncated_sides"]:
        raise ValueError(f"{name} line mask is truncated or lacks a completeness record")
    return result


def _reference_common(session, reference: dict, required: set[str]) -> dict:
    if not isinstance(reference, dict) or set(reference) != required:
        raise ValueError(f"reference must contain exactly {sorted(required)}")
    for key in ("reference_id", "source", "coordinate_frame", "image_sha256"):
        if not isinstance(reference[key], str) or not reference[key].strip():
            raise ValueError(f"reference {key} must be a non-empty string")
    if reference["coordinate_frame"] != "original_image_pixels":
        raise ValueError("reference must use original_image_pixels")
    image_sha = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
    if reference["image_sha256"] != image_sha:
        raise ValueError("reference is not bound to the current image")
    return reference


def t2_program(session, spec, outputs, inputs):
    unit = _device(outputs, "central_venous_catheter")
    width, height = _frame(session)
    target = _reference_common(session, inputs.get("cvc_target_reference"),
                               {"reference_id", "source", "coordinate_frame", "image_sha256", "point_xy"})
    tip = _point(unit.value["distal_tip_xy"], "CVC distal tip", width, height)
    target_point = _point(target["point_xy"], "CVC target point", width, height)
    line = _line_mask(session, unit, "Device_central_venous_catheter")
    distance = math.hypot(tip[0] - target_point[0], tip[1] - target_point[1])
    reference = add_observation(
        session, "cvc_target_reference", target, evidence_type="landmark",
        source_tool="caller_reference",
        validity_conditions=["caller-supplied anatomical target; not independently verified by A26"])
    value = {
        "device_presence": {"status": "present", "presence_raw_score": unit.value.get("presence_raw_score"),
                            "screening_threshold": unit.value.get("screening_threshold"),
                            "probability_calibrated": False},
        "cvc_distal_tip_xy_px": tip,
        "target_reference": {**target, "observation_id": reference.id},
        "tip_target_distance_px": distance,
        "position_evidence": {"status": "measurement_only", "clinical_position": None,
                               "limitations": ["target is caller-provided", "distance is in pixels",
                                               "no acceptable-position threshold is frozen"]},
    }
    support = [unit.id, reference.id]
    relation = add_relation(session, "cvc_tip_to_target", value, input_ids=support,
                            rule_id=f"{spec.function_id}:tip_point_distance:{spec.version}")
    return value, relation, support


def _polyline_points(reference: dict, width: int, height: int) -> list[list[float]]:
    points = reference["polyline_xy"]
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("diaphragm reference polyline needs at least two points")
    return [_point(point, "diaphragm polyline point", width, height) for point in points]


def _curve_y(points: list[list[float]], x: float) -> float | None:
    for first, second in zip(points, points[1:]):
        x0, x1 = first[0], second[0]
        if min(x0, x1) <= x <= max(x0, x1):
            if x1 == x0:
                return (first[1] + second[1]) / 2
            fraction = (x - x0) / (x1 - x0)
            return first[1] + fraction * (second[1] - first[1])
    return None


def _crossing_state(line: np.ndarray, points: list[list[float]]) -> str:
    """Require a local crossing along one connected 8-neighbour line path."""
    ys, xs = np.nonzero(line)
    if not len(xs):
        return "indeterminate_discontinuous_line"
    signed: dict[tuple[int, int], float] = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        curve_y = _curve_y(points, float(x))
        if curve_y is not None:
            signed[(y, x)] = y - curve_y
    if not signed:
        return "no_reference_overlap"

    for (y, x), distance in signed.items():
        if abs(distance) <= 1e-9:
            return "crosses"
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                other = signed.get((y + dy, x + dx))
                if other is not None and distance * other < 0:
                    return "crosses"

    # If the mask is broken, separate fragments on opposite sides do not
    # establish that the projected device path actually crossed the curve.
    from scipy.ndimage import label

    _, component_count = label(line, structure=np.ones((3, 3), dtype=bool))
    if component_count > 1:
        return "indeterminate_discontinuous_line"
    return "does_not_cross"


def t3_program(session, spec, outputs, inputs):
    unit = _device(outputs, "nasogastric_tube")
    width, height = _frame(session)
    reference = _reference_common(session, inputs.get("diaphragm_reference"),
                                 {"reference_id", "source", "coordinate_frame", "image_sha256", "polyline_xy"})
    points = _polyline_points(reference, width, height)
    line = _line_mask(session, unit, "Device_nasogastric_tube")
    crossing_state = _crossing_state(line, points)
    if crossing_state == "no_reference_overlap":
        raise ValueError("NGT line has no overlap with the diaphragm reference x-range")
    crosses = {"crosses": True, "does_not_cross": False}.get(crossing_state)
    endpoint = _point(unit.value["distal_tip_xy"], "NGT lowest visible endpoint", width, height)
    ref_node = add_observation(
        session, "diaphragm_reference", reference, evidence_type="landmark",
        source_tool="caller_reference",
        validity_conditions=["caller-supplied projected diaphragm reference; not A5 lung base"])
    value = {
        "device_presence": {"status": "present", "presence_raw_score": unit.value.get("presence_raw_score"),
                            "screening_threshold": unit.value.get("screening_threshold"),
                            "probability_calibrated": False},
        "ngt_line": {"mask_name": "Device_nasogastric_tube", "observation_id": unit.id},
        "lowest_visible_endpoint_xy_px": endpoint,
        "diaphragm_reference": {**reference, "observation_id": ref_node.id},
        "crosses_diaphragm": crosses,
        "crossing_status": crossing_state,
        "position_evidence": {"status": "projected_crossing_only", "clinical_position": None,
                               "true_tip_status": "not_available_from_A26_lowest_visible_endpoint",
                               "limitations": ["crossing is a 2D projection relation", "no acceptable-position threshold is frozen"]},
    }
    support = [unit.id, ref_node.id]
    relation = add_relation(session, "ngt_line_to_diaphragm", value, input_ids=support,
                            rule_id=f"{spec.function_id}:polyline_crossing:{spec.version}")
    return value, relation, support


DEVICE_PROGRAMS = {"T2": t2_program, "T3": t3_program}
