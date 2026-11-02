"""T4 chest-drain evidence adapter.

The current A26 model does not detect chest drains. T4 therefore accepts a
caller-supplied, image-bound drain path/side-hole/tip artifact and performs
only deterministic 2-D projection checks. The adapter is deliberately
source-explicit so a future drain-capable model can replace the artifact
producer without changing the function contract.
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
from ..perception.geometry import relate_masks


def _lung_units(outputs):
    units = [unit for values in outputs.values() for unit in values
             if unit.source_tool == "segment_lungs"]
    found = {unit.finding_or_anatomy: unit for unit in units
             if unit.finding_or_anatomy in {"Right_Lung", "Left_Lung"}}
    if set(found) != {"Right_Lung", "Left_Lung"}:
        raise ValueError("T4 requires current Right_Lung and Left_Lung observations")
    return found


def _point(value, name, width, height):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real)
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f"{name} must contain two finite coordinates")
    point = [float(value[0]), float(value[1])]
    if not (0 <= point[0] <= width and 0 <= point[1] <= height):
        raise ValueError(f"{name} is outside the original image frame")
    return point


def _mask(session, name, owner=None):
    raw = session.masks.get(name)
    if raw is None:
        raise ValueError(f"required mask {name} is missing")
    frame = inspect_image(session.image_ref)
    mask = np.asarray(raw, dtype=bool)
    if mask.shape != (frame["height"], frame["width"]) or not mask.any():
        raise ValueError(f"{name} mask is empty or outside the original image frame")
    meta = session.mask_metadata.get(name, {})
    if meta.get("coordinate_frame") != "original_image_pixels":
        raise ValueError(f"{name} mask is not in original_image_pixels")
    image_sha = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
    if meta.get("image_sha256") not in {None, image_sha}:
        raise ValueError(f"{name} mask is not bound to the current image")
    if not isinstance(meta.get("fov_truncated_sides"), list) or meta["fov_truncated_sides"]:
        raise ValueError(f"{name} mask is truncated or lacks a completeness record")
    if owner is not None and meta.get("observation_id") != owner.id:
        raise ValueError(f"{name} mask is not bound to the current observation")
    return mask, meta


def _source_reference(session, reference):
    required = {"reference_id", "source", "coordinate_frame", "image_sha256",
                "drain_mask_name", "tip_xy", "side_hole_xy"}
    if not isinstance(reference, dict) or set(reference) != required:
        raise ValueError(f"drain_reference must contain exactly {sorted(required)}")
    for key in ("reference_id", "source", "coordinate_frame", "image_sha256", "drain_mask_name"):
        if not isinstance(reference[key], str) or not reference[key].strip():
            raise ValueError(f"drain_reference {key} must be a non-empty string")
    if reference["coordinate_frame"] != "original_image_pixels":
        raise ValueError("drain_reference must use original_image_pixels")
    image_sha = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
    if reference["image_sha256"] != image_sha:
        raise ValueError("drain_reference is not bound to the current image")
    if not isinstance(reference["side_hole_xy"], list) or not reference["side_hole_xy"]:
        raise ValueError("drain_reference side_hole_xy must be a non-empty list")
    return reference


def _point_in_mask(point, mask):
    x, y = int(round(point[0])), int(round(point[1]))
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return False
    return bool(mask[y, x])


def t4_program(session, spec, outputs, inputs):
    reference = _source_reference(session, inputs.get("drain_reference"))
    frame = inspect_image(session.image_ref)
    width, height = frame["width"], frame["height"]
    lungs = _lung_units(outputs)
    lung_masks = {name: _mask(session, name, owner)[0] for name, owner in lungs.items()}
    drain_name = reference["drain_mask_name"]
    drain_mask, drain_meta = _mask(session, drain_name)
    drain_owner_id = drain_meta.get("observation_id")
    if not isinstance(drain_owner_id, str) or drain_owner_id not in session.graph.nodes:
        raise ValueError("drain mask lacks an observation owner")
    drain_owner = session.graph.nodes[drain_owner_id]
    if drain_owner.category != "observation":
        raise ValueError("drain mask owner is not an observation")
    tip = _point(reference["tip_xy"], "drain tip", width, height)
    side_holes = [_point(point, "side-hole", width, height)
                  for point in reference["side_hole_xy"]]
    if not _point_in_mask(tip, drain_mask):
        raise ValueError("drain tip is not on the supplied drain path mask")
    hole_records = [{"point_xy": point,
                     "on_path": _point_in_mask(point, drain_mask),
                     "image_side": ("image_left" if _point_in_mask(point, lung_masks["Right_Lung"])
                                     else "image_right" if _point_in_mask(point, lung_masks["Left_Lung"])
                                     else "outside_lungs")}
                    for point in side_holes]
    if not all(record["on_path"] for record in hole_records):
        raise ValueError("every side-hole reference must lie on the supplied drain path")
    relations = {
        image_side: relate_masks(drain_mask, lung_masks[name])
        for name, image_side in (("Right_Lung", "image_left"), ("Left_Lung", "image_right"))
    }
    source = add_observation(
        session, "chest_drain_reference", reference, evidence_type="region",
        source_tool="caller_reference",
        validity_conditions=["caller-supplied drain path, side-hole and tip; no automatic drain model is asserted"])
    value = {
        "device_presence": {"status": "present_by_reference", "probability_calibrated": False,
                            "source_observation_id": source.id},
        "drain_path": {"mask_name": drain_name, "observation_id": drain_owner_id,
                        "source": reference["source"], "area_px": int(drain_mask.sum())},
        "side_holes": {"count": len(hole_records), "points": hole_records},
        "distal_tip": {"point_xy": tip, "on_path": True},
        "pleural_projection_relation": {
            "by_image_side": relations,
            "scope": "2D projected drain-path to lung-mask relation",
            "clinical_position": None,
        },
        "applicability": {
            "status": "reference_adapter_only",
            "automatic_model": False,
            "clinical_interpretation": "not_assessed",
            "limitations": [
                "current A26 backend does not detect chest drains",
                "2D projection does not establish pleural-space or intrathoracic position",
                "side-hole annotations are caller-supplied and not independently validated",
            ],
        },
    }
    support = [source.id, drain_owner_id, *[unit.id for unit in lungs.values()]]
    relation = add_relation(session, "chest_drain_projected_relation", value,
                            input_ids=support,
                            rule_id=f"{spec.function_id}:reference_adapter:{spec.version}")
    return value, relation, support


DRAIN_PROGRAMS = {"T4": t4_program}
