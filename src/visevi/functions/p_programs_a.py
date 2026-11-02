"""Experimental evidence programs for P1, P2, P4 and P5.

These combine observable model outputs and deterministic measurements without
inventing disease probabilities, severity grades or clinical interpretations.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Real

import numpy as np

from ..backends import inspect_image
from ..construction.evidence import add_relation


def _step(outputs, step_id, count=1):
    units = outputs.get(step_id, [])
    if len(units) != count:
        raise ValueError(f"{step_id} requires exactly {count} observations")
    return units[0] if count == 1 else units


def _candidate(session, outputs, finding):
    # Import lazily to avoid a module cycle when runner registers these programs.
    from .runner import _candidate_binding

    unit = _step(outputs, "candidates")
    name, boxes = _candidate_binding(session, unit, finding)
    return unit, name, boxes


def _mask_hash(session, mask_name):
    mask = session.masks.get(mask_name)
    if mask is None:
        raise ValueError(f"required mask {mask_name} is missing")
    # Deterministic metrics and A18 both hash the normalized boolean mask.
    from ..perception.geometry import _mask
    return hashlib.sha256(np.ascontiguousarray(_mask(mask, mask_name)).tobytes()).hexdigest()


def _roi(session, outputs, step_id, mask_name, candidate):
    unit = _step(outputs, step_id)
    if (unit.value.get("parameters", {}).get("mask_name") != mask_name
            or candidate.id not in unit.provenance):
        raise ValueError(f"{step_id} is not bound to the current candidate mask")
    snapshot = unit.value.get("input_masks", {}).get(mask_name, {})
    if (snapshot.get("binary_sha256") != _mask_hash(session, mask_name)
            or snapshot.get("source", {}).get("observation_id") != candidate.id):
        raise ValueError(f"{step_id} input mask digest or owner disagrees with the current mask")
    return unit


def _attributes(session, outputs, mask_name, candidate, expected):
    units = outputs.get("attributes", [])
    if len(units) != len(expected):
        raise ValueError("required ROI attributes are missing or duplicated")
    found = {}
    for unit in units:
        attribute = unit.value.get("attribute")
        if attribute not in expected or attribute in found:
            raise ValueError("ROI attributes differ from the declared attributes")
        if (unit.value.get("roi_mask_name") != mask_name
                or candidate.id not in unit.provenance):
            raise ValueError("ROI attribute is not bound to the current candidate mask")
        if unit.value.get("roi_mask_sha256") != _mask_hash(session, mask_name):
            raise ValueError("ROI attribute mask digest disagrees with the current candidate mask")
        if unit.value.get("answer") not in {"present", "absent", "uncertain"}:
            raise ValueError("ROI attribute answer is unparsed or malformed")
        found[attribute] = unit
    return found


def _points(session, outputs, names, polyline=False):
    units = outputs.get("landmarks", [])
    frame = inspect_image(session.image_ref)
    found = {}
    for name in sorted(names):
        matches = [u for u in units if u.finding_or_anatomy == name]
        if len(matches) != 1:
            raise ValueError(f"missing or ambiguous landmark {name}")
        unit = matches[0]
        value = unit.value
        if polyline:
            coords = value.get("polyline_xy")
            if not isinstance(coords, list) or len(coords) < 2:
                raise ValueError(f"invalid lung base contour {name}")
        else:
            if value.get("status") != "ok":
                raise ValueError(f"landmark {name} is unavailable or outside image")
            coords = [value.get("point_xy")]
        for xy in coords:
            if (not isinstance(xy, (list, tuple)) or len(xy) != 2
                    or any(isinstance(v, bool) or not isinstance(v, Real)
                           or not math.isfinite(v) for v in xy)
                    or not (-0.5 <= xy[0] <= frame["width"] - 0.5)
                    or not (-0.5 <= xy[1] <= frame["height"] - 0.5)):
                raise ValueError(f"{name} contains invalid or out-of-frame coordinates")
        found[name] = unit
    return found


def _relation(session, spec, value, units, label):
    ids = list(dict.fromkeys(unit.id for unit in units))
    relation = add_relation(session, label, value, input_ids=ids,
                            rule_id=f"{spec.function_id}:descriptive_evidence:{spec.version}")
    return value, relation, ids


def _appearance_program(session, spec, outputs, finding, attr_names, use_intensity):
    candidate, mask_name, boxes = _candidate(session, outputs, finding)
    texture = _roi(session, outputs, "texture", mask_name, candidate)
    attributes = _attributes(session, outputs, mask_name, candidate, attr_names)
    observations = [candidate, texture, *attributes.values()]
    appearance = {"texture": texture.value}
    if use_intensity:
        intensity = _roi(session, outputs, "intensity", mask_name, candidate)
        observations.append(intensity)
        appearance["intensity"] = intensity.value
    attr_values = {key: unit.value for key, unit in attributes.items()}
    case = {"finding": finding, "detector_status": "detected",
            "diagnosis": "not_assessed", "disease_absence_established": False}
    # The detector and attribute model are the same backend and need not address
    # the same concept (especially broad ILD vs reticular opacity). Do not fuse.
    conflict = {"status": "not_assessed", "reason": "no frozen independent channel comparison rule"}
    value = {"case_evidence": case, "candidate_boxes": boxes,
             "attributes": attr_values, "conflict": conflict,
             "limitations": ["box union is not lesion segmentation",
                             "ROI statistics and attribute answers are not a diagnosis",
                             "model outputs are uncalibrated and may share a source"]}
    if use_intensity:
        value["roi_appearance"] = appearance
    else:
        value["phenotype"] = {"detector_target": finding, "attribute_target": "reticular opacity",
                              "equivalence_established": False}
        value["texture"] = texture.value
    return _relation(session, spec, value, observations, "candidate_appearance_attributes")


def p1_program(session, spec, outputs, inputs):
    return _appearance_program(session, spec, outputs, "consolidation",
                               {"consolidation", "air bronchogram"}, True)


def p2_program(session, spec, outputs, inputs):
    return _appearance_program(session, spec, outputs, "interstitial lung disease",
                               {"reticular opacity"}, False)


def p4_program(session, spec, outputs, inputs):
    bases = _points(session, outputs, {"right_lung_base_contour", "left_lung_base_contour"}, polyline=True)
    inspiration = [u for u in outputs.get("quality", []) if u.finding_or_anatomy == "inspiration"]
    if len(inspiration) != 1 or inspiration[0].value.get("status") != "ok":
        raise ValueError("inspiration rib-count context is missing or unknown")
    rib_counts = inspiration[0].value
    for side in ("posterior", "anterior"):
        data = rib_counts.get(side)
        if not isinstance(data, dict):
            raise ValueError(f"{side} rib-count context is missing")
        ribs, count = data.get("ribs_above"), data.get("count_above")
        if (not isinstance(ribs, list) or any(type(i) is not int or not 1 <= i <= 12 for i in ribs)
                or len(set(ribs)) != len(ribs) or type(count) is not int or count != len(ribs)
                or data.get("highest_number_above") != (max(ribs) if ribs else None)):
            raise ValueError(f"{side} rib-count context is internally inconsistent")
    dome = rib_counts.get("right_hemidiaphragm_dome_xy")
    frame = inspect_image(session.image_ref)
    if (not isinstance(dome, (tuple, list)) or len(dome) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in dome)
            or not (0 <= dome[0] < frame["width"] and 0 <= dome[1] < frame["height"])):
        raise ValueError("right hemidiaphragm dome is invalid or outside the image")
    lung_units = outputs.get("lungs", [])
    lungs = {}
    grayscale = {}
    support = [*bases.values(), inspiration[0]]
    for side in ("Right", "Left"):
        name = f"{side}_Lung"
        matches = [u for u in lung_units if u.finding_or_anatomy == name]
        if len(matches) != 1 or session.mask_metadata.get(name, {}).get("observation_id") != matches[0].id:
            raise ValueError(f"{name} is not bound to the current lung observation")
        mask = np.asarray(session.masks.get(name), dtype=bool)
        frame = inspect_image(session.image_ref)
        if mask.shape != (frame["height"], frame["width"]) or not mask.any() or session.fov_truncated([name]):
            raise ValueError(f"{name} is empty, truncated or outside the image frame")
        intensity = _roi(session, outputs, f"intensity_{side.lower()}", name, matches[0])
        lungs[side.lower()] = matches[0]
        grayscale[side.lower()] = intensity.value
        support.extend([matches[0], intensity])
    candidate_units = outputs.get("candidates", [])
    candidate = {"status": "not_run"}
    if candidate_units:
        if len(candidate_units) != 1 or candidate_units[0].value.get("finding") != "emphysema":
            raise ValueError("invalid emphysema candidate evidence")
        unit = candidate_units[0]
        status = unit.value.get("status")
        if status == "detected":
            _, _, boxes = _candidate(session, outputs, "emphysema")
            candidate = {"status": status, "boxes_xyxy_px": boxes}
        elif status in {"not_detected", "unparsed"}:
            candidate = {"status": status, "boxes_xyxy_px": []}
        else:
            raise ValueError("invalid emphysema candidate status")
        support.append(unit)
    value = {
        "lung_base_geometry": {name: unit.value for name, unit in bases.items()},
        "rib_counts": rib_counts,
        "grayscale_statistics": grayscale,
        "case_evidence": {"emphysema_candidate": candidate,
                          "emphysema_diagnosis": "not_assessed",
                          "hyperinflation_diagnosis": "not_assessed"},
        "applicability": {"status": "limited", "clinical_grade": None,
                          "limitations": ["lung base contour is predicted, not a diaphragm grade",
                                          "display gray depends on processing",
                                          "rib counts and emphysema detection have no frozen disease rule"]},
    }
    return _relation(session, spec, value, support, "hyperinflation_descriptive_context")


def p5_program(session, spec, outputs, inputs):
    landmarks = _points(session, outputs, {"right_costophrenic_angle", "left_costophrenic_angle"})
    candidate, mask_name, boxes = _candidate(session, outputs, "pleural effusion")
    intensity = _roi(session, outputs, "intensity", mask_name, candidate)
    sharpness = _roi(session, outputs, "sharpness", mask_name, candidate)
    value = {
        "costophrenic_landmarks": {name: unit.value for name, unit in landmarks.items()},
        "candidate_boxes": boxes,
        "roi_appearance": {"intensity": intensity.value, "sharpness": sharpness.value},
        "case_evidence": {"finding": "pleural effusion", "detector_status": "detected",
                          "diagnosis": "not_assessed", "disease_absence_established": False},
        "conflict": {"status": "not_assessed", "reason": "no frozen angle or effusion comparison rule"},
        "limitations": ["costophrenic points do not establish blunting",
                        "box union is not an effusion contour", "gray and sharpness are not disease grades"],
    }
    return _relation(session, spec, value, [*landmarks.values(), candidate, intensity, sharpness],
                     "effusion_candidate_and_landmarks")


P_PROGRAMS_A = {'P1': p1_program, 'P2': p2_program, 'P4': p4_program, 'P5': p5_program}
