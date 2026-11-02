"""Evidence programs for the spatial G2/G3 functions.

These programs keep the result at the level supported by the underlying
geometry tools: image-side lung zones and two-dimensional projected spatial
relations. They do not infer pulmonary lobes, patient laterality, or three-
dimensional occlusion from a projection.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..backends import inspect_image
from ..construction.evidence import add_observation, add_relation
from ..core.units import EvidenceUnit


def units_for_tool(outputs: dict[str, list[EvidenceUnit]], tool: str) -> list[EvidenceUnit]:
    return [unit for values in outputs.values() for unit in values
            if unit.source_tool == tool]


def _one(outputs: dict[str, list[EvidenceUnit]], tool: str) -> EvidenceUnit:
    units = units_for_tool(outputs, tool)
    if len(units) != 1:
        raise ValueError(f"expected exactly one {tool} observation")
    return units[0]


def _lung_units(outputs: dict[str, list[EvidenceUnit]]) -> dict[str, EvidenceUnit]:
    units = units_for_tool(outputs, "segment_lungs")
    found = {unit.finding_or_anatomy: unit for unit in units
             if unit.finding_or_anatomy in {"Right_Lung", "Left_Lung"}}
    if set(found) != {"Right_Lung", "Left_Lung"} or len(units) != 2:
        raise ValueError("G2/G3 require exactly one Right_Lung and one Left_Lung observation")
    return found


def _require_current_mask(session, name: str, owner: EvidenceUnit | None = None) -> np.ndarray:
    mask = session.masks.get(name)
    if mask is None:
        raise ValueError(f"required mask {name} is missing")
    frame = inspect_image(session.image_ref)
    array = np.asarray(mask, dtype=bool)
    if array.shape != (frame["height"], frame["width"]) or not array.any():
        raise ValueError(f"{name} is empty or not in the original image frame")
    if session.fov_truncated([name]):
        raise ValueError(f"{name} is truncated by the model field of view")
    metadata = session.mask_metadata.get(name, {})
    observation_id = metadata.get("observation_id")
    if not isinstance(observation_id, str) or observation_id not in session.graph.nodes:
        raise ValueError(f"{name} has no observation owner")
    if session.graph.nodes[observation_id].category != "observation":
        raise ValueError(f"{name} owner is not an observation")
    if owner is not None and observation_id != owner.id:
        raise ValueError(f"{name} is not bound to the current observation")
    if metadata.get("coordinate_frame") != "original_image_pixels":
        raise ValueError(f"{name} is not in the original image coordinate frame")
    if not isinstance(metadata.get("fov_truncated_sides"), list):
        raise ValueError(f"{name} lacks a field-of-view completeness record")
    return array


def _candidate(session, outputs, finding: str):
    # Imported lazily because runner registers these programs at module load.
    from .runner import _candidate_binding
    candidate = _one(outputs, "detect_abnormalities")
    mask_name, boxes = _candidate_binding(session, candidate, finding)
    return candidate, mask_name, boxes


def _verify_metric_binding(session, unit: EvidenceUnit, mask_names: list[str], parameters: dict) -> None:
    if unit.value.get("parameters") != parameters:
        raise ValueError(f"{unit.source_tool} parameters disagree with the requested masks")
    snapshots = unit.value.get("input_masks")
    if not isinstance(snapshots, dict) or set(snapshots) != set(mask_names):
        raise ValueError(f"{unit.source_tool} input mask snapshot is incomplete")
    for name in mask_names:
        mask = _require_current_mask(session, name)
        metadata = session.mask_metadata[name]
        owner = metadata["observation_id"]
        snapshot = snapshots[name]
        digest = hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()
        if (owner not in unit.provenance
                or snapshot.get("binary_sha256") != digest
                or snapshot.get("source") != metadata
                or snapshot.get("shape_hw") != list(mask.shape)):
            raise ValueError(f"{unit.source_tool} is not bound to the current {name} mask and source")


def _image_sides(zones: dict) -> list[str]:
    return [image_side for mask_side, image_side in
            (("right", "image_left"), ("left", "image_right"))
            if zones[mask_side].get("overlap_px", 0) > 0]


def _patient_sides(image_sides: list[str], inputs: dict) -> list[str] | None:
    if not inputs.get("orientation_verified"):
        return None
    mapping = (({"image_left": "patient_right", "image_right": "patient_left"})
               if inputs["patient_right_is_image_left"] else
               ({"image_left": "patient_left", "image_right": "patient_right"}))
    return [mapping[side] for side in image_sides]


def g2_program(session, spec, outputs, inputs):
    finding = inputs.get("finding")
    if not isinstance(finding, str) or not finding.strip():
        raise ValueError("G2 requires a finding")
    candidate, mask_name, boxes = _candidate(session, outputs, finding)
    lungs = _lung_units(outputs)
    for side, unit in lungs.items():
        _require_current_mask(session, side, unit)
    if float(np.nonzero(session.masks["Right_Lung"])[1].mean()) >= float(
            np.nonzero(session.masks["Left_Lung"])[1].mean()):
        raise ValueError("A1 Right_Lung/Left_Lung labels disagree with the image-axis convention")
    _require_current_mask(session, mask_name, candidate)
    zones = _one(outputs, "assign_lung_zones")
    metric_masks = ["Right_Lung", "Left_Lung", mask_name]
    _verify_metric_binding(session, zones, metric_masks,
                           {"region_mask_name": mask_name})
    if not {unit.id for unit in lungs.values()}.issubset(set(zones.provenance)):
        raise ValueError("A20 result is not bound to both current lung observations")
    if candidate.id not in zones.provenance:
        raise ValueError("A20 result is not bound to the current candidate observation")
    value = zones.value
    if value.get("status") != "ok":
        raise ValueError("candidate has no overlap with either lung")
    for side in ("right", "left"):
        record = value.get(side)
        if not isinstance(record, dict) or not isinstance(record.get("zone_overlap_px"), dict):
            raise ValueError(f"A20 {side} zone output is malformed")
        if set(record["zone_overlap_px"]) != {"upper", "middle", "lower"}:
            raise ValueError(f"A20 {side} zone output is incomplete")
    image_sides = _image_sides(value)
    if not image_sides:
        raise ValueError("candidate does not overlap either lung")
    support = [unit.id for values in outputs.values() for unit in values]
    if inputs.get("orientation_verified"):
        orientation = add_observation(
            session, "patient_image_orientation",
            {"patient_right_is_image_left": inputs["patient_right_is_image_left"],
             "orientation_source": inputs["orientation_source"], "verified": True},
            evidence_type="global_condition", source_tool="verified_orientation_input",
            validity_conditions=["trusted caller-supplied orientation record"])
        support.append(orientation.id)
    result = {
        "finding": finding,
        "candidate_boxes": boxes,
        "lung_side": image_sides,
        "patient_sides": _patient_sides(image_sides, inputs),
        "upper_middle_lower_zones": {
            image_side: value[mask_side]["zone_overlap_px"]
            for mask_side, image_side in (("right", "image_left"),
                                          ("left", "image_right"))
        },
        "zone_overlap": {
            image_side: value[mask_side]
            for mask_side, image_side in (("right", "image_left"),
                                          ("left", "image_right"))
        },
        "coordinate_frame": "original_image_pixels",
        "applicability": {
            "status": "limited" if value.get("outside_lung_overlap_px", 0) else "applicable",
            "outside_lung_overlap_px": value.get("outside_lung_overlap_px", 0),
            "zone_convention": "three equal vertical bands of each lung extent",
            "clinical_interpretation": "not_assessed",
            "patient_side_available": bool(inputs.get("orientation_verified")),
            "limitations": [
                "zones are geometric image regions, not pulmonary lobes",
                "candidate is a union of detector boxes, not a lesion segmentation",
            ],
        },
    }
    relation = add_relation(session, "candidate_lung_zone", result,
                            input_ids=support,
                            rule_id=f"{spec.function_id}:lung_zones:{spec.version}")
    return result, relation, support


def _external_mask_source(session, name: str) -> tuple[np.ndarray, str]:
    mask = _require_current_mask(session, name)
    metadata = session.mask_metadata[name]
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{name} requires a traceable caller-supplied source")
    owner = session.graph.nodes[metadata["observation_id"]]
    if (owner.category != "observation" or owner.finding_or_anatomy != name
            or not owner.source_tool.strip()):
        raise ValueError(f"{name} source owner does not match the named region")
    image_sha = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
    if metadata.get("image_sha256") != image_sha:
        raise ValueError(f"{name} source is not bound to the current image")
    return mask, source


def _region_mask_source(session, name: str, known_owners: dict[str, EvidenceUnit]) -> str:
    if name in known_owners:
        _require_current_mask(session, name, known_owners[name])
        return known_owners[name].source_tool
    _, source = _external_mask_source(session, name)
    return source


def g3_program(session, spec, outputs, inputs):
    lesion_name = inputs.get("lesion_mask_name")
    anatomy_name = inputs.get("anatomy_mask_name")
    region_names = inputs.get("region_mask_names")
    if (not isinstance(lesion_name, str) or not isinstance(anatomy_name, str)
            or lesion_name == anatomy_name):
        raise ValueError("G3 requires distinct lesion_mask_name and anatomy_mask_name")
    if (not isinstance(region_names, list) or len(region_names) < 2
            or len(region_names) != len(set(region_names))):
        raise ValueError("G3 requires at least two unique region_mask_names")
    if lesion_name not in region_names or anatomy_name not in region_names:
        raise ValueError("G3 region_mask_names must include the lesion and anatomy masks")
    lungs = _lung_units(outputs)
    for side, unit in lungs.items():
        _require_current_mask(session, side, unit)
    heart = _one(outputs, "segment_heart")
    mediastinum = _one(outputs, "segment_mediastinum")
    _require_current_mask(session, "Heart", heart)
    _require_current_mask(session, "Mediastinum", mediastinum)
    known_owners = {**lungs, "Heart": heart, "Mediastinum": mediastinum}
    required_regions = {lesion_name, anatomy_name, "Right_Lung", "Left_Lung",
                        "Heart", "Mediastinum"}
    if not required_regions.issubset(set(region_names)):
        raise ValueError("A22 region_mask_names must include lesion, target anatomy, both lungs, heart and mediastinum")
    region_owners = {name: session.mask_metadata.get(name, {}).get("observation_id")
                     for name in region_names}
    region_sources = {name: _region_mask_source(session, name, known_owners)
                      for name in region_names}
    lesion_relation = _one(outputs, "relate_lesion_to_anatomy")
    _verify_metric_binding(session, lesion_relation, [lesion_name, anatomy_name],
                           {"lesion_mask_name": lesion_name,
                            "anatomy_mask_name": anatomy_name})
    pairwise = _one(outputs, "relate_multiple_regions")
    _verify_metric_binding(session, pairwise, region_names,
                           {"region_mask_names": region_names})
    required_ids = {owner for owner in region_owners.values() if isinstance(owner, str)}
    if not required_ids.issubset(set(pairwise.provenance)):
        raise ValueError("A22 result is not bound to the current anatomy observations")
    lesion_owner = session.mask_metadata[lesion_name]["observation_id"]
    anatomy_owner = session.mask_metadata[anatomy_name]["observation_id"]
    if not {lesion_owner, anatomy_owner}.issubset(set(lesion_relation.provenance)):
        raise ValueError("A21 result is not bound to the caller lesion and target anatomy masks")
    pairwise_relations = pairwise.value.get("relations")
    expected_pairs = {frozenset((first, second))
                      for index, first in enumerate(region_names)
                      for second in region_names[index + 1:]}
    observed_pairs = ({frozenset((item.get("region_a"), item.get("region_b")))
                       for item in pairwise_relations if isinstance(item, dict)}
                      if isinstance(pairwise_relations, list) else set())
    if len(pairwise_relations) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError("A22 pairwise relations do not cover exactly the requested regions")
    adjacency = [item for item in pairwise_relations
                 if item.get("topology") in {"edge_adjacent", "corner_adjacent"}]
    support = [unit.id for values in outputs.values() for unit in values]
    support.extend(sorted(required_ids))
    result = {
        "lesion_mask_name": lesion_name,
        "anatomy_mask_name": anatomy_name,
        "region_sources": region_sources,
        "overlap_2d": lesion_relation.value,
        "containment_2d": {
            "lesion_in_anatomy": lesion_relation.value.get("b_covers_a", False),
            "anatomy_in_lesion": lesion_relation.value.get("a_covers_b", False),
        },
        "adjacency_2d": adjacency,
        "pairwise_relations_2d": pairwise_relations,
        "gap_distance_px": lesion_relation.value.get("gap_distance_px"),
        "image_axis_direction": {
            "horizontal": lesion_relation.value.get("relative_image_horizontal"),
            "vertical": lesion_relation.value.get("relative_image_vertical"),
        },
        "interpretation_scope": "2D projected pixel-cell topology only",
        "coordinate_frame": "original_image_pixels",
        "applicability": {
            "status": "applicable",
            "clinical_interpretation": "not_assessed",
            "limitations": [
                "2D overlap does not establish 3D occlusion, invasion, or causality",
                "caller-supplied region masks remain predictions or annotations according to their source",
            ],
        },
    }
    relation = add_relation(session, "projected_spatial_relation", result,
                            input_ids=support,
                            rule_id=f"{spec.function_id}:projected_relation:{spec.version}")
    return result, relation, support


SPATIAL_PROGRAMS = {"G2": g2_program, "G3": g3_program}
