"""G2/G3 spatial integration contracts over hand-authored masks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import add_observation, validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.session import EvidenceSession
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit
from visevi.functions.runner import IMPLEMENTED_FUNCTIONS, _PROGRAMS, run_function_episode
from visevi.perception.metrics import build_metric_tools


class FakeModelTool:
    def __init__(self, name, harness, properties=None):
        self.name = name
        self.harness = harness
        self.properties = {"output_dir": {"type": "string"}, **(properties or {})}

    def describe(self):
        return ToolSpec(self.name, "perception", "synthetic spatial test model",
                        parameter_schema(self.properties, ("output_dir",)),
                        evidence_result_schema())

    def run(self, session, **params):
        return getattr(self.harness, self.name)(session, params)


class SpatialHarness:
    def __init__(self, image):
        self.image = image
        self.run_index = 0
        self.candidate_box = [4.0, 4.0, 7.0, 12.0]
        self.candidate_status = "detected"
        self.truncate_right_lung = False
        self.registry = ToolRegistry()
        for tool in build_metric_tools():
            self.registry.register(tool)
        for name, properties in {
            "segment_lungs": {},
            "segment_heart": {},
            "segment_mediastinum": {},
            "detect_abnormalities": {
                "findings": {"type": "array", "items": {"type": "string"}}
            },
        }.items():
            self.registry.register(FakeModelTool(name, self, properties))

    def emit(self, session, tool, target, value):
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation", evidence_type="region",
            finding_or_anatomy=target, value=value, source_tool=tool,
            validity_conditions=["synthetic prediction; not a clinical reference"],
        )
        session.add_observation(unit)
        return unit

    @staticmethod
    def attach(session, name, mask, owner, **extra):
        session.attach_masks({name: mask}, {name: {
            "observation_id": owner.id,
            "coordinate_frame": "original_image_pixels",
            "fov_truncated_sides": [],
            "full_image_observed": True,
            **extra,
        }})

    def segment_lungs(self, session, params):
        units = []
        for name, columns in (("Right_Lung", slice(2, 8)), ("Left_Lung", slice(12, 18))):
            mask = np.zeros((18, 20), dtype=bool)
            mask[2:16, columns] = True
            owner = self.emit(session, "segment_lungs", name, {"area_px": int(mask.sum())})
            sides = ["right"] if self.truncate_right_lung and name == "Right_Lung" else []
            self.attach(session, name, mask, owner, fov_truncated_sides=sides)
            units.append(owner)
        return units

    def detect_abnormalities(self, session, params):
        finding = params["findings"][0]
        boxes = [self.candidate_box] if self.candidate_status == "detected" else []
        owner = self.emit(session, "detect_abnormalities", f"candidate_{finding}", {
            "finding": finding, "status": self.candidate_status, "boxes_xyxy_px": boxes,
        })
        if boxes:
            from visevi.perception.vlm import box_mask
            name = "Candidate_" + finding.replace(" ", "_")
            mask = box_mask(boxes, width=20, height=18)
            self.attach(session, name, mask, owner,
                        source="A6 detect_abnormalities box union", image_sha256=
                        hashlib.sha256(Path(self.image).read_bytes()).hexdigest())
        return [owner]

    def segment_heart(self, session, params):
        mask = np.zeros((18, 20), dtype=bool)
        mask[6:12, 8:12] = True
        owner = self.emit(session, "segment_heart", "Heart", {"area_px": int(mask.sum())})
        self.attach(session, "Heart", mask, owner)
        return [owner]

    def segment_mediastinum(self, session, params):
        mask = np.zeros((18, 20), dtype=bool)
        mask[4:15, 8:12] = True
        owner = self.emit(session, "segment_mediastinum", "Mediastinum", {"area_px": int(mask.sum())})
        self.attach(session, "Mediastinum", mask, owner)
        return [owner]

    def run(self, function_id, tmp_path, *, inputs=None, session=None):
        self.run_index += 1
        return run_function_episode(
            function_id, image_ref=None if session else str(self.image), session=session,
            registry=self.registry, inputs=inputs, out_dir=tmp_path / f"run-{self.run_index}",
        )


@pytest.fixture
def harness(tmp_path):
    image = tmp_path / "synthetic.png"
    Image.new("L", (20, 18), 90).save(image)
    return SpatialHarness(image)


def _external_region(session, image, name, mask, source):
    owner = add_observation(session, name, {"source": source}, source_tool="caller_annotation",
                            evidence_type="region",
                            validity_conditions=["synthetic caller-supplied region"])
    session.attach_masks({name: mask}, {name: {
        "observation_id": owner.id,
        "coordinate_frame": "original_image_pixels",
        "fov_truncated_sides": [],
        "full_image_observed": True,
        "source": source,
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }})


def test_g2_reports_multiple_zones_and_image_side_with_bound_provenance(harness, tmp_path):
    assert {"G2", "G3"} <= set(IMPLEMENTED_FUNCTIONS)
    assert {"G2", "G3"} <= set(_PROGRAMS)
    result = harness.run("G2", tmp_path, inputs={"finding": "nodule"})
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["lung_side"] == ["image_left"]
    assert output["patient_sides"] is None
    assert output["upper_middle_lower_zones"]["image_left"] == {
        "upper": 9, "middle": 12, "lower": 3,
    }
    assert output["upper_middle_lower_zones"]["image_right"] == {
        "upper": 0, "middle": 0, "lower": 0,
    }
    assert "not pulmonary lobes" in " ".join(output["applicability"]["limitations"])
    relation = result.session.graph.nodes[result.functions[0].evidence_refs.relation_ids[0]]
    assert {result.session.graph.nodes[node_id].source_tool for node_id in relation.provenance} == {
        "segment_lungs", "detect_abnormalities", "assign_lung_zones",
    }
    assert validate_graph(result.session) == []


def test_g2_maps_to_patient_side_only_after_verified_orientation(harness, tmp_path):
    result = harness.run("G2", tmp_path, inputs={
        "finding": "nodule", "orientation_verified": True,
        "patient_right_is_image_left": True, "orientation_source": "verified synthetic marker",
    })
    assert result.status == "ok"
    assert result.functions[0].outputs["patient_sides"] == ["patient_right"]
    relation = result.session.graph.nodes[result.functions[0].evidence_refs.relation_ids[0]]
    assert any(result.session.graph.nodes[node_id].source_tool == "verified_orientation_input"
               for node_id in relation.provenance)


def test_g2_refuses_truncated_lung_and_unparsed_or_absent_candidates(harness, tmp_path):
    harness.truncate_right_lung = True
    truncated = harness.run("G2", tmp_path, inputs={"finding": "nodule"})
    assert truncated.status == "refused"
    assert "field-of-view" in truncated.functions[0].refusal.reason

    harness.truncate_right_lung = False
    harness.candidate_status = "not_detected"
    missing = harness.run("G2", tmp_path, inputs={"finding": "nodule"})
    assert missing.status == "refused"
    assert not missing.session.graph.by_category("claim")


def test_g2_refuses_candidates_outside_both_lungs(harness, tmp_path):
    harness.candidate_box = [0.0, 0.0, 1.0, 1.0]
    result = harness.run("G2", tmp_path, inputs={"finding": "nodule"})
    assert result.status == "refused"
    assert "no overlap with either lung" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


def test_g3_reports_only_projected_relations_and_uses_all_region_owners(harness, tmp_path):
    session = EvidenceSession(image_ref=str(harness.image))
    lesion = np.zeros((18, 20), dtype=bool)
    lesion[5:8, 4:7] = True
    _external_region(session, harness.image, "Lesion", lesion, "synthetic hand annotation")
    inputs = {
        "lesion_mask_name": "Lesion",
        "anatomy_mask_name": "Heart",
        "region_mask_names": ["Lesion", "Heart", "Right_Lung", "Left_Lung", "Mediastinum"],
    }
    result = harness.run("G3", tmp_path, inputs=inputs, session=session)
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["interpretation_scope"] == "2D projected pixel-cell topology only"
    assert output["containment_2d"]["lesion_in_anatomy"] is False
    assert output["gap_distance_px"] == 1.0
    assert len(output["pairwise_relations_2d"]) == 10
    assert all(item["topology"] in {"edge_adjacent", "corner_adjacent"}
               for item in output["adjacency_2d"])
    relation = result.session.graph.nodes[result.functions[0].evidence_refs.relation_ids[0]]
    assert {result.session.graph.nodes[node_id].source_tool for node_id in relation.provenance} == {
        "segment_lungs", "segment_heart", "segment_mediastinum",
        "relate_lesion_to_anatomy", "relate_multiple_regions", "caller_annotation",
    }
    spatial_observations = [node for node in result.session.graph.nodes.values()
                            if node.source_tool in {"relate_lesion_to_anatomy",
                                                    "relate_multiple_regions"}]
    assert any(result.session.graph.nodes[node_id].source_tool == "caller_annotation"
               for node_id in relation.provenance)
    assert all(any(result.session.graph.nodes[node_id].source_tool == "caller_annotation"
                   for node_id in node.provenance) for node in spatial_observations)
    assert result.functions[0].outputs["region_sources"]["Lesion"] == "synthetic hand annotation"
    assert validate_graph(result.session) == []


def test_g3_refuses_unbound_external_region_source(harness, tmp_path):
    session = EvidenceSession(image_ref=str(harness.image))
    lesion = np.zeros((18, 20), dtype=bool)
    lesion[5:8, 4:7] = True
    _external_region(session, harness.image, "Lesion", lesion, "synthetic hand annotation")
    session.mask_metadata["Lesion"].pop("image_sha256")
    result = harness.run("G3", tmp_path, inputs={
        "lesion_mask_name": "Lesion", "anatomy_mask_name": "Heart",
        "region_mask_names": ["Lesion", "Heart", "Right_Lung", "Left_Lung", "Mediastinum"],
    }, session=session)
    assert result.status == "refused"
    assert "current image" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")
