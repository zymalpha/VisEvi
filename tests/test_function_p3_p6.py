"""P3/P6 representative function programs over hand-placed synthetic masks."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit
from visevi.functions.runner import run_function_episode
from visevi.perception.metrics import build_metric_tools


class FakeTool:
    def __init__(self, name, harness, properties=None):
        self.name, self.harness = name, harness
        self.properties = {"output_dir": {"type": "string"}, "device": {"type": "string"},
                           **(properties or {})}

    def describe(self):
        return ToolSpec(self.name, "perception", "synthetic test model",
                        parameter_schema(self.properties, ("output_dir",)), evidence_result_schema())

    def run(self, session, **params):
        self.harness.calls.append((self.name, params))
        return getattr(self.harness, self.name)(session, params)


class Harness:
    def __init__(self, image):
        self.image = image
        self.calls = []
        self.a6_status = "detected"
        self.a8_region = True
        self.a8_raw_score = 0.82
        self.a8_max_probability = 0.91
        self.a8_area_override = None
        self.candidate_outside_lungs = False
        self.candidate_mask_mismatch = False
        self.registry = ToolRegistry()
        for tool in build_metric_tools():
            self.registry.register(tool)
        self.registry.register(FakeTool("segment_lungs", self))
        self.registry.register(FakeTool(
            "detect_abnormalities", self,
            {"findings": {"type": "array", "items": {"type": "string"}}}))
        self.registry.register(FakeTool("segment_pneumothorax", self))

    def emit(self, session, tool, target, value, evidence_type="region", provenance=()):
        unit = EvidenceUnit(session.next_id("obs"), "observation", evidence_type, target,
                            value, source_tool=tool, provenance=list(provenance),
                            validity_conditions=["synthetic prediction; not a clinical reference"])
        session.add_observation(unit)
        return unit

    @staticmethod
    def attach(session, name, mask, owner):
        session.attach_masks(
            {name: mask},
            {name: {"observation_id": owner.id, "coordinate_frame": "original_image_pixels",
                    "fov_truncated_sides": [], "full_image_observed": True,
                    "is_ground_truth": False}})

    def segment_lungs(self, session, params):
        units = []
        for name, columns in (("Right_Lung", slice(2, 8)), ("Left_Lung", slice(12, 18))):
            mask = np.zeros((18, 20), dtype=bool)
            mask[2:17, columns] = True
            owner = self.emit(session, "segment_lungs", name, {"area_px": int(mask.sum())})
            self.attach(session, name, mask, owner)
            units.append(owner)
        return units

    def detect_abnormalities(self, session, params):
        finding = params["findings"][0]
        if self.candidate_outside_lungs:
            box = [8.0, 4.0, 11.0, 7.0]
        elif finding == "pneumothorax":
            box = [2.0, 6.0, 4.0, 14.0]
        else:
            box = [3.0, 4.0, 6.0, 7.0]
        boxes = [box] if self.a6_status == "detected" else []
        owner = self.emit(session, "detect_abnormalities", f"candidate_{finding}",
                          {"finding": finding, "status": self.a6_status,
                           "boxes_xyxy_px": boxes, "raw_text": self.a6_status})
        if boxes:
            mask = np.zeros((18, 20), dtype=bool)
            x0, y0, x1, y1 = map(int, box)
            mask[y0:y1, x0:x1] = True
            if self.candidate_mask_mismatch:
                mask = np.roll(mask, 1, axis=1)
            self.attach(session, "Candidate_" + finding.replace(" ", "_"), mask, owner)
        return [owner]

    def segment_pneumothorax(self, session, params):
        status = "region_above_threshold" if self.a8_region else "no_region_above_threshold"
        mask = np.zeros((18, 20), dtype=bool)
        if self.a8_region:
            mask[6:14, 2:4] = True
        value = {"case_raw_score": self.a8_raw_score, "mask_threshold": 0.5,
                 "max_pixel_probability": self.a8_max_probability,
                 "area_px": int(mask.sum()) if self.a8_area_override is None else self.a8_area_override,
                 "status": status,
                 "score_meaning": "raw sigmoid outputs; not calibrated probabilities"}
        if mask.any():
            value["bbox_inclusive_px"] = {"x0": 2, "y0": 6, "x1": 3, "y1": 13,
                                           "w": 2, "h": 8}
        owner = self.emit(session, "segment_pneumothorax", "pneumothorax_candidate", value)
        self.attach(session, "Pneumothorax_Candidate", mask, owner)
        return [owner]

    def run(self, function_id, tmp_path, inputs=None):
        return run_function_episode(
            function_id, image_ref=str(self.image), registry=self.registry,
            inputs=inputs, out_dir=tmp_path / "run")

    def names(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def harness(tmp_path):
    image = tmp_path / "synthetic.png"
    Image.new("L", (20, 18), 96).save(image)
    return Harness(image)


@pytest.mark.parametrize("finding", ["nodule", "mass"])
def test_p3_reports_box_geometry_and_lung_zone_without_claiming_true_size(harness, tmp_path, finding):
    result = harness.run("P3", tmp_path, {"finding": finding})
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["candidate_boxes"] == [[3.0, 4.0, 6.0, 7.0]]
    assert output["center_xy_px"] == [[4.5, 5.5]]
    assert output["box_extent_px"] == [{"width": 3.0, "height": 3.0}]
    assert output["box_shape"]["scope"] == "candidate_box_union_approximation"
    assert output["lung_zones"]["right"]["overlap_px"] == 9
    assert output["lung_zones"]["left"]["overlap_px"] == 0
    assert output["applicability"]["physical_size_available"] is False
    assert output["applicability"]["clinical_interpretation"] == "not_assessed"
    relation = result.session.graph.nodes[result.functions[0].evidence_refs.relation_ids[0]]
    assert {result.session.graph.nodes[node_id].source_tool for node_id in relation.provenance} == {
        "segment_lungs", "detect_abnormalities", "describe_region_shape",
        "assign_lung_zones", "measure_region_extent"}
    assert validate_graph(result.session) == []


def test_p3_rejects_unsupported_finding_before_running_a_tools(harness, tmp_path):
    result = harness.run("P3", tmp_path, {"finding": "fracture"})
    assert result.status == "refused"
    assert "nodule or mass" in result.functions[0].refusal.reason
    assert not harness.calls


@pytest.mark.parametrize("status", ["not_detected", "unparsed"])
def test_p3_does_not_turn_missing_or_unparsed_candidates_into_negative_claims(harness, tmp_path, status):
    harness.a6_status = status
    result = harness.run("P3", tmp_path, {"finding": "nodule"})
    assert result.status == "refused"
    assert not result.session.graph.by_category("claim")


def test_p3_candidate_outside_both_lungs_is_not_localized(harness, tmp_path):
    harness.candidate_outside_lungs = True
    result = harness.run("P3", tmp_path, {"finding": "nodule"})
    assert result.status == "refused"
    assert "does not overlap either lung" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


def test_p3_rejects_candidate_mask_that_disagrees_with_detector_boxes(harness, tmp_path):
    harness.candidate_mask_mismatch = True
    result = harness.run("P3", tmp_path, {"finding": "nodule"})
    assert result.status == "refused"
    assert "union of detector boxes" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


def test_p6_concordant_positive_preserves_raw_channels_and_geometry_context(harness, tmp_path):
    result = harness.run("P6", tmp_path)
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["pneumothorax_mask"] == {
        "mask_name": "Pneumothorax_Candidate", "status": "region_above_threshold",
        "area_px": 16, "bbox_inclusive_px": {"x0": 2, "y0": 6, "x1": 3,
                                               "y1": 13, "w": 2, "h": 8}}
    assert output["case_raw_score"] == 0.82
    assert output["score_calibrated"] is False
    assert output["candidate_boxes"] == [[2.0, 6.0, 4.0, 14.0]]
    assert output["lung_mirror_similarity"]["mirror_dice"] == 1.0
    assert output["lung_mirror_similarity"]["disease_evidence"] is False
    assert output["conflict"]["status"] == "concordant_positive"
    assert output["clinical_interpretation"] == "not_assessed"
    assert validate_graph(result.session) == []


@pytest.mark.parametrize(
    "a8_region,a6_status,expected",
    [(True, "not_detected", "discordant_a8_positive"),
     (False, "detected", "discordant_a6_positive"),
     (False, "not_detected", "concordant_no_region")],
)
def test_p6_retains_channel_direction_and_conflict_without_silent_averaging(
        harness, tmp_path, a8_region, a6_status, expected):
    harness.a8_region, harness.a6_status = a8_region, a6_status
    if not a8_region:
        harness.a8_raw_score, harness.a8_max_probability = 0.18, 0.22
    result = harness.run("P6", tmp_path)
    assert result.status == "ok"
    conflict = result.functions[0].outputs["conflict"]
    assert conflict == {"status": expected, "a8_region": a8_region,
                        "a6_candidate": a6_status == "detected"}
    assert result.functions[0].outputs["disease_absence_established"] is False


@pytest.mark.parametrize("field,value", [
    ("a8_raw_score", float("nan")), ("a8_raw_score", 1.2),
    ("a8_max_probability", -0.1),
])
def test_p6_rejects_nonfinite_or_out_of_range_raw_scores(harness, tmp_path, field, value):
    setattr(harness, field, value)
    result = harness.run("P6", tmp_path)
    assert result.status == "refused"
    assert not result.session.graph.by_category("claim")


def test_p6_rejects_a8_mask_summary_that_disagrees_with_the_attached_mask(harness, tmp_path):
    harness.a8_area_override = 999
    result = harness.run("P6", tmp_path)
    assert result.status == "refused"
    assert "area" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


def test_p6_unparsed_semantic_channel_is_not_treated_as_negative(harness, tmp_path):
    harness.a6_status = "unparsed"
    result = harness.run("P6", tmp_path)
    assert result.status == "refused"
    assert "unparsed" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


def test_p6_negative_rerun_replaces_an_earlier_positive_mask(harness, tmp_path):
    first = run_function_episode(
        "P6", image_ref=str(harness.image), registry=harness.registry,
        out_dir=tmp_path / "positive")
    assert first.status == "ok"
    assert first.session.masks["Pneumothorax_Candidate"].sum() == 16

    harness.a8_region = False
    harness.a8_raw_score, harness.a8_max_probability = 0.18, 0.22
    second = run_function_episode(
        "P6", session=first.session, registry=harness.registry, reuse=False,
        out_dir=tmp_path / "negative")
    assert second.status == "ok"
    assert second.functions[0].outputs["pneumothorax_mask"]["area_px"] == 0
    assert not second.session.masks["Pneumothorax_Candidate"].any()
    a8_id = next(node_id for node_id in second.functions[0].evidence_refs.observation_ids
                 if second.session.graph.nodes[node_id].source_tool == "segment_pneumothorax")
    assert second.session.mask_metadata["Pneumothorax_Candidate"]["observation_id"] == a8_id


def test_p6_relation_is_bound_to_both_lungs_and_both_prediction_channels(harness, tmp_path):
    result = harness.run("P6", tmp_path)
    relation = result.session.graph.nodes[result.functions[0].evidence_refs.relation_ids[0]]
    sources = [result.session.graph.nodes[node_id].source_tool for node_id in relation.provenance]
    assert sources.count("segment_lungs") == 2
    assert set(sources) == {"segment_lungs", "segment_pneumothorax",
                            "detect_abnormalities", "measure_lung_symmetry"}
    symmetry = next(result.session.graph.nodes[node_id] for node_id in relation.provenance
                    if result.session.graph.nodes[node_id].source_tool == "measure_lung_symmetry")
    assert set(symmetry.provenance) == set(result.session.mask_provenance(["Right_Lung", "Left_Lung"]))
    assert math.isfinite(symmetry.value["mirror_dice"])
