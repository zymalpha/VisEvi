"""T1 ETT-position integration contracts using hand-checkable point geometry."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit, ToolRefusal
from visevi.functions.runner import run_function_episode
from visevi.functions.specs import get_function_spec


class FakeDevices:
    def __init__(self):
        self.calls = 0
        self.refuse = False
        self.presence_raw_score = 0.9
        self.screening_threshold = 0.5
        self.ett_status = "present"
        self.tip = [5.0, 3.0]
        self.carina = [5.0, 8.0]
        self.carina_status = "ok"
        self.position_raw_scores = {
            "ETT - Normal": 0.1, "ETT - Borderline": 0.2, "ETT - Abnormal": 0.7,
            "NGT - Normal": 0.8,
        }

    def describe(self):
        return ToolSpec(
            "localize_devices", "perception", "A26 synthetic device output",
            parameter_schema({"output_dir": {"type": "string"},
                              "device": {"type": "string"}}, ("output_dir",)),
            evidence_result_schema())

    @staticmethod
    def _emit(session, target, evidence_type, value):
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation", evidence_type=evidence_type,
            finding_or_anatomy=target, value=value, source_tool="localize_devices",
            provenance=["cached:cxr-lines-and-tubes:synthetic"],
            validity_conditions=["synthetic A26 output; raw scores are uncalibrated"])
        session.add_observation(unit)
        return unit

    def run(self, session, **params):
        self.calls += 1
        if self.refuse:
            return ToolRefusal("localize_devices", "synthetic A26 failure")
        ett = self._emit(session, "endotracheal_tube", "region", {
            "device_family": "ETT",
            "presence_raw_score": self.presence_raw_score,
            "screening_threshold": self.screening_threshold,
            "status": self.ett_status,
            "line_mask_area_px": 6,
            "distal_tip_xy": self.tip,
            "distal_tip_peak_probability": 0.8,
            "score_meaning": "raw sigmoid score; uncalibrated screening policy",
        })
        # The line deliberately reaches the carina. Its region distance is zero,
        # while the hand-checkable tip-to-carina point distance is five pixels.
        if self.ett_status == "present":
            line = np.zeros((12, 16), bool)
            line[2:9, 5] = True
            session.attach_masks(
                {"Device_endotracheal_tube": line},
                {"Device_endotracheal_tube": {
                    "observation_id": ett.id,
                    "coordinate_frame": "original_image_pixels",
                    "full_image_observed": True,
                    "fov_truncated_sides": [],
                }})
        carina = self._emit(session, "carina", "landmark", {
            "point_xy": self.carina,
            "heatmap_peak": 0.7,
            "status": self.carina_status,
        })
        position = self._emit(session, "device_position_classification", "attribute", {
            "raw_scores": self.position_raw_scores,
            "fold_ids": {"classifier": [0, 1]},
            "label_source": "synthetic RANZCR-style labels",
        })
        ngt = self._emit(session, "nasogastric_tube", "region", {
            "device_family": "NGT", "presence_raw_score": 0.1,
            "screening_threshold": 0.5, "status": "below_screening_threshold",
            "distal_tip_xy": [3.0, 10.0], "distal_tip_peak_probability": 0.2,
        })
        return [ett, ngt, carina, position]


@pytest.fixture
def t1(tmp_path):
    image = tmp_path / "image.png"
    Image.new("L", (16, 12), 80).save(image)
    tool = FakeDevices()
    registry = ToolRegistry()
    registry.register(tool)

    def run(**kwargs):
        return run_function_episode(
            "T1", image_ref=str(image), registry=registry,
            out_dir=tmp_path / f"run-{tool.calls}", **kwargs)
    return tool, run


def test_t1_spec_uses_a26_points_instead_of_a11_region_distance():
    spec = get_function_spec("T1")
    assert spec.required_tools == ("localize_devices",)
    assert [step.step_id for step in spec.steps] == ["devices"]
    assert "measure_region_distance" not in spec.required_tools
    assert spec.outputs == (
        "device_presence", "ett_distal_tip_xy_px", "carina_xy_px",
        "tip_carina_distance_px", "position_evidence")


def test_t1_measures_tip_to_carina_points_and_keeps_raw_scores_uncalibrated(t1):
    tool, run = t1
    result = run()
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["ett_distal_tip_xy_px"] == [5.0, 3.0]
    assert output["carina_xy_px"] == [5.0, 8.0]
    assert output["tip_carina_distance_px"] == 5.0
    assert output["device_presence"] == {
        "status": "present", "presence_raw_score": 0.9,
        "screening_threshold": 0.5, "probability_calibrated": False}
    position = output["position_evidence"]
    assert position["carina_minus_tip_vertical_px"] == 5.0
    assert position["tip_minus_carina_vector_xy_px"] == [0.0, -5.0]
    assert position["clinical_position"] is None
    assert position["status"] == "measurement_only"
    assert not any("cm" in key or "physical" in key for key in position)
    assert result.executed_tools == ["localize_devices"]
    assert tool.calls == 1

    run_ref = result.functions[0]
    relation = result.session.graph.nodes[run_ref.evidence_refs.relation_ids[0]]
    assert {result.session.graph.nodes[node_id].finding_or_anatomy
            for node_id in relation.provenance} == {
                "endotracheal_tube", "carina", "device_position_classification"}
    assert {result.session.graph.nodes[node_id].finding_or_anatomy
            for node_id in run_ref.evidence_refs.observation_ids} == {
                "endotracheal_tube", "carina", "device_position_classification"}
    assert validate_graph(result.session) == []


def test_t1_does_not_replace_point_distance_with_the_line_mask_distance(t1):
    _, run = t1
    result = run()
    assert result.status == "ok"
    assert result.functions[0].outputs["tip_carina_distance_px"] == 5.0
    # The synthetic ETT line covers (5, 8), so any A11 line-to-carina-mask
    # calculation would be zero and would fail this contract.
    assert result.functions[0].outputs["tip_carina_distance_px"] != 0.0


def test_t1_position_classifier_remains_auxiliary_and_does_not_grade_geometry(t1):
    _, run = t1
    result = run()
    output = result.functions[0].outputs
    assert output["position_evidence"]["clinical_position"] is None
    assert output["position_evidence"]["position_classification_raw_scores"] == {
        "ETT - Abnormal": 0.7, "ETT - Borderline": 0.2, "ETT - Normal": 0.1}
    claim = result.session.graph.nodes[result.functions[0].evidence_refs.claim_ids[0]]
    assert claim.value["clinical_position"] is None


@pytest.mark.parametrize("scores", [
    {"ETT - Normal": 0.1, "ETT - Abnormal": 0.9},
    {"ETT - Normal": 0.1, "ETT - Borderline": float("nan"), "ETT - Abnormal": 0.9},
])
def test_t1_rejects_missing_or_invalid_ett_position_scores(t1, scores):
    tool, run = t1
    tool.position_raw_scores = scores
    result = run()
    assert result.status == "refused"
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("score,threshold,status", [
    (0.49, 0.5, "below_screening_threshold"),
    (0.9, 0.5, "below_screening_threshold"),
    (0.49, 0.5, "present"),
])
def test_t1_rejects_absent_or_internally_inconsistent_presence(t1, score, threshold, status):
    tool, run = t1
    tool.presence_raw_score = score
    tool.screening_threshold = threshold
    tool.ett_status = status
    result = run()
    assert result.status == "refused"
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("tip,carina,carina_status", [
    (None, [5.0, 8.0], "ok"),
    ([5.0], [5.0, 8.0], "ok"),
    ([5.0, 3.0], None, "outside_image"),
    ([5.0, 3.0], [5.0, 8.0], "outside_image"),
    ([5.0, 3.0], [float("inf"), 8.0], "ok"),
])
def test_t1_invalid_tip_or_carina_cannot_create_a_claim(t1, tip, carina, carina_status):
    tool, run = t1
    tool.tip, tool.carina, tool.carina_status = tip, carina, carina_status
    result = run()
    assert result.status == "refused"
    assert not result.session.graph.by_category("claim")


def test_t1_tool_refusal_is_structured_and_stops_the_function(t1):
    tool, run = t1
    tool.refuse = True
    result = run()
    assert result.status == "refused"
    assert result.functions[0].refusal.tool == "localize_devices"
    assert result.session.trace[-1]["result"] == "refusal"
    assert not result.session.graph.nodes


def test_t1_distance_is_finite_and_matches_euclidean_definition(t1):
    tool, run = t1
    tool.tip, tool.carina = [2.5, 2.0], [5.5, 6.0]
    result = run()
    distance = result.functions[0].outputs["tip_carina_distance_px"]
    assert math.isfinite(distance)
    assert distance == 5.0
