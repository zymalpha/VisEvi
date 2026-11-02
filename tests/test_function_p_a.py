"""Hand-checkable integration contracts for P1, P2, P4 and P5 pilots."""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit
from visevi.functions import runner
from visevi.perception.metrics import build_metric_tools
from visevi.perception.vlm import box_mask


class FakeTool:
    def __init__(self, name, harness, properties=None):
        self.name, self.harness = name, harness
        self.properties = {"output_dir": {"type": "string"}, "device": {"type": "string"},
                           **(properties or {})}

    def describe(self):
        return ToolSpec(self.name, "perception", "synthetic fixture",
                        parameter_schema(self.properties, ("output_dir",)), evidence_result_schema())

    def run(self, session, **params):
        return getattr(self.harness, self.name)(session, params)


class Harness:
    def __init__(self, image):
        self.image = image
        self.detector_status = "detected"
        self.bad_candidate_mask = False
        self.bad_roi_owner = False
        self.bad_roi_digest = False
        self.attribute_answer = "present"
        self.bad_point = False
        self.quality_status = "ok"
        self.registry = ToolRegistry()
        for tool in build_metric_tools():
            self.registry.register(tool)
        for name, props in {
            "segment_lungs": {},
            "locate_lung_landmarks": {},
            "detect_abnormalities": {"findings": {"type": "array", "items": {"type": "string"}}},
            "classify_roi_attributes": {"roi_mask_name": {"type": "string"},
                                        "attributes": {"type": "array", "items": {"type": "string"}}},
            "assess_image_quality": {},
        }.items():
            self.registry.register(FakeTool(name, self, props))

    def emit(self, session, tool, target, value, provenance=()):
        u = EvidenceUnit(id=session.next_id("obs"), category="observation", evidence_type="attribute",
                         finding_or_anatomy=target, value=value, source_tool=tool,
                         provenance=list(provenance), validity_conditions=["synthetic output"])
        session.add_observation(u)
        return u

    @staticmethod
    def attach(session, name, mask, owner):
        session.attach_masks({name: mask}, {name: {"observation_id": owner.id,
            "coordinate_frame": "original_image_pixels", "fov_truncated_sides": []}})

    def segment_lungs(self, session, params):
        result = []
        for name, x in (("Right_Lung", (1, 9)), ("Left_Lung", (11, 19))):
            mask = np.zeros((18, 20), bool)
            mask[2:16, x[0]:x[1]] = True
            unit = self.emit(session, "segment_lungs", name, {"area_px": int(mask.sum())})
            self.attach(session, name, mask, unit)
            result.append(unit)
        return result

    def detect_abnormalities(self, session, params):
        finding = params["findings"][0]
        boxes = [[3., 4., 7., 9.]] if self.detector_status == "detected" else []
        unit = self.emit(session, "detect_abnormalities", "candidate_" + finding,
                         {"finding": finding, "status": self.detector_status, "boxes_xyxy_px": boxes})
        if boxes:
            mask = box_mask(boxes, 20, 18)
            if self.bad_candidate_mask:
                mask = np.roll(mask, 1, axis=1)
            self.attach(session, "Candidate_" + finding.replace(" ", "_"), mask, unit)
        return [unit]

    def classify_roi_attributes(self, session, params):
        name = params["roi_mask_name"]
        provenance = [] if self.bad_roi_owner else session.mask_provenance([name])
        return [self.emit(session, "classify_roi_attributes", "roi_attribute_" + a.replace(" ", "_"),
                          {"attribute": a, "answer": self.attribute_answer, "roi_mask_name": name,
                           "roi_mask_sha256": "0" * 64 if self.bad_roi_digest else
                               hashlib.sha256(np.ascontiguousarray(session.masks[name], dtype=bool).tobytes()).hexdigest(),
                           "option_raw_probabilities": {"A": .8, "B": .1, "C": .1}},
                          provenance) for a in params["attributes"]]

    def locate_lung_landmarks(self, session, params):
        result = []
        for side, x in (("right", 3), ("left", 15)):
            result.append(self.emit(session, "locate_lung_landmarks", side + "_costophrenic_angle",
                        {"point_xy": [x, -10 if self.bad_point else 12], "status": "ok"}))
            result.append(self.emit(session, "locate_lung_landmarks", side + "_lung_base_contour",
                        {"polyline_xy": [[x, 12], [x + 1, 13], [x + 2, 12]]}))
        return result

    def assess_image_quality(self, session, params):
        return [self.emit(session, "assess_image_quality", "inspiration",
                          {"status": self.quality_status, "grade": None,
                           "posterior": {"count_above": 8, "ribs_above": list(range(1, 9)), "highest_number_above": 8},
                           "anterior": {"count_above": 5, "ribs_above": list(range(1, 6)), "highest_number_above": 5},
                           "right_hemidiaphragm_dome_xy": [4, 12]})]

    def run(self, function, tmp_path, **kwargs):
        return runner.run_function_episode(function, image_ref=str(self.image), registry=self.registry,
                                           out_dir=tmp_path / "run", **kwargs)


@pytest.fixture
def harness(tmp_path):
    image = tmp_path / "x.png"
    Image.new("L", (20, 18), 80).save(image)
    return Harness(image)


@pytest.mark.parametrize("function", ["P1", "P2", "P4", "P5"])
def test_four_functions_build_bounded_evidence_graph(harness, tmp_path, function):
    result = harness.run(function, tmp_path)
    assert result.status == "ok", result.functions[0].refusal
    run = result.functions[0]
    assert validate_graph(result.session) == []
    assert len(run.evidence_refs.claim_ids) == 1
    relation = result.session.graph.nodes[run.evidence_refs.relation_ids[0]]
    assert set(relation.provenance) == set(run.evidence_refs.observation_ids)
    assert "not_assessed" in str(run.outputs)
    if function == "P1":
        assert set(run.outputs["attributes"]) == {"consolidation", "air bronchogram"}
        assert run.outputs["roi_appearance"]["intensity"]["mean"] == 80
        assert run.outputs["conflict"]["status"] == "not_assessed"
    elif function == "P2":
        assert run.outputs["phenotype"]["equivalence_established"] is False
        assert run.outputs["phenotype"]["attribute_target"] == "reticular opacity"
    elif function == "P4":
        assert run.outputs["rib_counts"]["posterior"]["count_above"] == 8
        assert run.outputs["case_evidence"]["hyperinflation_diagnosis"] == "not_assessed"
        assert run.outputs["case_evidence"]["emphysema_candidate"]["status"] == "not_run"
    else:
        assert run.outputs["costophrenic_landmarks"]["right_costophrenic_angle"]["point_xy"] == [3, 12]
        assert run.outputs["roi_appearance"]["sharpness"]["tenengrad_mean"] == 0


@pytest.mark.parametrize("function", ["P1", "P2", "P5"])
@pytest.mark.parametrize("status", ["not_detected", "unparsed"])
def test_missing_candidate_never_becomes_disease_negative(harness, tmp_path, function, status):
    harness.detector_status = status
    result = harness.run(function, tmp_path)
    assert result.status == "refused"
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("function", ["P1", "P2", "P5"])
def test_candidate_box_mask_disagreement_refused(harness, tmp_path, function):
    harness.bad_candidate_mask = True
    result = harness.run(function, tmp_path)
    assert result.status == "refused"
    assert "union of detector boxes" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("function", ["P1", "P2"])
def test_attribute_cannot_refer_to_wrong_candidate(harness, tmp_path, function):
    harness.bad_roi_owner = True
    result = harness.run(function, tmp_path)
    assert result.status == "refused"
    assert "attribute is not bound" in result.functions[0].refusal.reason


@pytest.mark.parametrize("function", ["P1", "P2"])
def test_attribute_cannot_refer_to_different_mask_contents(harness, tmp_path, function):
    harness.bad_roi_digest = True
    result = harness.run(function, tmp_path)
    assert result.status == "refused"
    assert "attribute mask digest" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("function,tool", [
    ("P1", "summarize_roi_intensity"), ("P2", "extract_roi_texture"),
    ("P4", "summarize_roi_intensity"), ("P5", "measure_roi_sharpness"),
])
def test_roi_metric_cannot_refer_to_different_mask_contents(harness, tmp_path, function, tool):
    metric = harness.registry._tools[tool]
    original = metric.run
    def corrupt(session, **params):
        result = original(session, **params)
        result[0].value["input_masks"][params["mask_name"]]["binary_sha256"] = "0" * 64
        return result
    metric.run = corrupt
    result = harness.run(function, tmp_path)
    assert result.status == "refused"
    assert "input mask digest or owner" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("function", ["P1", "P2"])
def test_unparsed_required_attribute_is_not_coerced(harness, tmp_path, function):
    harness.attribute_answer = "unparsed"
    result = harness.run(function, tmp_path)
    assert result.status == "refused"
    assert not result.session.graph.by_category("claim")


def test_p5_out_of_frame_costophrenic_point_refused(harness, tmp_path):
    harness.bad_point = True
    result = harness.run("P5", tmp_path)
    assert result.status == "refused"
    assert "invalid or out-of-frame" in result.functions[0].refusal.reason


def test_p4_unknown_inspiration_refused(harness, tmp_path):
    harness.quality_status = "unknown"
    result = harness.run("P4", tmp_path)
    assert result.status == "refused"
    assert "inspiration" in result.functions[0].refusal.reason


def test_p4_optional_detector_is_reported_when_requested(harness, tmp_path):
    result = harness.run("P4", tmp_path, include_optional=True)
    assert result.status == "ok", result.functions[0].refusal
    assert result.functions[0].outputs["case_evidence"]["emphysema_candidate"]["status"] == "detected"
    assert result.functions[0].outputs["case_evidence"]["emphysema_diagnosis"] == "not_assessed"
