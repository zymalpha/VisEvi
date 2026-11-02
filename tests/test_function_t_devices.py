"""T2/T3 caller-reference device geometry contracts."""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit, ToolRefusal
from visevi.functions.runner import run_function_episode


class DeviceHarness:
    def __init__(self, image):
        self.image = image
        self.calls = 0
        self.registry = ToolRegistry()
        self.registry.register(FakeDevices(self))

    def run(self, function_id, tmp_path, inputs):
        self.calls += 1
        return run_function_episode(function_id, image_ref=str(self.image), registry=self.registry,
                                    inputs=inputs, out_dir=tmp_path / f"{function_id}-{self.calls}")


class FakeDevices:
    def __init__(self, harness):
        self.harness = harness
        self.properties = {"output_dir": {"type": "string"}}

    def describe(self):
        return ToolSpec("localize_devices", "perception", "synthetic A26",
                        parameter_schema(self.properties, ("output_dir",)), evidence_result_schema())

    def run(self, session, **params):
        frame = (20, 18)
        units = []
        for family, key, tip, row in (("CVC", "central_venous_catheter", [5.0, 5.0], 5),
                                      ("NGT", "nasogastric_tube", [15.0, 13.0], 13)):
            value = {"device_family": family, "status": "present",
                     "presence_raw_score": 0.9, "screening_threshold": 0.5,
                     "distal_tip_xy": tip, "distal_tip_peak_probability": 0.8}
            unit = EvidenceUnit(session.next_id("obs"), "observation", "region", key, value,
                                source_tool="localize_devices",
                                provenance=["cached:synthetic-a26"],
                                validity_conditions=["synthetic raw A26 output"])
            session.add_observation(unit)
            mask = np.zeros((frame[1], frame[0]), bool)
            if family == "CVC":
                mask[2:6, 5] = True
                name = "Device_central_venous_catheter"
            else:
                mask[5:14, 15] = True
                name = "Device_nasogastric_tube"
            session.attach_masks({name: mask}, {name: {"observation_id": unit.id,
                "coordinate_frame": "original_image_pixels", "fov_truncated_sides": [],
                "full_image_observed": True}})
            units.append(unit)
        units.append(EvidenceUnit(session.next_id("obs"), "observation", "landmark", "carina",
                                  {"point_xy": [5.0, 8.0], "status": "ok"},
                                  source_tool="localize_devices", provenance=["cached:synthetic-a26"],
                                  validity_conditions=["synthetic raw A26 output"]))
        session.add_observation(units[-1])
        units.append(EvidenceUnit(session.next_id("obs"), "observation", "attribute",
                                  "device_position_classification", {"raw_scores": {}},
                                  source_tool="localize_devices", provenance=["cached:synthetic-a26"],
                                  validity_conditions=["synthetic raw A26 output"]))
        session.add_observation(units[-1])
        return units


def _ref(image, **extra):
    value = {"reference_id": "ref-001", "source": "synthetic annotation",
             "coordinate_frame": "original_image_pixels",
             "image_sha256": hashlib.sha256(Path(image).read_bytes()).hexdigest()}
    value.update(extra)
    return value


@pytest.fixture
def harness(tmp_path):
    image = tmp_path / "device.png"
    Image.new("L", (20, 18), 80).save(image)
    return DeviceHarness(image)


def test_t2_point_distance_uses_explicit_target_reference(harness, tmp_path):
    result = harness.run("T2", tmp_path, {"cvc_target_reference": _ref(harness.image, point_xy=[8.0, 5.0])})
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["tip_target_distance_px"] == 3.0
    assert output["position_evidence"]["clinical_position"] is None
    assert validate_graph(result.session) == []


def test_t2_rejects_missing_or_wrong_frame_target(harness, tmp_path):
    missing = harness.run("T2", tmp_path, {})
    assert missing.status == "refused"
    wrong = _ref(harness.image, point_xy=[8.0, 5.0]); wrong["image_sha256"] = "0" * 64
    result = harness.run("T2", tmp_path, {"cvc_target_reference": wrong})
    assert result.status == "refused"
    assert "current image" in result.functions[0].refusal.reason


def test_t3_reports_projected_crossing_and_true_tip_limit(harness, tmp_path):
    curve = _ref(harness.image, polyline_xy=[[0.0, 10.0], [19.0, 10.0]])
    result = harness.run("T3", tmp_path, {"diaphragm_reference": curve})
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["crosses_diaphragm"] is True
    assert output["position_evidence"]["true_tip_status"] == "not_available_from_A26_lowest_visible_endpoint"
    assert validate_graph(result.session) == []


def test_t3_rejects_reference_not_in_original_frame(harness, tmp_path):
    curve = _ref(harness.image, polyline_xy=[[0.0, 10.0], [19.0, 10.0]])
    curve["coordinate_frame"] = "resized_model_pixels"
    result = harness.run("T3", tmp_path, {"diaphragm_reference": curve})
    assert result.status == "refused"
    assert "original_image_pixels" in result.functions[0].refusal.reason
