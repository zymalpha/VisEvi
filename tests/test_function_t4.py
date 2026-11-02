"""T4 chest-drain reference-adapter contracts.

The adapter is intentionally not an automatic detector. These tests exercise
the source/image binding and deterministic projected relations that a future
drain-capable backend must supply.
"""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import add_observation, validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.session import EvidenceSession
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit
from visevi.functions.runner import IMPLEMENTED_FUNCTIONS, _PROGRAMS, run_function_episode


class FakeLungs:
    def describe(self):
        return ToolSpec("segment_lungs", "perception", "synthetic lungs",
                        parameter_schema({"output_dir": {"type": "string"}}, ("output_dir",)),
                        evidence_result_schema())

    def run(self, session, **params):
        units = []
        for name, columns in (("Right_Lung", slice(2, 9)), ("Left_Lung", slice(12, 19))):
            mask = np.zeros((18, 20), dtype=bool)
            mask[2:16, columns] = True
            unit = EvidenceUnit(session.next_id("obs"), "observation", "region", name,
                                {"area_px": int(mask.sum())}, source_tool="segment_lungs",
                                validity_conditions=["synthetic model output"])
            session.add_observation(unit)
            session.attach_masks({name: mask}, {name: {
                "observation_id": unit.id, "coordinate_frame": "original_image_pixels",
                "fov_truncated_sides": [], "full_image_observed": True,
            }})
            units.append(unit)
        return units


@pytest.fixture
def case(tmp_path):
    image = tmp_path / "drain.png"
    Image.new("L", (20, 18), 90).save(image)
    session = EvidenceSession(image_ref=str(image))
    drain = np.zeros((18, 20), dtype=bool)
    drain[3:13, 5] = True
    owner = add_observation(session, "Chest_Drain", {"source": "synthetic annotation"},
                            source_tool="caller_annotation", evidence_type="region",
                            validity_conditions=["synthetic caller supplied drain path"])
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    session.attach_masks({"Chest_Drain": drain}, {"Chest_Drain": {
        "observation_id": owner.id, "coordinate_frame": "original_image_pixels",
        "fov_truncated_sides": [], "full_image_observed": True,
        "source": "synthetic annotation", "image_sha256": image_sha,
    }})
    registry = ToolRegistry()
    registry.register(FakeLungs())
    return image, session, registry, image_sha


def reference(image, image_sha):
    return {"reference_id": "drain-ref-001", "source": "synthetic annotation",
            "coordinate_frame": "original_image_pixels", "image_sha256": image_sha,
            "drain_mask_name": "Chest_Drain", "tip_xy": [5.0, 12.0],
            "side_hole_xy": [[5.0, 9.0], [5.0, 10.0]]}


def test_t4_reference_adapter_is_registered_and_reports_projected_relation(case, tmp_path):
    image, session, registry, image_sha = case
    assert "T4" in IMPLEMENTED_FUNCTIONS and "T4" in _PROGRAMS
    result = run_function_episode(
        "T4", session=session, registry=registry,
        inputs={"drain_reference": reference(image, image_sha)}, out_dir=tmp_path / "run")
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["device_presence"]["status"] == "present_by_reference"
    assert output["applicability"]["automatic_model"] is False
    assert output["distal_tip"]["on_path"] is True
    assert output["side_holes"]["count"] == 2
    assert validate_graph(result.session) == []


def test_t4_refuses_missing_or_wrong_image_bound_reference(case, tmp_path):
    image, session, registry, image_sha = case
    missing = run_function_episode("T4", session=session, registry=registry,
                                   inputs={}, out_dir=tmp_path / "missing")
    assert missing.status == "refused"
    assert "drain_reference" in missing.functions[0].refusal.reason
    wrong = reference(image, image_sha)
    wrong["image_sha256"] = "0" * 64
    result = run_function_episode("T4", session=session, registry=registry,
                                  inputs={"drain_reference": wrong}, out_dir=tmp_path / "wrong")
    assert result.status == "refused"
    assert "current image" in result.functions[0].refusal.reason
