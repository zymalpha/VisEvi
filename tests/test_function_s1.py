"""S1 paired-study contracts with synthetic, caller-bound lesion matches.

These tests verify pair orchestration and raw comparison semantics.  They do
not validate registration, lesion matching, or a clinical progression rule.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit, ToolRefusal
from visevi.functions.runner import run_function_episode
from visevi.functions.specs import get_function_spec
from visevi.perception.metrics import build_metric_tools


def pair_metadata() -> dict:
    return {
        "patient_id_hash": "sha256:" + "a" * 64,
        "identity_verified": True,
        "prior_study_id": "study-prior",
        "current_study_id": "study-current",
        "prior_time": "2025-01-02T10:00:00Z",
        "current_time": "2025-02-03T10:00:00Z",
        "chronology_verified": True,
        "projection_comparable": True,
        "registration": {
            "status": "verified",
            "method": "synthetic_identity_transform",
            "id": "registration-001",
            "common_pixel_frame_verified": True,
            "applied_to_inputs": True,
            "frame_id": "registered-20x16",
        },
        "lesion_match": {
            "status": "verified",
            "id": "match-001",
            "source": "synthetic blinded correspondence",
        },
    }


class FakePairTool:
    def __init__(self, name: str, harness: "PairHarness", properties=None):
        self.name = name
        self.harness = harness
        self.properties = {
            "output_dir": {"type": "string"},
            "device": {"type": "string"},
            **(properties or {}),
        }

    def describe(self):
        return ToolSpec(
            self.name, "perception", "synthetic paired-study model",
            parameter_schema(self.properties, ("output_dir",)),
            evidence_result_schema(),
        )

    def run(self, session, **params):
        role = self.harness.role(session.image_ref)
        self.harness.calls.append((role, self.name, params))
        if (role, self.name) in self.harness.refuse:
            return ToolRefusal(self.name, f"synthetic {role} evidence unavailable")
        return getattr(self.harness, self.name)(session, params, role)


class PairHarness:
    def __init__(self, prior_image, current_image):
        self.prior_image = prior_image
        self.current_image = current_image
        self.calls = []
        self.refuse = set()
        self.candidate_status = {"prior": "detected", "current": "detected"}
        self.box_count = {"prior": 1, "current": 1}
        self.candidate_mask_mismatch = set()
        self.registry = ToolRegistry()
        for tool in build_metric_tools():
            self.registry.register(tool)
        for name, properties in {
            "segment_lungs": {},
            "segment_heart": {},
            "detect_abnormalities": {"findings": {"type": "array"}},
        }.items():
            self.registry.register(FakePairTool(name, self, properties))

    def role(self, image_ref):
        path = str(image_ref)
        if path == str(self.prior_image):
            return "prior"
        if path == str(self.current_image):
            return "current"
        raise AssertionError(f"tool received an unbound study: {path}")

    @staticmethod
    def emit(session, tool, finding, value, *, provenance=()):
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation",
            evidence_type="attribute", finding_or_anatomy=finding,
            value=value, source_tool=tool, provenance=list(provenance),
            validity_conditions=["synthetic prediction; not a clinical reference"],
        )
        session.add_observation(unit)
        return unit

    @staticmethod
    def attach(session, name, mask, owner):
        session.attach_masks(
            {name: mask},
            {name: {
                "observation_id": owner.id,
                "coordinate_frame": "original_image_pixels",
                "fov_truncated_sides": [],
                "is_ground_truth": False,
            }},
        )

    def segment_lungs(self, session, params, role):
        units = []
        for name, columns in (("Right_Lung", slice(1, 8)), ("Left_Lung", slice(12, 19))):
            mask = np.zeros((16, 20), bool)
            mask[2:15, columns] = True
            unit = self.emit(session, "segment_lungs", name, {"study_role": role})
            self.attach(session, name, mask, unit)
            units.append(unit)
        return units

    def segment_heart(self, session, params, role):
        mask = np.zeros((16, 20), bool)
        mask[6:14, 7:13] = True
        unit = self.emit(session, "segment_heart", "Heart", {"study_role": role})
        self.attach(session, "Heart", mask, unit)
        return [unit]

    def detect_abnormalities(self, session, params, role):
        finding = params["findings"][0]
        status = self.candidate_status[role]
        count = self.box_count[role]
        boxes = []
        if status == "detected":
            boxes = ([[2, 5, 4, 7]] if role == "prior" else [[2, 4, 6, 8]]) * count
        unit = self.emit(session, "detect_abnormalities", "candidate_" + finding, {
            "finding": finding,
            "status": status,
            "boxes_xyxy_px": boxes,
            "raw_text": f"synthetic {role} candidate",
        })
        if status == "detected":
            mask = np.zeros((16, 20), bool)
            mask[5:7, 2:4] = True if role == "prior" else False
            if role == "current":
                mask[4:8, 2:6] = True
            if role in self.candidate_mask_mismatch:
                mask = np.roll(mask, 1, axis=1)
            self.attach(session, "Candidate_" + finding.replace(" ", "_"), mask, unit)
        return [unit]

    def run(self, *, metadata=None, **kwargs):
        return run_function_episode(
            "S1",
            image_ref=str(self.current_image),
            prior_image_ref=str(self.prior_image),
            registry=self.registry,
            inputs={"finding": "nodule", "pair_metadata": metadata or pair_metadata()},
            **kwargs,
        )


@pytest.fixture
def pair_harness(tmp_path):
    prior = tmp_path / "prior.png"
    current = tmp_path / "current.png"
    Image.new("L", (20, 16), 40).save(prior)
    Image.new("L", (20, 16), 80).save(current)
    return PairHarness(prior, current)


def test_s1_spec_explicitly_routes_every_step_to_prior_or_current():
    spec = get_function_spec("S1")
    assert spec.input_type == "paired_images"
    assert {step.input_role for step in spec.steps} == {"prior", "current"}
    assert all(step.step_id.startswith(step.input_role + "_") for step in spec.steps)


def test_s1_reports_raw_paired_deltas_without_inventing_progression(pair_harness, tmp_path):
    result = pair_harness.run(out_dir=tmp_path / "run")
    assert result.status == "ok"
    run = result.functions[0]
    assert run.outputs["change_direction"] == "not_assessed"
    assert run.outputs["matched_lesions"] == {
        "finding": "nodule",
        "match_id": "match-001",
        "matching_source": "synthetic blinded correspondence",
        "prior_candidate_count": 1,
        "current_candidate_count": 1,
    }
    extent = run.outputs["change_magnitude"]["extent"]
    assert extent["bbox_width_px"] == {"prior": 2, "current": 4, "delta": 2}
    assert extent["bbox_height_px"] == {"prior": 2, "current": 4, "delta": 2}
    intensity = run.outputs["change_magnitude"]["intensity"]
    assert intensity["mean_display_gray"] == {"prior": 40.0, "current": 80.0, "delta": 40.0}
    assert run.outputs["comparability"]["registration_id"] == "registration-001"
    assert run.outputs["comparability"]["clinical_direction_threshold_frozen"] is False
    assert result.study_sessions["prior"] is not result.study_sessions["current"]
    assert result.study_sessions["prior"].image_ref == str(pair_harness.prior_image)
    assert result.study_sessions["current"].image_ref == str(pair_harness.current_image)
    assert validate_graph(result.session) == []
    assert len(result.session.graph.by_category("claim")) == 1
    assert all(word not in json.dumps(run.outputs, ensure_ascii=False).lower()
               for word in ("improved", "progressed", "stable"))


def test_s1_keeps_tool_caches_and_runtime_masks_separate_by_study(pair_harness, tmp_path):
    result = pair_harness.run(out_dir=tmp_path / "run")
    assert result.status == "ok"
    calls = [(role, tool) for role, tool, _ in pair_harness.calls]
    assert calls.count(("prior", "detect_abnormalities")) == 1
    assert calls.count(("current", "detect_abnormalities")) == 1
    assert not {"segment_lungs", "segment_heart"} & {tool for _, tool in calls}
    prior = result.study_sessions["prior"]
    current = result.study_sessions["current"]
    for session in (prior, current):
        traced = [item["tool"] for item in session.trace]
        assert traced.count("measure_region_extent") == 1
        assert traced.count("summarize_roi_intensity") == 1
    assert prior.masks["Candidate_nodule"].sum() == 4
    assert current.masks["Candidate_nodule"].sum() == 16


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda m: m.update(identity_verified=False), "identity"),
        (lambda m: m.update(chronology_verified=False), "chronology"),
        (lambda m: m.update(current_time=m["prior_time"]), "strictly later"),
        (lambda m: m.update(current_study_id=m["prior_study_id"]), "distinct studies"),
        (lambda m: m.update(projection_comparable=False), "projection"),
        (lambda m: m["registration"].update(status="missing"), "registration"),
        (lambda m: m["lesion_match"].update(status="missing"), "lesion match"),
    ],
)
def test_s1_refuses_unverified_or_incomparable_pair_metadata(
        pair_harness, tmp_path, mutate, reason):
    metadata = deepcopy(pair_metadata())
    mutate(metadata)
    result = pair_harness.run(metadata=metadata, out_dir=tmp_path / "run")
    assert result.status == "refused"
    assert reason in result.functions[0].refusal.reason.lower()
    assert not pair_harness.calls
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("field", ["common_pixel_frame_verified", "applied_to_inputs", "frame_id"])
def test_s1_requires_a_recorded_common_pixel_frame(pair_harness, tmp_path, field):
    metadata = deepcopy(pair_metadata())
    metadata["registration"].pop(field)
    result = pair_harness.run(metadata=metadata, out_dir=tmp_path / "run")
    assert result.status == "refused"
    assert "common pixel" in result.functions[0].refusal.reason.lower()
    assert not pair_harness.calls


def test_s1_rejects_plaintext_patient_identifier_before_persistence(pair_harness, tmp_path):
    metadata = pair_metadata()
    plaintext_id = "patient-name-or-hospital-id"
    metadata["patient_id_hash"] = plaintext_id
    out = tmp_path / "run"
    result = pair_harness.run(metadata=metadata, out_dir=out)
    assert result.status == "refused"
    assert "sha-256" in result.functions[0].refusal.reason.lower()
    assert not pair_harness.calls
    serialized = (out / "episode.json").read_text()
    assert plaintext_id not in serialized
    assert "redacted_invalid_pair_metadata" in serialized


def test_s1_refuses_studies_with_different_raster_dimensions(pair_harness, tmp_path):
    Image.new("L", (21, 16), 80).save(pair_harness.current_image)
    result = pair_harness.run(out_dir=tmp_path / "run")
    assert result.status == "refused"
    assert "dimensions" in result.functions[0].refusal.reason.lower()
    assert not pair_harness.calls


def test_s1_refuses_the_same_raster_as_both_studies(pair_harness, tmp_path):
    result = run_function_episode(
        "S1",
        image_ref=str(pair_harness.current_image),
        prior_image_ref=str(pair_harness.current_image),
        registry=pair_harness.registry,
        inputs={"finding": "nodule", "pair_metadata": pair_metadata()},
        out_dir=tmp_path / "run",
    )
    assert result.status == "refused"
    assert "distinct images" in result.functions[0].refusal.reason.lower()
    assert not pair_harness.calls


@pytest.mark.parametrize("role,status", [
    ("prior", "not_detected"), ("current", "not_detected"),
    ("prior", "unparsed"), ("current", "unparsed"),
])
def test_s1_requires_one_parsed_candidate_in_each_study(pair_harness, tmp_path, role, status):
    pair_harness.candidate_status[role] = status
    result = pair_harness.run(out_dir=tmp_path / "run")
    assert result.status == "refused"
    assert "candidate" in result.functions[0].refusal.reason.lower()
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("role", ["prior", "current"])
def test_s1_refuses_ambiguous_box_unions_without_a_matching_program(pair_harness, tmp_path, role):
    pair_harness.box_count[role] = 2
    result = pair_harness.run(out_dir=tmp_path / "run")
    assert result.status == "refused"
    assert "exactly one" in result.functions[0].refusal.reason.lower()
    assert not result.session.graph.by_category("claim")


@pytest.mark.parametrize("role", ["prior", "current"])
def test_s1_rejects_candidate_mask_that_disagrees_with_its_box(pair_harness, tmp_path, role):
    pair_harness.candidate_mask_mismatch.add(role)
    result = pair_harness.run(out_dir=tmp_path / "run")
    assert result.status == "refused"
    assert "union of detector boxes" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


def test_s1_required_tool_refusal_names_study_and_stops_pair_claim(pair_harness, tmp_path):
    pair_harness.refuse.add(("prior", "detect_abnormalities"))
    result = pair_harness.run(out_dir=tmp_path / "run")
    assert result.status == "refused"
    assert "prior" in result.functions[0].refusal.reason.lower()
    assert not result.session.graph.by_category("claim")


def test_s1_saves_both_study_sessions_and_pair_binding(pair_harness, tmp_path):
    out = tmp_path / "run"
    result = pair_harness.run(out_dir=out)
    assert result.status == "ok"
    assert (out / "session.json").is_file()
    assert (out / "prior_session.json").is_file()
    payload = json.loads((out / "episode.json").read_text())
    assert payload["study_sessions"]["prior"]["image_ref"] == str(pair_harness.prior_image)
    assert payload["study_sessions"]["current"]["image_ref"] == str(pair_harness.current_image)
    assert payload["functions"][0]["outputs"]["change_direction"] == "not_assessed"


def test_s1_new_prior_study_cannot_reuse_imported_nodes_from_an_older_pair(
        pair_harness, tmp_path):
    first = pair_harness.run(out_dir=tmp_path / "first")
    assert first.status == "ok"
    assert first.functions[0].outputs["change_magnitude"]["intensity"]["mean_display_gray"]["prior"] == 40.0
    first_prior_ids = {node_id for node_id in first.session.graph.nodes if node_id.startswith("prior:")}

    second_prior = tmp_path / "second-prior.png"
    Image.new("L", (20, 16), 10).save(second_prior)
    pair_harness.prior_image = second_prior
    metadata = pair_metadata()
    metadata["prior_study_id"] = "study-earlier-prior"
    metadata["prior_time"] = "2024-12-01T10:00:00Z"
    metadata["registration"]["id"] = "registration-002"
    metadata["lesion_match"]["id"] = "match-002"
    second = run_function_episode(
        "S1", session=first.session, prior_image_ref=str(second_prior),
        registry=pair_harness.registry,
        inputs={"finding": "nodule", "pair_metadata": metadata},
        out_dir=tmp_path / "second",
    )
    assert second.status == "ok"
    assert second.functions[0].outputs["change_magnitude"]["intensity"]["mean_display_gray"]["prior"] == 10.0
    second_prior_ids = {node_id for node_id in second.functions[0].evidence_refs.observation_ids
                        if node_id.startswith("prior:")}
    assert second_prior_ids
    assert second_prior_ids.isdisjoint(first_prior_ids)
