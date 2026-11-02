"""S2/S3 conservative temporal integration contracts."""
from copy import deepcopy

import pytest

from visevi.construction.evidence import validate_graph
from visevi.functions.runner import IMPLEMENTED_FUNCTIONS, _PROGRAMS
from visevi.functions.runner import run_function_episode
from test_function_s1 import PairHarness, pair_metadata


@pytest.fixture
def pair_harness(tmp_path):
    prior = tmp_path / "prior-temporal.png"
    current = tmp_path / "current-temporal.png"
    from PIL import Image
    Image.new("L", (20, 16), 40).save(prior)
    Image.new("L", (20, 16), 80).save(current)
    return PairHarness(prior, current)


def _run(pair_harness, function_id, tmp_path, metadata=None, **inputs):
    return run_function_episode(
        function_id,
        image_ref=str(pair_harness.current_image),
        prior_image_ref=str(pair_harness.prior_image),
        registry=pair_harness.registry,
        inputs={"finding": "nodule", "pair_metadata": metadata or pair_metadata(), **inputs},
        out_dir=tmp_path / f"{function_id}-run",
    )


def test_s2_is_registered_and_preserves_uncertain_prior_absence(pair_harness, tmp_path):
    assert {"S2", "S3"} <= set(IMPLEMENTED_FUNCTIONS)
    assert {"S2", "S3"} <= set(_PROGRAMS)
    pair_harness.candidate_status["prior"] = "not_detected"
    result = _run(pair_harness, "S2", tmp_path)
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["new_candidate_regions"]["status"] == "candidate_current_but_prior_absence_unverified"
    assert output["new_candidate_regions"]["disease_absence_or_new_lesion_established"] is False
    assert output["temporal_matches"]["status"] == "not_applicable"
    assert output["comparability"]["registration_recomputed"] is False
    assert validate_graph(result.session) == []


def test_s2_accepts_independent_prior_absence_only_as_candidate_newness(pair_harness, tmp_path):
    pair_harness.candidate_status["prior"] = "not_detected"
    metadata = deepcopy(pair_metadata())
    metadata.pop("lesion_match")
    result = _run(pair_harness, "S2", tmp_path, metadata=metadata,
                  prior_reference={"status": "independently_absent", "source": "reader-1",
                                   "id": "prior-read-001"})
    assert result.status == "ok"
    output = result.functions[0].outputs["new_candidate_regions"]
    assert output["status"] == "candidate_new_with_independent_prior_absence"
    assert output["interpretation"] == "candidate_newness_only"
    assert output["independent_prior_reference"]["source"] == "reader-1"


@pytest.mark.parametrize("role", ["prior", "current"])
def test_s2_never_converts_detector_negative_into_disease_change(pair_harness, tmp_path, role):
    pair_harness.candidate_status[role] = "not_detected"
    result = _run(pair_harness, "S2", tmp_path)
    assert result.status == "ok"
    output = result.functions[0].outputs["new_candidate_regions"]
    assert output["disease_absence_or_new_lesion_established"] is False
    if role == "current":
        assert output["status"] == "no_current_candidate_observed"


def test_s3_requires_verified_one_to_one_match_and_reports_raw_change(pair_harness, tmp_path):
    result = _run(pair_harness, "S3", tmp_path)
    assert result.status == "ok"
    output = result.functions[0].outputs
    assert output["stability"] == "not_assessed"
    assert output["change_magnitude"]["bbox"]["delta"]["width_px"] == 2.0
    assert output["matched_lesions"]["match_id"] == "match-001"
    assert validate_graph(result.session) == []


@pytest.mark.parametrize("role", ["prior", "current"])
def test_s3_refuses_multiple_candidates(pair_harness, tmp_path, role):
    pair_harness.box_count[role] = 2
    result = _run(pair_harness, "S3", tmp_path)
    assert result.status == "refused"
    assert "exactly one" in result.functions[0].refusal.reason
    assert not result.session.graph.by_category("claim")


def test_s3_refuses_without_verified_match(pair_harness, tmp_path):
    metadata = deepcopy(pair_metadata())
    metadata["lesion_match"]["status"] = "not_available"
    result = _run(pair_harness, "S3", tmp_path, metadata=metadata)
    assert result.status == "refused"
    assert "lesion match" in result.functions[0].refusal.reason.lower()
