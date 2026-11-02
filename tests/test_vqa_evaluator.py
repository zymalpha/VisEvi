"""Tests for raw-answer scoring and patient-clustered VQA estimates."""
import json
from pathlib import Path

from PIL import Image

import scripts.evaluate_vqa_q4 as evaluator
from scripts.evaluate_vqa_q4 import _patient_cluster_summary, _raw_answer


def test_raw_answer_score_is_independent_of_verifier_extras():
    raw = ('{"schema_version":"old","answer":"PA","citations":[], '
           '"limitations":["the image shows AP and an extra diagnosis"]}')
    assert _raw_answer(raw) == ("PA", "valid")
    assert _raw_answer("not-json") == (None, "invalid_json")
    assert _raw_answer({"answer": "PA, likely normal"}) == (None, "invalid_answer")


def test_cluster_summary_bootstraps_patients_without_exporting_ids():
    cases = [
        {"vqa_status": "answered", "answer": "PA", "reference_view": "PA"},
        {"vqa_status": "abstained", "answer": None, "reference_view": "AP"},
        {"vqa_status": "answered", "answer": "AP", "reference_view": "AP"},
    ]
    summary = _patient_cluster_summary(cases, ["patient-a", "patient-a", "patient-b"],
                                       samples=100, seed=7)
    assert summary["available"] is True
    assert summary["patient_count"] == 2
    assert summary["metrics"]["answer_coverage"] == 0.75
    assert summary["metrics"]["end_to_end_accuracy"] == 0.75
    assert set(summary["confidence_intervals_95_percentile"]) == {
        "answer_coverage", "end_to_end_accuracy", "conditional_accuracy",
    }
    assert "patient-a" not in str(summary)
    assert _patient_cluster_summary(cases, ["patient-a", None, "patient-b"])["available"] is False


def test_image_only_baseline_persists_raw_output_without_tool_evidence(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("L", (4, 3), color=100).save(image_root / "one.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{
        "relative_path": "one.png", "view_position": "PA", "subject_id": "private-patient",
    }]}))

    class BaselineConsumer:
        model_id = "local-test-model"
        family = "qwen3_vl"
        max_new_tokens = 12

        def __init__(self):
            self.call = None

        def generate_image_only(self, *, question, image_ref):
            self.call = {"question": question, "image_ref": image_ref}
            return "PA"

    consumer = BaselineConsumer()
    monkeypatch.setattr(evaluator, "_build_consumer", lambda *args: consumer)
    output = tmp_path / "run"
    summary = evaluator.evaluate(manifest, image_root, output, device="cuda:4",
                                 consumer_name="qwen3-vl", input_mode="image_only")

    result_path = output / "case_00000" / "vqa_result.json"
    audit = json.loads(result_path.read_text())
    case = json.loads((output / "cases.jsonl").read_text().splitlines()[0])
    assert consumer.call == {"question": "What is the view position: PA or AP?",
                             "image_ref": str(image_root / "one.png")}
    assert audit["baseline"] == "image_only"
    assert audit["raw_consumer_output"] == "PA"
    assert "evidence_card" not in audit
    assert "private-patient" not in result_path.read_text()
    assert "private-patient" not in json.dumps(case)
    assert case["route_status"] == "baseline_direct"
    assert summary["metrics"]["end_to_end_accuracy"] == 1.0
    assert summary["counts"]["tool_ok"] == 0
