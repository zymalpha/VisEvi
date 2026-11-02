#!/usr/bin/env python3
"""Evaluate the Q4 integration path on a private development manifest.

This is a workflow/tool smoke evaluator. The manifest's view_position metadata
is a development reference for A24 and is not the radiologist-authored VQA
evaluation set. Per-case output intentionally contains only an ordinal and
status fields; patient/study identifiers and image paths are never written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable

from visevi.eval.runner import build_a_registry
from visevi.vqa import (TransformersLingshuConsumer, TransformersMedGemmaConsumer,
                        TransformersQwen3VLConsumer, VQARequest, run_vqa_episode)
from visevi.vqa.consumer import build_image_only_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "original_data/_manifests/20261031-mimic-integration-development-v1/manifest.json"
DEFAULT_IMAGE_ROOT = ROOT / "original_data/mimic-cxr-jpg/2.1.0"
LABELS = ("PA", "AP", "lateral", "unknown")


class EvidenceEchoConsumer:
    """Consume only the Claim; this makes the script measure integration/A24."""

    model_id = "evidence-echo-smoke-only"

    def generate(self, *, question: str, evidence_card, image_ref=None):
        del question
        del image_ref
        claims = evidence_card.claims
        if len(claims) != 1:
            return {"schema_version": "q4-response-2", "answer": "unknown",
                    "citations": [], "limitations": [], "visual_assessment": "not_assessed"}
        claim_id, claim = next(iter(claims.items()))
        value = claim.get("value", {})
        answer = value.get("view") if isinstance(value, dict) else None
        if answer not in LABELS:
            answer = "unknown"
        return {"schema_version": "q4-response-2", "answer": answer,
                "citations": [claim_id], "limitations": [],
                "visual_assessment": "not_assessed"}


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_write(path: Path, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(data)


def _canonical_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {"pa": "PA", "posteroanterior": "PA", "ap": "AP",
               "anteroposterior": "AP", "lateral": "lateral", "unknown": "unknown"}
    return aliases.get(normalized)


def _raw_answer(value: Any) -> tuple[str | None, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, "invalid_json"
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        return None, "missing_answer"
    answer = _canonical_label(value["answer"])
    return answer, "valid" if answer is not None else "invalid_answer"


def _patient_cluster_summary(cases: list[dict[str, Any]], patient_ids: list[Any], *,
                             samples: int = 2000, seed: int = 20261102) -> dict[str, Any]:
    if len(cases) != len(patient_ids) or any(
            not isinstance(patient_id, (str, int)) or not str(patient_id)
            for patient_id in patient_ids):
        return {"available": False, "reason": "manifest lacks a patient identifier for every selected image"}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for patient_id, case in zip(patient_ids, cases):
        grouped.setdefault(str(patient_id), []).append(case)
    if len(grouped) < 2:
        return {"available": False, "reason": "at least two patients are needed for cluster intervals",
                "patient_count": len(grouped), "image_count": len(cases)}

    def patient_metrics(rows: list[dict[str, Any]]) -> tuple[float, float | None, float | None]:
        answered = [row for row in rows if row.get("vqa_status") == "answered"]
        coverage = len(answered) / len(rows)
        paired = [row for row in rows if row.get("reference_view") in LABELS]
        e2e_accuracy = (sum(row.get("answer") == row.get("reference_view")
                            for row in paired) / len(paired) if paired else None)
        paired_answered = [row for row in answered if row.get("reference_view") in LABELS]
        conditional = (sum(row.get("answer") == row.get("reference_view")
                           for row in paired_answered) / len(paired_answered)
                       if paired_answered else None)
        return coverage, e2e_accuracy, conditional

    groups = list(grouped.values())
    observed = [patient_metrics(rows) for rows in groups]
    point = {
        "answer_coverage": sum(row[0] for row in observed) / len(observed),
        "end_to_end_accuracy": (sum(row[1] for row in observed if row[1] is not None)
                                / sum(row[1] is not None for row in observed)
                                if any(row[1] is not None for row in observed) else None),
        "conditional_accuracy": (sum(row[2] for row in observed if row[2] is not None)
                                 / sum(row[2] is not None for row in observed)
                                 if any(row[2] is not None for row in observed) else None),
    }
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(samples):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        estimates = [patient_metrics(rows) for rows in selected]
        draws["answer_coverage"].append(sum(row[0] for row in estimates) / len(estimates))
        e2e = [row[1] for row in estimates if row[1] is not None]
        if e2e:
            draws["end_to_end_accuracy"].append(sum(e2e) / len(e2e))
        conditional = [row[2] for row in estimates if row[2] is not None]
        if conditional:
            draws["conditional_accuracy"].append(sum(conditional) / len(conditional))

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return [ordered[int(0.025 * (len(ordered) - 1))],
                ordered[int(0.975 * (len(ordered) - 1))]]

    return {"available": True, "patient_count": len(groups), "image_count": len(cases),
            "estimand": "equal-weight mean of patient-level metrics",
            "bootstrap_replicates": samples,
            "seed": seed,
            "metrics": point,
            "confidence_intervals_95_percentile": {key: interval(values)
                                                     for key, values in draws.items()}}


def _macro_f1(rows: Iterable[dict[str, Any]]) -> float | None:
    pairs = [(row["reference_view"], row["answer"])
             for row in rows if row.get("reference_view") in LABELS and row.get("answer") in LABELS]
    if not pairs:
        return None
    values = []
    for label in LABELS:
        tp = sum(reference == label and answer == label for reference, answer in pairs)
        fp = sum(reference != label and answer == label for reference, answer in pairs)
        fn = sum(reference == label and answer != label for reference, answer in pairs)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return sum(values) / len(values) if values else None


def _build_consumer(name: str, input_mode: str, model_id: str | None,
                    device: str):
    if name == "evidence-echo":
        return EvidenceEchoConsumer()
    model_classes = {
        "qwen3-vl": TransformersQwen3VLConsumer,
        "lingshu": TransformersLingshuConsumer,
        "medgemma": TransformersMedGemmaConsumer,
    }
    consumer_class = model_classes[name]
    resolved_model_id = model_id or consumer_class.DEFAULT_MODEL_ID
    backend_input_mode = "image_and_evidence" if input_mode == "image_only" else input_mode
    return consumer_class(
        resolved_model_id, device=device, input_mode=backend_input_mode,
        local_files_only=True,
    )


def evaluate(manifest_path: Path, image_root: Path, output: Path, *, device: str,
             limit: int | None = None, consumer_name: str = "evidence-echo",
             input_mode: str = "image_and_evidence", model_id: str | None = None) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output must be a new directory; existing results are never overwritten")
    if consumer_name == "evidence-echo" and input_mode == "image_only":
        raise ValueError("image_only baseline requires a local vision-language model")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest has no image entries")
    selected = rows if limit is None else rows[:limit]
    consumer = EvidenceEchoConsumer() if consumer_name == "evidence-echo" else None
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    registry = build_a_registry()
    cases: list[dict[str, Any]] = []
    selected_patient_ids = [row.get("subject_id") for row in selected]
    for ordinal, row in enumerate(selected):
        relative_path = row.get("relative_path")
        image = image_root / relative_path if isinstance(relative_path, str) else None
        reference = _canonical_label(row.get("view_position"))
        case: dict[str, Any] = {
            "case_ordinal": ordinal,
            "reference_view": reference,
            "route_status": None,
            "tool_status": None,
            "tool_view": None,
            "vqa_status": None,
            "answer": None,
            "verification_accepted": None,
            "raw_answer": None,
            "raw_answer_parse_status": "not_generated",
            "input_mode": input_mode if consumer_name != "evidence-echo" else "evidence_claim_echo",
        }
        if image is None:
            case["route_status"] = "invalid_manifest_entry"
            case["vqa_status"] = "abstained"
            case["error_code"] = "MANIFEST_IMAGE_PATH_INVALID"
            cases.append(case)
            continue
        case_dir = output / f"case_{ordinal:05d}"
        try:
            if image.is_file() and consumer is None:
                consumer = _build_consumer(consumer_name, input_mode, model_id, device)
            if image.is_file() and consumer is not None and input_mode == "image_only":
                case_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
                question = "What is the view position: PA or AP?"
                raw_output = consumer.generate_image_only(question=question, image_ref=str(image))
                answer = _canonical_label(raw_output)
                vqa_status = "answered" if answer is not None else "abstained"
                config = {
                    "model_id": consumer.model_id,
                    "consumer_class": type(consumer).__name__,
                    "family": getattr(consumer, "family", None),
                    "device": device,
                    "input_mode": "image_only",
                    "max_new_tokens": getattr(consumer, "max_new_tokens", 128),
                    "torch_dtype": getattr(consumer, "torch_dtype", "bfloat16"),
                    "local_files_only": True,
                    "transformers_version": getattr(consumer, "transformers_version", None),
                    "torch_version": getattr(consumer, "torch_version", None),
                }
                prompt_hash = hashlib.sha256(build_image_only_prompt(question).encode("utf-8")).hexdigest()
                _private_write(case_dir / "vqa_result.json", {
                    "baseline": "image_only",
                    "question": question,
                    "image_sha256": _manifest_sha256(image),
                    "consumer_configuration": config,
                    "prompt_schema_version": "q4-image-only-1",
                    "prompt_sha256": prompt_hash,
                    "raw_consumer_output": raw_output,
                    "parsed_answer": answer,
                    "response_status": vqa_status,
                })
                case.update({
                    "route_status": "baseline_direct",
                    "vqa_status": vqa_status,
                    "answer": answer,
                    "raw_answer": answer,
                    "raw_answer_parse_status": "valid" if answer is not None else "invalid_answer",
                    "verification_accepted": None,
                    "audit_path": str((case_dir / "vqa_result.json").relative_to(output)),
                })
                cases.append(case)
                continue
            result = run_vqa_episode(
                VQARequest(str(image), "What is the view position: PA or AP?"),
                consumer=consumer, registry=registry, device=device,
                out_dir=case_dir,
            )
        except Exception as exc:
            case["route_status"] = "runner_exception"
            case["error_type"] = type(exc).__name__
            case["audit_path"] = str(case_dir.relative_to(output))
            case_dir.mkdir(mode=0o700, exist_ok=True)
            _private_write(case_dir / "evaluator_error.json", {
                "error_type": type(exc).__name__, "case_ordinal": ordinal,
            })
            cases.append(case)
            continue
        case["route_status"] = result.route.status
        case["vqa_status"] = result.status
        case["answer"] = result.response.answer
        raw_answer, raw_status = _raw_answer(result.raw_consumer_output)
        case["raw_answer"] = raw_answer
        case["raw_answer_parse_status"] = raw_status
        case["error_code"] = result.error_code
        case["audit_path"] = str((case_dir / "vqa_result.json").relative_to(output))
        case["verification_accepted"] = (result.verification or {}).get("accepted")
        case["input_mode"] = result.consumer_input_mode or case["input_mode"]
        if result.function_episode and result.function_episode.functions:
            run = result.function_episode.functions[0]
            case["tool_status"] = run.status
            case["tool_view"] = (run.outputs or {}).get("view")
        cases.append(case)

    references = [case["reference_view"] for case in cases if case.get("reference_view") in LABELS]
    answers = [case["answer"] for case in cases if case.get("answer") in LABELS]
    answerable = [case for case in cases if case.get("vqa_status") == "answered"]
    answerable_paired = [case for case in answerable
                         if case.get("reference_view") in LABELS]
    tool_ok = [case for case in cases if case.get("tool_status") == "ok"]
    paired = [case for case in cases if case.get("reference_view") in LABELS]
    confusion = Counter((case["reference_view"], case.get("answer") or "refused") for case in paired)
    configured_model_id = (consumer.model_id if consumer is not None else
                           model_id or {"qwen3-vl": TransformersQwen3VLConsumer,
                                        "lingshu": TransformersLingshuConsumer,
                                        "medgemma": TransformersMedGemmaConsumer}[consumer_name].DEFAULT_MODEL_ID)
    model_configuration = {
        "model_id": configured_model_id,
        "consumer_class": type(consumer).__name__ if consumer is not None else None,
        "family": getattr(consumer, "family", None),
        "device": device,
        "input_mode": input_mode if consumer_name != "evidence-echo" else "evidence_claim_echo",
        "max_new_tokens": getattr(consumer, "max_new_tokens", 128),
        "torch_dtype": getattr(consumer, "torch_dtype", "bfloat16"),
        "local_files_only": True,
        "transformers_version": getattr(consumer, "transformers_version", None),
        "torch_version": getattr(consumer, "torch_version", None),
    }
    summary: dict[str, Any] = {
        "purpose": "q4_development_model_comparison_not_clinical_validation",
        "manifest_sha256": _manifest_sha256(manifest_path),
        "manifest_images": len(rows),
        "selected_images": len(selected),
        "device": device,
        "consumer": configured_model_id,
        "consumer_configuration": model_configuration,
        "input_mode": input_mode if consumer_name != "evidence-echo" else "evidence_claim_echo",
        "counts": {
            "tool_ok": len(tool_ok),
            "tool_refused": sum(case.get("tool_status") == "refused" for case in cases),
            "answered": len(answerable),
            "abstained": sum(case.get("vqa_status") == "abstained" for case in cases),
            "missing_input": sum(case.get("error_code") == "INPUT_IMAGE_NOT_FOUND"
                                 for case in cases),
            "invalid_manifest_entry": sum(case.get("error_code") == "MANIFEST_IMAGE_PATH_INVALID"
                                           for case in cases),
            "runner_exception": sum(case.get("route_status") == "runner_exception" for case in cases),
        },
        "metrics": {
            "tool_coverage": len(tool_ok) / len(cases) if cases else None,
            "answer_coverage": len(answerable) / len(cases) if cases else None,
            "conditional_answer_accuracy": (
                sum(case.get("answer") == case.get("reference_view") for case in answerable_paired)
                / len(answerable_paired) if answerable_paired else None
            ),
            "raw_answer_accuracy_all_paired": (
                sum(case.get("raw_answer") == case.get("reference_view") for case in paired)
                / len(paired) if paired else None
            ),
            "end_to_end_accuracy": (
                sum(case.get("answer") == case.get("reference_view") for case in paired) / len(paired)
                if paired else None
            ),
            "macro_f1_answerable": _macro_f1(answerable),
        },
        "reference_counts": dict(sorted(Counter(references).items())),
        "answer_counts": dict(sorted(Counter(answers).items())),
        "raw_answer_counts": dict(sorted(Counter(case["raw_answer"] for case in cases
                                                  if case.get("raw_answer") in LABELS).items())),
        "raw_answer_parse_counts": dict(sorted(Counter(case["raw_answer_parse_status"]
                                                       for case in cases).items())),
        "patient_clustered": _patient_cluster_summary(cases, selected_patient_ids),
        "confusion": {f"{reference}->{answer}": count
                      for (reference, answer), count in sorted(confusion.items())},
        "limitations": [
            "view_position is development metadata, not the physician-authored PhysioNet VQA gold set",
            ("the evidence-echo consumer is deterministic and is not a language model"
             if consumer_name == "evidence-echo" else
             "VLM comparisons are local development results and do not establish clinical validity"),
            "evidence_only and image_and_evidence are separate input conditions and must not be pooled",
            "the development manifest is enriched, nonrandom and may overlap upstream training sources",
            "per-case output omits patient/study identifiers and image paths",
        ],
    }
    _private_write(output / "summary.json", summary)
    cases_path = output / "cases.jsonl"
    fd = os.open(cases_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:4", "cuda:5", "cuda:6", "cuda:7"), default="cuda:4")
    parser.add_argument("--consumer", choices=("evidence-echo", "qwen3-vl", "lingshu", "medgemma"),
                        default="evidence-echo")
    parser.add_argument("--input-mode", choices=("evidence_only", "image_and_evidence", "image_only"),
                        default="image_and_evidence")
    parser.add_argument("--model-id")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    summary = evaluate(args.manifest, args.image_root, args.output,
                       device=args.device, limit=args.limit,
                       consumer_name=args.consumer, input_mode=args.input_mode,
                       model_id=args.model_id)
    print(json.dumps({"selected_images": summary["selected_images"],
                      "counts": summary["counts"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
