#!/usr/bin/env python3
"""Run one registered MedVisEvi function through the VQA evidence pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from visevi.eval.runner import build_a_registry
from visevi.functions.specs import FUNCTION_IDS
from visevi.vqa import (TransformersLingshuConsumer, TransformersMedGemmaConsumer,
                        TransformersQwen3VLConsumer, VQARequest, run_vqa_episode)


class EvidenceEchoConsumer:
    """Deterministic transport smoke test; it does not evaluate model quality."""

    model_id = "evidence-echo-smoke-only"
    input_mode = "evidence_only"

    def generate(self, *, question, evidence_card, image_ref=None):
        del question, image_ref
        citations = list(evidence_card.claim_ids)
        if evidence_card.function_id == "Q4":
            claim = next(iter(evidence_card.claims.values()), {})
            answer = claim.get("value", {}).get("view", "unknown")
            if answer not in {"PA", "AP", "lateral", "unknown"}:
                answer = "unknown"
            return {"schema_version": "q4-response-2", "answer": answer,
                    "citations": citations, "limitations": [],
                    "visual_assessment": "not_assessed"}
        return {"schema_version": "function-response-1",
                "function_output": evidence_card.function_output,
                "citations": citations}


def _json_object(path: Path | None, name: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {name} JSON file: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{name} JSON file must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--function-id", choices=FUNCTION_IDS,
                        help="optional explicit FunctionSpec selection")
    parser.add_argument("--prior-image", type=Path,
                        help="required by S1-S3 paired-image functions")
    parser.add_argument("--inputs-json", type=Path,
                        help="JSON object of function-specific values and references")
    parser.add_argument("--slots-json", type=Path,
                        help="JSON object of declared FunctionSpec semantic slots")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new private output directory; existing paths are refused")
    parser.add_argument("--consumer", choices=("medgemma", "qwen3-vl", "lingshu", "evidence-echo"),
                        default="qwen3-vl")
    parser.add_argument("--input-mode", choices=("evidence_only", "image_and_evidence"),
                        default="image_and_evidence")
    parser.add_argument("--model-id")
    parser.add_argument("--device", choices=("cpu", "cuda:4", "cuda:5", "cuda:6", "cuda:7"),
                        default="cuda:4")
    parser.add_argument("--dataset-id")
    parser.add_argument("--sample-id")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    question = f"{args.function_id} {args.question}" if args.function_id else args.question
    inputs = _json_object(args.inputs_json, "inputs")
    slots = _json_object(args.slots_json, "slots")
    consumer_cls = {
        "medgemma": TransformersMedGemmaConsumer,
        "qwen3-vl": TransformersQwen3VLConsumer,
        "lingshu": TransformersLingshuConsumer,
    }.get(args.consumer)
    if consumer_cls is None:
        consumer = EvidenceEchoConsumer()
    elif not args.image.expanduser().is_file():
        consumer = None
    else:
        consumer = consumer_cls(
            args.model_id or consumer_cls.DEFAULT_MODEL_ID,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            local_files_only=True,
            input_mode=args.input_mode,
        )

    result = run_vqa_episode(
        VQARequest(
            str(args.image), question, args.dataset_id, args.sample_id,
            slots=slots, inputs=inputs,
            prior_image_ref=str(args.prior_image) if args.prior_image else None,
        ),
        consumer=consumer,
        registry=build_a_registry(),
        out_dir=args.out_dir,
        device=args.device,
    )
    function_runs = result.function_episode.functions if result.function_episode else []
    print(json.dumps({
        "status": result.status,
        "route_status": result.route.status,
        "function_id": result.route.selected_function_ids[0]
        if result.route.selected_function_ids else None,
        "function_status": function_runs[0].status if function_runs else None,
        "consumer_model": result.consumer_model,
        "consumer_input_mode": result.consumer_input_mode,
        "answer": result.response.answer,
        "citations": list(result.response.citations),
        "refusal_reason": result.response.refusal_reason,
        "verification": result.verification,
        "error_code": result.error_code,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
