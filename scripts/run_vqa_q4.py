#!/usr/bin/env python3
"""Run one local Q4 chest-X-ray VQA episode and print a PHI-minimal summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visevi.eval.runner import build_a_registry
from visevi.vqa import (TransformersLingshuConsumer, TransformersMedGemmaConsumer,
                        TransformersQwen3VLConsumer, VQARequest, run_vqa_episode)


class EvidenceEchoConsumer:
    """Deterministic test consumer; verifies plumbing, never medical accuracy."""

    model_id = "evidence-echo-smoke-only"

    def generate(self, *, question, evidence_card, image_ref=None):
        del question
        del image_ref
        claims = evidence_card.claims
        if len(claims) != 1:
            return {"schema_version": "q4-response-2", "answer": "unknown",
                    "citations": [], "limitations": [], "visual_assessment": "not_assessed"}
        claim_id, claim = next(iter(claims.items()))
        view = claim.get("value", {}).get("view")
        if view not in {"PA", "AP", "lateral", "unknown"}:
            view = "unknown"
        return {"schema_version": "q4-response-2", "answer": view,
                "citations": [claim_id], "limitations": [],
                "visual_assessment": "not_assessed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", default="What is the view position of this chest X-ray? Is it PA or AP?")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new private output directory; existing directories are refused")
    parser.add_argument("--consumer", choices=("medgemma", "qwen3-vl", "lingshu", "evidence-echo"),
                        default="qwen3-vl")
    parser.add_argument("--input-mode", choices=("evidence_only", "image_and_evidence"),
                        default="image_and_evidence")
    parser.add_argument("--model-id")
    parser.add_argument("--device", choices=("cuda:4", "cuda:5", "cuda:6", "cuda:7"), default="cuda:4")
    parser.add_argument("--dataset-id")
    parser.add_argument("--sample-id")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    consumer_cls = None
    if not args.image.expanduser().is_file():
        consumer = None
    elif args.consumer == "medgemma":
        consumer_cls = TransformersMedGemmaConsumer
        kwargs = {"input_mode": args.input_mode}
    elif args.consumer == "qwen3-vl":
        consumer_cls = TransformersQwen3VLConsumer
        kwargs = {"input_mode": args.input_mode}
    elif args.consumer == "lingshu":
        consumer_cls = TransformersLingshuConsumer
        kwargs = {"input_mode": args.input_mode}
    else:
        consumer = EvidenceEchoConsumer()
        consumer_cls = None

    if consumer_cls is not None:
        model_id = args.model_id or consumer_cls.DEFAULT_MODEL_ID
        consumer = consumer_cls(
            model_id, device=args.device, max_new_tokens=args.max_new_tokens,
            local_files_only=True, **kwargs,
        )

    result = run_vqa_episode(
        VQARequest(str(args.image), args.question, args.dataset_id, args.sample_id),
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
        "verification_accepted": (result.verification or {}).get("accepted"),
        "error_code": result.error_code,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
