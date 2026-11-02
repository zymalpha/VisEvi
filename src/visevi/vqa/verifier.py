"""Deterministic answer validation against the evidence card."""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .evidence_card import EvidenceCard


_VIEW_ALIASES = {
    "pa": "PA", "posteroanterior": "PA", "后前位": "PA",
    "ap": "AP", "anteroposterior": "AP", "前后位": "AP",
    "lateral": "lateral", "侧位": "lateral", "unknown": "unknown", "不确定": "unknown",
}

_LIMITATION_TEXT = {
    "image_quality_not_assessed": "Image quality was not independently assessed.",
    "projection_metadata_not_used": "Projection metadata was not used as answer evidence.",
}
_VISUAL_ASSESSMENTS = {"consistent", "conflict", "uncertain", "not_assessed"}
_RESPONSE_SCHEMA = "q4-response-2"
_FUNCTION_RESPONSE_SCHEMA = "function-response-1"


def _payload(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("consumer output is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("consumer output must be a JSON object")
    allowed = {"schema_version", "answer", "citations", "limitations", "visual_assessment"}
    if set(value) != allowed:
        raise ValueError("consumer output does not match the q4-response-2 schema")
    if value["schema_version"] != _RESPONSE_SCHEMA:
        raise ValueError("unsupported consumer response schema")
    if not isinstance(value["answer"], str) or not value["answer"].strip():
        raise ValueError("answer must be non-empty text")
    if not isinstance(value["citations"], list) or any(not isinstance(x, str) for x in value["citations"]):
        raise ValueError("citations must be a list of claim IDs")
    if (not isinstance(value["limitations"], list)
            or any(not isinstance(x, str) or x not in _LIMITATION_TEXT for x in value["limitations"])):
        raise ValueError("limitations must contain only registered limitation codes")
    if (not isinstance(value["visual_assessment"], str)
            or value["visual_assessment"] not in _VISUAL_ASSESSMENTS):
        raise ValueError("visual_assessment must be a registered Q4 state")
    return dict(value)


def _canonical_view(answer: str) -> str | None:
    normalized = re.sub(r"[\s._-]+", "", answer.strip().lower())
    return _VIEW_ALIASES.get(normalized)


def verify_q4_answer(card: EvidenceCard, output: Mapping[str, Any] | str, *,
                     input_mode: str = "evidence_only") -> dict[str, Any]:
    """Validate a closed Q4 response contract against the cited claim."""
    if card.function_id != "Q4":
        raise ValueError("Q4 verifier received a different function card")
    if card.status != "answerable":
        raise ValueError("cannot verify an answer from a refused or insufficient card")
    if input_mode not in {"evidence_only", "image_and_evidence"}:
        raise ValueError("unsupported consumer input mode")
    try:
        parsed = _payload(output)
    except ValueError as exc:
        return {"accepted": False, "reason": str(exc), "answer": None,
                "citations": [], "limitations": [], "parsed_output": None}
    claim_nodes = card.claims
    citations = list(dict.fromkeys(parsed["citations"]))
    if not citations:
        return {"accepted": False, "reason": "answer has no claim citation", "answer": None,
                "citations": [], "limitations": parsed["limitations"], "parsed_output": parsed}
    if any(citation not in claim_nodes for citation in citations):
        return {"accepted": False, "reason": "answer cites a missing or non-claim evidence node",
                "answer": None, "citations": citations, "limitations": parsed["limitations"],
                "parsed_output": parsed}
    visual_assessment = parsed["visual_assessment"]
    if input_mode == "image_and_evidence" and visual_assessment != "consistent":
        return {"accepted": False, "reason": f"visual assessment is {visual_assessment}",
                "answer": None, "citations": citations, "limitations": parsed["limitations"],
                "parsed_output": parsed}
    if input_mode == "evidence_only" and visual_assessment != "not_assessed":
        return {"accepted": False, "reason": "evidence-only response must not claim visual assessment",
                "answer": None, "citations": citations, "limitations": parsed["limitations"],
                "parsed_output": parsed}
    supported_views = {
        claim_nodes[citation].get("value", {}).get("view")
        for citation in citations
    }
    canonical = _canonical_view(parsed["answer"])
    if canonical is None:
        return {"accepted": False, "reason": "Q4 answer must be one supported view label",
                "answer": None, "citations": citations, "limitations": parsed["limitations"],
                "parsed_output": parsed}
    if len(supported_views) != 1 or canonical not in supported_views:
        return {"accepted": False, "reason": "answer conflicts with cited Q4 claim",
                "answer": None, "citations": citations, "limitations": parsed["limitations"],
                "parsed_output": parsed}
    return {"accepted": True, "reason": None, "answer": canonical,
            "citations": citations,
            "limitations": [_LIMITATION_TEXT[code] for code in parsed["limitations"]],
            "limitation_codes": parsed["limitations"], "parsed_output": parsed}


def verify_function_answer(card: EvidenceCard, output: Mapping[str, Any] | str) -> dict[str, Any]:
    """Accept only an exact structured copy bound to a function Claim.

    This verifies transport and provenance. It does not assess medical
    correctness or the validity of the upstream tools.
    """
    if card.function_id in {"Q4", "unselected"}:
        raise ValueError("generic function verifier received a Q4 or unselected card")
    if card.status != "answerable":
        raise ValueError("cannot verify an answer from a refused or insufficient card")
    try:
        if isinstance(output, str):
            output = json.loads(output)
        if not isinstance(output, Mapping):
            raise ValueError("consumer output must be a JSON object")
        allowed = {"schema_version", "function_output", "citations"}
        if set(output) != allowed:
            raise ValueError("consumer output does not match the function-response-1 schema")
        if output["schema_version"] != _FUNCTION_RESPONSE_SCHEMA:
            raise ValueError("unsupported consumer response schema")
        if not isinstance(output["function_output"], Mapping):
            raise ValueError("function_output must be a JSON object")
        if not isinstance(output["citations"], list) or any(
                not isinstance(value, str) for value in output["citations"]):
            raise ValueError("citations must be a list of claim IDs")
        parsed = dict(output)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"accepted": False, "reason": str(exc), "answer": None,
                "citations": [], "parsed_output": None,
                "verification_level": "structured_output_and_claim_binding"}

    citations = list(dict.fromkeys(parsed["citations"]))
    if not citations:
        reason = "answer has no claim citation"
    elif any(citation not in card.claims for citation in citations):
        reason = "answer cites a missing or non-claim evidence node"
    elif dict(parsed["function_output"]) != card.function_output:
        reason = "function_output differs from the evidence card"
    else:
        metadata_fields = {"function_id", "spec_version", "proposition", "support_ids",
                           "counter_ids", "rule_id"}
        bound = False
        for citation in citations:
            value = card.claims[citation].get("value", {})
            if (value.get("function_id") == card.function_id
                    and value.get("spec_version") == card.provenance.get("function_version")
                    and {key: item for key, item in value.items()
                         if key not in metadata_fields} == card.function_output):
                bound = True
                break
        reason = None if bound else "cited claim is not bound to the function output"

    accepted = reason is None
    answer = (json.dumps(card.function_output, ensure_ascii=False, sort_keys=True,
                         allow_nan=False) if accepted else None)
    return {"accepted": accepted, "reason": reason, "answer": answer,
            "citations": citations,
            "limitations": (["Verification checks exact output copying and claim binding only; "
                             "it does not establish medical accuracy."] if accepted else []),
            "parsed_output": parsed,
            "verification_level": "structured_output_and_claim_binding",
            "medical_accuracy_verified": False}
