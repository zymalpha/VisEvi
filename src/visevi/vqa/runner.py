"""Single-image VQA episode runner over FunctionSpec and evidence episodes."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ..core.registry import ToolRegistry
from ..functions.runner import FunctionEpisodeResult, run_function_episode
from ..functions.specs import load_function_specs
from ..eval.runner import build_a_registry
from .consumer import AnswerConsumer, build_answer_prompt
from .evidence_card import EvidenceCard
from .gate import FunctionGate, RuleFunctionGate
from .schema import RouteDecision, VQARequest, VQAResponse
from .verifier import verify_q4_answer


SUPPORTED_VQA_FUNCTIONS = frozenset({"Q4"})


@dataclass
class VQAEpisodeResult:
    status: str
    request: VQARequest
    route: RouteDecision
    evidence_card: EvidenceCard
    response: VQAResponse
    consumer_model: str | None
    function_episode: FunctionEpisodeResult | None = None
    verification: dict[str, Any] | None = None
    consumer_input_mode: str | None = None
    raw_consumer_output: Any = None
    consumer_config: dict[str, Any] | None = None
    prompt_sha256: str | None = None
    error_code: str | None = None
    _temporary_workspace: tempfile.TemporaryDirectory | None = field(
        default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        card_json = self.evidence_card.to_json()
        image_digest = self.evidence_card.provenance.get("image_sha256")
        function_episode = _function_episode_audit(self.function_episode, image_digest)
        return {
            "status": self.status,
            "request": {"question": self.request.question,
                        "dataset_id": self.request.dataset_id,
                        "image_sha256": image_digest},
            "route": self.route.to_dict(),
            "evidence_card": self.evidence_card.to_dict(),
            "evidence_card_sha256": hashlib.sha256(card_json.encode("utf-8")).hexdigest(),
            "response": self.response.to_dict(),
            "consumer_model": self.consumer_model,
            "consumer_input_mode": self.consumer_input_mode,
            "consumer_config": self.consumer_config,
            "prompt_schema_version": "q4-response-2",
            "prompt_sha256": self.prompt_sha256,
            "raw_consumer_output": self.raw_consumer_output,
            "function_episode": function_episode,
            "verification": self.verification,
            "error_code": self.error_code,
        }


def _function_episode_audit(episode: FunctionEpisodeResult | None,
                            image_digest: str | None) -> dict[str, Any] | None:
    if episode is None:
        return None
    payload = episode.to_dict()

    sanitized = _redact_image_paths(payload)
    session = sanitized.get("session")
    if isinstance(session, dict):
        session["image_sha256"] = image_digest
    return sanitized


def _redact_image_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_image_paths(item) for key, item in value.items()
                if key not in {"image_ref", "image_path", "prior_image_ref", "current_image_ref"}}
    if isinstance(value, list):
        return [_redact_image_paths(item) for item in value]
    return value


def _sanitize_function_episode_files(out_dir: Path, image_digest: str | None) -> None:
    for name in ("session.json", "prior_session.json", "episode.json"):
        path = out_dir / name
        if not path.is_file():
            continue
        payload = _redact_image_paths(json.loads(path.read_text(encoding="utf-8")))
        if name == "session.json" and isinstance(payload, dict):
            payload["image_sha256"] = image_digest
        elif name == "episode.json" and isinstance(payload, dict):
            session = payload.get("session")
            if isinstance(session, dict):
                session["image_sha256"] = image_digest
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             indent=2, allow_nan=False) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(data)


def _abstained(reason: str) -> VQAResponse:
    return VQAResponse(status="abstained", refusal_reason=reason)


def _consumer_config(consumer: AnswerConsumer | None) -> dict[str, Any] | None:
    if consumer is None:
        return None
    keys = ("model_id", "input_mode", "family", "device", "max_new_tokens",
            "local_files_only", "torch_dtype", "trust_remote_code",
            "transformers_version", "torch_version")
    config = {key: getattr(consumer, key) for key in keys if getattr(consumer, key, None) is not None}
    config["consumer_class"] = type(consumer).__name__
    return config


def _raw_output_record(value: Any) -> Any:
    if isinstance(value, str):
        return value
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return {"unserializable_type": type(value).__name__}
    return value


def run_vqa_episode(
    request: VQARequest,
    *,
    consumer: AnswerConsumer | None = None,
    gate: FunctionGate | None = None,
    registry: ToolRegistry | None = None,
    out_dir: str | Path | None = None,
    device: str | None = None,
) -> VQAEpisodeResult:
    """Run routing, one registered function, evidence rendering, and verification.

    The first VQA implementation supports Q4. Unsupported, ambiguous, and
    unknown questions abstain before any A tool is called. ``out_dir``, when
    supplied, must be new and is created with owner-only directory/file modes.
    """
    if not isinstance(request, VQARequest):
        raise TypeError("request must be a VQARequest")
    output = Path(out_dir).resolve() if out_dir is not None else None
    if output is not None and output.exists():
        raise ValueError("out_dir must be new; VQA results are never overwritten")

    image = Path(request.image_ref).expanduser().resolve()
    image_exists = image.is_file()
    digest = _sha256(image) if image_exists else None
    if output is not None:
        try:
            output.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("out_dir must be new; VQA results are never overwritten") from exc

    if not image_exists:
        route = RouteDecision("invalid_input", reason="input image file does not exist",
                              matched_by="input_validation")
        card = EvidenceCard.routing_refusal(request.question, route.to_dict())
        card = EvidenceCard(card.card_version, card.function_id, card.question, card.status,
                            card.route, card.evidence_nodes, card.claim_ids, card.function_output,
                            card.tool_steps, card.refusal,
                            {**card.provenance, "image_sha256": None})
        result = VQAEpisodeResult("abstained", request, route, card,
                                  _abstained("input image file does not exist"), None,
                                  error_code="INPUT_IMAGE_NOT_FOUND")
        if output is not None:
            _write_private_json(output / "vqa_result.json", result.to_dict())
        return result

    gate = gate or RuleFunctionGate()
    route = gate.route(request.question)
    if route.status == "matched" and route.selected_function_ids[0] not in SUPPORTED_VQA_FUNCTIONS:
        route = RouteDecision("unsupported", candidates=route.candidates,
                              reason=f"VQA execution is not implemented for {route.selected_function_ids[0]}",
                              matched_by=route.matched_by, gate_version=route.gate_version,
                              category_route=route.category_route)

    if route.status == "matched":
        function_id = route.selected_function_ids[0]
        declared_slots = set(load_function_specs()[function_id].slots)
        extra_slots = set(request.slots) - declared_slots
        if extra_slots:
            route = RouteDecision(
                "invalid_input", candidates=route.candidates,
                reason=f"undeclared input slots: {', '.join(sorted(extra_slots))}",
                matched_by=route.matched_by, gate_version=route.gate_version,
                category_route=route.category_route,
            )

    if route.abstain:
        card = EvidenceCard.routing_refusal(request.question, route.to_dict())
        card = EvidenceCard(card.card_version, card.function_id, card.question, card.status,
                            card.route, card.evidence_nodes, card.claim_ids, card.function_output,
                            card.tool_steps, card.refusal,
                            {**card.provenance, "image_sha256": digest})
        result = VQAEpisodeResult("abstained", request, route, card,
                                  _abstained(route.reason), None)
        if output is not None:
            _write_private_json(output / "vqa_result.json", result.to_dict())
        return result

    function_id = route.selected_function_ids[0]
    def execute(function_out: Path) -> FunctionEpisodeResult:
        return run_function_episode(
            function_id=function_id,
            image_ref=str(image),
            query=request.question,
            inputs=dict(request.slots),
            registry=registry or build_a_registry(),
            out_dir=function_out,
            device=device,
        )

    temporary_workspace = None
    function_out_dir: Path | None = None
    try:
        if output is not None:
            function_out_dir = output / "function_episode"
        else:
            temporary_workspace = tempfile.TemporaryDirectory(prefix="visevi-vqa-")
            function_out_dir = Path(temporary_workspace.name) / "function_episode"
        function_episode = execute(function_out_dir)
    except Exception as exc:
        if function_out_dir is not None:
            _sanitize_function_episode_files(function_out_dir, digest)
        card = EvidenceCard(
            "1.0", function_id, request.question, "refused", route.to_dict(), (), (), {}, (),
            {"reason": f"function execution failed ({type(exc).__name__})"},
            {"image_sha256": digest, "spec_hash": None, "function_version": None},
        )
        response = _abstained(f"function execution failed ({type(exc).__name__})")
        result = VQAEpisodeResult("abstained", request, route, card, response, None,
                                  _temporary_workspace=temporary_workspace)
        if output is not None:
            _write_private_json(output / "vqa_result.json", result.to_dict())
        return result
    if function_out_dir is not None:
        _sanitize_function_episode_files(function_out_dir, digest)
    run = next((item for item in function_episode.functions if item.function_id == function_id), None)
    if run is None:
        card = EvidenceCard.routing_refusal(request.question,
                                             {**route.to_dict(), "reason": "function run missing"})
        response = _abstained("function run returned no result")
        result = VQAEpisodeResult("abstained", request, route, card, response, None,
                                  function_episode)
    else:
        card = EvidenceCard.from_episode(request.question, function_episode, run)
        if card.provenance.get("image_sha256") != digest:
            raise ValueError("image changed between VQA request validation and function execution")
        if card.status != "answerable":
            reason = run.refusal.reason if run.refusal else f"function status: {run.status}"
            response = _abstained(reason)
            result = VQAEpisodeResult("abstained", request, route, card, response,
                                      None, function_episode)
        elif consumer is None:
            response = _abstained("local answer consumer is not configured")
            result = VQAEpisodeResult("abstained", request, route, card, response,
                                      None, function_episode)
        else:
            raw_output = None
            try:
                raw_output = consumer.generate(question=request.question, evidence_card=card,
                                               image_ref=str(image))
                verification = verify_q4_answer(
                    card, raw_output,
                    input_mode=getattr(consumer, "input_mode", "evidence_only"),
                )
            except Exception as exc:
                verification = {"accepted": False,
                                "reason": f"answer consumer failed ({type(exc).__name__})",
                                "error_type": type(exc).__name__, "parsed_output": None}
            if verification["accepted"]:
                response = VQAResponse("answered", answer=verification["answer"],
                                        citations=tuple(verification["citations"]),
                                        limitations=tuple(verification["limitations"]))
                result = VQAEpisodeResult("answered", request, route, card, response,
                                          getattr(consumer, "model_id", type(consumer).__name__),
                                          function_episode, verification,
                                          getattr(consumer, "input_mode", None),
                                          _raw_output_record(raw_output),
                                          _consumer_config(consumer),
                                          hashlib.sha256(build_answer_prompt(
                                              request.question, card,
                                              include_image=getattr(consumer, "input_mode", None)
                                              == "image_and_evidence").encode("utf-8")).hexdigest())
            else:
                response = _abstained(verification["reason"])
                result = VQAEpisodeResult("abstained", request, route, card, response,
                                          getattr(consumer, "model_id", type(consumer).__name__),
                                          function_episode, verification,
                                          getattr(consumer, "input_mode", None),
                                          _raw_output_record(raw_output),
                                          _consumer_config(consumer),
                                          hashlib.sha256(build_answer_prompt(
                                              request.question, card,
                                              include_image=getattr(consumer, "input_mode", None)
                                              == "image_and_evidence").encode("utf-8")).hexdigest())

    if output is not None:
        _write_private_json(output / "vqa_result.json", result.to_dict())
    result._temporary_workspace = temporary_workspace
    return result
