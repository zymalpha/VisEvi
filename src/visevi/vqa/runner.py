"""VQA episode runner over single- and paired-image FunctionSpec workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from ..core.registry import ToolRegistry
from ..core.session import EvidenceSession
from ..functions.runner import FunctionEpisodeResult, run_function_episode
from ..functions.specs import FUNCTION_IDS, FunctionSpec, load_function_specs
from ..eval.runner import build_a_registry
from .consumer import AnswerConsumer, build_answer_prompt
from .evidence_card import EvidenceCard
from .gate import FunctionGate, RuleFunctionGate
from .schema import RouteDecision, VQARequest, VQAResponse
from .verifier import verify_function_answer, verify_q4_answer


SUPPORTED_VQA_FUNCTIONS = frozenset(FUNCTION_IDS)
FUNCTION_RESPONSE_SCHEMA = "function-response-1"

_FUNCTION_RUNTIME_INPUTS = {
    "G1": {"finding", "lesion_mask_name", "orientation_verified",
           "patient_right_is_image_left", "orientation_source"},
    "G2": {"finding", "lesion_mask_name", "orientation_verified",
           "patient_right_is_image_left", "orientation_source"},
    "G3": {"lesion_mask_name", "anatomy_mask_name", "region_mask_names"},
    "P3": {"finding", "lesion_mask_name"},
    "P8": {"right_hilum_mask_name", "left_hilum_mask_name", "hilum_mask_names"},
    "P10": {"aortic_knob_mask_name"},
    "S1": {"finding", "pair_metadata"},
    "S2": {"finding", "pair_metadata", "prior_reference"},
    "S3": {"finding", "pair_metadata"},
    "T2": {"cvc_target_reference"},
    "T3": {"diaphragm_reference"},
    "T4": {"drain_reference"},
}

_REQUIRED_FUNCTION_INPUTS = {
    "G1": {"finding"}, "G2": {"finding"},
    "G3": {"lesion_mask_name", "anatomy_mask_name", "region_mask_names"},
    "P3": {"finding"},
    "P8": {"right_hilum_mask_name", "left_hilum_mask_name", "hilum_mask_names"},
    "P10": {"aortic_knob_mask_name"},
    "S1": {"finding", "pair_metadata"},
    "S2": {"finding", "pair_metadata"},
    "S3": {"finding", "pair_metadata"},
    "T2": {"cvc_target_reference"},
    "T3": {"diaphragm_reference"},
    "T4": {"drain_reference"},
}


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
    prior_image_sha256: str | None = None
    _temporary_workspace: tempfile.TemporaryDirectory | None = field(
        default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        card_json = self.evidence_card.to_json()
        image_digest = self.evidence_card.provenance.get("image_sha256")
        function_episode = _function_episode_audit(
            self.function_episode, image_digest, self.prior_image_sha256)
        return {
            "status": self.status,
            "request": {"question": self.request.question,
                        "dataset_id": self.request.dataset_id,
                        "image_sha256": image_digest,
                        "prior_image_sha256": self.prior_image_sha256},
            "route": self.route.to_dict(),
            "evidence_card": self.evidence_card.to_dict(),
            "evidence_card_sha256": hashlib.sha256(card_json.encode("utf-8")).hexdigest(),
            "response": self.response.to_dict(),
            "consumer_model": self.consumer_model,
            "consumer_input_mode": self.consumer_input_mode,
            "consumer_config": self.consumer_config,
            "prompt_schema_version": ("q4-response-2" if self.evidence_card.function_id == "Q4"
                                       else FUNCTION_RESPONSE_SCHEMA),
            "prompt_sha256": self.prompt_sha256,
            "raw_consumer_output": self.raw_consumer_output,
            "function_episode": function_episode,
            "verification": self.verification,
            "error_code": self.error_code,
        }


def _function_episode_audit(episode: FunctionEpisodeResult | None,
                            image_digest: str | None,
                            prior_image_digest: str | None = None) -> dict[str, Any] | None:
    if episode is None:
        return None
    payload = episode.to_dict()

    sanitized = _redact_image_paths(payload)
    session = sanitized.get("session")
    if isinstance(session, dict):
        session["image_sha256"] = image_digest
    sessions = sanitized.get("study_sessions")
    if isinstance(sessions, dict):
        if isinstance(sessions.get("current"), dict):
            sessions["current"]["image_sha256"] = image_digest
        if prior_image_digest and isinstance(sessions.get("prior"), dict):
            sessions["prior"]["image_sha256"] = prior_image_digest
    return sanitized


def _redact_image_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_image_paths(item) for key, item in value.items()
                if key not in {"image_ref", "image_path", "prior_image_ref", "current_image_ref"}}
    if isinstance(value, list):
        return [_redact_image_paths(item) for item in value]
    return value


def _sanitize_function_episode_files(out_dir: Path, image_digest: str | None,
                                     prior_image_digest: str | None = None) -> None:
    for name in ("session.json", "prior_session.json", "episode.json"):
        path = out_dir / name
        if not path.is_file():
            continue
        payload = _redact_image_paths(json.loads(path.read_text(encoding="utf-8")))
        if name == "session.json" and isinstance(payload, dict):
            payload["image_sha256"] = image_digest
        elif name == "prior_session.json" and isinstance(payload, dict):
            payload["image_sha256"] = prior_image_digest
        elif name == "episode.json" and isinstance(payload, dict):
            session = payload.get("session")
            if isinstance(session, dict):
                session["image_sha256"] = image_digest
            sessions = payload.get("study_sessions")
            if isinstance(sessions, dict):
                if isinstance(sessions.get("current"), dict):
                    sessions["current"]["image_sha256"] = image_digest
                if prior_image_digest and isinstance(sessions.get("prior"), dict):
                    sessions["prior"]["image_sha256"] = prior_image_digest
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


def _parameter_input_names(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        match = re.fullmatch(r"\$input\.([A-Za-z_][A-Za-z0-9_]*)(?:\..*)?", value)
        if match:
            found.add(match.group(1))
    elif isinstance(value, Mapping):
        for item in value.values():
            found.update(_parameter_input_names(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_parameter_input_names(item))
    return found


def _function_inputs(request: VQARequest, function_id: str, spec: FunctionSpec,
                     *, has_prior_image: bool) -> tuple[dict[str, Any] | None, str | None]:
    undeclared_slots = set(request.slots) - set(spec.slots)
    if undeclared_slots:
        return None, f"undeclared semantic slots: {', '.join(sorted(undeclared_slots))}"
    allowed_inputs = _FUNCTION_RUNTIME_INPUTS.get(function_id, set()).copy()
    for step in spec.steps:
        allowed_inputs.update(_parameter_input_names(step.parameters))
    extra_inputs = set(request.inputs) - allowed_inputs
    if extra_inputs:
        return None, f"unsupported function inputs: {', '.join(sorted(extra_inputs))}"

    merged = dict(request.slots)
    for key, value in request.inputs.items():
        if key in merged and merged[key] != value:
            return None, f"conflicting values for function input {key}"
        merged[key] = value
    missing = _REQUIRED_FUNCTION_INPUTS.get(function_id, set()) - set(merged)
    if missing:
        return None, f"missing required function inputs: {', '.join(sorted(missing))}"

    if spec.input_type == "paired_images" and not has_prior_image:
        return None, f"{function_id} requires prior_image_ref or prior_session"
    if spec.input_type != "paired_images" and has_prior_image:
        return None, "prior image/session is only valid for paired-image functions"
    if function_id in {"G1", "G2", "P3", "S1", "S2", "S3"}:
        finding = merged.get("finding")
        if not isinstance(finding, str) or not finding.strip():
            return None, f"{function_id} finding must be a non-empty string"
    if function_id == "P3" and merged["finding"] not in {"nodule", "mass"}:
        return None, "P3 finding must be nodule or mass"
    if function_id == "P8":
        right, left = merged.get("right_hilum_mask_name"), merged.get("left_hilum_mask_name")
        if (not isinstance(right, str) or not isinstance(left, str) or right == left
                or merged.get("hilum_mask_names") != [right, left]):
            return None, "P8 requires distinct right/left mask names and their ordered list"
    if function_id == "G3":
        if (not isinstance(merged.get("lesion_mask_name"), str)
                or not isinstance(merged.get("anatomy_mask_name"), str)):
            return None, "G3 lesion_mask_name and anatomy_mask_name must be strings"
        names = merged.get("region_mask_names")
        if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
                or len(names) != len(set(names))):
            return None, "G3 region_mask_names must be a list of unique mask names"
    if function_id == "P10" and not isinstance(merged.get("aortic_knob_mask_name"), str):
        return None, "P10 aortic_knob_mask_name must be a string"
    reference_inputs = {
        "S1": ("pair_metadata",), "S2": ("pair_metadata",), "S3": ("pair_metadata",),
        "T2": ("cvc_target_reference",), "T3": ("diaphragm_reference",),
        "T4": ("drain_reference",),
    }.get(function_id, ())
    for name in reference_inputs:
        if not isinstance(merged.get(name), Mapping):
            return None, f"{function_id} {name} must be an object"
    if function_id == "S2" and "prior_reference" in merged and not isinstance(
            merged["prior_reference"], Mapping):
        return None, "S2 prior_reference must be an object"
    return merged, None


def _invalid_route(route: RouteDecision, reason: str) -> RouteDecision:
    return RouteDecision("invalid_input", candidates=route.candidates, reason=reason,
                         matched_by=route.matched_by, gate_version=route.gate_version,
                         category_route=route.category_route)


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
    session: EvidenceSession | None = None,
    prior_session: EvidenceSession | None = None,
) -> VQAEpisodeResult:
    """Run one registered function through evidence-card consumption and verification.

    ``inputs`` carries function-specific values; ``slots`` carries declared
    FunctionSpec slots. Paired functions also need ``prior_image_ref`` or a
    matching ``prior_session``. Seeded sessions support caller-sourced masks.
    ``out_dir``, when supplied, must be new and uses owner-only file modes.
    """
    if not isinstance(request, VQARequest):
        raise TypeError("request must be a VQARequest")
    output = Path(out_dir).resolve() if out_dir is not None else None
    if output is not None and output.exists():
        raise ValueError("out_dir must be new; VQA results are never overwritten")

    image = Path(request.image_ref).expanduser().resolve()
    image_exists = image.is_file()
    digest = _sha256(image) if image_exists else None
    prior_image: Path | None = None
    prior_digest: str | None = None
    prior_validation_error: str | None = None
    if request.prior_image_ref is not None:
        prior_image = Path(request.prior_image_ref).expanduser().resolve()
    if prior_session is not None:
        session_prior_image = Path(prior_session.image_ref).expanduser().resolve()
        if prior_image is not None and prior_image != session_prior_image:
            prior_validation_error = "prior_image_ref does not match prior_session"
        prior_image = session_prior_image
    if prior_image is not None and prior_image.is_file():
        prior_digest = _sha256(prior_image)
    elif prior_image is not None:
        prior_validation_error = "prior image file does not exist"
    session_validation_error = None
    if session is not None and Path(session.image_ref).expanduser().resolve() != image:
        session_validation_error = "image_ref does not match the supplied evidence session"
    if output is not None:
        try:
            output.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("out_dir must be new; VQA results are never overwritten") from exc

    def persist(result: VQAEpisodeResult) -> VQAEpisodeResult:
        result.prior_image_sha256 = prior_digest
        if output is not None:
            _write_private_json(output / "vqa_result.json", result.to_dict())
        return result

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
        return persist(result)

    gate = gate or RuleFunctionGate()
    route = gate.route(request.question)
    runtime_inputs: dict[str, Any] = {}
    if route.status == "matched":
        function_id = route.selected_function_ids[0]
        if function_id not in SUPPORTED_VQA_FUNCTIONS:
            route = _invalid_route(route, f"unknown registered function: {function_id}")
        elif prior_validation_error:
            route = _invalid_route(route, prior_validation_error)
        elif session_validation_error:
            route = _invalid_route(route, session_validation_error)
        else:
            spec = load_function_specs()[function_id]
            runtime_inputs, input_error = _function_inputs(
                request, function_id, spec,
                has_prior_image=prior_image is not None or prior_session is not None)
            if input_error:
                route = _invalid_route(route, input_error)

    if route.abstain:
        card = EvidenceCard.routing_refusal(request.question, route.to_dict())
        card = EvidenceCard(card.card_version, card.function_id, card.question, card.status,
                            card.route, card.evidence_nodes, card.claim_ids, card.function_output,
                            card.tool_steps, card.refusal,
                            {**card.provenance, "image_sha256": digest,
                             "study_image_sha256s": ({"current": digest,
                                                      **({"prior": prior_digest} if prior_digest else {})})})
        result = VQAEpisodeResult("abstained", request, route, card,
                                  _abstained(route.reason), None)
        return persist(result)

    function_id = route.selected_function_ids[0]
    def execute(function_out: Path) -> FunctionEpisodeResult:
        return run_function_episode(
            function_id=function_id,
            image_ref=str(image),
            session=session,
            prior_image_ref=str(prior_image) if prior_image is not None else None,
            prior_session=prior_session,
            query=request.question,
            inputs=runtime_inputs,
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
            _sanitize_function_episode_files(function_out_dir, digest, prior_digest)
        card = EvidenceCard(
            "1.0", function_id, request.question, "refused", route.to_dict(), (), (), {}, (),
            {"reason": f"function execution failed ({type(exc).__name__})"},
            {"image_sha256": digest,
             "study_image_sha256s": ({"current": digest,
                                      **({"prior": prior_digest} if prior_digest else {})}),
             "spec_hash": None, "function_version": None},
        )
        response = _abstained(f"function execution failed ({type(exc).__name__})")
        result = VQAEpisodeResult("abstained", request, route, card, response, None,
                                  _temporary_workspace=temporary_workspace)
        return persist(result)
    if function_out_dir is not None:
        _sanitize_function_episode_files(function_out_dir, digest, prior_digest)
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
        if prior_digest is not None and card.provenance.get("study_image_sha256s", {}).get("prior") != prior_digest:
            raise ValueError("prior image changed between VQA request validation and function execution")
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
                if card.function_id == "Q4":
                    verification = verify_q4_answer(
                        card, raw_output,
                        input_mode=getattr(consumer, "input_mode", "evidence_only"),
                    )
                else:
                    verification = verify_function_answer(card, raw_output)
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

    result._temporary_workspace = temporary_workspace
    return persist(result)
