"""Contract tests for the first VQA vertical slice; uses a synthetic A24 tool."""
from __future__ import annotations

import json
from pathlib import Path
import stat
from contextlib import nullcontext

import pytest

from visevi.core.registry import ToolRegistry
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit
from visevi.functions.specs import FUNCTION_IDS
from visevi.vqa import (EvidenceCard, RuleFunctionGate, VQARequest,
                        TransformersMedGemmaConsumer, run_vqa_episode,
                        verify_function_answer, verify_q4_answer)


class FakeClassifyView:
    name = "classify_view"

    def __init__(self, view="PA", status="ok", assumed=False):
        self.view = view
        self.status = status
        self.assumed = assumed
        self.calls = 0

    def describe(self):
        return ToolSpec(
            self.name, "perception", "synthetic view classifier",
            parameter_schema({"output_dir": {"type": "string"},
                              "device": {"type": "string"}}, ("output_dir",)),
            evidence_result_schema(),
        )

    def run(self, session, **params):
        self.calls += 1
        return [EvidenceUnit(
            id=session.next_id("obs"), category="observation",
            evidence_type="global_condition", finding_or_anatomy="view_position",
            value={"view": self.view, "status": self.status, "assumed": self.assumed,
                   "raw_text": "synthetic", "option_raw_probabilities": {
                       "A": 0.91, "B": 0.08, "C": 0.01}},
            confidence=None, source_tool=self.name,
            validity_conditions=["synthetic test output"],
        )]


class FakeConsumer:
    model_id = "test-consumer"

    def __init__(self, answer="PA", citation=None):
        self.answer = answer
        self.citation = citation
        self.received_card = None
        self.input_mode = "evidence_only"

    def generate(self, *, question, evidence_card, image_ref=None):
        del image_ref
        self.received_card = evidence_card
        claim_id = self.citation or evidence_card.claim_ids[0]
        return {"schema_version": "q4-response-2", "answer": self.answer,
                "citations": [claim_id], "limitations": [],
                "visual_assessment": ("consistent" if self.input_mode == "image_and_evidence"
                                      else "not_assessed")}


class FakeFunctionConsumer:
    model_id = "test-function-consumer"
    input_mode = "evidence_only"

    def __init__(self):
        self.received_card = None

    def generate(self, *, question, evidence_card, image_ref=None):
        del question, image_ref
        self.received_card = evidence_card
        return {"schema_version": "function-response-1",
                "function_output": evidence_card.function_output,
                "citations": list(evidence_card.claim_ids)}


def _image(path: Path) -> Path:
    path.write_bytes(b"synthetic-image-bytes")
    return path


def _registry(tool: FakeClassifyView) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def test_q4_end_to_end_routes_runs_builds_lossless_card_and_verifies(tmp_path):
    image = _image(tmp_path / "image.bin")
    output = tmp_path / "run"
    tool = FakeClassifyView()
    consumer = FakeConsumer()

    result = run_vqa_episode(
        VQARequest(str(image), "这张胸片是 PA 还是 AP 投照？", dataset_id="synthetic",
                   sample_id="patient-123"),
        consumer=consumer,
        registry=_registry(tool),
        out_dir=output,
    )

    assert result.status == "answered"
    assert result.route.selected_function_ids == ("Q4",)
    assert result.response.answer == "PA"
    assert result.response.citations == result.evidence_card.claim_ids
    assert result.to_dict()["consumer_input_mode"] == "evidence_only"
    assert tool.calls == 1
    assert len(result.evidence_card.evidence_nodes) == 3
    view = next(node for node in result.evidence_card.evidence_nodes
                if node["finding_or_anatomy"] == "view_position")
    assert view["value"]["option_raw_probabilities"] == {"A": 0.91, "B": 0.08, "C": 0.01}
    assert json.loads(result.evidence_card.to_json())["function_output"]["view"] == "PA"
    assert (output / "function_episode" / "episode.json").is_file()
    assert (output / "vqa_result.json").is_file()
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    audit = json.loads((output / "vqa_result.json").read_text())
    assert audit["raw_consumer_output"]["schema_version"] == "q4-response-2"
    assert audit["verification"]["parsed_output"] == audit["raw_consumer_output"]
    assert len(audit["evidence_card_sha256"]) == 64
    assert audit["prompt_sha256"]
    audit_json = json.dumps(audit, ensure_ascii=False)
    assert str(image) not in audit_json
    assert "patient-123" not in audit_json
    assert "sample_id" not in audit["request"]
    assert audit["function_episode"]["session"]["image_sha256"] == audit["request"]["image_sha256"]
    assert "image_ref" not in audit["function_episode"]["session"]
    for path in output.rglob("*"):
        if path.is_file():
            persisted = path.read_text(encoding="utf-8")
            assert str(image) not in persisted
            assert "patient-123" not in persisted
    saved_episode = json.loads((output / "function_episode" / "episode.json").read_text())
    saved_session = json.loads((output / "function_episode" / "session.json").read_text())
    assert "image_ref" not in saved_episode["session"]
    assert "image_ref" not in saved_session
    assert saved_session["image_sha256"] == audit["request"]["image_sha256"]


def test_p9_runs_through_generic_card_consumer_and_claim_bound_verifier(tmp_path):
    from PIL import Image
    from test_function_episodes import Harness

    image = tmp_path / "p9.png"
    Image.new("L", (16, 12), 80).save(image)
    harness = Harness(image)
    consumer = FakeFunctionConsumer()
    result = run_vqa_episode(
        VQARequest(str(image), "P9"), consumer=consumer, registry=harness.registry,
    )

    assert result.status == "answered"
    assert result.route.selected_function_ids == ("P9",)
    assert result.evidence_card.function_id == "P9"
    assert result.response.answer == json.dumps(
        result.evidence_card.function_output, ensure_ascii=False, sort_keys=True)
    assert result.to_dict()["prompt_schema_version"] == "function-response-1"
    assert result.verification["verification_level"] == "structured_output_and_claim_binding"
    assert result.verification["medical_accuracy_verified"] is False
    assert consumer.received_card is result.evidence_card

    prompt = __import__("visevi.vqa.consumer", fromlist=["build_answer_prompt"]).build_answer_prompt(
        "CTR?", result.evidence_card)
    assert "without changing, omitting, or adding any field" in prompt
    assert result.evidence_card.to_text() in prompt

    good = {"schema_version": "function-response-1",
            "function_output": result.evidence_card.function_output,
            "citations": list(result.evidence_card.claim_ids)}
    bad_output = {**good, "function_output": {**good["function_output"], "CTR": 9.0}}
    bad_citation = {**good, "citations": ["obs-9999"]}
    assert verify_function_answer(result.evidence_card, good)["accepted"]
    assert not verify_function_answer(result.evidence_card, bad_output)["accepted"]
    assert not verify_function_answer(result.evidence_card, bad_citation)["accepted"]


def test_function_required_inputs_refuse_before_a_tools_run(tmp_path):
    from PIL import Image
    from test_function_episodes import Harness

    image = tmp_path / "missing-finding.png"
    Image.new("L", (16, 12), 80).save(image)
    harness = Harness(image)
    result = run_vqa_episode(
        VQARequest(str(image), "G1"), consumer=FakeFunctionConsumer(),
        registry=harness.registry,
    )

    assert result.route.status == "invalid_input"
    assert "finding" in result.route.reason
    assert result.status == "abstained"
    assert harness.calls == []


@pytest.mark.parametrize("function_id", FUNCTION_IDS)
def test_all_registered_functions_enter_vqa_without_q4_allowlist_block(tmp_path, function_id):
    image = _image(tmp_path / f"{function_id}.bin")
    result = run_vqa_episode(
        VQARequest(str(image), function_id), registry=ToolRegistry(),
    )

    assert result.route.status != "unsupported"
    assert result.route.status in {"matched", "invalid_input"}
    assert result.response.status == "abstained"


def test_malformed_function_input_becomes_route_refusal_not_python_error(tmp_path):
    image = _image(tmp_path / "bad-input.bin")
    result = run_vqa_episode(
        VQARequest(str(image), "P3", slots={"finding": ["nodule"]}),
        registry=ToolRegistry(),
    )

    assert result.route.status == "invalid_input"
    assert "finding must be a non-empty string" in result.route.reason


def test_s2_paired_image_pipeline_records_both_hashes_without_paths(tmp_path):
    from PIL import Image
    from test_function_s1 import PairHarness, pair_metadata

    prior = tmp_path / "prior.png"
    current = tmp_path / "current.png"
    Image.new("L", (20, 16), 40).save(prior)
    Image.new("L", (20, 16), 80).save(current)
    harness = PairHarness(prior, current)
    consumer = FakeFunctionConsumer()
    result = run_vqa_episode(
        VQARequest(str(current), "S2", inputs={"finding": "nodule",
                                               "pair_metadata": pair_metadata()},
                   prior_image_ref=str(prior)),
        consumer=consumer, registry=harness.registry,
    )

    assert result.status == "answered"
    assert result.route.selected_function_ids == ("S2",)
    hashes = result.evidence_card.provenance["study_image_sha256s"]
    assert hashes["current"] == result.to_dict()["request"]["image_sha256"]
    assert hashes["prior"] == result.to_dict()["request"]["prior_image_sha256"]
    audit = json.dumps(result.to_dict(), ensure_ascii=False)
    assert str(prior) not in audit and str(current) not in audit
    assert result.function_episode.study_sessions["prior"].image_ref == str(prior.resolve())


def test_paired_function_requires_prior_image_before_tools_run(tmp_path):
    image = _image(tmp_path / "single.png")
    result = run_vqa_episode(
        VQARequest(str(image), "S2", inputs={"finding": "nodule",
                                               "pair_metadata": {}}),
        consumer=FakeFunctionConsumer(), registry=ToolRegistry(),
    )

    assert result.route.status == "invalid_input"
    assert "prior_image_ref" in result.route.reason
    assert result.status == "abstained"


def test_q4_answer_conflict_with_claim_is_rejected(tmp_path):
    image = _image(tmp_path / "image.bin")
    result = run_vqa_episode(
        VQARequest(str(image), "PA 还是 AP？"),
        consumer=FakeConsumer(answer="AP"),
        registry=_registry(FakeClassifyView(view="PA")),
    )

    assert result.status == "abstained"
    assert "conflicts" in result.response.refusal_reason


def test_vqa_runner_passes_resolved_original_image_to_consumer(tmp_path):
    import os

    class RecordingConsumer(FakeConsumer):
        received_image_ref = None

        def generate(self, *, question, evidence_card, image_ref=None):
            self.received_image_ref = image_ref
            return super().generate(question=question, evidence_card=evidence_card,
                                    image_ref=image_ref)

    image = _image(tmp_path / "image.bin")
    consumer = RecordingConsumer()

    result = run_vqa_episode(
        VQARequest(os.path.relpath(image, Path.cwd()), "这张胸片是 PA 还是 AP 投照？"),
        consumer=consumer, registry=_registry(FakeClassifyView()),
    )

    assert result.status == "answered"
    assert consumer.received_image_ref == str(image.resolve())
    assert Path(consumer.received_image_ref).is_absolute()


@pytest.mark.parametrize("include_image", [False, True], ids=["evidence-only", "image-and-evidence"])
def test_answer_prompt_separates_evidence_only_and_image_instructions(include_image):
    from visevi.vqa.consumer import build_answer_prompt

    card = EvidenceCard("1.0", "Q4", "view?", "answerable", {}, (), (), {}, (), None, {})
    prompt = build_answer_prompt("  view?  ", card, include_image=include_image)

    assert "Question: view?\n" in prompt
    assert card.to_text() in prompt
    assert "Cite one or more existing claim IDs" in prompt
    assert '"schema_version", "answer", "citations", "limitations", "visual_assessment"' in prompt
    assert '"q4-response-2"' in prompt
    assert "Do not include free-text limitations" in prompt
    if include_image:
        assert "inspecting the provided image and the evidence card" in prompt
        assert "use conflict or uncertain otherwise" in prompt
        assert "using only the evidence card" not in prompt
        assert "Do not add visual findings" not in prompt
    else:
        assert "using only the evidence card" in prompt
        assert "inspecting the provided image" not in prompt


def test_multimodal_consumer_defaults_to_image_and_evidence():
    from visevi.vqa import TransformersVQAConsumer

    class FakeProcessor:
        pass

    class FakeModel:
        def eval(self):
            pass

    consumer = TransformersVQAConsumer(
        "test", family="qwen3_vl", model=FakeModel(), processor=FakeProcessor(),
        torch_module=type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})(),
    )

    assert consumer.input_mode == "image_and_evidence"


def test_medgemma_default_points_to_the_existing_local_checkpoint():
    assert TransformersMedGemmaConsumer.DEFAULT_MODEL_ID == \
        "/ssd/visual_reasoning/models/medgemma-1.5-4b-it"


def test_vqa_result_records_consumer_input_mode(tmp_path):
    image = _image(tmp_path / "image.bin")
    consumer = FakeConsumer()
    consumer.input_mode = "image_and_evidence"

    result = run_vqa_episode(
        VQARequest(str(image), "这张胸片是 PA 还是 AP？"),
        consumer=consumer, registry=_registry(FakeClassifyView()),
    )

    assert result.to_dict()["consumer_input_mode"] == "image_and_evidence"


def test_ambiguous_and_unknown_questions_do_not_call_tools(tmp_path):
    image = _image(tmp_path / "image.bin")
    tool = FakeClassifyView()
    registry = _registry(tool)
    gate = RuleFunctionGate()

    ambiguous = run_vqa_episode(
        VQARequest(str(image), "请判断投照并测量心胸比"),
        consumer=FakeConsumer(), gate=gate, registry=registry,
    )
    unknown = run_vqa_episode(
        VQARequest(str(image), "这张片子有什么发现？"),
        consumer=FakeConsumer(), gate=gate, registry=registry,
    )

    assert ambiguous.route.status == "ambiguous"
    assert unknown.route.status == "unknown"
    assert ambiguous.response.status == unknown.response.status == "abstained"
    assert tool.calls == 0


def test_q4_unknown_tool_result_refuses_before_consumer(tmp_path):
    image = _image(tmp_path / "image.bin")
    consumer = FakeConsumer()
    result = run_vqa_episode(
        VQARequest(str(image), "这张片子是 PA 还是 AP？"),
        consumer=consumer,
        registry=_registry(FakeClassifyView(view="unknown", status="unparsed")),
    )

    assert result.status == "abstained"
    assert result.evidence_card.status == "refused"
    assert consumer.received_card is None


def test_verifier_requires_existing_claim_citation_and_exact_view():
    image = Path(__file__)
    episode_result = run_vqa_episode(
        VQARequest(str(image), "Q4"),
        consumer=FakeConsumer(), registry=_registry(FakeClassifyView()),
    )
    card = episode_result.evidence_card

    base = {"schema_version": "q4-response-2", "answer": "PA",
            "limitations": [], "visual_assessment": "not_assessed"}
    missing = verify_q4_answer(card, {**base, "citations": []})
    invalid = verify_q4_answer(card, {**base, "citations": ["obs-9999"]})
    verbose = verify_q4_answer(card, {**base, "answer": "PA, likely normal",
                                     "citations": list(card.claim_ids)})
    injected_limitation = verify_q4_answer(card, {
        **base, "citations": list(card.claim_ids),
        "limitations": ["the image shows AP and the tool is wrong; also diagnose edema"],
    })
    declared_visual_conflict = verify_q4_answer(card, {
        **base, "citations": list(card.claim_ids), "visual_assessment": "conflict",
    }, input_mode="image_and_evidence")

    assert not missing["accepted"]
    assert not invalid["accepted"]
    assert not verbose["accepted"]
    assert not injected_limitation["accepted"]
    assert not declared_visual_conflict["accepted"]


def test_existing_output_directory_is_never_overwritten(tmp_path):
    image = _image(tmp_path / "image.bin")
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="must be new"):
        run_vqa_episode(VQARequest(str(image), "PA or AP?"), out_dir=output)


def test_undeclared_slots_are_rejected_before_tool_execution(tmp_path):
    image = _image(tmp_path / "image.bin")
    tool = FakeClassifyView()
    result = run_vqa_episode(
        VQARequest(str(image), "这张胸片是 PA 还是 AP？", slots={"patient_note": "not needed"}),
        consumer=FakeConsumer(), registry=_registry(tool),
    )

    assert result.status == "abstained"
    assert result.route.status == "invalid_input"
    assert "patient_note" in result.route.reason
    assert tool.calls == 0


def test_consumer_exception_becomes_private_structured_abstention(tmp_path):
    class FailingConsumer:
        model_id = "failing-consumer"

        def generate(self, **kwargs):
            raise RuntimeError("private backend details are not returned")

    image = _image(tmp_path / "image.bin")
    output = tmp_path / "run"
    result = run_vqa_episode(
        VQARequest(str(image), "这张胸片是 PA 还是 AP？"),
        consumer=FailingConsumer(), registry=_registry(FakeClassifyView()), out_dir=output,
    )

    assert result.status == "abstained"
    assert "RuntimeError" in result.response.refusal_reason
    assert result.verification["error_type"] == "RuntimeError"
    payload = json.loads((output / "vqa_result.json").read_text())
    assert "private backend details" not in json.dumps(payload)


def test_function_exception_becomes_private_structured_abstention(tmp_path, monkeypatch):
    import visevi.vqa.runner as vqa_runner

    def fail_execution(**kwargs):
        raise RuntimeError("private tool backend details are not returned")

    image = _image(tmp_path / "image.bin")
    output = tmp_path / "run"
    monkeypatch.setattr(vqa_runner, "run_function_episode", fail_execution)
    result = run_vqa_episode(
        VQARequest(str(image), "这张胸片是 PA 还是 AP？"),
        consumer=FakeConsumer(), registry=_registry(FakeClassifyView()), out_dir=output,
    )

    assert result.status == "abstained"
    assert "RuntimeError" in result.response.refusal_reason
    payload = json.loads((output / "vqa_result.json").read_text())
    assert payload["evidence_card"]["status"] == "refused"
    assert "private tool backend details" not in json.dumps(payload)


@pytest.mark.parametrize("input_mode", ["evidence_only", "image_and_evidence"])
def test_medgemma_consumer_sends_requested_inputs_and_decodes_only_new_tokens(tmp_path, input_mode):
    class FakeTensor:
        shape = (1, 2)

    class FakeInputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class FakeProcessor:
        def __init__(self):
            self.messages = None
            self.decoded = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            assert kwargs["add_generation_prompt"] is True
            return FakeInputs(input_ids=FakeTensor())

        def decode(self, tokens, *, skip_special_tokens):
            self.decoded = (tokens, skip_special_tokens)
            return ('{"schema_version":"q4-response-2","answer":"PA",'
                    '"citations":["claim-1"],"limitations":[],'
                    '"visual_assessment":"not_assessed"}')

    class FakeModel:
        device = "cpu"

        def eval(self):
            self.evaluated = True

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 12
            return [[1, 2, 3, 4]]

    processor = FakeProcessor()
    consumer = TransformersMedGemmaConsumer(
        "local-test-model", max_new_tokens=12, model=FakeModel(), processor=processor,
        device="cpu", input_mode=input_mode,
        torch_module=type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})(),
    )
    assert consumer.model.evaluated is True
    card = EvidenceCard(
        "1.0", "Q4", "view?", "answerable", {}, (), ("claim-1",), {}, (), None, {},
    )

    from PIL import Image
    image_path = tmp_path / "original.png"
    Image.new("L", (3, 2), color=123).save(image_path)
    output = consumer.generate(question="view?", evidence_card=card, image_ref=str(image_path))

    assert '"answer":"PA"' in output
    assert processor.decoded == ([3, 4], True)
    content = processor.messages[0]["content"]
    if input_mode == "image_and_evidence":
        assert [part["type"] for part in content] == ["image", "text"]
        assert isinstance(content[0]["image"], Image.Image)
        assert content[0]["image"].mode == "RGB"
        assert content[0]["image"].size == (3, 2)
        assert content[0]["image"].getpixel((0, 0)) == (123, 123, 123)
        assert "inspecting the provided image" in content[1]["text"]
        assert card.to_text() in content[1]["text"]
    else:
        assert [part["type"] for part in content] == ["text"]
        assert "using only the evidence card" in content[0]["text"]


@pytest.mark.parametrize("family", ["medgemma", "qwen3_vl", "qwen2_5_vl"])
def test_image_and_evidence_consumer_requires_image_ref(family):
    from visevi.vqa import TransformersVQAConsumer

    kwargs = {"device": "cpu", "input_mode": "image_and_evidence",
              "model": object(), "processor": object()}
    if family == "medgemma":
        consumer = TransformersMedGemmaConsumer("test", **kwargs)
    else:
        consumer = TransformersVQAConsumer("test", family=family, **kwargs)
    card = EvidenceCard("1.0", "Q4", "view?", "answerable", {}, (), (), {}, (), None, {})

    with pytest.raises(ValueError, match="image_ref is required"):
        consumer.generate(question="view?", evidence_card=card)


def test_multimodel_consumer_supports_evidence_only_and_image_modes(tmp_path):
    class FakeTensor:
        shape = (1, 2)

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return FakeInputs(input_ids=FakeTensor())

        def decode(self, tokens, *, skip_special_tokens):
            return "{}"

    class FakeModel:
        device = "cpu"

        def eval(self):
            pass

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    from visevi.vqa import TransformersVQAConsumer
    image = tmp_path / "small.png"
    from PIL import Image
    Image.new("RGB", (2, 2), color="white").save(image)
    card = EvidenceCard("1.0", "Q4", "view?", "answerable", {}, (), (), {}, (), None, {})
    evidence_processor = FakeProcessor()
    evidence = TransformersVQAConsumer(
        "test", family="qwen3_vl", input_mode="evidence_only",
        model=FakeModel(), processor=evidence_processor,
        torch_module=type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})(),
    )
    evidence.generate(question="view?", evidence_card=card)
    assert [part["type"] for part in evidence_processor.messages[0]["content"]] == ["text"]
    assert "using only the evidence card" in evidence_processor.messages[0]["content"][0]["text"]

    image_processor = FakeProcessor()
    image_consumer = TransformersVQAConsumer(
        "test", family="qwen2_5_vl", input_mode="image_and_evidence",
        model=FakeModel(), processor=image_processor,
        torch_module=type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})(),
    )
    image_consumer.generate(question="view?", evidence_card=card, image_ref=str(image))
    assert [part["type"] for part in image_processor.messages[0]["content"]] == ["image", "text"]
    assert "inspecting the provided image" in image_processor.messages[0]["content"][1]["text"]

    image_only_processor = FakeProcessor()
    image_only_processor.decode = lambda tokens, *, skip_special_tokens: "PA"
    image_only_consumer = TransformersVQAConsumer(
        "test", family="qwen3_vl", input_mode="image_and_evidence",
        model=FakeModel(), processor=image_only_processor,
        torch_module=type("FakeTorch", (), {"inference_mode": staticmethod(nullcontext)})(),
    )
    assert image_only_consumer.generate_image_only(question="What view?", image_ref=str(image)) == "PA"
    only_content = image_only_processor.messages[0]["content"]
    assert [part["type"] for part in only_content] == ["image", "text"]
    assert "What view?" in only_content[1]["text"]
    assert "evidence card" not in only_content[1]["text"].lower()
    assert "PA, AP, lateral, or unknown" in only_content[1]["text"]


def test_vqa_request_rejects_non_json_slots():
    with pytest.raises(ValueError, match="finite JSON"):
        VQARequest("image.png", "question?", slots={"condition": float("nan")})


def test_missing_image_returns_structured_abstention_and_persists_result(tmp_path):
    output = tmp_path / "missing-image-run"
    result = run_vqa_episode(
        VQARequest(str(tmp_path / "missing.png"), "PA or AP?"), out_dir=output,
    )

    assert result.status == "abstained"
    assert result.error_code == "INPUT_IMAGE_NOT_FOUND"
    assert result.route.status == "invalid_input"
    saved = json.loads((output / "vqa_result.json").read_text())
    assert saved["error_code"] == "INPUT_IMAGE_NOT_FOUND"
    assert saved["response"]["status"] == "abstained"


def test_temporary_function_artifacts_live_through_consumer_call(tmp_path, monkeypatch):
    import gc
    import visevi.vqa.runner as vqa_runner

    image = _image(tmp_path / "image.bin")
    original = vqa_runner.run_function_episode
    artifacts = []

    def create_artifact(**kwargs):
        episode = original(**kwargs)
        artifact = Path(kwargs["out_dir"]) / "consumer-visible.txt"
        artifact.write_text("available until result release")
        artifacts.append(artifact)
        return episode

    class ArtifactConsumer(FakeConsumer):
        def generate(self, *, question, evidence_card, image_ref=None):
            assert artifacts and artifacts[0].is_file()
            return super().generate(question=question, evidence_card=evidence_card,
                                    image_ref=image_ref)

    monkeypatch.setattr(vqa_runner, "run_function_episode", create_artifact)
    result = run_vqa_episode(
        VQARequest(str(image), "这张胸片是 PA 还是 AP 投照？"), consumer=ArtifactConsumer(),
        registry=_registry(FakeClassifyView()),
    )
    assert result.status == "answered"
    assert artifacts[0].is_file()
    assert result._temporary_workspace is not None
    result = None
    gc.collect()
    assert not artifacts[0].exists()


def test_evidence_card_preserves_nested_fields_and_serializes_stably():
    card = EvidenceCard(
        "1.0", "Q4", "view?", "answerable", {"status": "matched"},
        ({"id": "obs-1", "category": "observation", "value": {"nested": [1, {"x": 2.0}]}},),
        ("claim-1",), {"view": "PA"}, (), None,
        {"image_sha256": "a" * 64, "spec_hash": "b" * 64, "function_version": "1.0"},
    )
    assert card.to_json() == card.to_json()
    assert json.loads(card.to_json())["evidence_nodes"][0]["value"]["nested"][1]["x"] == 2.0
