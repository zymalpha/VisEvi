"""Consumer interface for evidence-only and image-assisted local reasoners."""
from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, Mapping, Protocol

from PIL import Image

from .evidence_card import EvidenceCard


class AnswerConsumer(Protocol):
    model_id: str

    def generate(self, *, question: str, evidence_card: EvidenceCard,
                 image_ref: str | None = None) -> Mapping[str, Any] | str:
        """Return JSON-like answer, citations, and limitations."""


def build_answer_prompt(question: str, evidence_card: EvidenceCard, *,
                        include_image: bool = False) -> str:
    """Build a deterministic prompt for evidence-only or image-assisted answering."""
    if include_image:
        instruction = (
            "Answer the user's chest X-ray question by inspecting the provided image and the evidence card. "
            "Use only the Q4 view labels PA, AP, lateral, or unknown. Cite an existing claim ID. "
            "Set visual_assessment to consistent only when the image agrees with the cited claim; "
            "use conflict or uncertain otherwise. "
            "Do not add diagnoses, history, or unrelated findings. "
        )
    else:
        instruction = (
            "Answer the user's chest X-ray question using only the evidence card below. "
            "Do not add visual findings, diagnoses, history, or measurements absent from the card. "
            "Set visual_assessment to not_assessed. "
        )
    return (
        instruction
        + "If the card status is not answerable or does not support a direct answer, abstain. "
        "Cite one or more existing claim IDs. Return one JSON object with exactly these keys: "
        '"schema_version", "answer", "citations", "limitations", "visual_assessment". '
        'Set schema_version to "q4-response-2". Answer with only PA, AP, lateral, or unknown. '
        "limitations must be a list containing only these codes: "
        "image_quality_not_assessed, projection_metadata_not_used. Do not include free-text limitations.\n\n"
        f"Question: {question.strip()}\n\n{evidence_card.to_text()}"
    )


def build_image_only_prompt(question: str) -> str:
    return (
        "Identify the view position in the provided chest X-ray. "
        "Return exactly one label: PA, AP, lateral, or unknown. "
        "Use unknown when the image does not support a clear choice. "
        "Do not add citations, diagnoses, history, or explanations.\n\n"
        f"Question: {question.strip()}"
    )


def _generate_messages(consumer, messages) -> str:
    inputs = consumer.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(consumer.model.device)
    input_length = inputs["input_ids"].shape[-1]
    with consumer._inference_mode():
        generated = consumer.model.generate(
            **inputs,
            max_new_tokens=consumer.max_new_tokens,
            do_sample=False,
        )
    return consumer.processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()


class CallableConsumer:
    """Adapt a local callable to the consumer contract, useful for test harnesses."""

    def __init__(self, model_id: str, generate_fn) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        self.model_id = model_id
        self._generate_fn = generate_fn

    def generate(self, *, question: str, evidence_card: EvidenceCard,
                 image_ref: str | None = None) -> Mapping[str, Any] | str:
        del image_ref
        return self._generate_fn(build_answer_prompt(question, evidence_card))


class TransformersMedGemmaConsumer:
    """Local EvidenceCard consumer backed by MedGemma 1.5 4B.

    The input mode selects evidence-only or image-and-evidence messages.
    Local-only loading prevents silent downloads or API calls.
    """

    DEFAULT_MODEL_ID = os.environ.get(
        "VISEVI_MEDGEMMA_MODEL", "/ssd/visual_reasoning/models/medgemma-1.5-4b-it")
    FAMILY = "gemma3_medgemma_1_5"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, *, device: str = "cuda:4",
                 max_new_tokens: int = 128, local_files_only: bool = True,
                 input_mode: str = "image_and_evidence",
                 model=None, processor=None, torch_module=None) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        if device != "cpu" and device not in {f"cuda:{index}" for index in range(4, 8)}:
            raise ValueError("device must be cpu or one of cuda:4 through cuda:7")
        if input_mode not in {"evidence_only", "image_and_evidence"}:
            raise ValueError("input_mode must be evidence_only or image_and_evidence")
        self.model_id = model_id
        self.family = self.FAMILY
        self.device = device
        self.input_mode = input_mode
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.torch_dtype = "bfloat16"
        self.trust_remote_code = False
        self._torch = torch_module
        self.model = model
        self.processor = processor
        self.transformers_version = None
        self.torch_version = getattr(self._torch, "__version__", None)
        if self.model is None or self.processor is None:
            try:
                import torch
                import transformers
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "TransformersMedGemmaConsumer requires torch and transformers in the active environment"
                ) from exc
            self._torch = torch
            self.torch_version = torch.__version__
            self.transformers_version = transformers.__version__
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
                trust_remote_code=False,
                use_safetensors=True,
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                trust_remote_code=False,
                use_fast=False,
            )
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()
        self._inference_mode = self._torch.inference_mode if self._torch else nullcontext

    def generate(self, *, question: str, evidence_card: EvidenceCard,
                 image_ref: str | None = None) -> str:
        prompt = build_answer_prompt(question, evidence_card,
                                     include_image=self.input_mode == "image_and_evidence")
        if self.input_mode == "image_and_evidence":
            if image_ref is None:
                raise ValueError("image_ref is required for image_and_evidence mode")
            with Image.open(Path(image_ref).expanduser().resolve()) as source:
                image = source.convert("RGB")
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with self._inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()

    def generate_image_only(self, *, question: str, image_ref: str) -> str:
        with Image.open(Path(image_ref).expanduser().resolve()) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_image_only_prompt(question)},
        ]}]
        return _generate_messages(self, messages)


class TransformersVQAConsumer:
    """Local Transformers VLM consumer for evidence-only or image+evidence runs."""

    def __init__(self, model_id: str, *, family: str, device: str = "cuda:4",
                 input_mode: str = "image_and_evidence", max_new_tokens: int = 128,
                 local_files_only: bool = True, model=None, processor=None,
                 torch_module=None) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if family not in {"qwen3_vl", "qwen2_5_vl", "medgemma"}:
            raise ValueError("unsupported model family")
        if device != "cpu" and device not in {f"cuda:{index}" for index in range(4, 8)}:
            raise ValueError("device must be cpu or one of cuda:4 through cuda:7")
        if input_mode not in {"evidence_only", "image_and_evidence"}:
            raise ValueError("input_mode must be evidence_only or image_and_evidence")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        self.model_id = model_id
        self.family = family
        self.device = device
        self.input_mode = input_mode
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.torch_dtype = "bfloat16"
        self.trust_remote_code = False
        self.model = model
        self.processor = processor
        self._torch = torch_module
        self.transformers_version = None
        self.torch_version = getattr(self._torch, "__version__", None)
        if self.model is None or self.processor is None:
            try:
                import torch
                import transformers
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError("local VQA consumers require torch and transformers") from exc
            self._torch = torch
            self.torch_version = torch.__version__
            self.transformers_version = transformers.__version__
            class_name = {
                "qwen3_vl": "Qwen3VLForConditionalGeneration",
                "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
            }.get(family)
            if class_name is None:
                model_class = AutoModelForImageTextToText
            else:
                model_class = getattr(transformers, class_name, None)
                if model_class is None:
                    required_version = "4.57.0" if family == "qwen3_vl" else "4.50.0"
                    raise RuntimeError(
                        f"{family} requires transformers>={required_version}; "
                        f"found {transformers.__version__}"
                    )
            self.model = model_class.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
                trust_remote_code=False,
                use_safetensors=True,
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                trust_remote_code=False,
                use_fast=False,
            )
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()
        self._inference_mode = self._torch.inference_mode if self._torch else nullcontext

    def generate(self, *, question: str, evidence_card: EvidenceCard,
                 image_ref: str | None = None) -> str:
        prompt = build_answer_prompt(question, evidence_card,
                                     include_image=self.input_mode == "image_and_evidence")
        if self.input_mode == "image_and_evidence":
            if image_ref is None:
                raise ValueError("image_ref is required for image_and_evidence mode")
            image_path = Path(image_ref).expanduser().resolve()
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with self._inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(generated[0][input_length:], skip_special_tokens=True).strip()

    def generate_image_only(self, *, question: str, image_ref: str) -> str:
        with Image.open(Path(image_ref).expanduser().resolve()) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_image_only_prompt(question)},
        ]}]
        return _generate_messages(self, messages)


class TransformersQwen3VLConsumer(TransformersVQAConsumer):
    DEFAULT_MODEL_ID = "/ssd/visual_reasoning/zym/models/qwen3-vl-8b-instruct"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs) -> None:
        super().__init__(model_id, family="qwen3_vl", **kwargs)


class TransformersLingshuConsumer(TransformersVQAConsumer):
    DEFAULT_MODEL_ID = "/ssd/visual_reasoning/zym/models/lingshu-7b"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs) -> None:
        super().__init__(model_id, family="qwen2_5_vl", **kwargs)
