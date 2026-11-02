"""Question-to-evidence VQA pipeline."""

from .consumer import (AnswerConsumer, CallableConsumer, TransformersLingshuConsumer,
                       TransformersMedGemmaConsumer, TransformersQwen3VLConsumer,
                       TransformersVQAConsumer, build_answer_prompt)
from .evidence_card import EvidenceCard
from .gate import FunctionGate, RuleFunctionGate
from .category_gate import (CATEGORY_IDS, CategoryGate, CategoryRouteDecision,
                            CategorySelectionPolicy)
from .category_function_gate import CategoryFunctionGate
from .runner import VQAEpisodeResult, run_vqa_episode
from .schema import RouteCandidate, RouteDecision, VQARequest, VQAResponse
from .verifier import verify_q4_answer

__all__ = [
    "AnswerConsumer", "CallableConsumer", "TransformersVQAConsumer",
    "TransformersQwen3VLConsumer", "TransformersLingshuConsumer",
    "TransformersMedGemmaConsumer", "build_answer_prompt", "EvidenceCard",
    "FunctionGate", "RuleFunctionGate", "VQAEpisodeResult", "run_vqa_episode",
    "RouteCandidate", "RouteDecision", "VQARequest", "VQAResponse", "verify_q4_answer",
    "CATEGORY_IDS", "CategoryGate", "CategoryRouteDecision", "CategorySelectionPolicy",
    "CategoryFunctionGate",
]
