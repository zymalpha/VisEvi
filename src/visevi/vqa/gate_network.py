"""Five independent small neural gates; imported only by routing users.

This module defines architecture and serving adapters, not a training loss,
optimizer, text encoder, pretrained checkpoint, or a differentiable tool runner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from .category_gate import (CATEGORY_IDS, CategoryRouteDecision,
                            CategorySelectionPolicy)


@dataclass(frozen=True)
class GateNetworkConfig:
    input_dim: int
    hidden_dim: int = 128
    architecture: str = "mlp"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.architecture not in {"linear", "mlp"}:
            raise ValueError("architecture must be linear or mlp")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return {**asdict(self), "categories": list(CATEGORY_IDS),
                "activation": "independent_sigmoid", "version": "1.0"}


class FiveCategoryGates(nn.Module):
    """One independently parameterized gate for each of Q/P/T/S/G.

    ``forward`` maps floating features [batch, input_dim] to logits [batch, 5]
    in CATEGORY_IDS order, without detaching the encoder or gate gradients.
    ``activation_scores`` is also differentiable and does not sum to one.
    """

    def __init__(self, config: GateNetworkConfig) -> None:
        super().__init__()
        if not isinstance(config, GateNetworkConfig):
            raise TypeError("config must be a GateNetworkConfig")
        self.config = config

        def gate() -> nn.Module:
            if config.architecture == "linear":
                return nn.Linear(config.input_dim, 1)
            return nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, 1))

        self.gates = nn.ModuleDict({category: gate() for category in CATEGORY_IDS})

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor) or not features.is_floating_point():
            raise TypeError("features must be a floating-point torch Tensor")
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError(f"features must have shape [batch, {self.config.input_dim}]")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        return torch.cat([self.gates[c](features) for c in CATEGORY_IDS], dim=-1)

    def activation_scores(self, features: Tensor) -> Tensor:
        return self(features).float().sigmoid()


class NeuralCategoryGate:
    """Question-to-category serving adapter with an injected feature encoder.

    ``encode_questions`` accepts a sequence of questions and returns a Tensor
    [batch, input_dim]. No encoder is downloaded or silently selected here.
    Set ``ready=True`` only after loading suitable trained weights explicitly;
    the default reports ``untrained`` before calling either encoder or gates.
    Training uses FiveCategoryGates directly, outside this inference adapter.
    """

    def __init__(self, network: FiveCategoryGates,
                 encode_questions: Callable[[Sequence[str]], Tensor], *,
                 model_id: str, ready: bool = False,
                 policy: CategorySelectionPolicy | None = None) -> None:
        if not isinstance(network, FiveCategoryGates):
            raise TypeError("network must be FiveCategoryGates")
        if not callable(encode_questions):
            raise TypeError("encode_questions must be callable")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must identify the gate weights and feature encoder")
        if type(ready) is not bool:
            raise TypeError("ready must be boolean")
        self.network, self.encode_questions = network, encode_questions
        self.model_id, self.ready = model_id, ready
        self.policy = policy if policy is not None else CategorySelectionPolicy()

    def route_categories(self, question: str) -> CategoryRouteDecision:
        if not isinstance(question, str) or not question.strip():
            return CategoryRouteDecision("invalid_input", reason="question is empty", model_id=self.model_id)
        if not self.ready:
            return CategoryRouteDecision("untrained", reason="trained category gate weights are not configured",
                                         model_id=self.model_id)
        if self.network.training:
            raise ValueError("call network.eval() before serving category routes")
        if isinstance(self.encode_questions, nn.Module) and self.encode_questions.training:
            raise ValueError("call the feature encoder's eval() before serving category routes")
        with torch.inference_mode():
            features = self.encode_questions([question])
            if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[0] != 1:
                raise ValueError("encode_questions must return one [1, input_dim] feature tensor")
            logits = self.network(features).float()[0]
            scores = logits.sigmoid()
            raw = dict(zip(CATEGORY_IDS, logits.cpu().tolist()))
            activations = dict(zip(CATEGORY_IDS, scores.cpu().tolist()))
        return self.policy.select(activations, logits=raw, model_id=self.model_id)
