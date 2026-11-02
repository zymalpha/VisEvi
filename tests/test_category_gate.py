"""Category routing contracts; CPU tensors and synthetic tools only."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from visevi.vqa import (CATEGORY_IDS, CategoryFunctionGate, CategoryRouteDecision,
                        CategorySelectionPolicy, VQARequest, run_vqa_episode)


def _scores(**overrides):
    return {**dict.fromkeys(CATEGORY_IDS, 0.1), **overrides}


class FixedCategoryGate:
    def __init__(self, *categories):
        self.decision = CategorySelectionPolicy().select(
            _scores(**dict.fromkeys(categories, 0.9)), model_id="synthetic-category-gate")

    def route_categories(self, question):
        return self.decision


def test_independent_thresholds_allow_one_many_or_no_experts():
    policy = CategorySelectionPolicy()
    assert policy.select(_scores(Q=0.9)).selected_categories == ("Q",)
    multi = policy.select(_scores(Q=0.9, P=0.8, G=0.5))
    assert multi.selected_categories == ("Q", "P", "G")
    assert not multi.abstain
    assert sum(multi.activation_scores.values()) > 1
    none = policy.select(_scores(Q=0.49))
    assert none.status == "unknown"
    assert none.selected_categories == ()
    assert none.abstain


def test_custom_thresholds_and_over_budget_routes_are_recorded_without_truncation():
    thresholds = _scores(Q=0.7, P=0.6, T=0.7, S=0.7, G=0.7)
    policy = CategorySelectionPolicy(thresholds, max_categories=1)
    decision = policy.select(_scores(Q=0.7, P=0.8), model_id="test-weights-and-encoder")
    assert decision.status == "ambiguous"
    assert decision.selected_categories == ()
    trace = json.loads(json.dumps(decision.to_dict(), allow_nan=False))
    assert trace["activation_scores"]["Q"] == 0.7
    assert trace["activation_scores"]["P"] == 0.8
    assert trace["selection_policy"] == {"thresholds": thresholds, "max_categories": 1}
    assert trace["model_id"] == "test-weights-and-encoder"
    assert trace["score_calibrated"] is False


@pytest.mark.parametrize("scores", [
    {"Q": 0.8}, {**_scores(), "E": 0.8}, _scores(Q=float("nan")),
    _scores(P=float("inf")), _scores(T=True), _scores(S=-0.1), _scores(G=1.1),
])
def test_selection_rejects_missing_extra_and_invalid_scores(scores):
    with pytest.raises(ValueError):
        CategorySelectionPolicy().select(scores)


@pytest.mark.parametrize("budget", [0, 6, 1.5, True])
def test_invalid_category_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="max_categories"):
        CategorySelectionPolicy(max_categories=budget)


def test_category_decision_rejects_inconsistent_or_unknown_selection():
    for status, selected in [("matched", ()), ("unknown", ("Q",)),
                             ("matched", ("E",)), ("matched", ("Q", "Q"))]:
        with pytest.raises(ValueError):
            CategoryRouteDecision(status, selected)


def test_category_bridge_limits_function_resolution_to_selected_category():
    q_gate = CategoryFunctionGate(FixedCategoryGate("Q"))
    assert q_gate.route("PA还是AP投照？").selected_function_ids == ("Q4",)
    assert q_gate.route("心胸比是多少？").status == "unknown"
    assert q_gate.route("P9").status == "unknown"
    p_gate = CategoryFunctionGate(FixedCategoryGate("P"))
    assert p_gate.route("心胸比是多少？").selected_function_ids == ("P9",)
    assert p_gate.route("Q4").status == "unknown"


def test_multi_category_selection_remains_visible_when_single_function_bridge_refuses():
    result = CategoryFunctionGate(FixedCategoryGate("Q", "P")).route("投照和心胸比？")
    assert result.status == "ambiguous"
    assert result.selected_function_ids == ()
    assert result.to_dict()["category_route"]["selected_categories"] == ["Q", "P"]


@pytest.mark.parametrize("question,category,slots,expected", [
    ("PA还是AP投照？", "Q", {}, "matched"),
    ("心胸比是多少？", "P", {}, "unsupported"),
    ("PA还是AP投照？", "Q", {"undeclared_slot": True}, "invalid_input"),
])
def test_vqa_keeps_category_trace_on_answers_and_execution_refusals(
        tmp_path, question, category, slots, expected):
    from test_vqa_pipeline import FakeClassifyView, FakeConsumer, _image, _registry

    image = _image(tmp_path / "synthetic.bin")
    tool = FakeClassifyView()
    gate = CategoryFunctionGate(FixedCategoryGate(category))
    result = run_vqa_episode(VQARequest(str(image), question, slots=slots),
                             gate=gate, registry=_registry(tool), consumer=FakeConsumer())
    assert result.route.status == expected
    trace = result.to_dict()["route"]["category_route"]
    assert trace["selected_categories"] == [category]
    assert trace["score_calibrated"] is False
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["route"]["category_route"] == trace
    assert tool.calls == (1 if expected == "matched" else 0)
    assert result.status == ("answered" if expected == "matched" else "abstained")


def test_untrained_gate_never_calls_encoder_or_a_tools(tmp_path):
    torch = pytest.importorskip("torch")
    from visevi.vqa.gate_network import FiveCategoryGates, GateNetworkConfig, NeuralCategoryGate
    from test_vqa_pipeline import FakeClassifyView, FakeConsumer, _image, _registry

    def forbidden_encoder(questions):
        pytest.fail("untrained gate must refuse before encoding questions")

    network = FiveCategoryGates(GateNetworkConfig(4))
    gate = NeuralCategoryGate(network, forbidden_encoder, model_id="untrained-test")
    assert gate.route_categories("投照？").status == "untrained"
    assert gate.route_categories(" ").status == "invalid_input"
    tool = FakeClassifyView()
    image = _image(tmp_path / "synthetic.bin")
    result = run_vqa_episode(VQARequest(str(image), "投照？"),
                             gate=CategoryFunctionGate(gate), registry=_registry(tool),
                             consumer=FakeConsumer())
    assert result.route.status == "unsupported"
    assert result.route.category_route["status"] == "untrained"
    assert result.response.status == "abstained"
    assert tool.calls == 0
    assert next(network.parameters()).device == torch.device("cpu")


@pytest.mark.parametrize("architecture", ["linear", "mlp"])
def test_five_heads_have_independent_parameters_and_stable_output_order(architecture):
    torch = pytest.importorskip("torch")
    from visevi.vqa.gate_network import FiveCategoryGates, GateNetworkConfig

    network = FiveCategoryGates(GateNetworkConfig(4, hidden_dim=3, architecture=architecture)).eval()
    all_parameter_ids = [id(p) for category in CATEGORY_IDS for p in network.gates[category].parameters()]
    assert len(all_parameter_ids) == len(set(all_parameter_ids))
    with torch.no_grad():
        for index, category in enumerate(CATEGORY_IDS):
            head = network.gates[category]
            for parameter in head.parameters():
                parameter.zero_()
            (head if architecture == "linear" else head[-1]).bias.fill_(index - 2)
    features = torch.zeros(2, 4)
    before = network(features)
    assert torch.equal(before, torch.arange(-2, 3).expand(2, -1))
    with torch.no_grad():
        q_head = network.gates["Q"]
        (q_head if architecture == "linear" else q_head[-1]).bias.add_(1)
    after = network(features)
    assert torch.equal(before[:, 1:], after[:, 1:])
    assert torch.equal(after[:, 0], before[:, 0] + 1)


@pytest.mark.parametrize("architecture", ["linear", "mlp"])
def test_network_keeps_encoder_and_all_gate_gradients_without_choosing_training_loss(architecture):
    torch = pytest.importorskip("torch")
    from visevi.vqa.gate_network import FiveCategoryGates, GateNetworkConfig

    # Deterministic positive parameters keep this gradient contract independent
    # of random initialisation, optimiser or a proposed supervision objective.
    encoder = torch.nn.Linear(2, 4)
    network = FiveCategoryGates(GateNetworkConfig(4, hidden_dim=3, architecture=architecture))
    with torch.no_grad():
        for parameter in list(encoder.parameters()) + list(network.parameters()):
            parameter.fill_(0.1)
    scores = network.activation_scores(encoder(torch.ones(2, 2)))
    scores.sum().backward()
    for parameter in list(encoder.parameters()) + list(network.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == parameter.numel()


def test_network_state_dict_round_trip_preserves_routes():
    torch = pytest.importorskip("torch")
    from visevi.vqa.gate_network import FiveCategoryGates, GateNetworkConfig

    config = GateNetworkConfig(4, hidden_dim=3)
    original, restored = FiveCategoryGates(config).eval(), FiveCategoryGates(config).eval()
    restored.load_state_dict(original.state_dict())
    features = torch.ones(2, 4)
    assert torch.equal(original(features), restored(features))
    assert config.to_dict()["categories"] == list(CATEGORY_IDS)
    assert config.to_dict()["activation"] == "independent_sigmoid"


def test_neural_adapter_uses_injected_encoder_and_records_raw_logits():
    torch = pytest.importorskip("torch")
    from visevi.vqa.gate_network import FiveCategoryGates, GateNetworkConfig, NeuralCategoryGate

    network = FiveCategoryGates(GateNetworkConfig(4, architecture="linear")).eval()
    with torch.no_grad():
        for category, head in network.gates.items():
            head.weight.zero_()
            head.bias.fill_(2 if category in {"Q", "P"} else -2)
    calls = []

    def encode(questions):
        calls.append(questions)
        return torch.ones(len(questions), 4)

    gate = NeuralCategoryGate(network, encode, model_id="synthetic-only", ready=True)
    result = gate.route_categories("投照和心胸比？")
    assert calls == [["投照和心胸比？"]]
    assert result.selected_categories == ("Q", "P")
    assert result.logits == dict(zip(CATEGORY_IDS, [2, 2, -2, -2, -2]))
    assert result.activation_scores["Q"] == pytest.approx(0.880797, abs=1e-6)
    assert result.to_dict()["score_calibrated"] is False


def test_neural_adapter_requires_eval_mode_and_valid_single_question_features():
    torch = pytest.importorskip("torch")
    from visevi.vqa.gate_network import FiveCategoryGates, GateNetworkConfig, NeuralCategoryGate

    network = FiveCategoryGates(GateNetworkConfig(4))
    gate = NeuralCategoryGate(network, lambda qs: torch.ones(1, 4), model_id="test", ready=True)
    with pytest.raises(ValueError, match="network.eval"):
        gate.route_categories("投照？")
    network.eval()
    for invalid in [torch.ones(4), torch.ones(2, 4), None]:
        gate = NeuralCategoryGate(network, lambda qs: invalid, model_id="test", ready=True)
        with pytest.raises(ValueError, match="one .* feature tensor"):
            gate.route_categories("投照？")
    encoder = torch.nn.Linear(4, 4)
    gate = NeuralCategoryGate(network, encoder, model_id="test", ready=True)
    with pytest.raises(ValueError, match="encoder's eval"):
        gate.route_categories("投照？")


def test_network_rejects_wrong_shapes_dtypes_and_nonfinite_features():
    torch = pytest.importorskip("torch")
    from visevi.vqa.gate_network import FiveCategoryGates, GateNetworkConfig

    network = FiveCategoryGates(GateNetworkConfig(4))
    for features in [torch.ones(4), torch.ones(2, 5), torch.ones(1, 1, 4),
                     torch.full((1, 4), float("nan"))]:
        with pytest.raises(ValueError):
            network(features)
    for features in [torch.ones(1, 4, dtype=torch.int64), [[1.0] * 4]]:
        with pytest.raises(TypeError):
            network(features)


def test_plain_vqa_import_does_not_require_torch():
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; import visevi.vqa; assert 'torch' not in sys.modules"],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
