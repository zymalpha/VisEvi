"""Static integration gates for the Q/P catalogue and the A-tool registry."""

from visevi.eval.runner import build_a_registry
from visevi.functions.runner import IMPLEMENTED_FUNCTIONS, _PROGRAMS, _json_hash
from visevi.functions.specs import load_function_specs
import numpy as np


def test_all_qp_programs_are_registered_and_have_a_tool_plan():
    specs = load_function_specs()
    qp_ids = {function_id for function_id in specs if function_id.startswith(("Q", "P"))}
    assert qp_ids <= set(IMPLEMENTED_FUNCTIONS)
    assert qp_ids <= set(_PROGRAMS)
    assert all(specs[function_id].steps for function_id in qp_ids)


def test_a_registry_covers_every_qp_required_step():
    registry = build_a_registry()
    available = {entry["name"] for entry in registry.describe_all()}
    specs = load_function_specs()
    required = {
        step.tool
        for function_id, spec in specs.items()
        if function_id.startswith(("Q", "P"))
        for step in spec.steps
        if step.required
    }
    assert required <= available


def test_runtime_numpy_landmark_cache_has_a_stable_fingerprint():
    value = {"nodes": np.arange(6, dtype=np.float32).reshape(2, 3)}
    assert _json_hash(value) == _json_hash({"nodes": value["nodes"].copy()})


def test_all_qp_programs_use_registered_runner_mappings():
    qp_ids = {function_id for function_id in load_function_specs()
              if function_id.startswith(("Q", "P"))}
    assert qp_ids <= set(_PROGRAMS)
