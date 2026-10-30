"""Audit the versioned 26-function catalogue against actual A-tool contracts."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import jsonschema
import pytest

from visevi.eval.runner import build_a_registry
from visevi.functions.planning import DependencyGraph
from visevi.functions.specs import (
    FUNCTION_IDS, SPEC_DIRECTORY, FunctionSpec, ToolStep,
    get_function_spec, load_function_specs,
)


def test_catalogue_has_exactly_the_26_business_functions_and_draft_protocols():
    specs = load_function_specs()
    expected = {f"{category}{number}"
                for category, count in (("Q", 4), ("P", 12), ("T", 4), ("S", 3), ("G", 3))
                for number in range(1, count + 1)}
    assert set(specs) == expected
    assert len(specs) == len(FUNCTION_IDS) == 26
    for function_id, spec in specs.items():
        assert spec.category == function_id[0]
        assert spec.status == "specified"
        assert spec.evaluation_protocol["status"] == "draft_not_frozen"
        assert spec.evaluation_protocol["primary_metrics"]
        assert spec.guards and spec.refusal_rules and spec.slots and spec.blockers
        if function_id not in {"Q4", "P3", "P6", "P9", "T1", "G1", "S1"}:
            assert any("executor is not implemented" in blocker for blocker in spec.blockers)
        assert spec.input_type == ("paired_images" if function_id.startswith("S") else "single_image")


def _resolve_example(value, schema):
    """Substitute runtime references with a type-valid example, not a tool result.

    The schema supplies parameter types independently from the catalogue, so
    misspelled names, wrong literal types and unsupported frozen vocabulary are
    detected.  Actual input binding and missing values are runner tests.
    """
    if isinstance(value, str) and value.startswith("$input."):
        if "enum" in schema:
            return schema["enum"][0]
        kind = schema.get("type", "string")
        if kind == "array":
            return [_resolve_example("$input.example", schema.get("items", {}))
                    for _ in range(max(1, schema.get("minItems", 1)))]
        return {"string": "example", "number": 1.0, "integer": 1,
                "boolean": True, "object": {}}.get(kind, "example")
    if isinstance(value, list):
        return [_resolve_example(item, schema.get("items", {})) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_example(item, schema.get("properties", {}).get(key, {}))
                for key, item in value.items()}
    return value


def test_all_spec_steps_use_actual_registered_tools_and_schema_parameters():
    registry = build_a_registry()
    tool_schemas = {tool["name"]: tool["input_schema"] for tool in registry.describe_all()}
    assert len(tool_schemas) == 27
    for spec in load_function_specs().values():
        graph = DependencyGraph(spec.steps)
        graph.validate_registry(registry)
        required_ids = {step.step_id for step in graph.required_closure()}
        assert {step.step_id for step in spec.steps if step.required} <= required_ids
        for step in graph.required_closure():
            assert set(step.depends_on) <= required_ids
        for step in spec.steps:
            schema = tool_schemas[step.tool]
            assert set(step.parameters) <= set(schema["properties"]), (spec.function_id, step.step_id)
            params = {key: _resolve_example(value, schema["properties"][key])
                      for key, value in step.parameters.items()}
            # Runtime supplies a distinct output path for each model step.
            if "output_dir" in schema["properties"]:
                params["output_dir"] = f"/tmp/example/{spec.function_id}/{step.step_id}"
            jsonschema.validate(params, schema)


def test_p9_requires_projection_quality_and_complete_heart_before_ctr():
    spec = get_function_spec("P9")
    assert [step.step_id for step in spec.steps] == ["view", "quality", "heart", "ctr"]
    assert all(step.required for step in spec.steps)
    assert spec.steps[-1].depends_on == ("heart",)
    assert set(spec.required_tools) == {
        "classify_view", "assess_image_quality", "segment_heart", "measure_ctr"}
    assert spec.steps[-1].parameters == {"heart_mask_name": "Heart"}
    assert "CTR" in spec.outputs
    assert spec.evaluation_protocol["units"]["CTR"] == "dimensionless"
    assert "enlargement" in spec.outputs and "evidence_claim" not in spec.outputs


def test_g1_preserves_two_lung_relations_and_runtime_finding_binding():
    spec = get_function_spec("G1")
    steps = {step.step_id: step for step in spec.steps}
    assert steps["candidates"].parameters == {"findings": ["$input.finding"]}
    for side, mask in (("right", "Right_Lung"), ("left", "Left_Lung")):
        relation = steps[f"{side}_relation"]
        assert relation.tool == "relate_lesion_to_anatomy"
        assert relation.parameters == {
            "lesion_mask_name": "$input.lesion_mask_name", "anatomy_mask_name": mask}
        assert set(relation.depends_on) == {"lungs", "candidates"}
    assert {"image_sides", "patient_sides", "orientation_verified", "scope"} <= set(spec.outputs)
    assert any("unverified" in rule for rule in spec.refusal_rules)


def test_unsupported_device_and_temporal_work_have_explicit_blockers():
    assert any("does not support chest drains" in text for text in get_function_spec("T4").blockers)

    s1 = get_function_spec("S1")
    assert s1.input_type == "paired_images"
    assert any("same original-image coordinate frame" in text for text in s1.guards)
    assert any("caller-verified registration" in text for text in s1.blockers)
    assert any("Independent longitudinal annotations" in text for text in s1.blockers)
    assert any("thresholds are not frozen" in text for text in s1.blockers)
    assert any("Same-tool steps" in text for text in s1.guards)

    for function_id in ("S2", "S3"):
        spec = get_function_spec(function_id)
        assert spec.input_type == "paired_images"
        assert any("orchestration" in text for text in spec.blockers)
        assert any("executor is not implemented" in text for text in spec.blockers)
        assert any("Same-tool steps" in text for text in spec.guards)


def test_specs_roundtrip_without_sharing_mutable_input_data():
    original = get_function_spec("P9")
    data = original.to_dict()
    restored = FunctionSpec.from_dict(data)
    assert restored == original
    data["steps"][-1]["parameters"]["heart_mask_name"] = "wrong"
    data["evaluation_protocol"]["units"]["CTR"] = "wrong"
    assert restored.steps[-1].parameters["heart_mask_name"] == "Heart"
    assert restored.evaluation_protocol["units"]["CTR"] == "dimensionless"
    separately_loaded = get_function_spec("P9")
    separately_loaded.steps[-1].parameters["heart_mask_name"] = "other"
    assert original.steps[-1].parameters["heart_mask_name"] == "Heart"


@pytest.mark.parametrize("change, message", [
    ({"function_id": "P99"}, "unknown function_id"),
    ({"category": "Q"}, "category must match"),
    ({"input_type": "video"}, "unsupported input_type"),
    ({"status": "finished"}, "unsupported specification status"),
    ({"outputs": []}, "must be non-empty"),
    ({"evaluation_protocol": {}}, "missing reference"),
    ({"unexpected_field": True}, "unexpected keyword"),
])
def test_schema_rejects_malformed_function_protocols(change, message):
    data = get_function_spec("P9").to_dict()
    data.update(change)
    with pytest.raises(ValueError, match=message):
        FunctionSpec.from_dict(data)


def test_schema_rejects_cycles_duplicate_steps_unknown_dependencies_and_nonfinite_values():
    data = get_function_spec("P9").to_dict()
    for steps, message in [
        ([data["steps"][0], data["steps"][0]], "duplicate step_id"),
        ([{**data["steps"][0], "depends_on": ["absent"]}], "unknown dependencies"),
        ([{**data["steps"][0], "depends_on": ["quality"]},
          {**data["steps"][1], "depends_on": ["view"]}], "cycle"),
    ]:
        changed = deepcopy(data)
        changed["steps"] = steps
        with pytest.raises(ValueError, match=message):
            FunctionSpec.from_dict(changed)
    with pytest.raises(ValueError, match="finite JSON"):
        ToolStep("test", "test", parameters={"bad": float("nan")})
    with pytest.raises(ValueError, match="boolean"):
        ToolStep("test", "test", required="true")


def test_loader_rejects_missing_malformed_duplicate_or_misnamed_specs(tmp_path):
    destination = tmp_path / "specs"
    shutil.copytree(SPEC_DIRECTORY, destination)
    (destination / "P9.json").unlink()
    with pytest.raises(ValueError, match="missing specifications.*P9"):
        load_function_specs(destination)
    shutil.copyfile(SPEC_DIRECTORY / "P9.json", destination / "P9.json")
    shutil.copyfile(SPEC_DIRECTORY / "P9.json", destination / "zz.json")
    with pytest.raises(ValueError, match="duplicate function_id"):
        load_function_specs(destination)
    (destination / "zz.json").unlink()
    (destination / "P9.json").rename(destination / "P0.json")
    with pytest.raises(ValueError, match="filename.*does not match"):
        load_function_specs(destination)
    (destination / "P0.json").unlink()
    (destination / "P9.json").write_text("{")
    with pytest.raises(ValueError, match="invalid specification"):
        load_function_specs(destination)
    with pytest.raises(KeyError, match="unknown function_id"):
        get_function_spec("P99")
