"""Planner/router tests use tiny independent catalogues, with no model assets."""
from types import SimpleNamespace

import pytest

from visevi.functions.planning import DependencyGraph, RuleRouter


def step(step_id, tool=None, depends_on=(), required=True):
    return SimpleNamespace(step_id=step_id, tool=tool or step_id,
                           depends_on=depends_on, parameters={}, required=required)


def test_dependency_order_and_required_closure_include_optional_upstream():
    steps = [step("measure", depends_on=("mask",)),
             step("extra", required=False), step("mask", required=False),
             step("view")]
    graph = DependencyGraph(steps)
    assert [item.step_id for item in graph.topological_order()] == [
        "extra", "mask", "measure", "view"]
    assert [item.step_id for item in graph.required_closure()] == [
        "mask", "measure", "view"]
    assert graph.topological_order() == graph.topological_order()


@pytest.mark.parametrize("steps, message", [
    ([step("a"), step("a")], "duplicate step_id"),
    ([step("a", depends_on=("absent",))], "missing dependency"),
    ([step("a", depends_on=("b",)), step("b", depends_on=("a",))], "cycle"),
    ([step("a", depends_on=("a",))], "cycle"),
    ([step("a"), step("b", depends_on=("a", "a"))], "duplicate dependency"),
])
def test_invalid_dependency_graphs_are_rejected_before_execution(steps, message):
    with pytest.raises(ValueError, match=message):
        DependencyGraph(steps)


def test_registry_checks_step_tools_including_repeated_calls():
    graph = DependencyGraph([step("mask1", "segment"), step("mask2", "segment"),
                             step("extra", "not_installed", required=False)])
    registry = SimpleNamespace(describe_all=lambda: [{"name": "segment"}])
    graph.validate_registry(registry, required_only=True)
    with pytest.raises(ValueError, match="unknown tool.*not_installed"):
        graph.validate_registry(registry)


def catalogue():
    return {function_id: SimpleNamespace(function_id=function_id, keywords=keywords)
            for function_id, keywords in {
                "Q4": ("投照", "PA", "AP", "projection"),
                "P9": ("心胸比", "CTR", "cardiomegaly"),
                "G1": ("左右定位", "laterality"),
                "S1": ("进展", "progression"),
            }.items()}


def test_explicit_route_supports_multiple_ids_and_case_normalization():
    router = RuleRouter(catalogue())
    assert router.resolve(function_id="p9").function_ids == ("P9",)
    result = router.route("请执行P9和q4，然后再做 P9")
    assert result.status == "matched"
    assert result.function_ids == ("P9", "Q4")
    assert result.matched_by == "explicit_id"
    assert router.resolve(function_id=["g1", "P9"]).function_ids == ("G1", "P9")
    assert result.to_dict()["function_ids"] == ["P9", "Q4"]


def test_unknown_explicit_route_never_falls_back_to_keywords():
    router = RuleRouter(catalogue())
    assert router.route("P99 心胸比").status == "unknown"
    assert router.route("Q4 P99").status == "unknown"
    assert router.resolve(function_id="P99", query="心胸比").status == "unknown"
    assert router.resolve(function_id="").status == "unknown"
    assert router.route("prefixP9suffix").status == "unknown"


@pytest.mark.parametrize("query, function_id", [
    ("计算心胸比", "P9"), ("CTR measurement", "P9"),
    ("投照是否合适", "Q4"), ("PA/AP", "Q4"),
    ("请做左右定位", "G1"), ("Assess progression", "S1"),
])
def test_chinese_and_english_keywords(query, function_id):
    result = RuleRouter(catalogue()).route(query)
    assert result.status == "matched"
    assert result.function_ids == (function_id,)


def test_english_keyword_boundaries_and_ambiguous_requests():
    router = RuleRouter(catalogue())
    assert router.route("apparatus control").status == "unknown"
    assert router.route("reprojection").status == "unknown"
    assert router.route("").status == "unknown"
    result = router.route("心胸比和投照")
    assert result.status == "ambiguous"
    assert result.function_ids == ("Q4", "P9")
    assert router.route("P9 心胸比和投照").function_ids == ("P9",)


def test_catalogue_keys_must_match_spec_ids():
    with pytest.raises(ValueError, match="does not match"):
        RuleRouter({"P9": SimpleNamespace(function_id="Q4", keywords=())})
