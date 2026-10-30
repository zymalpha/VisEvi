import pytest

from visevi.construction.evidence import (
    Claim,
    Observation,
    Relation,
    add_claim,
    add_observation,
    add_relation,
    validate_graph,
    is_claim,
    is_observation,
    is_relation,
)
from visevi.core.session import EvidenceSession


def test_three_layer_nodes_and_edges_are_serializable():
    session = EvidenceSession(image_ref="synthetic:test")
    obs = add_observation(session, "heart_mask", {"area_px": 10},
                          source_tool="segment_heart")
    view = add_observation(session, "view_position", {"view": "PA"},
                           evidence_type="attribute", source_tool="classify_view")
    ctr = add_relation(session, "cardiothoracic_ratio", {"CTR": 0.5},
                      input_ids=[obs.id], rule_id="ctr_from_mask")
    claim = add_claim(session, "heart_size_measurable", {"status": "supported"},
                      support_ids=[ctr.id, view.id], rule_id="p9_gate")
    assert [n.category for n in session.graph.nodes.values()] == [
        "observation", "observation", "relation", "claim"]
    assert isinstance(obs, Observation) and is_observation(obs)
    assert isinstance(ctr, Relation) and is_relation(ctr)
    assert isinstance(claim, Claim) and is_claim(claim)
    assert {e["relation"] for e in session.graph.edges} == {"derived_from", "supports"}
    assert validate_graph(session) == []
    loaded = session.to_dict()
    assert loaded["graph"]["nodes"][-1]["id"] == claim.id


def test_missing_input_and_conflicting_claim_inputs_are_rejected():
    session = EvidenceSession(image_ref="synthetic:test")
    with pytest.raises(ValueError, match="missing"):
        add_relation(session, "distance", {"px": 1}, input_ids=["obs-9999"], rule_id="r")
    obs = add_observation(session, "finding", {}, source_tool="a")
    with pytest.raises(ValueError, match="support and contradict"):
        add_claim(session, "p", {}, support_ids=[obs.id], counter_ids=[obs.id], rule_id="r")


def test_graph_validation_reports_cycle_and_bad_edge():
    session = EvidenceSession(image_ref="synthetic:test")
    a = add_observation(session, "a", {}, source_tool="a")
    b = add_observation(session, "b", {}, source_tool="b")
    session.graph.add_edge(a.id, b.id, "derived_from", "manual")
    session.graph.add_edge(b.id, a.id, "derived_from", "manual")
    session.graph.edges.append({"src": "missing", "dst": a.id, "relation": "supports"})
    errors = validate_graph(session)
    assert any("cycle" in error for error in errors)
    assert any("source missing" in error for error in errors)


def test_observation_external_source_is_retained_without_dangling_edge():
    session = EvidenceSession(image_ref="synthetic:test")
    obs = add_observation(session, "Heart", {"area": 10}, source_tool="segment_heart",
                          provenance=["cached:PSPNet:heart.png", "tool_call:infer-1"])
    assert obs.provenance == ["cached:PSPNet:heart.png", "tool_call:infer-1"]
    assert session.graph.edges == []
    assert validate_graph(session) == []
    with pytest.raises(ValueError, match="missing"):
        add_observation(session, "Heart", {}, source_tool="segment_heart",
                        provenance=["obs-0009"])
    with pytest.raises(ValueError, match="missing"):
        add_relation(session, "CTR", {}, input_ids=["cached:PSPNet:heart.png"], rule_id="r")
    # A missing graph reference must not pass validation merely because there
    # is no edge for it (e.g. a malformed externally loaded session).
    obs.provenance.append("obs-deleted")
    assert any("provenance node missing: obs-deleted" in error for error in validate_graph(session))


@pytest.mark.parametrize("field,value", [
    ("proposition", "different proposition"),
    ("support_ids", ["obs-9999"]),
    ("counter_ids", ["obs-9999"]),
    ("rule_id", "forged-rule"),
])
def test_claim_payload_cannot_forge_graph_contract(field, value):
    session = EvidenceSession(image_ref="synthetic:test")
    obs = add_observation(session, "Heart", {}, source_tool="segment_heart")
    with pytest.raises(ValueError, match="conflicts"):
        add_claim(session, "valid-proposition", {field: value}, support_ids=[obs.id], rule_id="r")
    assert len(session.graph.nodes) == 1
    assert session.graph.edges == []


def test_relation_payload_and_claim_empty_proposition_are_rejected_atomically():
    session = EvidenceSession(image_ref="synthetic:test")
    obs = add_observation(session, "Heart", {}, source_tool="segment_heart")
    with pytest.raises(ValueError, match="conflicts"):
        add_relation(session, "CTR", {"rule_id": "forged"}, input_ids=[obs.id], rule_id="r")
    with pytest.raises(ValueError, match="proposition"):
        add_claim(session, " ", {}, support_ids=[obs.id], rule_id="r")
    with pytest.raises(TypeError, match="not a string"):
        add_relation(session, "CTR", {}, input_ids=obs.id, rule_id="r")
    assert len(session.graph.nodes) == 1
    assert session.graph.edges == []


@pytest.mark.parametrize("relation", ["attribute_of", "spatial", "comparison", "temporal"])
def test_legacy_edge_types_remain_valid(relation):
    session = EvidenceSession(image_ref="synthetic:test")
    a = add_observation(session, "a", {}, source_tool="a")
    b = add_observation(session, "b", {}, source_tool="b")
    session.graph.add_edge(a.id, b.id, relation, "legacy-rule")
    assert validate_graph(session) == []


def test_custom_id_gaps_do_not_collide_and_claim_tampering_is_detected():
    session = EvidenceSession(image_ref="synthetic:test")
    a = add_observation(session, "a", {}, source_tool="a", node_id="obs-0002")
    b = add_observation(session, "b", {}, source_tool="b")
    assert a.id != b.id
    claim = add_claim(session, "test", {}, support_ids=[a.id], counter_ids=[b.id], rule_id="r")
    assert validate_graph(session) == []
    claim.value["support_ids"] = [b.id]
    errors = validate_graph(session)
    assert any("both supports and contradicts" in error for error in errors)
    assert any("supports edges disagree" in error for error in errors)
