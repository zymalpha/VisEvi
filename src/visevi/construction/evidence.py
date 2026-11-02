"""Typed helpers for the three-layer evidence graph.

The perception tools already emit :class:`~visevi.core.units.EvidenceUnit`
objects.  This module adds a small, deterministic construction API for the
function layer:

``observation``
    A fact directly returned by an A tool (mask, point, box, attribute, ...).
``relation``
    A measurement or spatial/temporal relation derived from one or more
    observations.
``claim``
    A function-level proposition supported or contradicted by relations and
    observations.

The helpers intentionally keep payloads as dictionaries so existing tools and
the serializable ``EvidenceSession`` contract remain backwards compatible.
Every derived node records its upstream IDs in ``provenance`` and creates an
explicit graph edge with a rule/program basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

from ..core.session import EvidenceSession
from ..core.units import EvidenceUnit


OBSERVATION = "observation"
RELATION = "relation"
CLAIM = "claim"
EXTERNAL_PROVENANCE_PREFIXES = ("cached:", "tool_call:", "artifact:")
EDGE_RELATIONS = {"derived_from", "supports", "contradicts", "attribute_of",
                  "spatial", "comparison", "temporal"}

# Runtime nodes remain ``EvidenceUnit`` for compatibility with the existing
# session serializer.  These aliases make the three layers explicit to type
# checkers and callers without introducing a second wire format.
Observation: TypeAlias = EvidenceUnit
Relation: TypeAlias = EvidenceUnit
Claim: TypeAlias = EvidenceUnit


def is_observation(node: EvidenceUnit) -> bool:
    return isinstance(node, EvidenceUnit) and node.category == OBSERVATION


def is_relation(node: EvidenceUnit) -> bool:
    return isinstance(node, EvidenceUnit) and node.category == RELATION


def is_claim(node: EvidenceUnit) -> bool:
    return isinstance(node, EvidenceUnit) and node.category == CLAIM


def _ids(values: Iterable[str] | None) -> list[str]:
    """Return stable, de-duplicated non-empty IDs."""
    if isinstance(values, (str, bytes)):
        raise TypeError("evidence node IDs must be an iterable of strings, not a string")
    out: list[str] = []
    for value in values or ():
        if not isinstance(value, str) or not value.strip():
            raise TypeError("evidence node IDs must be non-empty strings")
        if value not in out:
            out.append(value)
    return out


def _check_upstream(session: EvidenceSession, ids: Sequence[str]) -> None:
    missing = [node_id for node_id in ids if node_id not in session.graph.nodes]
    if missing:
        raise ValueError(f"upstream evidence node(s) missing: {', '.join(missing)}")


def _new_id(session: EvidenceSession, prefix: str) -> str:
    # A caller can supply sparse/custom IDs, so the session's count is only a
    # starting point rather than a guarantee that the returned ID is unused.
    suffix = len(session.graph.nodes) + 1
    candidate = session.next_id(prefix)
    while candidate in session.graph.nodes:
        candidate = f"{prefix}-{suffix:04d}"
        suffix += 1
    return candidate


def _text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _node_id(session: EvidenceSession, supplied: str | None, prefix: str) -> str:
    if supplied is not None:
        _text(supplied, "node_id")
        return supplied
    return _new_id(session, prefix)


def _external_reference(value: str) -> bool:
    return isinstance(value, str) and any(value.startswith(prefix) and bool(value[len(prefix):].strip())
               for prefix in EXTERNAL_PROVENANCE_PREFIXES)


def _payload(value: Mapping[str, Any], **contract: Any) -> dict[str, Any]:
    """Prevent payload fields from disagreeing with the actual graph edges."""
    result = dict(value)
    for key, expected in contract.items():
        if key in result and result[key] != expected:
            raise ValueError(f"payload {key} conflicts with the evidence graph contract")
        result[key] = expected
    return result


def _add_edges(session: EvidenceSession, src_ids: Sequence[str], dst: str,
               relation: str, basis: str) -> None:
    for src in src_ids:
        if src == dst:
            raise ValueError("evidence graph cannot contain a self-edge")
        session.graph.add_edge(src, dst, relation=relation, basis=basis)


def add_observation(
    session: EvidenceSession,
    finding_or_anatomy: str,
    value: Mapping[str, Any] | dict[str, Any],
    *,
    evidence_type: str = "observation",
    source_tool: str,
    node_id: str | None = None,
    region: dict[str, Any] | None = None,
    confidence: float | None = None,
    provenance: Iterable[str] = (),
    validity_conditions: Iterable[str] = (),
) -> EvidenceUnit:
    """Add a direct A-tool observation to ``session``.

    ``provenance`` may contain existing node IDs or explicit external references
    prefixed by ``cached:``, ``tool_call:`` or ``artifact:``.  External references
    remain labels, not graph edges; all node references must already exist.
    The source tool is mandatory so a function result cannot silently lose its
    origin.
    """
    _text(finding_or_anatomy, "finding_or_anatomy")
    _text(source_tool, "source_tool")
    _text(evidence_type, "evidence_type")
    upstream = _ids(provenance)
    _check_upstream(session, [p for p in upstream if not _external_reference(p)])
    unit = EvidenceUnit(
        id=_node_id(session, node_id, "obs"),
        category=OBSERVATION,
        evidence_type=evidence_type,
        finding_or_anatomy=finding_or_anatomy,
        value=dict(value),
        region=region,
        confidence=confidence,
        source_tool=source_tool,
        provenance=upstream,
        validity_conditions=list(validity_conditions),
    )
    session.add_observation(unit)
    # Provenance can refer to an earlier observation/tool call.  If it is a
    # graph node, make the dependency explicit; tool-call strings are retained
    # in the node but cannot become graph edges.
    _add_edges(session, [p for p in upstream if p in session.graph.nodes], unit.id,
               relation="derived_from", basis=source_tool)
    return unit


def add_relation(
    session: EvidenceSession,
    finding_or_anatomy: str,
    value: Mapping[str, Any] | dict[str, Any],
    *,
    input_ids: Iterable[str],
    rule_id: str,
    evidence_type: str = "relation",
    source_tool: str = "function_program",
    node_id: str | None = None,
    region: dict[str, Any] | None = None,
    confidence: float | None = None,
    validity_conditions: Iterable[str] = (),
) -> EvidenceUnit:
    """Add a measurement/spatial relation derived from existing nodes."""
    inputs = _ids(input_ids)
    if not inputs:
        raise ValueError("a relation requires at least one input node")
    _text(finding_or_anatomy, "finding_or_anatomy")
    _text(rule_id, "rule_id")
    _text(source_tool, "source_tool")
    _text(evidence_type, "evidence_type")
    _check_upstream(session, inputs)
    unit = EvidenceUnit(
        id=_node_id(session, node_id, "rel"),
        category=RELATION,
        evidence_type=evidence_type,
        finding_or_anatomy=finding_or_anatomy,
        value=_payload(value, rule_id=rule_id, input_ids=inputs),
        region=region,
        confidence=confidence,
        source_tool=source_tool,
        provenance=inputs,
        validity_conditions=list(validity_conditions),
    )
    session.graph.add(unit)
    _add_edges(session, inputs, unit.id, relation="derived_from", basis=rule_id)
    return unit


def add_claim(
    session: EvidenceSession,
    proposition: str,
    value: Mapping[str, Any] | dict[str, Any],
    *,
    support_ids: Iterable[str] = (),
    counter_ids: Iterable[str] = (),
    rule_id: str,
    source_tool: str = "function_program",
    node_id: str | None = None,
    confidence: float | None = None,
    validity_conditions: Iterable[str] = (),
) -> EvidenceUnit:
    """Add a function-level claim with explicit support and counter evidence."""
    supports, counters = _ids(support_ids), _ids(counter_ids)
    if not supports and not counters:
        raise ValueError("a claim requires support_ids or counter_ids")
    if set(supports) & set(counters):
        raise ValueError("a node cannot support and contradict the same claim")
    _text(proposition, "proposition")
    _text(rule_id, "rule_id")
    _text(source_tool, "source_tool")
    _check_upstream(session, supports + counters)
    payload = _payload(value, proposition=proposition, support_ids=supports,
                       counter_ids=counters, rule_id=rule_id)
    unit = EvidenceUnit(
        id=_node_id(session, node_id, "claim"),
        category=CLAIM,
        evidence_type="claim",
        finding_or_anatomy=proposition,
        value=payload,
        confidence=confidence,
        source_tool=source_tool,
        provenance=supports + counters,
        validity_conditions=list(validity_conditions),
    )
    session.add_claim(unit)
    _add_edges(session, supports, unit.id, relation="supports", basis=rule_id)
    _add_edges(session, counters, unit.id, relation="contradicts", basis=rule_id)
    return unit


def validate_graph(session: EvidenceSession) -> list[str]:
    """Return graph contract violations; an empty list means valid.

    This is deliberately non-throwing so an episode runner can include the
    validation result in its reproducible output.  It checks node categories,
    provenance and edge references, plus the acyclic dependency property.
    """
    errors: list[str] = []
    nodes = session.graph.nodes
    allowed = {OBSERVATION, RELATION, CLAIM}
    for node_id, node in nodes.items():
        if node.id != node_id:
            errors.append(f"{node_id}: node ID disagrees with graph key")
        if node.category not in allowed:
            errors.append(f"{node_id}: invalid category {node.category!r}")
        if not isinstance(node.source_tool, str) or not node.source_tool.strip():
            errors.append(f"{node_id}: missing source_tool")
        if not isinstance(node.finding_or_anatomy, str) or not node.finding_or_anatomy.strip():
            errors.append(f"{node_id}: missing finding_or_anatomy")
        for upstream in node.provenance:
            if upstream == node_id:
                errors.append(f"{node_id}: self provenance")
            elif upstream not in nodes and not _external_reference(upstream):
                errors.append(f"{node_id}: provenance node missing: {upstream}")
        if node.category in {RELATION, CLAIM}:
            references = [p for p in node.provenance if p in nodes]
            if not references:
                errors.append(f"{node_id}: derived node has no upstream graph nodes")
        if node.category == RELATION:
            if not isinstance(node.value, Mapping):
                errors.append(f"{node_id}: relation payload must be a mapping")
                relation_value: Mapping[str, Any] = {}
            else:
                relation_value = node.value

            inputs = relation_value.get("input_ids")
            rule_id = relation_value.get("rule_id")
            valid_inputs = isinstance(inputs, list) and all(
                isinstance(item, str) and bool(item.strip()) for item in inputs
            )
            if not valid_inputs or not inputs:
                errors.append(f"{node_id}: relation input_ids must be a non-empty list of IDs")
            else:
                if len(set(inputs)) != len(inputs):
                    errors.append(f"{node_id}: relation input_ids contain duplicates")
                missing_inputs = [item for item in inputs if item not in nodes]
                if missing_inputs:
                    errors.append(f"{node_id}: relation input node missing: {', '.join(missing_inputs)}")
                if node.provenance != inputs:
                    errors.append(f"{node_id}: relation input_ids disagree with provenance")

            if not isinstance(rule_id, str) or not rule_id.strip():
                errors.append(f"{node_id}: relation is missing rule_id")

            incoming = [edge for edge in session.graph.edges
                        if edge.get("dst") == node_id and edge.get("relation") == "derived_from"]
            incoming_ids = [edge.get("src") for edge in incoming]
            if valid_inputs and incoming_ids != inputs:
                errors.append(f"{node_id}: derived_from edges disagree with relation input_ids")
            if isinstance(rule_id, str) and rule_id.strip() and any(
                edge.get("basis") != rule_id for edge in incoming
            ):
                errors.append(f"{node_id}: derived_from edge basis disagrees with relation rule_id")
        if node.category == CLAIM:
            # Old claim nodes omit these fields; if present, they must agree
            # with provenance and support/counter edges (no forged references).
            supports = node.value.get("support_ids")
            counters = node.value.get("counter_ids")
            if supports is not None or counters is not None:
                if not isinstance(supports, list) or not isinstance(counters, list):
                    errors.append(f"{node_id}: claim references must be lists")
                elif any(not isinstance(p, str) or not p.strip() for p in supports + counters):
                    errors.append(f"{node_id}: claim references must be non-empty strings")
                else:
                    if set(supports) & set(counters):
                        errors.append(f"{node_id}: claim evidence both supports and contradicts")
                    if list(dict.fromkeys(supports + counters)) != node.provenance:
                        errors.append(f"{node_id}: claim references disagree with provenance")
                    for relation, references in (("supports", supports), ("contradicts", counters)):
                        actual = [e.get("src") for e in session.graph.edges
                                  if e.get("dst") == node_id and e.get("relation") == relation]
                        if set(actual) != set(references):
                            errors.append(f"{node_id}: {relation} edges disagree with claim references")
            if "proposition" in node.value and node.value["proposition"] != node.finding_or_anatomy:
                errors.append(f"{node_id}: claim proposition disagrees with finding_or_anatomy")
    for edge in session.graph.edges:
        src, dst = edge.get("src"), edge.get("dst")
        if src not in nodes:
            errors.append(f"edge source missing: {src}")
        if dst not in nodes:
            errors.append(f"edge destination missing: {dst}")
        if not edge.get("basis"):
            errors.append(f"edge {src}->{dst}: missing basis")
        if edge.get("relation") not in EDGE_RELATIONS:
            errors.append(f"edge {src}->{dst}: invalid relation {edge.get('relation')!r}")
    # Kahn traversal over graph edges catches accidental cycles.
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    indegree: dict[str, int] = {node_id: 0 for node_id in nodes}
    for edge in session.graph.edges:
        src, dst = edge.get("src"), edge.get("dst")
        if src in nodes and dst in nodes:
            outgoing[src].append(dst)
            indegree[dst] += 1
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node_id = queue.pop()
        visited += 1
        for dst in outgoing[node_id]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    if visited != len(nodes):
        errors.append("evidence graph contains a cycle")
    return errors


@dataclass(frozen=True)
class EvidenceRefs:
    """Convenience return value for programs that build all three layers."""

    observation_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {"observations": list(self.observation_ids),
                "relations": list(self.relation_ids),
                "claims": list(self.claim_ids)}
