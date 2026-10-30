"""Model-agnostic serving layer — 消费端无关 (CVPR_PAPER_PLAN.md §0.1, axis 3).

Evidence is built ONCE by the two-layer pipeline and frozen as session.json;
this module renders that artifact into consumption formats for any reasoner,
with no model-specific tailoring (证据不变性). The card carries finding-level
claims with their own uncertainty ([Bel, Pl], K) and validity conditions, so
the consuming model needs no knowledge of DS internals.

  evidence_card(session_dict) -> str   markdown card for prompt consumption
  to_prompt_block(session_dict) -> str fenced block, ready to paste

Takes a plain dict (loaded session.json): the serving side never imports the
perception stack — serialization is the contract. 组合≠诊断: the diagnostic
step stays with the consuming reasoner.
"""
from __future__ import annotations

from pathlib import Path


def _node_summary(nodes: dict) -> str:
    counts: dict[str, int] = {}
    for n in nodes.values():
        counts[n["evidence_type"]] = counts.get(n["evidence_type"], 0) + 1
    return ", ".join(f"{k}×{v}" for k, v in counts.items())


def _fmt(value: dict, skip: tuple = ("method",)) -> str:
    parts = []
    for k, v in value.items():
        if k in skip or isinstance(v, (dict, list)):
            continue
        parts.append(f"{k}={round(v, 4) if isinstance(v, float) else v}")
    return "; ".join(parts)


def _claim_block(n: dict, nodes: dict) -> list[str]:
    v = n["value"]
    lines = [f"### 证据声明 [{n['id']}] {v.get('proposition', n['finding_or_anatomy'])}",
             f"- 支持方向: {v.get('supported_direction', '?')} | "
             f"[Bel, Pl] = [{v.get('bel')}, {v.get('pl')}]"
             f"（区间宽 {v.get('interval_width')}）| "
             f"源冲突 K={v.get('conflict_k')} | 组合规则: {v.get('combination_rule', '-')}"]
    sup = [pid for pid in n.get("provenance", []) if pid in nodes]
    if sup:
        lines.append("- 上游测量:")
        for pid in sup:
            m = nodes[pid]
            lines.append(f"  - [{pid}] {m['finding_or_anatomy']} — {_fmt(m['value'])}")
            up = [q for q in m.get("provenance", []) if q in nodes]
            if up:
                tools = sorted({nodes[q].get("source_tool", "?") for q in up})
                lines.append(f"    (溯源: {', '.join(up)} ← {', '.join(tools)})")
    if n.get("validity_conditions"):
        lines.append("- 适用条件: " + "；".join(n["validity_conditions"]))
    return lines


def evidence_card(session: dict) -> str:
    """Deterministic markdown rendering of a frozen session artifact."""
    nodes = {n["id"]: n for n in session["graph"]["nodes"]}
    claims = [n for n in session["graph"]["nodes"] if n["category"] == "claim"]
    d = session.get("demand", {})
    out = ["## VisEvi 视觉证据卡",
           f"- 图像: {Path(session.get('image_ref', '?')).name}",
           f"- 推理需求: {d.get('proposition', '?')}",
           f"- 元件: {_node_summary(nodes)}"]
    v = session.get("verdict") or {}
    out += ["", "### 判定",
            (f"- answerable={v.get('answerable')} — {v.get('answer', '')}"
             if v.get("answerable") else
             f"- answerable=False（拒答）— {v.get('abstain_reason', '')}"),
            f"- 证据引用: {', '.join(v.get('evidence_refs', [])) or '-'}", ""]
    if claims:
        out += [line for n in claims for line in _claim_block(n, nodes)] + [""]
    else:
        out += ["### 证据声明",
                "- 无可交付证据声明（需求未满足或证据不足，见判定与工具轨迹）", ""]
    out += ["---",
            "注: 组合≠诊断 — 以上为发现级证据（自带不确定度与适用条件），"
            "诊断结论由消费方推理模型作出。"]
    return "\n".join(out)


def to_prompt_block(session: dict) -> str:
    """Fenced block for direct injection into any LLM prompt."""
    return "```markdown\n" + evidence_card(session) + "\n```"
