"""Render the EvidenceGraph of a session.json as a PNG (node-link diagram).

Usage:
    python scripts/plot_evidence_graph.py <session.json> <out.png>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# column per element type (left -> right = Layer1 -> measurement -> Layer2 claim)
COL = {"region": 0, "landmark": 0.02, "measurement": 2, "claim": 4}
BAND = {"region": +1, "landmark": -1}          # landmarks stacked below regions
COLOR = {"region": "#cfe8ff", "landmark": "#d5f0d5", "measurement": "#ffe3b3",
         "claim": "#ffc9c9"}
EDGE_COLOR = {"P-CTR-01": "#2a6fbd", "P-SYM-01": "#2e8b57"}

session = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
nodes = {n["id"]: n for n in session["graph"]["nodes"]}
edges = session["graph"]["edges"]

# --- layout ---------------------------------------------------------------
groups: dict[str, list[str]] = {}
for nid, n in nodes.items():
    key = "claim" if n["category"] == "claim" else n["evidence_type"]
    groups.setdefault(key, []).append(nid)

pos: dict[str, tuple[float, float]] = {}
for gtype, ids in groups.items():
    x = COL.get(gtype, 3)
    for i, nid in enumerate(ids):
        y = (len(ids) - 1) / 2 - i
        y += BAND.get(gtype, 0) * 3.2          # separate band for landmarks
        pos[nid] = (x, y)

fig, ax = plt.subplots(figsize=(15, 9))


def label(n: dict) -> str:
    v = n["value"]
    t = f"{n['id']}\n{n['finding_or_anatomy']}"
    if n["evidence_type"] == "region":
        t += f"\n面积 {v['area_px']:,}px²"
    elif n["evidence_type"] == "landmark":
        t += f"\n{v['n_points']} 点"
    elif "CTR" in v:
        t += f"\nCTR = {v['CTR']}"
    elif "mirror_dice" in v:
        t += f"\ndice = {v['mirror_dice']}"
    if n["category"] == "claim":
        t += f"\n支持: {v.get('supported_direction')}  Bel={v['bel']}  Pl={v['pl']}"
    return t


for nid, n in nodes.items():
    x, y = pos[nid]
    gtype = "claim" if n["category"] == "claim" else n["evidence_type"]
    w = 2.6 if n["category"] != "claim" else 2.9
    h = 1.15 if n["category"] != "claim" else 1.5
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.08", linewidth=1.4,
                                edgecolor="#444", facecolor=COLOR.get(gtype, "#eee")))
    ax.text(x, y, label(n), ha="center", va="center", fontsize=8.5)

for e in edges:
    x0, y0 = pos[e["src"]]
    x1, y1 = pos[e["dst"]]
    # trim arrows to box edges (rough horizontal trim)
    dx = x1 - x0
    shrink = 1.45 if abs(dx) > 1 else 0.05
    ax.add_patch(FancyArrowPatch((x0 + shrink, y0), (x1 - shrink, y1),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3,
                                 color=EDGE_COLOR.get(e["basis"], "#888")))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.14, e["relation"],
            ha="center", fontsize=7.5, color="#555")

ax.set_xlim(-1.6, 5.7)
ax.set_ylim(min(y for _, y in pos.values()) - 1.4, max(y for _, y in pos.values()) + 1.6)
ax.axis("off")

# legend + verdict
handles = [plt.Line2D([], [], marker="s", ls="", markersize=11, markeredgecolor="#444",
                      markerfacecolor=c, label=t)
           for t, c in [("region 元件 (PSPNet)", COLOR["region"]),
                        ("landmark 元件 (HybridGNet)", COLOR["landmark"]),
                        ("measurement (确定性几何)", COLOR["measurement"]),
                        ("claim 证据 [Bel,Pl]", COLOR["claim"])]]
handles += [plt.Line2D([], [], color=c, label=b)
            for b, c in EDGE_COLOR.items()]
ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.9)

v = session.get("verdict")
if v:
    ax.set_title("EvidenceSession 证据图  |  " + session["image_ref"].split("/")[-1]
                 + f"  |  {len(nodes)} 节点 · {len(edges)} 边（每边带 basis）"
                 + (f"\nVerdict: {v['answer']}" if v.get("answer") else ""),
                 fontsize=11, loc="left")

fig.tight_layout()
out = Path(sys.argv[2])
fig.savefig(out, dpi=150)
print("saved:", out.resolve())
