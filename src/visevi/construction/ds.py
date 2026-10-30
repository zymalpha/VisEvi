"""Dempster–Shafer evidence kernel (LAYER2_ALGORITHM_SPEC.md §1–§3).

Mass functions are plain dicts over {"F", "NOT_F", "THETA"} (binary frame;
missing keys = 0). Semantics:
  discount  — input unreliability widens ignorance, never flips direction (§2.2)
  dempster  — independent-source combination; returns conflict K (§3.2)
  yager     — high-conflict alternative: conflict mass goes to THETA (§3.3)
  bel_pl    — report interval [Bel, Pl]
Calibration curves g_* are explicit, versioned data with parameters flagged
"to be calibrated" (v0 uses literature thresholds).
"""
from __future__ import annotations

KEYS = ("F", "NOT_F", "THETA")


def _norm(m: dict) -> dict:
    out = {k: float(m.get(k, 0.0)) for k in KEYS}
    s = sum(out.values())
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"mass does not sum to 1: {m}")
    return out


def discount(m: dict, alpha: float) -> dict:
    """Shafer discounting: m'(A)=α·m(A), m'(Θ)=1−α+α·m(Θ). alpha = product of
    per-input reliability factors (view, mask quality, rotation...)."""
    m = _norm(m)
    return {k: round(alpha * m[k], 6) for k in ("F", "NOT_F")} | {
        "THETA": round(1 - alpha + alpha * m["THETA"], 6)}


def dempster(m1: dict, m2: dict) -> tuple[dict, float]:
    """Orthogonal sum for independent sources. Returns (combined, conflict K)."""
    m1, m2 = _norm(m1), _norm(m2)
    k = m1["F"] * m2["NOT_F"] + m1["NOT_F"] * m2["F"]
    if k >= 1.0:
        raise ValueError("fully conflicting sources (K=1)")
    m = {
        "F": (m1["F"] * m2["F"] + m1["F"] * m2["THETA"] + m1["THETA"] * m2["F"]) / (1 - k),
        "NOT_F": (m1["NOT_F"] * m2["NOT_F"] + m1["NOT_F"] * m2["THETA"]
                  + m1["THETA"] * m2["NOT_F"]) / (1 - k),
    }
    m["THETA"] = 1 - m["F"] - m["NOT_F"]
    return {kk: round(v, 6) for kk, v in m.items()}, round(k, 6)


def yager(m1: dict, m2: dict) -> tuple[dict, float]:
    """Conflict mass assigned to ignorance instead of being normalized away
    (spec §3.3: do not average / hide conflict)."""
    m1, m2 = _norm(m1), _norm(m2)
    k = m1["F"] * m2["NOT_F"] + m1["NOT_F"] * m2["F"]
    m = {
        "F": m1["F"] * m2["F"] + m1["F"] * m2["THETA"] + m1["THETA"] * m2["F"],
        "NOT_F": m1["NOT_F"] * m2["NOT_F"] + m1["NOT_F"] * m2["THETA"]
                 + m1["THETA"] * m2["NOT_F"],
        "THETA": m1["THETA"] * m2["THETA"] + k,
    }
    return {kk: round(v, 6) for kk, v in m.items()}, round(k, 6)


def combine(m1: dict, m2: dict, conflict_threshold: float = 0.6) -> tuple[dict, float, str]:
    """Policy of spec §3.3: K > threshold -> yager (and K stays a first-class
    output); else dempster."""
    _, k = yager(m1, m2)                      # K identical under both rules
    if k > conflict_threshold:
        m, _ = yager(m1, m2)
        return m, k, "yager"
    m, _ = dempster(m1, m2)
    return m, k, "dempster"


def bel_pl(m: dict) -> dict:
    m = _norm(m)
    return {"bel_F": round(m["F"], 6), "pl_F": round(m["F"] + m["THETA"], 6),
            "bel_not_f": round(m["NOT_F"], 6), "pl_not_f": round(m["NOT_F"] + m["THETA"], 6),
            "interval_width": round(m["THETA"], 6)}


# --- calibration curves (explicit data; v0 params from literature) ---------

def g_ctr(ctr: float, strength: float = 0.85) -> dict:
    """CTR -> mass over {cardiac-enlargement-pattern F, ¬F, Θ}. PA thresholds
    0.45/0.55, linear between. Params TO BE CALIBRATED (spec §2.1)."""
    if ctr <= 0.45:
        return {"NOT_F": strength, "THETA": round(1 - strength, 6)}
    if ctr >= 0.55:
        return {"F": strength, "THETA": round(1 - strength, 6)}
    t = (ctr - 0.45) / 0.10
    return {"F": round(strength * t, 6), "NOT_F": round(strength * (1 - t), 6),
            "THETA": round(1 - strength, 6)}


def g_symmetry(dice: float, strength: float = 0.8) -> dict:
    """Mirror-Dice -> mass over {high-bilateral-symmetry F, ¬F, Θ}.
    dice>=0.93 strong symmetry; <=0.80 strong asymmetry; linear between.
    Params TO BE CALIBRATED."""
    if dice >= 0.93:
        return {"F": strength, "THETA": round(1 - strength, 6)}
    if dice <= 0.80:
        return {"NOT_F": strength, "THETA": round(1 - strength, 6)}
    t = (dice - 0.80) / 0.13
    return {"F": round(strength * t, 6), "NOT_F": round(strength * (1 - t), 6),
            "THETA": round(1 - strength, 6)}


# v0 discount factors (placeholder values, every one flagged to-calibrate)
ALPHA = {"view_assumed_PA": 0.95,   # A24 unavailable -> explicit assumption (spec §6)
         "pspnet_mask": 0.85,
         "hybridgnet_landmarks": 0.85}
