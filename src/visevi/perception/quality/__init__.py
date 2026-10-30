"""Layer-1 image-quality & basic-condition tools (Capability_Matrix A25 family).

  inspiration_metrics / assess_inspiration  — A25 吸气程度（几何代理）
  rotation_metrics    / assess_rotation     — A25 体位旋转（双侧对称性代理）
  exposure_metrics    / assess_exposure      — A25 曝光度与对比度（强度统计）
  carry_view_assumption                    — A24 占位：CheXagent 未接 -> 显式假设

Deterministic only (masks + pixel statistics; no classifier in v0.1 — the
"专用质量分类器" part of A25 is a later increment). ALL bands/thresholds are
v0 placeholders to be calibrated on JSRT/Montgomery; every unit carries the
method and placeholder disclaimers in validity_conditions. Quality conditions
are global_condition evidence: they condition the whole session (cf.
view_assumed_PA alpha=0.95 already used by construction/ds.py).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ...core.spec import ToolSpec, parameter_schema, evidence_result_schema
from ...core.units import EvidenceUnit, ToolRefusal, fov_refusal
from .. import geometry as geo


# ---------------------------- pure metrics ---------------------------------

def _apex_base(m: np.ndarray) -> tuple[int, int]:
    rows = np.where(m.any(axis=1))[0]
    return int(rows.min()), int(rows.max())


def inspiration_metrics(m_right: np.ndarray, m_left: np.ndarray,
                        m_union: np.ndarray) -> dict:
    """吸气程度几何代理: 肺野高度 / 胸廓内宽（自动代理；经典标准是前肋计数，
    需肋骨检测，v0.1 未部署）。band 阈值为占位值。"""
    ra, rb = _apex_base(m_right)
    la, lb = _apex_base(m_left)
    rh, lh = rb - ra + 1, lb - la + 1          # inclusive, consistent with geo.bbox
    tw, _ = geo.max_row_width(m_union)
    mean_h = (rh + lh) / 2
    idx = mean_h / tw
    if idx < 0.80:
        band = "低通气嫌疑（浅吸气）"
    elif idx > 1.00:
        band = "过度通气嫌疑（深吸气）"
    else:
        band = "常见范围"
    return {"right_apex_y": ra, "right_base_y": rb, "right_height": rh,
            "left_apex_y": la, "left_base_y": lb, "left_height": lh,
            "mean_lung_height": round(mean_h, 1), "thorax_inner_width": tw,
            "inspiration_index": round(idx, 4),
            "diaphragm_dy_px": abs(rb - lb),
            "diaphragm_dy_ratio": round(abs(rb - lb) / mean_h, 4),
            "band_hint": band,
            "method": "mean lung-field height / thorax inner width "
                      "(automated proxy for anterior-rib counting; PLACEHOLDER bands)"}


def rotation_metrics(m_right: np.ndarray, m_left: np.ndarray,
                     m_heart: np.ndarray, midline_x: float) -> dict:
    """体位旋转代理: 双侧成对结构的纵向错位 + 面积/镜像对称性。band 阈值占位。"""
    ra, rb = _apex_base(m_right)
    la, lb = _apex_base(m_left)
    mean_h = ((rb - ra + 1) + (lb - la + 1)) / 2   # inclusive, as above
    apex_dy, base_dy = abs(ra - la), abs(rb - lb)
    proxy = max(apex_dy, base_dy) / mean_h
    hb = geo.bbox(m_heart)
    heart_cx = (hb["x0"] + hb["x1"]) / 2
    if proxy < 0.05:
        band = "轻微/无旋转"
    elif proxy < 0.10:
        band = "轻中度旋转嫌疑"
    else:
        band = "显著旋转嫌疑"
    return {"apex_dy_px": apex_dy, "base_dy_px": base_dy,
            "rotation_proxy": round(proxy, 4),
            "area_ratio_R_over_L": round(m_right.sum() / m_left.sum(), 4),
            "mirror_dice": round(geo.mirror_dice(m_right, m_left, midline_x)["mirror_dice"], 4),
            "heart_center_offset_px": round(abs(heart_cx - midline_x), 1),
            "band_hint": band,
            "method": "max(apex_dy, base_dy)/mean lung height + bilateral "
                      "symmetry context; PLACEHOLDER bands"}


def exposure_metrics(gray: np.ndarray, m_union: np.ndarray,
                     m_right: np.ndarray, m_left: np.ndarray) -> dict:
    """曝光度与对比度: 肺野内强度统计 + 纵隔带穿透代理（两肺内缘之间逐行间隙）
    + 全图削波比例。band 阈值占位。CXR 极性: 肺野暗、纵隔/骨亮。"""
    lung = gray[m_union]
    p5, p95 = np.percentile(lung, [5, 95])
    lung_mean, lung_std = float(lung.mean()), float(lung.std())
    # mediastinal strip: per-row gap between the two lungs' inner edges
    vals = []
    for y in np.where(m_right.any(axis=1) & m_left.any(axis=1))[0]:
        xr = np.where(m_right[y])[0].max()      # right lung inner edge (image-left organ)
        xl = np.where(m_left[y])[0].min()
        if xl > xr + 1:
            vals.append(gray[y, xr + 1:xl])
    if vals:
        strip = np.concatenate(vals)
        strip_mean, ratio = float(strip.mean()), float(strip.mean() / max(lung_mean, 1e-6))
        pen = "纵隔/脊柱可辨（占位）" if ratio >= 1.15 else "穿透不足嫌疑（占位）"
    else:
        strip_mean, ratio, pen = float("nan"), float("nan"), "无间隙行（掩膜重叠，代理不可用）"
    if lung_mean < 60:
        exp = "偏黑（曝光过度嫌疑，占位）"
    elif lung_mean > 110:
        exp = "偏白（曝光不足嫌疑，占位）"
    else:
        exp = "占位可接受范围"
    ci = (p95 - p5) / 255.0
    con = "对比度偏低嫌疑（占位）" if ci < 0.50 else "对比度可接受（占位）"
    return {"lung_mean": round(lung_mean, 1), "lung_std": round(lung_std, 1),
            "lung_p5": round(float(p5), 1), "lung_p95": round(float(p95), 1),
            "contrast_index": round(ci, 4),
            "clip_low_frac": round(float((gray == 0).mean()), 5),
            "clip_high_frac": round(float((gray == 255).mean()), 5),
            "mediastinum_strip_mean": round(strip_mean, 1) if strip_mean == strip_mean else None,
            "penetration_ratio": round(ratio, 3) if ratio == ratio else None,
            "exposure_hint": exp, "contrast_hint": con, "penetration_hint": pen,
            "method": "intensity stats in lung union + per-row inter-lung gap strip "
                      "+ global clipping fractions; PLACEHOLDER bands"}


# ---------------------------- registry tools --------------------------------

def _require_masks(session, keys):
    masks = session.masks
    missing = [k for k in keys if k not in masks]
    if missing:
        return None, ToolRefusal(tool="quality_tool", reason="required Layer-1 "
                               "elements missing (D6: run segment_anatomy first)",
                               missing=[f"mask:{m}" for m in missing])
    truncated = session.fov_truncated(keys)
    if truncated:
        return None, fov_refusal("quality_tool", truncated)
    return masks, None


class AssessInspiration:
    """A25-吸气: lung-height/thorax-width proxy as global_condition evidence."""

    def describe(self) -> ToolSpec:
        return ToolSpec(name="assess_inspiration", category="perception",
                        description="Inspiration level proxy: lung field height over "
                                    "thorax inner width (A25; placeholder bands)",
                        input_schema=parameter_schema(), output_schema=evidence_result_schema())

    def run(self, session, **_):
        masks, refusal = _require_masks(session, ("Right_Lung", "Left_Lung"))
        if refusal:
            refusal.tool = "assess_inspiration"
            return refusal
        v = inspiration_metrics(masks["Right_Lung"], masks["Left_Lung"],
                                masks["Right_Lung"] | masks["Left_Lung"])
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation",
            evidence_type="global_condition", finding_or_anatomy="inspiration_level",
            value={k: v[k] for k in v},
            source_tool="assess_inspiration", provenance=session.mask_provenance(("Right_Lung", "Left_Lung")),
            validity_conditions=[v["method"], "bands are v0 placeholders"])
        session.add_observation(unit)
        return [unit]


class AssessRotation:
    """A25-旋转: bilateral symmetry proxies as global_condition evidence."""

    def describe(self) -> ToolSpec:
        return ToolSpec(name="assess_rotation", category="perception",
                        description="Body-position rotation proxy from bilateral "
                                    "structure offsets (A25; placeholder bands)",
                        input_schema=parameter_schema(), output_schema=evidence_result_schema())

    def run(self, session, **_):
        masks, refusal = _require_masks(session, ("Right_Lung", "Left_Lung", "Heart"))
        if refusal:
            refusal.tool = "assess_rotation"
            return refusal
        union = masks["Right_Lung"] | masks["Left_Lung"]
        mid = (geo.bbox(union)["x0"] + geo.bbox(union)["x1"]) / 2.0
        v = rotation_metrics(masks["Right_Lung"], masks["Left_Lung"],
                             masks["Heart"], mid)
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation",
            evidence_type="global_condition", finding_or_anatomy="rotation",
            value=v, source_tool="assess_rotation",
            provenance=session.mask_provenance(("Right_Lung", "Left_Lung", "Heart")),
            validity_conditions=[v["method"], "bands are v0 placeholders",
                                 "heart offset is context (cardiac position ≠ rotation)"])
        session.add_observation(unit)
        return [unit]


class AssessExposure:
    """A25-曝光/对比度: lung-field intensity stats + penetration proxy."""

    def describe(self) -> ToolSpec:
        return ToolSpec(name="assess_exposure", category="perception",
                        description="Exposure & contrast: intensity stats in lung "
                                    "fields + mediastinal-strip penetration (A25)",
                        input_schema=parameter_schema(), output_schema=evidence_result_schema())

    def run(self, session, **_):
        masks, refusal = _require_masks(session, ("Right_Lung", "Left_Lung"))
        if refusal:
            refusal.tool = "assess_exposure"
            return refusal
        with Image.open(session.image_ref) as image:
            if image.mode in ("I", "F") or image.mode.startswith("I;16"):
                return ToolRefusal(tool="assess_exposure", reason="exposure proxy bands are "
                                   "defined for 8-bit display rasters; high-bit-depth input "
                                   "requires an explicit recorded windowing conversion")
            gray = np.asarray(image.convert("L"))
        v = exposure_metrics(gray, masks["Right_Lung"] | masks["Left_Lung"],
                             masks["Right_Lung"], masks["Left_Lung"])
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation",
            evidence_type="global_condition", finding_or_anatomy="exposure_contrast",
            value=v, source_tool="assess_exposure", provenance=session.mask_provenance(("Right_Lung", "Left_Lung")),
            validity_conditions=[v["method"], "bands are v0 placeholders",
                                 "CXR polarity assumed dark-lung/bright-mediastinum"])
        session.add_observation(unit)
        return [unit]


class CarryViewAssumption:
    """A24 占位: no view classifier deployed -> the PA assumption becomes an
    explicit global_condition unit (instead of an invisible constant)."""

    def describe(self) -> ToolSpec:
        return ToolSpec(name="carry_view_assumption", category="perception",
                        description="A24 placeholder: carry view as EXPLICIT "
                                    "assumed_PA (classifier unavailable)",
                        input_schema=parameter_schema(), output_schema=evidence_result_schema())

    def run(self, session, **_):
        unit = EvidenceUnit(
            id=session.next_id("obs"), category="observation",
            evidence_type="global_condition", finding_or_anatomy="view_position",
            value={"view": "assumed_PA", "assumed": True,
                   "classifier": "none (A24/CheXagent not deployed)",
                   "downstream_effect": "alpha=0.95 discount in construction/ds.py"},
            source_tool="carry_view_assumption", provenance=[],
            validity_conditions=["assumption, not measurement; AP view would "
                                 "magnify heart 5-10% and shift CTR thresholds"])
        session.add_observation(unit)
        return [unit]
