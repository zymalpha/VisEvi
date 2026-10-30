"""Anatomy artifact loaders; A1-A4 live/cached interfaces are in tools.py.

segment_anatomy   — PSPNet organ masks (A1–A4 outputs) as region Elements
detect_landmarks  — legacy image-bound HybridGNet LH-120 cache (not the formal A5)

SegmentAnatomy retains the legacy three-target default. New named A1-A4 tools
can run PSPNet live or consume cached predictions whose record binds image,
model, documented weights and preprocessing. Predictions are
not independent ground truth. Artifacts are mapped into the original frame:
  PSPNet 512-frame of center-crop square s=min(W,H), offsets (ox, oy)
  HybridGNet [0,1] of letterbox square s=max(W,H), offset oy_pad; (x,y)=(col,row)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.spec import ToolSpec, parameter_schema, evidence_result_schema
from ...core.units import EvidenceUnit, ToolRefusal
from .. import geometry as geo

# HybridGNet LH-120 landmark groups (validated in scripts/landmark_axis_check.py)
LM_GROUPS = {"right_lung": (0, 44), "left_lung": (44, 94), "heart": (94, 120)}

ORGANS = ("Right_Lung", "Left_Lung", "Heart")  # retain legacy default
ANATOMY_GROUPS = {
    "segment_lungs": ("A1", ("Right_Lung", "Left_Lung")),
    "segment_heart": ("A2", ("Heart",)),
    "segment_mediastinum": ("A3", ("Mediastinum",)),
    "segment_shoulder_bones": ("A4", ("Left_Clavicle", "Right_Clavicle", "Left_Scapula", "Right_Scapula")),
}
SUPPORTED_ORGANS = {name for _, names in ANATOMY_GROUPS.values() for name in names}


def _frame_params(img_w: int, img_h: int) -> dict:
    s_crop, s_pad = min(img_w, img_h), max(img_w, img_h)
    return {"s_crop": s_crop, "s_pad": s_pad,
            "ox": img_w // 2 - s_crop // 2, "oy": img_h // 2 - s_crop // 2,   # XRayCenterCrop
            "ox_pad": (s_pad - img_w) // 2,
            "oy_pad": (s_pad - img_h) // 2}


def load_mask_original_frame(masks_dir: str | Path, name: str,
                             img_w: int, img_h: int) -> np.ndarray:
    """512 model frame -> resize to s_crop -> paste at (ox, oy). Bool array."""
    p = _frame_params(img_w, img_h)
    with Image.open(Path(masks_dir) / f"{name}.png") as source:
        if source.size != (512, 512):
            raise ValueError("PSPNet artifacts must be in the declared 512x512 model frame")
        binary = geo._mask(np.asarray(source), name)
    m = Image.fromarray(binary.astype('uint8') * 255).resize(
        (p["s_crop"], p["s_crop"]), Image.Resampling.NEAREST)
    canvas = np.zeros((img_h, img_w), bool)
    canvas[p["oy"]:p["oy"] + p["s_crop"], p["ox"]:p["ox"] + p["s_crop"]] = np.asarray(m) > 127
    return canvas


class SegmentAnatomy:
    """A1–A4: expose lung, heart, mediastinum, and shoulder-bone masks."""

    def describe(self) -> ToolSpec:
        return ToolSpec(
            name="segment_anatomy", category="perception",
            description="Anatomy masks for lungs, heart, mediastinum, clavicles, and scapulae; "
                        "original-image pixel frame; consumes PSPNet prediction artifacts",
            input_schema=parameter_schema({"masks_dir": {"type": "string"},
                "image_ref": {"type": "string"},
                "organs": {"type": "array", "items": {"type": "string"}}},
                ("masks_dir", "image_ref")),
            output_schema=evidence_result_schema())

    def run(self, session, *, masks_dir: str, image_ref: str,
            organs: tuple[str, ...] = ORGANS) -> list[EvidenceUnit] | ToolRefusal:
        from .pspnet import file_sha256, verify_cache_record

        try:
            if not organs or len(set(organs)) != len(organs) or any(name not in SUPPORTED_ORGANS for name in organs):
                raise ValueError("organs must contain unique supported A1-A4 target names")
            if Path(image_ref).resolve() != Path(session.image_ref).resolve():
                raise ValueError("image_ref must match the session image")
            with Image.open(image_ref) as image:
                w, h = image.size
            image_hash = file_sha256(image_ref)
            record_path = Path(masks_dir) / 'cache_record.json'
            if not record_path.is_file():
                return ToolRefusal(tool='segment_anatomy', reason=(
                    'cache_record.json missing; unbound masks are not accepted as PSPNet output '
                    '(attach supplied masks with session.attach_masks and explicit source metadata)'),
                    missing=[str(record_path)])
            record = json.loads(record_path.read_text())
            expected_hashes = verify_cache_record(record, image_ref, [f'{name}.png' for name in organs])
            staged = {}
            metadata = {}
            fp = _frame_params(w, h)
            valid_bbox = [fp['ox'], fp['oy'], fp['ox'] + fp['s_crop'], fp['oy'] + fp['s_crop']]
            for name in organs:
                path = Path(masks_dir) / f'{name}.png'
                if not path.is_file():
                    return ToolRefusal(tool='segment_anatomy', reason=f'mask not found: {name}', missing=[str(path)])
                digest = file_sha256(path)
                if digest != expected_hashes[path.name]:
                    raise ValueError(f'cached mask checksum mismatch: {name}')
                mask = load_mask_original_frame(masks_dir, name, w, h)
                if not mask.any():
                    return ToolRefusal(tool='segment_anatomy', reason=f'empty mask: {name}', missing=[f'nonempty:{name}'])
                staged[name] = mask
                metadata[name] = {
                    'image_sha256': image_hash, 'source_mask_sha256': digest,
                    'source_mask_path': str(path.resolve()), 'coordinate_frame': 'original_image_pixels',
                    'source_frame': '512x512 center crop', 'original_shape_hw': [h, w],
                    'valid_fov_bbox_xyxy_exclusive': valid_bbox,
                    'full_image_observed': w == h,
                    'fov_truncated_sides': fov_truncated_sides(mask, fp),
                    'cache_binding_verified': True,
                    'producer': record['binding']['producer'],
                    'laterality_basis': 'PSPNet target names; Right_* lies on image left in standard PA display',
                }
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return ToolRefusal(tool='segment_anatomy', reason=str(exc))
        # Commit only after every requested mask has passed validation.
        units = []
        for name, m in staged.items():
            f = Path(masks_dir) / f'{name}.png'
            mw, my = geo.max_row_width(m)
            bounds = geo.bbox(m)
            conditions = [
                "pixel units only; no mm spacing available for PNG",
                "region bbox has exclusive upper bounds; value.bbox retains legacy inclusive indices",
                "outside center-crop field of view is unobserved, not anatomical background",
                "cached prediction mask; accuracy not independently validated"]
            if metadata[name]['fov_truncated_sides']:
                conditions.append("mask touches the field-of-view edge; extent measurements are refused")
            unit = EvidenceUnit(
                id=session.next_id("obs"), category="observation", evidence_type="region",
                finding_or_anatomy=name,
                value={"area_px": geo.area(m), "max_width_px": mw, "max_width_row_y": my,
                       "bbox": bounds, "bbox_convention": "legacy inclusive pixel indices",
                       "mask_metadata": dict(metadata[name])},
                region={"x0": bounds['x0'], "y0": bounds['y0'],
                        "x1": bounds['x1'] + 1, "y1": bounds['y1'] + 1}, confidence=None,
                source_tool="segment_anatomy",
                provenance=[f"cached:PSPNet:{f.name}"],
                validity_conditions=conditions)
            session.add_observation(unit)
            metadata[name]['observation_id'] = unit.id
            units.append(unit)
        # masks themselves are kept for downstream geometry via session runtime cache
        session.attach_masks(staged, metadata)
        return units


def fov_truncated_sides(mask: np.ndarray, fp: dict) -> list[str]:
    """Crop edges lying inside the image that the mask touches."""
    h, w = mask.shape
    x0, y0, s = fp["ox"], fp["oy"], fp["s_crop"]
    edges = {"left": (x0 > 0, mask[:, x0]), "right": (x0 + s < w, mask[:, x0 + s - 1]),
             "top": (y0 > 0, mask[y0, :]), "bottom": (y0 + s < h, mask[y0 + s - 1, :])}
    return [side for side, (inside, line) in edges.items() if inside and line.any()]


class DetectLandmarks:
    """A5: HybridGNet LH-120 landmarks as landmark Elements (cached v0)."""

    def describe(self) -> ToolSpec:
        return ToolSpec(
            name="detect_landmarks", category="perception",
            description="LH-120 anatomical landmarks (groups: right_lung 0-44, "
                        "left_lung 44-94, heart 94-120), original pixel frame",
            input_schema=parameter_schema({"landmarks_json": {"type": "string"},
                "image_ref": {"type": "string"}, "groups": {"type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "integer"},
                    "minItems": 2, "maxItems": 2}}}, ("landmarks_json", "image_ref")),
            output_schema=evidence_result_schema())

    def run(self, session, *, landmarks_json: str, image_ref: str,
            groups: dict | None = None) -> list[EvidenceUnit] | ToolRefusal:
        from .pspnet import file_sha256

        p = Path(landmarks_json)
        if not p.exists():
            return ToolRefusal(tool="detect_landmarks", reason="landmarks file missing",
                               missing=[str(p)])
        try:
            with Image.open(image_ref) as image:
                w, h = image.size
            payload = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "landmarks_norm" not in payload:
                raise ValueError("landmarks file must contain landmarks_norm")
            if payload.get("image_sha256") != file_sha256(image_ref):
                return ToolRefusal(tool="detect_landmarks",
                                   reason="landmarks file is not bound to this image (image_sha256)",
                                   missing=["image_sha256"])
            lm = np.array(payload["landmarks_norm"], dtype=float)
            groups = groups or LM_GROUPS
            if (lm.ndim != 2 or lm.shape[1] != 2 or not np.isfinite(lm).all()
                    or any(not isinstance(a, int) or not isinstance(b, int) or a < 0 or b <= a or b > len(lm)
                           for a, b in groups.values())):
                raise ValueError("invalid landmark coordinates or groups")
        except (OSError, ValueError, TypeError) as exc:
            return ToolRefusal(tool="detect_landmarks", reason=str(exc))
        fp = _frame_params(w, h)
        lm_px = np.stack([lm[:, 0] * fp["s_pad"] - fp["ox_pad"],
                          lm[:, 1] * fp["s_pad"] - fp["oy_pad"]], 1)
        session.attach_landmarks({"lm_px": lm_px, "groups": groups})
        units = []
        for g, (a, b) in groups.items():
            pts = lm_px[a:b]
            xs, ys = pts[:, 0], pts[:, 1]
            unit = EvidenceUnit(
                id=session.next_id("obs"), category="observation", evidence_type="landmark",
                finding_or_anatomy=g,
                value={"n_points": int(b - a),
                       "centroid_xy": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                       "x_extent_px": round(float(xs.max() - xs.min()), 1)},
                region={"x0": float(xs.min()), "y0": float(ys.min()),
                        "x1": float(xs.max()), "y1": float(ys.max())},
                confidence=None, source_tool="detect_landmarks",
                provenance=[f"cached:HybridGNet:{p.name}"],
                validity_conditions=[
                    "legacy cached landmarks; not the formal A5 interface",
                    "axis convention (x,y)=(col,row) validated vs PSPNet centroids "
                    "(98px mean error vs 677px transposed)"])
            session.add_observation(unit)
            units.append(unit)
        return units
