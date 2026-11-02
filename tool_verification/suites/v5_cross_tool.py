"""Cross-tool consistency distributions on the frozen core40 set.

These are agreement diagnostics, not ground-truth accuracy scores.  Different
models have different labels and image support, so each pair is reported with
an explicit mapping and per-case values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
try:
    from .case_manifest import verify_frozen_cases
except ImportError:
    from case_manifest import verify_frozen_cases

ROOT = Path(__file__).resolve().parents[2]
TV = ROOT / "tool_verification"


def dice(a, b):
    den = int(a.sum() + b.sum())
    return 1.0 if den == 0 else float(2 * (a & b).sum() / den)


def load_png(path):
    with Image.open(path) as im:
        return np.asarray(im) > 0


def load_psp(path, shape):
    """Expand a 512 center-crop mask into the original image frame."""
    mask = load_png(path)
    h, w = shape
    side = min(h, w)
    with Image.fromarray(mask.astype(np.uint8) * 255) as im:
        square = np.asarray(im.resize((side, side), Image.Resampling.NEAREST)) > 0
    out = np.zeros((h, w), bool)
    oy, ox = (h - side) // 2, (w - side) // 2
    out[oy:oy + side, ox:ox + side] = square
    return out


def load_cxas(record_dir, label, shape):
    rec = json.loads((record_dir / "cache_record.json").read_text())
    d = np.load(record_dir / "labels_model_frame.npz")
    shp = tuple(int(v) for v in d["shape"])
    masks = np.unpackbits(d["packed"], axis=-1)[..., :shp[2]].astype(bool)
    idx = next(int(k) for k, v in rec["result"]["labels"].items() if v == label)
    h, w = shape
    rows = np.minimum(np.floor(np.arange(h) * shp[1] / h).astype(int), shp[1] - 1)
    cols = np.minimum(np.floor(np.arange(w) * shp[2] / w).astype(int), shp[2] - 1)
    return masks[idx][np.ix_(rows, cols)]


def centroid(mask):
    y, x = np.nonzero(mask)
    return None if not len(x) else [float(x.mean()), float(y.mean())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be a new directory")
    args.out.mkdir(parents=True)
    cases = json.loads((TV / "cases/core40.json").read_text())["cases"]
    verify_frozen_cases(cases, ROOT / "original_data")
    psp_root = TV / "runs/20261030/v5_ctr_crosscheck_retry4"
    model_root = TV / "runs/20261030/v3_models_retry1/ours"
    rows = []
    for case in cases:
        stem = Path(case["image_path"]).stem
        h, w = case["size_wh"][1], case["size_wh"][0]
        psp = psp_root / stem / "pspnet"
        cxas = psp_root / stem / "cxas"
        a5_path = model_root / stem / "a5/cache_record.json"
        a26_path = model_root / stem / "a26/cache_record.json"
        row = {"case": stem, "source": case["source"]}
        try:
            heart = load_psp(psp / "Heart.png", (h, w))
            lung_r = load_psp(psp / "Right_Lung.png", (h, w))
            lung_l = load_psp(psp / "Left_Lung.png", (h, w))
            row["psp_heart_vs_cxas_heart_dice"] = dice(heart, load_cxas(cxas, "heart", (h, w)))
            # CXAS uses patient-side names, matching PSPNet's image-side convention.
            row["psp_right_lung_vs_cxas_right_lung_dice"] = dice(lung_r, load_cxas(cxas, "right lung", (h, w)))
            row["psp_left_lung_vs_cxas_left_lung_dice"] = dice(lung_l, load_cxas(cxas, "left lung", (h, w)))
        except Exception as exc:
            row["psp_cxas_error"] = str(exc)
        try:
            a5 = json.loads(a5_path.read_text())["result"]
            n = np.asarray(a5["nodes_xy"], dtype=float)
            # A5 contour polygon is a light-weight structural comparison;
            # reuse cv2 only here to avoid treating its point contour as a mask.
            import cv2
            pred_r, pred_l = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
            cv2.fillPoly(pred_r, [np.rint(n[:44]).astype(np.int32)], 1)
            cv2.fillPoly(pred_l, [np.rint(n[44:94]).astype(np.int32)], 1)
            row["hybridgnet_right_lung_vs_psp_right_lung_dice"] = dice(pred_r.astype(bool), load_psp(psp / "Right_Lung.png", (h, w)))
            row["hybridgnet_left_lung_vs_psp_left_lung_dice"] = dice(pred_l.astype(bool), load_psp(psp / "Left_Lung.png", (h, w)))
        except Exception as exc:
            row["hybridgnet_psp_error"] = str(exc)
        try:
            a26 = json.loads(a26_path.read_text())["result"]
            carina = np.asarray(a26["carina_xy"], dtype=float)
            cxas_carina = centroid(load_cxas(cxas, "tracheal bifurcation", (h, w)))
            row["a26_carina_xy"] = carina.tolist()
            row["cxas_tracheal_bifurcation_centroid_xy"] = cxas_carina
            row["a26_cxas_carina_distance_frac_width"] = (
                None if cxas_carina is None else float(np.linalg.norm(carina - np.asarray(cxas_carina)) / w))
        except Exception as exc:
            row["a26_cxas_error"] = str(exc)
        rows.append(row)
    numeric = {}
    for key in ("psp_heart_vs_cxas_heart_dice", "psp_right_lung_vs_cxas_right_lung_dice",
                "psp_left_lung_vs_cxas_left_lung_dice", "hybridgnet_right_lung_vs_psp_right_lung_dice",
                "hybridgnet_left_lung_vs_psp_left_lung_dice", "a26_cxas_carina_distance_frac_width"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        numeric[key] = {"n": len(vals), "median": float(np.median(vals)) if vals else None,
                        "p10": float(np.quantile(vals, .1)) if vals else None,
                        "p90": float(np.quantile(vals, .9)) if vals else None,
                        "min": float(np.min(vals)) if vals else None,
                        "max": float(np.max(vals)) if vals else None}
    summary = {"cases": len(rows), "distributions": numeric,
               "note": "agreement diagnostics only; no pair is treated as ground truth"}
    (args.out / "cross_tool_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (args.out / "cross_tool_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
