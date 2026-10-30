"""V5 implementation checks on documented or author-training data.

This is an implementation/representation check, not a medical accuracy study.
Run individual tasks into a new directory, for example:

  .venv/bin/python tool_verification/suites/v5_expected.py --task montgomery --out ... --device cuda:4

Tasks: montgomery, jsrt, ctr, nih-view.  Results are append-only and preserve
per-case inputs, model outputs, and the exact denominator used for each metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visevi import backends
from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import build_a_registry

ROOT = Path(__file__).resolve().parents[2]
TV = ROOT / "tool_verification"


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    den = int(a.sum() + b.sum())
    return 1.0 if den == 0 else float(2 * (a & b).sum() / den)


def load_mask(path, shape=None):
    with Image.open(path) as image:
        m = np.asarray(image) > 0
    if shape is not None and tuple(m.shape) != tuple(shape):
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m


def result_or_refusal(result):
    if isinstance(result, ToolRefusal):
        return {"status": "refusal", "reason": result.reason, "missing": result.missing}
    return {"status": "ok", "units": [u.to_dict() for u in result]}


def run_segment(reg, image, out, device, tool):
    session = EvidenceSession(image_ref=str(image))
    result = reg.run(tool, session, output_dir=str(out), device=device)
    return result, session


def montgomery(args, reg):
    manifest = json.loads((TV / "cases/montgomery138.json").read_text())
    root = ROOT / "original_data/montgomery-nlm/data/MontgomerySet"
    rows, node_rows = [], []
    for index, case in enumerate(manifest["cases"], 1):
        image = ROOT / "original_data" / case["image_path"]
        stem = image.stem
        out = args.out / "montgomery" / stem
        result, session = run_segment(reg, image, out, args.device, "segment_lungs")
        row = {"case": stem, "image": str(image), "result": result_or_refusal(result)}
        if not isinstance(result, ToolRefusal):
            with Image.open(image) as im:
                w, h = im.size
            side = min(w, h); ox, oy = (w - side) // 2, (h - side) // 2
            fov = np.zeros((h, w), bool); fov[oy:oy + side, ox:ox + side] = True
            gt_left = load_mask(root / "ManualMask/leftMask" / f"{stem}.png")
            gt_right = load_mask(root / "ManualMask/rightMask" / f"{stem}.png")
            pred_right, pred_left = session.masks["Right_Lung"], session.masks["Left_Lung"]
            # Montgomery leftMask is image-left (radiological patient right),
            # so it maps to PSPNet Right_Lung.
            row.update({
                "right_dice_in_fov": dice(pred_right & fov, gt_left & fov),
                "left_dice_in_fov": dice(pred_left & fov, gt_right & fov),
                "right_gt_outside_fraction": float((gt_left & ~fov).sum() / max(1, gt_left.sum())),
                "left_gt_outside_fraction": float((gt_right & ~fov).sum() / max(1, gt_right.sum())),
                "pred_right_fov_sides": session.mask_metadata["Right_Lung"].get("fov_truncated_sides", []),
                "pred_left_fov_sides": session.mask_metadata["Left_Lung"].get("fov_truncated_sides", []),
            })
        rows.append(row)
        print(f"montgomery {index}/{len(manifest['cases'])} {stem}", flush=True)
    ok = [r for r in rows if "right_dice_in_fov" in r]
    summary = {
        "task": "montgomery",
        "cases": len(rows),
        "successful": len(ok),
        "right_dice_median_in_fov": float(np.median([r["right_dice_in_fov"] for r in ok])) if ok else None,
        "left_dice_median_in_fov": float(np.median([r["left_dice_in_fov"] for r in ok])) if ok else None,
        "right_gt_outside_median": float(np.median([r["right_gt_outside_fraction"] for r in ok])) if ok else None,
        "left_gt_outside_median": float(np.median([r["left_gt_outside_fraction"] for r in ok])) if ok else None,
        "note": "implementation check on documented/author-related data; not an independent medical accuracy result",
    }
    return summary, rows


def polygon_mask(points, shape):
    mask = np.zeros(shape, np.uint8)
    pts = np.rint(np.asarray(points)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def jsrt(args, reg):
    manifest = json.loads((TV / "cases/jsrt_scr247.json").read_text())
    rows = []
    for index, case in enumerate(manifest["cases"], 1):
        image = ROOT / "original_data" / case["image_path"]
        stem = image.stem; out = args.out / "jsrt" / stem
        result, session = run_segment(reg, image, out, args.device, "segment_lungs")
        # Reuse the shared PSPNet prediction cache and the same session for A2.
        heart_result = reg.run("segment_heart", session, output_dir=str(out), device=args.device) if not isinstance(result, ToolRefusal) else result
        row = {"case": stem, "result": result_or_refusal(result)}
        if not isinstance(result, ToolRefusal) and not isinstance(heart_result, ToolRefusal):
            with Image.open(image) as im: shape = (im.height, im.width)
            # The mirror follows the usual image-side naming: left_lung is on
            # image-left, so it maps to PSPNet Right_Lung (same convention as
            # Montgomery). Keep the explicit image-side mapping in the row.
            gt_image_left = load_mask(ROOT / "original_data" / case["masks"]["left_lung"], shape)
            gt_image_right = load_mask(ROOT / "original_data" / case["masks"]["right_lung"], shape)
            gt_heart = load_mask(ROOT / "original_data" / case["masks"]["heart"], shape)
            # The mirror masks are image-side labels in the SCR convention;
            # report the best explicit side pairing and retain both values.
            row.update({
                "right_lung_dice": dice(session.masks["Right_Lung"], gt_image_left),
                "left_lung_dice": dice(session.masks["Left_Lung"], gt_image_right),
                "heart_dice": dice(session.masks.get("Heart", np.zeros(shape, bool)), gt_heart),
            })
        rows.append(row)
        print(f"jsrt {index}/{len(manifest['cases'])} {stem}", flush=True)
    ok = [r for r in rows if "heart_dice" in r]
    summary = {
        "task": "jsrt",
        "cases": len(rows), "successful": len(ok),
        "right_lung_dice_median": float(np.median([r["right_lung_dice"] for r in ok])) if ok else None,
        "left_lung_dice_median": float(np.median([r["left_lung_dice"] for r in ok])) if ok else None,
        "heart_dice_median": float(np.median([r["heart_dice"] for r in ok])) if ok else None,
        "note": "third-party SCR mirror used for implementation checking only; not formal GT",
    }
    return summary, rows


def nih_view(args, reg):
    manifest = json.loads((TV / "cases/nih_view292.json").read_text())
    rows = []
    for index, case in enumerate(manifest["cases"], 1):
        image = ROOT / "original_data" / case["image_path"]
        stem = image.stem; out = args.out / "nih" / stem
        session = EvidenceSession(image_ref=str(image))
        result = reg.run("classify_view", session, output_dir=str(out), device=args.device)
        value = result_or_refusal(result)
        predicted = None
        if not isinstance(result, ToolRefusal) and result:
            predicted = result[0].value.get("view")
        rows.append({"case": stem, "expected": case["view_position"], "predicted": predicted,
                     "result": value})
        print(f"nih-view {index}/{len(manifest['cases'])} {stem}", flush=True)
    ok = [r for r in rows if r["predicted"]]
    accuracy = sum(r["predicted"] == r["expected"] for r in ok) / len(ok) if ok else None
    return {"task": "nih-view", "cases": len(rows), "successful": len(ok),
            "accuracy": accuracy, "note": "NIH is in CheXOne training sources; implementation check only"}, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("montgomery", "jsrt", "nih-view"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:4")
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be a new directory")
    args.out.mkdir(parents=True)
    reg = build_a_registry()
    try:
        summary, rows = (montgomery(args, reg) if args.task == "montgomery" else
                         jsrt(args, reg) if args.task == "jsrt" else nih_view(args, reg))
    finally:
        backends.shutdown_workers()
    (args.out / f"{args.task}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (args.out / f"{args.task}_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
