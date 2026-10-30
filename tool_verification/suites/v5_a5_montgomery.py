"""V5 A5 semantic-node and contour check on Montgomery.

This is an implementation/representation check on the model's documented
training-related data, not an independent medical-accuracy estimate.
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


def load_mask(path: Path, shape):
    with Image.open(path) as image:
        mask = np.asarray(image) > 0
    if tuple(mask.shape) != tuple(shape):
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    return mask


def polygon_mask(points, shape):
    out = np.zeros(shape, np.uint8)
    cv2.fillPoly(out, [np.rint(np.asarray(points)).astype(np.int32)], 1)
    return out.astype(bool)


def semantic_checks(nodes):
    """Check the frozen table against the contour's geometric roles.

    A5's node order walks each contour from apex down the outer border, across
    the base, and back up the inner border.  The ranges below are therefore
    part of the semantic contract, rather than a nearest-neighbour heuristic.
    """
    n = np.asarray(nodes, dtype=float)
    return {
        "right_lung_apex_is_min_y": int(np.argmin(n[:44, 1])) == 0,
        "right_costophrenic_is_max_y_outer": int(np.argmax(n[:22, 1])) == 21,
        "right_cardiophrenic_is_max_x_base": int(np.argmax(n[21:30, 0]) + 21) == 29,
        "left_lung_apex_is_min_y": int(np.argmin(n[44:94, 1]) + 44) == 44,
        "left_costophrenic_is_max_y_outer": int(np.argmax(n[44:66, 1]) + 44) == 65,
        "left_cardiophrenic_is_min_x_base": int(np.argmin(n[65:72, 0]) + 65) == 71,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:4")
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be a new directory")
    args.out.mkdir(parents=True)

    manifest = json.loads((TV / "cases/montgomery138.json").read_text())
    gt_root = ROOT / "original_data/montgomery-nlm/data/MontgomerySet/ManualMask"
    reg = build_a_registry()
    rows = []
    try:
        for index, case in enumerate(manifest["cases"], 1):
            image = ROOT / "original_data" / case["image_path"]
            stem = image.stem
            case_out = args.out / stem
            session = EvidenceSession(image_ref=str(image))
            result = reg.run("locate_lung_landmarks", session,
                             output_dir=str(case_out), device=args.device)
            row = {"case": stem, "image": str(image)}
            if isinstance(result, ToolRefusal):
                row.update({"status": "refusal", "reason": result.reason})
            else:
                nodes = np.asarray(session.landmarks["lm_px"], dtype=float)
                with Image.open(image) as im:
                    width, height = im.size
                shape = (height, width)
                side = min(width, height)
                ox, oy = (width - side) // 2, (height - side) // 2
                fov = np.zeros(shape, bool)
                fov[oy:oy + side, ox:ox + side] = True
                gt_left = load_mask(gt_root / "leftMask" / f"{stem}.png", shape)
                gt_right = load_mask(gt_root / "rightMask" / f"{stem}.png", shape)
                # A5 node groups follow image-side naming: right lung is left.
                pred_right = polygon_mask(nodes[0:44], shape)
                pred_left = polygon_mask(nodes[44:94], shape)
                semantic = semantic_checks(nodes)
                row.update({
                    "status": "ok",
                    "right_dice_in_fov": dice(pred_right & fov, gt_left & fov),
                    "left_dice_in_fov": dice(pred_left & fov, gt_right & fov),
                    "right_gt_outside_fraction": float((gt_left & ~fov).sum() / max(1, gt_left.sum())),
                    "left_gt_outside_fraction": float((gt_right & ~fov).sum() / max(1, gt_right.sum())),
                    "semantic_checks": semantic,
                    "semantic_all_ok": all(semantic.values()),
                    "fov_bbox_xyxy_exclusive": [ox, oy, ox + side, oy + side],
                })
            rows.append(row)
            print(f"a5-montgomery {index}/{len(manifest['cases'])} {stem}", flush=True)
    finally:
        backends.shutdown_workers()

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "task": "a5-montgomery",
        "cases": len(rows),
        "successful": len(ok),
        "right_dice_median_in_fov": float(np.median([r["right_dice_in_fov"] for r in ok])) if ok else None,
        "left_dice_median_in_fov": float(np.median([r["left_dice_in_fov"] for r in ok])) if ok else None,
        "right_dice_min_in_fov": float(np.min([r["right_dice_in_fov"] for r in ok])) if ok else None,
        "left_dice_min_in_fov": float(np.min([r["left_dice_in_fov"] for r in ok])) if ok else None,
        "semantic_all_ok_cases": sum(r.get("semantic_all_ok", False) for r in ok),
        "semantic_rates": {
            key: float(np.mean([r["semantic_checks"][key] for r in ok]))
            for key in (list(ok[0]["semantic_checks"]) if ok else [])
        },
        "thresholds": {"dice_median": 0.90, "semantic_rate": 1.0},
        "note": "implementation/representation check on documented or author-related data; not independent medical accuracy",
    }
    (args.out / "a5_montgomery_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (args.out / "a5_montgomery_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    if summary["successful"] != summary["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
