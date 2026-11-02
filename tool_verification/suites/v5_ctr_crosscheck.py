"""Cross-check A12 against the original CXAS CTR extractor on core40."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from visevi import backends
from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import build_a_registry
try:
    from .case_manifest import verify_frozen_cases
except ImportError:
    from case_manifest import verify_frozen_cases

ROOT = Path(__file__).resolve().parents[2]
TV = ROOT / "tool_verification"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:4")
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be a new directory")
    args.out.mkdir(parents=True)
    cases = json.loads((TV / "cases/core40.json").read_text())["cases"]
    verify_frozen_cases(cases, ROOT / "original_data")
    reg = build_a_registry()
    manifest = {"cases": []}
    rows = []
    try:
        for index, case in enumerate(cases, 1):
            image = ROOT / "original_data" / case["image_path"]
            stem = image.stem
            case_out = args.out / stem
            session = EvidenceSession(image_ref=str(image))
            # A12 depends on the A2 heart mask; PSPNet runs on CPU here to
            # leave the model GPU available for the CXAS worker.
            heart = reg.run("segment_heart", session,
                            output_dir=str(case_out / "pspnet"), device="cpu")
            result = (reg.run("measure_ctr", session,
                              output_dir=str(case_out / "cxas"), device=args.device)
                      if not isinstance(heart, ToolRefusal) else heart)
            row = {"case": stem, "image": str(image), "status": "refusal" if isinstance(result, ToolRefusal) else "ok"}
            if isinstance(result, ToolRefusal):
                row["reason"] = result.reason
            else:
                value = next(u.value for u in result if u.finding_or_anatomy == "cardiothoracic_ratio")
                row["ours_ctr"] = float(value["CTR"])
                row["labels_model_frame"] = str(case_out / "cxas/labels_model_frame.npz")
                row["size_hw"] = list(reversed(case["size_wh"]))
                manifest["cases"].append({"case": stem, "labels_model_frame": row["labels_model_frame"],
                                          "size_hw": row["size_hw"]})
            rows.append(row)
            print(f"ctr {index}/{len(cases)} {stem}", flush=True)
    finally:
        backends.shutdown_workers()

    manifest_path = args.out / "oracle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    oracle_out = args.out / "cxas_official_ctr.json"
    source = next((ROOT / "tools/src/cxas").glob("ChestXRayAnatomySegmentation-*/cxas"))
    cmd = [str(ROOT / ".venv-oracle/bin/python"), str(TV / "oracles/cxas_ctr_official.py"),
           "--manifest", str(manifest_path), "--source", str(source), "--out", str(oracle_out)]
    run = subprocess.run(cmd, text=True, capture_output=True)
    (args.out / "oracle_console.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
    if run.returncode:
        raise SystemExit(f"CXAS oracle failed; see {args.out / 'oracle_console.txt'}")
    official = {r["case"]: r["score"] for r in json.loads(oracle_out.read_text())}
    compared = []
    for row in rows:
        if row.get("status") == "ok" and row["case"] in official:
            row["official_ctr"] = official[row["case"]]
            row["abs_difference"] = abs(row["ours_ctr"] - row["official_ctr"])
            compared.append(row)
    ours = np.array([r["ours_ctr"] for r in compared], dtype=float)
    ref = np.array([r["official_ctr"] for r in compared], dtype=float)
    rho, pval = spearmanr(ours, ref) if len(compared) >= 2 else (float("nan"), float("nan"))
    large = sorted(compared, key=lambda r: r["abs_difference"], reverse=True)[:10]
    summary = {
        "task": "a12-cxas-crosscheck",
        "cases": len(rows), "successful": len(compared),
        "spearman_rho": float(rho), "spearman_pvalue": float(pval),
        "ours_median": float(np.median(ours)) if len(ours) else None,
        "official_median": float(np.median(ref)) if len(ref) else None,
        "mae": float(np.mean(np.abs(ours - ref))) if len(ours) else None,
        "max_abs_difference": float(np.max(np.abs(ours - ref))) if len(ours) else None,
        "large_difference_cases": [{k: r[k] for k in ("case", "ours_ctr", "official_ctr", "abs_difference")}
                                   for r in large],
        "pass": bool(len(compared) == len(rows) and rho >= 0.8),
        "threshold": {"spearman_rho": 0.8},
        "note": "CXAS and A12 use different thoracic-boundary definitions; this is an implementation cross-check, not a clinical accuracy result",
    }
    (args.out / "a12_ctr_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (args.out / "a12_ctr_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
