#!/usr/bin/env python3
"""Run aggregate-only Q/P workflow smoke tests on the frozen MIMIC manifest.

The output contains no image, patient, study, or per-case identifiers. This is
workflow validation only; it does not measure medical accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "original_data/_manifests/20261031-mimic-integration-development-v1/manifest.json"
FUNCTION_IDS = tuple([f"Q{i}" for i in range(1, 5)] + [f"P{i}" for i in range(1, 13)])


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(data)


def run_shard(manifest_path: Path, image_root: Path, output: Path,
              shard_index: int, shard_count: int, device: str,
              limit: int | None = None) -> dict:
    from visevi.eval.runner import build_a_registry
    from visevi.functions.runner import run_function_episode

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest has no image entries")
    if limit is not None:
        rows = rows[:limit]
    selected = [(index, row) for index, row in enumerate(rows)
                if index % shard_count == shard_index]
    counts = {function_id: Counter() for function_id in FUNCTION_IDS}
    reasons = {function_id: Counter() for function_id in FUNCTION_IDS}
    optional = {function_id: Counter() for function_id in FUNCTION_IDS}
    projections = Counter()
    exceptions = Counter()
    with tempfile.TemporaryDirectory(prefix=f"visevi-pq-smoke-{shard_index}-") as temp:
        temp_root = Path(temp)
        registry = build_a_registry()
        for ordinal, (index, row) in enumerate(selected):
            projection = row.get("view_position")
            projections[str(projection or "unknown")] += 1
            image = image_root / row["relative_path"]
            if not image.is_file():
                exceptions["missing_image"] += 1
                for function_id in FUNCTION_IDS:
                    counts[function_id]["input_missing"] += 1
                continue
            inputs = {"finding": "nodule"}
            try:
                result = run_function_episode(
                    FUNCTION_IDS, image_ref=str(image), registry=registry,
                    inputs=inputs,
                    out_dir=temp_root / f"case_{ordinal:05d}",
                    device=device,
                )
            except Exception as exc:  # keep only an error class, never a case payload
                exceptions[type(exc).__name__] += 1
                for function_id in FUNCTION_IDS:
                    counts[function_id]["runner_exception"] += 1
                continue
            by_id = {run.function_id: run for run in result.functions}
            for function_id in FUNCTION_IDS:
                run = by_id.get(function_id)
                if run is None:
                    counts[function_id]["missing_result"] += 1
                    continue
                counts[function_id][run.status] += 1
                if run.refusal is not None:
                    reasons[function_id][run.refusal.reason] += 1
                optional[function_id].update(run.missing_optional)
    payload = {
        "purpose": "development_workflow_smoke_only_not_medical_accuracy",
        "manifest_sha256": _digest(manifest_path),
        "manifest_images": len(manifest["files"]),
        "selected_images": len(selected),
        "shard": {"index": shard_index, "count": shard_count},
        "device": device,
        "projection_counts": dict(sorted(projections.items())),
        "functions": {
            function_id: {
                "status_counts": dict(sorted(counts[function_id].items())),
                "refusal_reason_counts": dict(sorted(reasons[function_id].items())),
                "missing_optional_counts": dict(sorted(optional[function_id].items())),
            }
            for function_id in FUNCTION_IDS
        },
        "runner_exception_counts": dict(sorted(exceptions.items())),
        "limitations": [
            "all results are workflow execution states, not medical accuracy",
            "P8 and P10 require source-bound hilar/aortic ROI masks absent from this manifest",
            "P3 uses the nodule query only; it does not score a nodule reference",
            "candidate-dependent functions may refuse when no candidate ROI is produced",
            "the manifest includes development samples and is not a locked test set",
        ],
    }
    _write_new(output, payload)
    return payload


def merge_shards(shards: list[Path], output: Path) -> dict:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    first = payloads[0]
    if any((item["manifest_sha256"], item["manifest_images"], item["shard"]["count"])
           != (first["manifest_sha256"], first["manifest_images"], len(payloads))
           for item in payloads):
        raise ValueError("shards do not belong to the same complete manifest partition")
    if {item["shard"]["index"] for item in payloads} != set(range(len(payloads))):
        raise ValueError("shard set is incomplete or duplicated")
    merged = {
        "purpose": first["purpose"], "manifest_sha256": first["manifest_sha256"],
        "manifest_images": first["manifest_images"],
        "selected_images": sum(item["selected_images"] for item in payloads),
        "projection_counts": {}, "functions": {}, "runner_exception_counts": {},
        "limitations": first["limitations"],
        "shards": [{"index": item["shard"]["index"], "device": item["device"]}
                   for item in sorted(payloads, key=lambda row: row["shard"]["index"])],
    }
    projection_totals = Counter()
    exception_totals = Counter()
    for item in payloads:
        projection_totals.update(item["projection_counts"])
        exception_totals.update(item["runner_exception_counts"])
    merged["projection_counts"] = dict(sorted(projection_totals.items()))
    merged["runner_exception_counts"] = dict(sorted(exception_totals.items()))
    for function_id in FUNCTION_IDS:
        statuses, refusals, missing_optional = Counter(), Counter(), Counter()
        for item in payloads:
            details = item["functions"][function_id]
            statuses.update(details["status_counts"])
            refusals.update(details["refusal_reason_counts"])
            missing_optional.update(details["missing_optional_counts"])
        merged["functions"][function_id] = {
            "status_counts": dict(sorted(statuses.items())),
            "refusal_reason_counts": dict(sorted(refusals.items())),
            "missing_optional_counts": dict(sorted(missing_optional.items())),
        }
    _write_new(output, merged)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path,
                        default=ROOT / "original_data/mimic-cxr-jpg/2.1.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge", nargs="+", type=Path)
    args = parser.parse_args()
    if args.merge:
        result = merge_shards(args.merge, args.output)
    else:
        if args.shard_count < 1 or not 0 <= (args.shard_index or 0) < args.shard_count:
            parser.error("shard index must be in [0, shard-count)")
        index = args.shard_index or 0
        result = run_shard(args.manifest, args.image_root, args.output,
                           index, args.shard_count, args.device, args.limit)
    print(json.dumps({"selected_images": result["selected_images"],
                      "output": str(args.output), "purpose": result["purpose"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
