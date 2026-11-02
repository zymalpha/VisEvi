#!/usr/bin/env python3
"""Create a deterministic, local-only MIMIC image candidate manifest.

This manifest uses Chest ImaGenome gold metadata already available locally.
It does not copy PHI/IDs to stdout, resolve paths, contact PhysioNet, or download
images. All selected paths remain candidates until checked against authorized
MIMIC-CXR-JPG v2.1.0 metadata and IMAGE_FILENAMES.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "original_data/chest-imagenome/1.0.0/gold_dataset"
ATTRIBUTE_FILE = GOLD / "gold_object_attribute_with_coordinates.txt"
SENTENCE_FILE = GOLD / "gold_all_sentences_500pts_1000studies.txt"
BBOX_FILE = GOLD / "gold_bbox_coordinate_annotations_1000images.csv"
SCHEMA_VERSION = "1.0"
SEED = 20261031
PATIENT_COUNT = 100

# Report-object attributes are used only to enrich the development candidate
# set. They are not independent image-reader labels.
TARGETS = {
    "cardiac": ("enlarged cardiac silhouette",),
    "consolidation": ("consolidation",),
    "effusion": ("pleural effusion",),
    "pneumothorax": ("pneumothorax",),
    "nodule_mass": ("mass/nodule (not otherwise specified)",
                    "multiple masses/nodules", "calcified nodule"),
    "hilar": ("enlarged hilum",),
    "mediastinal": ("mediastinal widening",),
    "device": ("endotracheal tube", "enteric tube", "ij line", "picc",
               "subclavian line", "chest tube"),
    "technical_quality": ("low lung volumes", "rotated"),
    "emphysema": ("copd/emphysema", "hyperaeration"),
}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_patient_key(patient_id: str) -> str:
    return hashlib.sha256(f"chest-imagenome:{patient_id}".encode()).hexdigest()


def build_manifest(output_dir: Path) -> dict:
    attrs = _read_tsv(ATTRIBUTE_FILE)
    sentences = _read_tsv(SENTENCE_FILE)
    boxes = _read_csv(BBOX_FILE)

    image_binding: dict[str, tuple[str, str, str]] = {}
    for row in sentences:
        current = (row["patient_id"], row["study_id"], row["StudyOrder"])
        prior = image_binding.setdefault(row["image_id"], current)
        if prior != current:
            raise ValueError("gold image ID maps to multiple patient/study records")

    bbox_images = {row["image_id"] for row in boxes}
    if len(bbox_images) != 1000:
        raise ValueError("unexpected gold bounding-box image count")
    if not bbox_images <= image_binding.keys():
        raise ValueError("some gold bbox images lack local patient/study metadata")

    patient_images: dict[str, dict[str, str]] = {}
    for image_id in bbox_images:
        patient_id, study_id, order = image_binding[image_id]
        previous = patient_images.setdefault(patient_id, {})
        if order in previous and previous[order] != image_id:
            raise ValueError("a gold patient has multiple bbox images in the same study order")
        previous[order] = image_id
    eligible = {pid: values for pid, values in patient_images.items()
                if set(values) == {"1", "2"}}
    if len(eligible) != 500:
        raise ValueError("gold metadata no longer describes 500 two-study patients")

    image_targets: dict[str, set[str]] = {image_id: set() for image_id in bbox_images}
    labels = {key: {name.lower() for name in names} for key, names in TARGETS.items()}
    image_patient = {image_id: image_binding[image_id][0] for image_id in bbox_images}
    for row in attrs:
        image_id = row.get("image_id", "")
        attribute = row.get("attribute", "").lower()
        if image_id not in image_targets or "|yes|" not in attribute:
            continue
        for target, names in labels.items():
            if any(name in attribute for name in names):
                image_targets[image_id].add(target)

    pool_by_patient: dict[str, dict[str, set[str]]] = {}
    for patient_id, images_by_order in eligible.items():
        image_ids = set(images_by_order.values())
        categories = {target for image_id in image_ids for target in image_targets[image_id]}
        pool_by_patient[patient_id] = {"images": image_ids, "categories": categories}

    remaining = set(eligible)
    selected: list[str] = []
    covered: set[str] = set()
    # Deterministic greedy max coverage gives MIMIC development samples a useful
    # spread over sparse positive labels, with seed-based tie breaking.
    rng = random.Random(SEED)
    tie_rank = {pid: rng.random() for pid in sorted(remaining)}
    while remaining and len(selected) < PATIENT_COUNT:
        best = max(remaining, key=lambda pid: (
            len(pool_by_patient[pid]["categories"] - covered),
            len(pool_by_patient[pid]["images"]),
            -tie_rank[pid],
        ))
        selected.append(best)
        covered.update(pool_by_patient[best]["categories"])
        remaining.remove(best)

    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    entries = []
    for patient_id in selected:
        orders = eligible[patient_id]
        for order in ("1", "2"):
            image_id = orders[order]
            study_id = image_binding[image_id][1]
            path_subject = f"p{patient_id[:2]}/p{patient_id}"
            jpg_path = f"files/{path_subject}/s{study_id}/{Path(image_id).stem}.jpg"
            entries.append({
                "patient_key_sha256": _safe_patient_key(patient_id),
                "study_order": int(order),
                "image_key_sha256": hashlib.sha256(image_id.encode()).hexdigest(),
                "candidate_relative_path": jpg_path,
                "path_status": "candidate_only_not_join_verified",
                "labels": sorted(image_targets[image_id]),
                "label_provenance": "chest_imagenome_gold_report_object_attributes",
                "split": "development",
                "seed": SEED,
                "limitations": [
                    "selected from report-annotation and bbox availability; not a random population sample",
                    "yes labels are report annotations and do not imply image blind-read truth",
                    "no for a finding may not mean confirmed visual absence",
                    "no MIMIC-CXR-JPG path/metadata/IMAGE_FILENAMES join has been verified",
                    "dataset/model training overlap has not been resolved",
                ],
            })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Chest ImaGenome 1.0.0 local gold metadata",
        "source_files": {
            str(path.relative_to(ROOT)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (ATTRIBUTE_FILE, SENTENCE_FILE, BBOX_FILE)
        },
        "sampling": {
            "method": "deterministic greedy positive-label max coverage by patient",
            "seed": SEED,
            "patient_count": len(selected),
            "candidate_image_count": len(entries),
            "one_image_per_gold_study_order": True,
            "study_orders": [1, 2],
            "development_only": True,
            "covered_report_annotation_categories": sorted(covered),
        },
        "image_path": {
            "root_relative_to_dataset": True,
            "status": "candidate path reconstructed from gold MIMIC IDs; verify against authorized v2.1.0 split metadata and IMAGE_FILENAMES before downloading",
            "expected_suffix": ".jpg",
        },
        "access": {
            "requires_mimic_cxr_jpg_access": True,
            "credential_or_dua_validation_done": False,
            "image_bytes_downloaded": False,
        },
        "files": entries,
    }
    path = output_dir / "manifest.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"manifest_path": str(path), "patient_count": len(selected),
            "image_count": len(entries), "bytes": path.stat().st_size,
            "sha256": _sha256(path), "covered_categories": sorted(covered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory under a local, access-controlled manifest root")
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else ROOT / args.out
    if output.exists():
        raise SystemExit("output directory already exists; refusing to overwrite")
    print(json.dumps(build_manifest(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
