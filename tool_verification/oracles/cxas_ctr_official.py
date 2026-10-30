"""Run the CXAS repository's original CTR extractor on saved CXAS masks.

This oracle intentionally does not import visevi.  The segmentation masks are
read from the worker's packed 512-frame artifact and mapped back with the
same nearest-index convention used by the author's resize-to-file-size path;
only the CTR arithmetic is delegated to the original function.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    # ``--source`` is the repository's ``cxas`` package directory.
    sys.path.insert(0, str(args.source.parent))
    if (args.source / "__init__.py").exists():
        # The package imports optional DICOM dependencies at top level.  Load
        # only the extraction module and its tiny helpers without importing
        # the package initializer.
        import types
        pkg = types.ModuleType("cxas")
        pkg.__path__ = [str(args.source)]
        sys.modules.setdefault("cxas", pkg)
    from cxas.extraction import cardiothoracic_ratio as ctr_module
    get_cardiothoracic_ratio = ctr_module.get_cardiothoracic_ratio
    # The original function indexes three labels by name.  Restrict the
    # temporary array to those labels so a 4892x4020 case does not allocate a
    # 159-channel multi-gigabyte array.
    ctr_module.label_mapper = {"lung": [0], "right hemidiaphragm": [1], "heart": [2]}

    rows = []
    for item in json.loads(args.manifest.read_text())["cases"]:
        data = np.load(item["labels_model_frame"])
        shape = tuple(int(v) for v in data["shape"])
        masks = np.unpackbits(data["packed"], axis=-1)[..., :shape[2]].astype(bool)
        height, width = item["size_hw"]
        rows_i = np.minimum(np.floor(np.arange(height) * shape[1] / height).astype(int), shape[1] - 1)
        cols_i = np.minimum(np.floor(np.arange(width) * shape[2] / width).astype(int), shape[2] - 1)
        # PAX-Ray++ ids: lung=134, right hemidiaphragm=107, heart=121.
        full = masks[[134, 107, 121]][:, rows_i][:, :, cols_i]
        value = get_cardiothoracic_ratio(full)
        rows.append({"case": item["case"], "score": float(value["score"]),
                     "labels_model_frame": item["labels_model_frame"]})
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "out": str(args.out)}, indent=1))


if __name__ == "__main__":
    main()
