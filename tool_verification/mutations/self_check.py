"""Small mutation self-checks for the implementation-verification harness.

These variants are local functions/monkeypatches only; src/visevi is never
modified.  Each injected defect must disagree with an independent expected
value or frozen parser contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visevi.perception import geometry, vlm


def _rect(x0, y0, x1, y1, shape=(20, 20)):
    m = np.zeros(shape, bool); m[y0:y1 + 1, x0:x1 + 1] = True; return m


def main(out: Path):
    rows = []

    # A9: old pixel-center Feret must disagree with the frozen pixel-cell
    # corner convention (a 1x10 bar is sqrt(101), not 9).
    bar = np.zeros((2, 10), bool); bar[0, :] = True
    expected = geometry.feret_max_diameter(bar)
    old_center = 9.0
    rows.append({'mutation': 'A9 Feret reverted to pixel centers',
                 'detected': not np.isclose(old_center, expected),
                 'expected': expected, 'mutated': old_center})

    # A13: swapping left/right channels changes the signed comparison.
    right, left = _rect(1, 1, 4, 8), _rect(10, 1, 16, 8)
    expected = geometry.lung_area_compare(right, left)['area_difference_right_minus_left_px']
    mutated = geometry.lung_area_compare(left, right)['area_difference_right_minus_left_px']
    rows.append({'mutation': 'A13 right/left channel swap', 'detected': expected != mutated,
                 'expected': expected, 'mutated': mutated})

    # A20: moving a zone boundary by one row changes a boundary-touching ROI.
    right_lung = _rect(2, 2, 8, 11, shape=(18, 18)); left_lung = _rect(10, 2, 16, 11, shape=(18, 18))
    region = _rect(3, 4, 7, 4, shape=(18, 18))
    expected = geometry.assign_lung_zones(right_lung, left_lung, region)
    shifted = np.zeros_like(region); shifted[5] = region[4]
    mutated = geometry.assign_lung_zones(right_lung, left_lung, shifted)
    rows.append({'mutation': 'A20 zone boundary shifted by one row',
                 'detected': expected != mutated, 'expected_zone': expected,
                 'mutated_zone': mutated})

    # CheXOne: using a 0-1000 grid instead of the frozen 0-100 grid changes
    # original-pixel coordinates by an order of magnitude.
    expected = vlm.grid_to_pixels((10, 20, 50, 80), 1000, 1000)
    mutated = [v / 10 for v in expected]
    rows.append({'mutation': 'A6/A7 box grid changed from 100 to 1000',
                 'detected': expected != mutated, 'expected': expected, 'mutated': mutated})

    # A8: changing the threshold from 0.5 to 0.4 changes a known borderline
    # probability map and must be caught by the mask identity check.
    prob = np.array([[0.39, 0.41], [0.50, 0.51]], dtype=np.float32)
    expected = prob > 0.5; mutated = prob > 0.4
    rows.append({'mutation': 'A8 mask threshold changed from 0.5 to 0.4',
                 'detected': not np.array_equal(expected, mutated),
                 'expected_mask': expected.tolist(), 'mutated_mask': mutated.tolist()})

    # Parser contract: invalid reversed boxes must remain unparsed, not be
    # repaired by sorting coordinates.
    expected = vlm.parse_boxes('<|box|>(60,10),(20,30)<|/box|>')['status']
    mutated = 'detected'
    rows.append({'mutation': 'A6 parser repairs reversed box coordinates',
                 'detected': expected != mutated, 'expected': expected, 'mutated': mutated})

    summary = {'mutations': len(rows), 'detected': sum(bool(r['detected']) for r in rows),
               'all_detected': all(bool(r['detected']) for r in rows), 'rows': rows,
               'note': 'local mutation probes; no project source files were modified'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not summary['all_detected']:
        raise SystemExit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); main(args.out)
