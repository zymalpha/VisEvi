"""V2 property tests: translation, left-right flip, integer upsampling, third-party GLCM, A25 rib rule.

Run: .venv/bin/python tool_verification/suites/v2_properties.py --out tool_verification/runs/<date>/v2_properties
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tool_verification/suites'))
from v2_deterministic import Recorder, close, lung_pair, random_mask   # noqa: E402

from visevi.core.session import EvidenceSession   # noqa: E402
from visevi.core.units import ToolRefusal        # noqa: E402
from visevi.eval.runner import build_a_registry   # noqa: E402
from visevi.perception import structures         # noqa: E402

REG = build_a_registry()


def session_for(tmp, gray, masks, tag):
    path = Path(tmp) / f'{tag}.png'
    Image.fromarray(gray).save(path)
    s = EvidenceSession(image_ref=str(path))
    s.attach_masks(masks)
    return s


def value(s, name, **params):
    out = REG.run(name, s, **params)
    return out if isinstance(out, ToolRefusal) else out[0].value


def geometry_values(s):
    e = value(s, 'measure_region_extent', mask_name='A')
    sh = value(s, 'describe_region_shape', mask_name='A')['largest_component']
    d = value(s, 'measure_region_distance', mask_a='A', mask_b='B')
    return {'area': value(s, 'measure_region_area', mask_name='A')['area_px'], 'bw': e['bbox_width_px'],
            'bh': e['bbox_height_px'], 'feret': e['feret_max_px'], 'perim': sh['perimeter_px'],
            'circ': sh['circularity'], 'solid': sh['solidity'], 'aspect': sh['rotated_aspect_ratio'],
            'gap': d['gap_distance_px'], 'center': d['pixel_center_distance_px']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n', type=int, default=300)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be new')
    args.out.mkdir(parents=True, exist_ok=True)
    rng, rec = random.Random(7), Recorder()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.n):
            h, w = rng.randint(8, 18), rng.randint(8, 20)
            gray = np.array([[rng.randrange(256) for _ in range(w)] for _ in range(h)], np.uint8)
            a, b = random_mask(rng, h, w), random_mask(rng, h, w)
            right, left = lung_pair(rng, h, w)
            base = session_for(tmp, gray, {'A': a, 'B': b, 'Right_Lung': right, 'Left_Lung': left}, f'b{i}')
            g0 = geometry_values(base)
            # translation (pad top/left)
            dy, dx = rng.randint(1, 5), rng.randint(1, 5)
            pad = lambda m: np.pad(m, ((dy, 0), (dx, 0)))
            moved = session_for(tmp, pad(gray), {'A': pad(a), 'B': pad(b)}, f't{i}')
            g1 = geometry_values(moved)
            rec.check('translation invariance (A9/A10/A11/A14)', i, all(close(g0[k], g1[k]) for k in g0), {'base': g0, 'moved': g1})
            i0 = value(base, 'summarize_roi_intensity', mask_name='A'); i1 = value(moved, 'summarize_roi_intensity', mask_name='A')
            rec.check('translation invariance (A15)', i, all(close(i0[k], i1[k]) for k in ('mean', 'variance_population', 'p50')))
            # integer upsampling
            k = rng.choice([2, 3])
            up = lambda m: np.kron(m, np.ones((k, k), m.dtype))
            big = session_for(tmp, up(gray), {'A': up(a), 'B': up(b)}, f'u{i}')
            g2 = geometry_values(big)
            ok = (g2['area'] == g0['area'] * k * k and g2['bw'] == g0['bw'] * k and g2['bh'] == g0['bh'] * k
                  and close(g2['feret'], g0['feret'] * k) and close(g2['perim'], g0['perim'] * k)
                  and close(g2['circ'], g0['circ']) and close(g2['solid'], g0['solid']) and close(g2['aspect'], g0['aspect'], 1e-7)
                  and close(g2['gap'], g0['gap'] * k))
            rec.check('k-fold upsampling scales area k^2, lengths k', i, ok, {'k': k, 'base': g0, 'up': g2})
            # left-right flip: swap roles of lungs
            fl = lambda m: m[:, ::-1]
            flipped = session_for(tmp, fl(gray), {'Right_Lung': fl(left), 'Left_Lung': fl(right), 'C': fl(a)}, f'f{i}')
            base.attach_masks({'C': a})
            c0, c1 = value(base, 'compare_lung_geometry'), value(flipped, 'compare_lung_geometry')
            rec.check('flip swaps A13 right/left', i, (c0['right_area_px'], c0['left_area_px'], c0['right_max_width_px'])
                      == (c1['left_area_px'], c1['right_area_px'], c1['left_max_width_px']))
            s0, s1 = value(base, 'measure_lung_symmetry'), value(flipped, 'measure_lung_symmetry')
            rec.check('flip preserves A19 mirror Dice', i, close(s0['mirror_dice'], s1['mirror_dice']))
            d0 = value(base, 'summarize_bilateral_distribution', lesion_mask_names=['C'])
            d1 = value(flipped, 'summarize_bilateral_distribution', lesion_mask_names=['C'])
            swap = {'right_only': 'left_only', 'left_only': 'right_only'}
            rec.check('flip swaps A23 sides', i, swap.get(d0['pattern'], d0['pattern']) == d1['pattern']
                      and d0['union_right_overlap_px'] == d1['union_left_overlap_px'])
            # third-party GLCM on a full-image ROI (skimage)
            full = np.ones((h, w), bool)
            base.attach_masks({'FULL': full})
            t = value(base, 'extract_roi_texture', mask_name='FULL')
            q = (gray // 16).astype(np.uint8)
            glcm = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16, symmetric=True, normed=True)
            props = {name: float(np.mean(graycoprops(glcm, name))) for name in
                     ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation')}
            props['angular_second_moment'] = props.pop('ASM')
            ok = all(close(t['features'][kk], props[kk], 1e-9) for kk in props if kk != 'correlation') and \
                (math.isnan(props['correlation']) or close(t['features']['correlation'], props['correlation'], 1e-9))
            rec.check('A16 equals skimage graycoprops (full ROI)', i, ok, {'tool': t['features'], 'skimage': props})
            # A25 rib rule: ribs whose centre in the dome column window lies above the dome
            masks = {}
            for n in range(1, 13):
                m = np.zeros((60, 30), bool)
                y0 = rng.randint(0, 55)
                m[y0:y0 + rng.randint(1, 4), rng.randint(0, 20):rng.randint(20, 30)] = True
                if rng.random() < 0.8:
                    masks[structures.rib_label('posterior', n, 'right')] = m
            x, y_lim, half = rng.uniform(5, 25), rng.randint(5, 55), 2
            r = structures.ribs_above(masks, 'posterior', 'right', x, y_lim, half)
            x0, x1 = max(0, int(round(x)) - half), int(round(x)) + half + 1
            exp = [n for n in range(1, 13) if structures.rib_label('posterior', n, 'right') in masks
                   and np.nonzero(masks[structures.rib_label('posterior', n, 'right')][:, x0:x1])[0].size
                   and np.nonzero(masks[structures.rib_label('posterior', n, 'right')][:, x0:x1])[0].mean() < y_lim]
            rec.check('A25 ribs-above-dome rule', i, r['ribs_above'] == exp and r['highest_number_above'] == (max(exp) if exp else None))
    summary = {'n': args.n, 'tools': rec.results, 'all_passed': all(v['failures'] == 0 for v in rec.results.values())}
    (args.out / 'v2_properties_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    for kk, v in rec.results.items():
        print(f'{kk:48s} {v["cases"] - v["failures"]}/{v["cases"]}')


if __name__ == '__main__':
    main()
