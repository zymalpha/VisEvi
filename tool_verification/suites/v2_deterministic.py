"""V2: deterministic A tools vs independent oracle on seeded random inputs.

Run: .venv/bin/python tool_verification/suites/v2_deterministic.py --out tool_verification/runs/<date>/v2 [--n 1000]
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tool_verification/oracles'))
import geometry_ref as ref                      # noqa: E402

from visevi.core.session import EvidenceSession  # noqa: E402
from visevi.core.units import ToolRefusal       # noqa: E402
from visevi.eval.runner import build_a_registry  # noqa: E402
from visevi.perception import structures        # noqa: E402

REL = 1e-9


def close(a, b, rel=REL):
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-9)


# ---------------------------------------------------------------- generators
def random_mask(rng, h, w, max_shapes=4):
    m = np.zeros((h, w), bool)
    while not m.any():
        for _ in range(rng.randint(1, max_shapes)):
            kind = rng.choice(['rect', 'ellipse', 'line', 'pixel', 'rect'])
            if kind == 'rect':
                x0, y0 = rng.randrange(w), rng.randrange(h)
                m[y0:y0 + rng.randint(1, h // 2), x0:x0 + rng.randint(1, w // 2)] = True
            elif kind == 'ellipse':
                cy, cx = rng.uniform(0, h), rng.uniform(0, w)
                ry, rx = rng.uniform(1, h / 3), rng.uniform(1, w / 3)
                yy, xx = np.mgrid[:h, :w]
                m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            elif kind == 'line':
                if rng.random() < .5:
                    m[rng.randrange(h), rng.randrange(w):rng.randrange(w) + 1] = True
                else:
                    m[rng.randrange(h):rng.randrange(h) + 1, rng.randrange(w)] = True
            else:
                m[rng.randrange(h), rng.randrange(w)] = True
        if rng.random() < .3 and m.sum() > 4:                     # punch a hole
            y0, x0 = rng.randrange(h), rng.randrange(w)
            m[y0:y0 + rng.randint(1, 3), x0:x0 + rng.randint(1, 3)] = False
    return m


def lung_pair(rng, h, w):
    mid = w // 2
    right = np.zeros((h, w), bool); left = np.zeros((h, w), bool)
    while not right.any():
        right[:, :mid - 1] = random_mask(rng, h, mid - 1, 3)
    while not left.any():
        left[:, mid + 1:] = random_mask(rng, h, w - mid - 1, 3)
    return right, left


def spacing(rng):
    return [rng.choice([0.5, 1.0, 0.143, 2.25]), rng.choice([0.5, 1.0, 0.175, 3.0])]


class Case:
    def __init__(self, tmp, rng, h, w, deep=False):
        self.gray = (np.array([[rng.randrange(65536 if deep else 256) for _ in range(w)] for _ in range(h)],
                              dtype=np.uint16 if deep else np.uint8))
        self.path = Path(tmp) / f'img_{rng.randrange(10**9)}.png'
        Image.fromarray(self.gray).save(self.path)
        self.session = EvidenceSession(image_ref=str(self.path))


# ---------------------------------------------------------------- checks
class Recorder:
    def __init__(self):
        self.results = {}

    def check(self, tool, case_id, ok, detail=None):
        r = self.results.setdefault(tool, {'cases': 0, 'failures': 0, 'first_failures': []})
        r['cases'] += 1
        if not ok:
            r['failures'] += 1
            if len(r['first_failures']) < 5:
                r['first_failures'].append({'case': case_id, 'detail': detail})


def run(registry, session, name, **params):
    out = registry.run(name, session, **params)
    return out if isinstance(out, ToolRefusal) else out[0].value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be new')
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    reg = build_a_registry()
    rec = Recorder()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.n):
            h, w = rng.randint(8, 22), rng.randint(8, 26)
            case = Case(tmp, rng, h, w, deep=(i % 5 == 0))
            s = case.session
            a, b = random_mask(rng, h, w), random_mask(rng, h, w)
            right, left = lung_pair(rng, h, w)
            sp = spacing(rng)
            s.attach_masks({'A': a, 'B': b, 'Right_Lung': right, 'Left_Lung': left, 'C': random_mask(rng, h, w)})

            # A9
            v = run(reg, s, 'measure_region_extent', mask_name='A', spacing_row_col=sp)
            bb = ref.bbox(a); mw = ref.max_row_width(a)
            ok = (v['bbox_width_px'], v['bbox_height_px'], v['max_row_width_px'], v['max_row_width_row_y']) == (bb['w'], bb['h'], mw[0], mw[1]) \
                and close(v['feret_max_px'], ref.feret(a)) and close(v['feret_max_physical'], ref.feret(a, sp[0], sp[1])) \
                and close(v['bbox_width_physical'], bb['w'] * sp[1]) and close(v['bbox_height_physical'], bb['h'] * sp[0])
            rec.check('A9 measure_region_extent', i, ok, {'tool': {k: v[k] for k in ('bbox_width_px', 'feret_max_px', 'max_row_width_px')}, 'ref_feret': ref.feret(a)})
            # A10
            v = run(reg, s, 'measure_region_area', mask_name='A', spacing_row_col=sp)
            rec.check('A10 measure_region_area', i, v['area_px'] == int(a.sum()) and close(v['area_physical'], a.sum() * sp[0] * sp[1]))
            # A11
            v = run(reg, s, 'measure_region_distance', mask_a='A', mask_b='B', spacing_row_col=sp)
            g, c = ref.gap_distance(a, b), (0.0 if (a & b).any() else ref.center_distance(a, b))
            gp = ref.gap_distance(a, b, sp[0], sp[1]); cp = 0.0 if (a & b).any() else ref.center_distance(a, b, sp[0], sp[1])
            ok = close(v['gap_distance_px'], g) and close(v['pixel_center_distance_px'], c) and close(v['gap_distance_physical'], gp) and close(v['pixel_center_distance_physical'], cp)
            rec.check('A11 measure_region_distance', i, ok, {'tool': [v['gap_distance_px'], v['pixel_center_distance_px'], v['gap_distance_physical']], 'ref': [g, c, gp]})
            # A12 arithmetic
            hb = sorted(rng.uniform(0, 100) for _ in range(2)); tb = sorted(rng.uniform(0, 100) for _ in range(2))
            v = run(reg, s, 'compute_ctr_from_borders', heart_borders_x=hb, thoracic_inner_borders_x=tb)
            inside = tb[0] <= hb[0] and hb[1] <= tb[1] and hb[1] > hb[0] and tb[1] > tb[0]
            ok = (isinstance(v, ToolRefusal) and not inside) or (not isinstance(v, ToolRefusal) and inside and close(v['CTR'], (hb[1] - hb[0]) / (tb[1] - tb[0])))
            rec.check('A12 compute_ctr_from_borders', i, ok, {'hb': hb, 'tb': tb})
            # A13
            v = run(reg, s, 'compare_lung_geometry')
            rw, lw = ref.max_row_width(right)[0], ref.max_row_width(left)[0]
            ok = (v['right_area_px'], v['left_area_px'], v['right_max_width_px'], v['left_max_width_px']) == (right.sum(), left.sum(), rw, lw) \
                and close(v['area_ratio_R_over_L'], right.sum() / left.sum()) and close(v['width_ratio_R_over_L'], rw / lw)
            rec.check('A13 compare_lung_geometry', i, ok)
            # A14
            v = run(reg, s, 'describe_region_shape', mask_name='A')
            exp = ref.shape(a)
            ok = v['component_count'] == len(exp) and all(
                t['area_px'] == e['area_px'] and close(t['perimeter_px'], e['perimeter_px']) and close(t['circularity'], e['circularity'])
                and close(t['convex_hull_area_px'], e['convex_hull_area_px']) and close(t['solidity'], e['solidity'])
                and close(t['minimum_rotated_rectangle_long_side_px'], e['long'], 1e-7) and close(t['minimum_rotated_rectangle_short_side_px'], e['short'], 1e-7)
                for t, e in zip(v['components'], exp))
            rec.check('A14 describe_region_shape', i, ok, {'tool': [{k: t[k] for k in ('area_px', 'perimeter_px', 'convex_hull_area_px', 'minimum_rotated_rectangle_long_side_px', 'minimum_rotated_rectangle_short_side_px')} for t in v['components']], 'ref': exp})
            # A15
            v = run(reg, s, 'summarize_roi_intensity', mask_name='A')
            e = ref.intensity(case.gray.astype(float), a)
            ok = all(close(v[k], e[k], 1e-9) for k in e)
            rec.check('A15 summarize_roi_intensity', i, ok, {'tool': {k: v[k] for k in e}, 'ref': e})
            # A16
            v = run(reg, s, 'extract_roi_texture', mask_name='A')
            mean, per = ref.glcm_features(case.gray.astype(float), a)
            if isinstance(v, ToolRefusal):
                ok = mean is None                       # refusal only when no neighbouring pair exists
            else:
                ok = mean is not None and all(close(v['features'][k], mean[k], 1e-9) for k in mean) and all(
                    (per[d] is None and v['directional_features'][d]['features'] is None) or
                    (per[d] is not None and v['directional_features'][d]['valid_pair_count'] == per[d]['pairs'])
                    for d in per)
            rec.check('A16 extract_roi_texture', i, ok)
            # A17
            scale = rng.choice([1.0, 255.0, 4.0])
            v = run(reg, s, 'measure_roi_sharpness', mask_name='A', intensity_scale=scale)
            t, mg = ref.tenengrad(case.gray.astype(float), a, scale)
            rec.check('A17 measure_roi_sharpness', i, close(v['tenengrad_mean'], t, 1e-9) and close(v['gradient_magnitude_mean'], mg, 1e-9), {'tool': v['tenengrad_mean'], 'ref': t})
            # A19
            mid = rng.randrange(2 * w) / 2
            v = run(reg, s, 'measure_lung_symmetry', midline_x=mid)
            ub = ref.bbox(right | left)
            v2 = run(reg, s, 'measure_lung_symmetry')
            ok = close(v['mirror_dice'], ref.mirror_dice(right, left, mid)) and close(v2['midline_x'], (ub['x0'] + ub['x1']) / 2) \
                and close(v2['mirror_dice'], ref.mirror_dice(right, left, (ub['x0'] + ub['x1']) / 2))
            rec.check('A19 measure_lung_symmetry', i, ok)
            # A20
            v = run(reg, s, 'assign_lung_zones', region_mask_name='C')
            cm = s.masks['C']
            ok = list(v['right']['zone_overlap_px'].values()) == ref.zones(right, cm) and list(v['left']['zone_overlap_px'].values()) == ref.zones(left, cm) \
                and v['outside_lung_overlap_px'] == int((cm & ~(right | left)).sum())
            rec.check('A20 assign_lung_zones', i, ok)
            # A21
            v = run(reg, s, 'relate_lesion_to_anatomy', lesion_mask_name='A', anatomy_mask_name='B')
            e = ref.relate(a, b); d = ref.lesion_direction(a, b)
            ok = v['topology'] == e['topology'] and all(close(v[k], e[k]) if isinstance(e[k], float) else v[k] == e[k] for k in e) \
                and (v['relative_image_horizontal'], v['relative_image_vertical']) == d
            rec.check('A21 relate_lesion_to_anatomy', i, ok, {'tool': {k: v[k] for k in e}, 'ref': e})
            # A22
            v = run(reg, s, 'relate_multiple_regions', region_mask_names=['A', 'B', 'C'])
            pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
            ok = all(all(close(r[k], e[k]) if isinstance(e[k], float) else r[k] == e[k] for k, e in [(k, ref.relate(s.masks[p], s.masks[q])) for k in ref.relate(s.masks[p], s.masks[q])])
                     for r, (p, q) in zip(v['relations'], pairs))
            rec.check('A22 relate_multiple_regions', i, ok)
            # A23
            v = run(reg, s, 'summarize_bilateral_distribution', lesion_mask_names=['A', 'C'])
            pattern, comps = ref.distribution(right, left, [a, s.masks['C']])
            ok = v['pattern'] == pattern and [(c['region_index'], c['area_px'], c['right_lung_overlap_px'], c['left_lung_overlap_px'], c['laterality']) for c in v['components']] == comps
            rec.check('A23 summarize_bilateral_distribution', i, ok, {'tool': v['pattern'], 'ref': pattern})
            # A12 auto rule: thoracic inner margin (synthetic rib masks)
            rr, rl = random_mask(rng, h, w), random_mask(rng, h, w)
            midl = w / 2 + rng.choice([0, 0.5])
            rows = ref.thoracic_inner_margin(rr, rl, midl, h - 1)
            try:
                t = structures.thoracic_inner_margin(rr, rl, midl, h - 1, min_rows=1)
                best = max(rows, key=lambda r: r[2] - r[1])
                ok = (t['row_y'], t['right_inner_x'], t['left_inner_x'], t['candidate_rows']) == (best[0], best[1], best[2], len(rows))
            except ValueError:
                ok = len(rows) == 0
            rec.check('A12 thoracic_inner_margin rule', i, ok)
            # A25 rotation rule
            cr, cl = random_mask(rng, h, w), random_mask(rng, h, w)
            sp_mask = np.zeros((h, w), bool); sp_mask[:, w // 2 - 1:w // 2 + 1] = True
            expected = ref.rotation_index(cr, cl, sp_mask)
            try:
                t = structures.rotation(cr, cl, sp_mask)
                ok = expected is not None and close(t['distance_right_px'], expected[0]) \
                    and close(t['distance_left_px'], expected[1]) and close(t['asymmetry_index'], expected[2])
            except ValueError:
                ok = expected is None
            rec.check('A25 rotation rule', i, ok)
            if i % 100 == 0:
                print(i, flush=True)
    summary = {'n': args.n, 'seed': args.seed, 'tolerance_rel': REL,
               'tools': rec.results, 'all_passed': all(r['failures'] == 0 for r in rec.results.values())}
    (args.out / 'v2_summary.json').write_text(json.dumps(summary, indent=1, default=str))
    for k, r in rec.results.items():
        print(f'{k:40s} {r["cases"] - r["failures"]}/{r["cases"]}')


if __name__ == '__main__':
    main()
