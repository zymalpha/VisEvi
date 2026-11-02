"""V4 metamorphic tests: outputs must follow known input transforms.

Transforms on 10 square 1024 images (5 ChestX-Det, 5 NIH from core40):
  pad   : black bands top and bottom (P = 128 each) -> portrait 1024 x 1280
  scale : 2x pixel upscaling (bilinear) -> 2048 x 2048
  flip  : left-right mirror
Run: .venv/bin/python tool_verification/suites/v4_metamorphic.py --out tool_verification/runs/<date>/v4 --device cuda:4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from visevi import backends
from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import build_a_registry
try:
    from .case_manifest import verify_frozen_cases
except ImportError:
    from case_manifest import verify_frozen_cases

ROOT = Path(__file__).resolve().parents[2]
TV = ROOT / 'tool_verification'
PAD = 128


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def transform(img: Image.Image, kind):
    g = img.convert('L')
    if kind == 'pad':
        out = Image.new('L', (g.width, g.height + 2 * PAD), 0); out.paste(g, (0, PAD)); return out
    if kind == 'scale':
        return g.resize((g.width * 2, g.height * 2), Image.Resampling.BILINEAR)
    if kind == 'flip':
        return g.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return g


def map_point(xy, kind, w):
    x, y = xy
    return {'pad': (x, y + PAD), 'scale': (2 * x + 0.5, 2 * y + 0.5), 'flip': (w - 1 - x, y)}[kind]


def map_mask(m, kind):
    if kind == 'pad':
        return np.pad(m, ((PAD, PAD), (0, 0)))
    if kind == 'scale':
        return np.kron(m, np.ones((2, 2), bool))
    return m[:, ::-1]


def run_all(reg, image_path, out, device):
    s = EvidenceSession(image_ref=str(image_path))
    res = {}
    for key, tool, params in [('psp', 'segment_heart', {'output_dir': str(out / 'psp'), 'device': device}),
                              ('psp_lungs', 'segment_lungs', {'output_dir': str(out / 'psp'), 'device': device}),
                              ('a5', 'locate_lung_landmarks', {'output_dir': str(out / 'a5'), 'device': device}),
                              ('a8', 'segment_pneumothorax', {'output_dir': str(out / 'a8'), 'device': device}),
                              ('a26', 'localize_devices', {'output_dir': str(out / 'a26'), 'device': device}),
                              ('a12', 'measure_ctr', {'output_dir': str(out / 'cxas'), 'device': device}),
                              ('a25', 'assess_image_quality', {'output_dir': str(out / 'cxas'), 'device': device}),
                              ('a6', 'detect_abnormalities', {'output_dir': str(out / 'a6'), 'device': device, 'findings': ['cardiomegaly']}),
                              ('a24', 'classify_view', {'output_dir': str(out / 'a24'), 'device': device})]:
        r = reg.run(tool, s, **params)
        res[key] = r if isinstance(r, ToolRefusal) else {u.finding_or_anatomy: u.value for u in r}
    res['masks'] = {k: v.copy() for k, v in s.masks.items()}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:4')
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('--out must be new')
    tol = json.loads((TV / 'cases/tolerances.json').read_text())['V4']
    frozen_cases = json.loads((TV / 'cases/core40.json').read_text())['cases']
    verify_frozen_cases(frozen_cases, ROOT / 'original_data')
    cases = [c for c in frozen_cases if c['source'] in ('chestx-det', 'nih')]
    cases = [c for c in cases if c['source'] == 'chestx-det'][:5] + [c for c in cases if c['source'] == 'nih'][:5]
    reg = build_a_registry()
    rows = []
    for case in cases:
        stem = Path(case['image_path']).stem
        with Image.open(ROOT / 'original_data' / case['image_path']) as im:
            base_img = transform(im, 'none')
        w, h = base_img.size
        img_dir = args.out / 'images'; img_dir.mkdir(parents=True, exist_ok=True)
        base_path = img_dir / f'{stem}_base.png'; base_img.save(base_path)
        base = run_all(reg, base_path, args.out / 'runs' / stem / 'base', args.device)
        for kind in ('pad', 'scale', 'flip'):
            tp = img_dir / f'{stem}_{kind}.png'; transform(base_img, kind).save(tp)
            t = run_all(reg, tp, args.out / 'runs' / stem / kind, args.device)
            row = {'case': stem, 'transform': kind}
            # PSPNet masks (A1/A2): pad must be exact (identical crop content); scale near-equal
            if kind != 'flip':
                row['psp_heart_iou'] = iou(map_mask(base['masks']['Heart'], kind), t['masks']['Heart'])
                row['psp_lungs_iou'] = min(iou(map_mask(base['masks'][n], kind), t['masks'][n]) for n in ('Right_Lung', 'Left_Lung'))
            else:
                row['psp_flip_right_vs_mirrored_left_iou'] = iou(map_mask(base['masks']['Left_Lung'], 'flip'), t['masks']['Right_Lung'])
            # A5 semantic points
            if kind != 'flip':
                dev = [np.hypot(*(np.subtract(map_point(base['a5'][k]['point_xy'], kind, w), t['a5'][k]['point_xy'])))
                       for k in base['a5'] if 'point_xy' in base['a5'][k]]
                row['a5_max_point_dev_px'] = float(max(dev)) / (2 if kind == 'scale' else 1)
            # A8 mask
            if kind != 'flip':
                bm = base['masks'].get('Pneumothorax_Candidate'); tm = t['masks'].get('Pneumothorax_Candidate')
                row['a8_mask_iou'] = None if bm is None and tm is None else iou(map_mask(bm if bm is not None else np.zeros((h, w), bool), kind),
                                                                               tm if tm is not None else np.zeros(map_mask(np.zeros((h, w), bool), kind).shape, bool))
            # A26 carina
            if kind != 'flip':
                bc, tc = base['a26']['carina']['point_xy'], t['a26']['carina']['point_xy']
                row['a26_carina_dev_frac_width'] = float(np.hypot(*np.subtract(map_point(bc, kind, w), tc))) / (w * (2 if kind == 'scale' else 1))
            # A12 / A25 invariants
            def ctr(r):
                return r['a12']['cardiothoracic_ratio']['CTR'] if isinstance(r['a12'], dict) else None
            row['a12_ctr_abs_diff'] = None if ctr(base) is None or ctr(t) is None else abs(ctr(base) - ctr(t))
            if kind != 'flip':
                rb, rt = base['a25']['rotation'], t['a25']['rotation']
                row['a25_rotation_index_abs_diff'] = (abs(rb['asymmetry_index'] - rt['asymmetry_index'])
                                                      if rb['status'] == rt['status'] == 'ok' else None)
            # CheXOne A6 box (grid cells) and A24 view
            if kind == 'scale':
                bb = base['a6']['candidate_cardiomegaly']['boxes_grid_0_100']; tb = t['a6']['candidate_cardiomegaly']['boxes_grid_0_100']
                row['a6_box_max_cell_dev'] = (float(np.abs(np.subtract(bb[0], tb[0])).max()) if bb and tb else (0.0 if not bb and not tb else None))
                row['a24_view_equal'] = base['a24']['view_position']['view'] == t['a24']['view_position']['view']
            rows.append(row)
            print(json.dumps(row), flush=True)
    backends.shutdown_workers()

    def worst(key, fn=max):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return fn(vals) if vals else None
    checks = {
        'PSPNet pad: masks identical after shift (IoU = 1)': all(r['psp_heart_iou'] == 1.0 and r['psp_lungs_iou'] == 1.0 for r in rows if r['transform'] == 'pad'),
        'PSPNet scale: mask IoU >= %.2f' % tol['mask_shift_iou']: all(r['psp_heart_iou'] >= tol['mask_shift_iou'] and r['psp_lungs_iou'] >= tol['mask_shift_iou'] for r in rows if r['transform'] == 'scale'),
        'A5 points follow pad/scale within %.1f px' % tol['hybridgnet_node_px_original']: all(r['a5_max_point_dev_px'] <= tol['hybridgnet_node_px_original'] for r in rows if 'a5_max_point_dev_px' in r),
        'A8 mask IoU >= %.2f (when a mask exists)' % tol['mask_shift_iou']: all(r['a8_mask_iou'] is None or r['a8_mask_iou'] >= tol['mask_shift_iou'] for r in rows if 'a8_mask_iou' in r),
        'A26 carina within %.0f%% of width' % (100 * tol['lines_tubes_point_frac_width']): all(r['a26_carina_dev_frac_width'] <= tol['lines_tubes_point_frac_width'] for r in rows if 'a26_carina_dev_frac_width' in r),
        'A12 CTR change <= %.2f (pad/scale/flip)' % tol['ctr_flip_abs']: all(r['a12_ctr_abs_diff'] is not None and r['a12_ctr_abs_diff'] <= tol['ctr_flip_abs'] for r in rows),
        'A6 box within %d grid cell after scale' % tol['chexone_box_grid_cells']: all(r['a6_box_max_cell_dev'] is not None and r['a6_box_max_cell_dev'] <= tol['chexone_box_grid_cells'] for r in rows if r['transform'] == 'scale'),
        'A24 view unchanged after scale': all(r['a24_view_equal'] for r in rows if r['transform'] == 'scale'),
    }
    info = {'PSPNet flip: Right_Lung(flipped) vs mirrored Left_Lung IoU (min)': worst('psp_flip_right_vs_mirrored_left_iou', min),
            'A25 rotation index change (max, pad/scale)': worst('a25_rotation_index_abs_diff')}
    summary = {'cases': len(cases), 'checks': checks, 'all_passed': all(checks.values()), 'informational': info,
               'worst': {k: worst(k) for k in ('a5_max_point_dev_px', 'a26_carina_dev_frac_width', 'a12_ctr_abs_diff', 'a6_box_max_cell_dev')}
               | {'psp_scale_min_iou': min(min(r['psp_heart_iou'], r['psp_lungs_iou']) for r in rows if r['transform'] == 'scale'),
                  'a8_min_iou': worst('a8_mask_iou', min)}}
    (args.out / 'v4_summary.json').write_text(json.dumps(summary, indent=1))
    (args.out / 'v4_rows.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(summary, indent=1))


def worst_kind(rows, kind, key, fn):
    vals = [r[key] for r in rows if r['transform'] == kind and r.get(key) is not None]
    return fn(vals) if vals else None


if __name__ == '__main__':
    main()
