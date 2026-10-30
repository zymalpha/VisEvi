"""V3 PSPNet: official TorchXRayVision pipeline vs our A1-A4 adapter (512 model frame).

Run: .venv/bin/python tool_verification/suites/v3_pspnet.py --out tool_verification/runs/<date>/v3_pspnet --device cuda:4
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import build_a_registry

ROOT = Path(__file__).resolve().parents[2]
TV = ROOT / 'tool_verification'


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:4')
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('--out must be new')
    official = args.out / 'official'
    subprocess.run([str(ROOT / '.venv/bin/python'), str(TV / 'oracles/pspnet_official.py'), '--cases', str(TV / 'cases/core40.json'),
                    '--data-root', str(ROOT / 'original_data'), '--weights', str(ROOT / 'tools/weights'),
                    '--out', str(official), '--device', args.device], check=True)
    tol = json.loads((TV / 'cases/tolerances.json').read_text())['V3']['pspnet_official_vs_adapter_iou_equivalent']
    reg = build_a_registry()
    rows = []
    for case in json.loads((TV / 'cases/core40.json').read_text())['cases']:
        stem = Path(case['image_path']).stem
        image = (ROOT / 'original_data' / case['image_path']).resolve()
        out_dir = args.out / 'ours' / stem
        result = reg.run('segment_lungs', EvidenceSession(image_ref=str(image)), output_dir=str(out_dir), device=args.device)
        if isinstance(result, ToolRefusal):
            rows.append({'case': stem, 'error': result.reason}); continue
        data = np.load(official / f'{stem}.npz')
        off = np.unpackbits(data['packed'], axis=-1)[..., :512].astype(bool)
        per = {}
        for k, target in enumerate(data['targets']):
            ours = np.asarray(Image.open(out_dir / (str(target).replace(' ', '_') + '.png'))) > 0
            per[str(target)] = {'iou': iou(ours, off[k]), 'area_ours': int(ours.sum()), 'area_official': int(off[k].sum())}
        rows.append({'case': stem, 'source': case['source'], 'size_wh': case['size_wh'], 'mode': case['mode'], 'targets': per})
    targets = [t for r in rows if 'targets' in r for t in r['targets']][:14]
    stats = {t: {'min_iou': min(r['targets'][t]['iou'] for r in rows if 'targets' in r),
                 'median_iou': float(np.median([r['targets'][t]['iou'] for r in rows if 'targets' in r]))} for t in targets}
    summary = {'tolerance_iou': tol, 'cases': len(rows), 'errors': [r for r in rows if 'error' in r], 'per_target': stats,
               'equivalent': all(v['min_iou'] >= tol for v in stats.values())}
    (args.out / 'v3_pspnet_summary.json').write_text(json.dumps(summary, indent=1))
    (args.out / 'v3_pspnet_rows.json').write_text(json.dumps(rows, indent=1))
    for t, v in stats.items():
        print(f'{t:24s} min IoU {v["min_iou"]:.4f}  median {v["median_iou"]:.4f}')
    print('equivalent at', tol, ':', summary['equivalent'])


if __name__ == '__main__':
    main()
