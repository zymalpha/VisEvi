"""Run every A1-A26 interface on a few images per local source and tabulate call status.

Engineering smoke test, not an accuracy study: it shows which interfaces run,
refuse or fail on differently sized/encoded inputs. Use a fresh --out directory.

Run: .venv/bin/python scripts/run_a_tools_multisource.py --out trials/<new> --per-source 3 --model-device cuda:4
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import a_capabilities, build_a_registry

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'chestx-det': 'original_data/chestx-det/images/test',
    'nih': 'original_data/nih-cxr14/kaggle-sample-v4/images',
    'jsrt': 'original_data/jsrt/kaggle-png-v1/data/images/images',
    'montgomery': 'original_data/montgomery-nlm/data/MontgomerySet/CXR_png',
}


def calls(model_dir: Path, device: str):
    """(tool, parameters) in dependency order; masks from earlier calls feed later ones."""
    model = lambda name: {'output_dir': str(model_dir / name), 'device': device}
    return [
        ('segment_lungs', {'output_dir': str(model_dir / 'pspnet'), 'device': 'cpu'}),
        ('segment_heart', {'output_dir': str(model_dir / 'pspnet'), 'device': 'cpu'}),
        ('segment_mediastinum', {'output_dir': str(model_dir / 'pspnet'), 'device': 'cpu'}),
        ('segment_shoulder_bones', {'output_dir': str(model_dir / 'pspnet'), 'device': 'cpu'}),
        ('locate_lung_landmarks', model('a5')),
        ('detect_abnormalities', {**model('chexone_a6'), 'findings': ['cardiomegaly', 'pleural effusion', 'nodule']}),
        ('ground_phrase', {**model('chexone_a7'), 'phrase': 'pleural effusion'}),
        ('segment_pneumothorax', model('a8')),
        ('measure_region_extent', {'mask_name': 'Heart'}),
        ('measure_region_area', {'mask_name': 'Right_Lung'}),
        ('measure_region_distance', {'mask_a': 'Heart', 'mask_b': 'Left_Lung'}),
        ('measure_ctr', model('cxas')),
        ('compare_lung_geometry', {}),
        ('describe_region_shape', {'mask_name': 'Heart'}),
        ('summarize_roi_intensity', {'mask_name': 'Right_Lung'}),
        ('extract_roi_texture', {'mask_name': 'Right_Lung'}),
        ('measure_roi_sharpness', {'mask_name': 'Right_Lung'}),
        ('classify_roi_attributes', {**model('chexone_a18'), 'roi_mask_name': 'Right_Lung',
                                     'attributes': ['consolidation', 'pleural effusion']}),
        ('measure_lung_symmetry', {}),
        ('assign_lung_zones', {'region_mask_name': 'Heart'}),
        ('relate_lesion_to_anatomy', {'lesion_mask_name': 'Heart', 'anatomy_mask_name': 'Left_Lung'}),
        ('relate_multiple_regions', {'region_mask_names': ['Heart', 'Right_Lung', 'Left_Lung']}),
        ('summarize_bilateral_distribution', {'lesion_mask_names': ['Heart']}),
        ('classify_view', model('chexone_a24')),
        ('assess_image_quality', model('cxas')),
        ('localize_devices', model('a26')),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=3)
    parser.add_argument('--model-device', default='cuda:4')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('--out must be empty or new')
    out.mkdir(parents=True, exist_ok=True)
    registry = build_a_registry()
    rows = []
    for source, folder in SOURCES.items():
        for image in sorted((ROOT / folder).glob('*.png'))[:args.per_source]:
            session = EvidenceSession(image_ref=str(image))
            sample_dir = out / f'{source}-{image.stem}'
            for tool, params in calls(sample_dir / 'models', args.model_device):
                started = time.perf_counter()
                result = registry.run(tool, session, **params)
                refused = isinstance(result, ToolRefusal)
                rows.append({'source': source, 'image': str(image.relative_to(ROOT)), 'tool': tool,
                             'status': session.trace[-1]['result'], 'seconds': round(time.perf_counter() - started, 2),
                             'reason': result.reason if refused else None,
                             'units': None if refused else len(result)})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            session.save(sample_dir / 'session.json')
    (out / 'calls.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    table = {}
    for row in rows:
        table.setdefault(row['tool'], Counter())[row['status']] += 1
    summary = {'images': len({r['image'] for r in rows}), 'calls': len(rows),
               'status_by_tool': {k: dict(v) for k, v in table.items()},
               'capabilities': a_capabilities(registry), 'medical_accuracy_evaluated': False}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary['status_by_tool'], ensure_ascii=False))


if __name__ == '__main__':
    main()
