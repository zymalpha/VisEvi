"""V1: interface contracts for all A interfaces (refusal semantics and schema).

Run: .venv/bin/python tool_verification/suites/v1_contracts.py --out tool_verification/runs/<date>/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from jsonschema import Draft202012Validator
import numpy as np
from PIL import Image

from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import a_capabilities, build_a_registry

MASK_PARAMS = {   # minimal valid parameters for mask-consuming tools; 'X' is the probe mask
    'measure_region_extent': {'mask_name': 'X'}, 'measure_region_area': {'mask_name': 'X'},
    'measure_region_distance': {'mask_a': 'X', 'mask_b': 'Heart'}, 'describe_region_shape': {'mask_name': 'X'},
    'summarize_roi_intensity': {'mask_name': 'X'}, 'extract_roi_texture': {'mask_name': 'X'},
    'measure_roi_sharpness': {'mask_name': 'X'}, 'assign_lung_zones': {'region_mask_name': 'X'},
    'relate_lesion_to_anatomy': {'lesion_mask_name': 'X', 'anatomy_mask_name': 'Heart'},
    'relate_multiple_regions': {'region_mask_names': ['X', 'Heart']},
    'summarize_bilateral_distribution': {'lesion_mask_names': ['X']},
}
MODEL_TOOLS = {'locate_lung_landmarks': {}, 'detect_abnormalities': {}, 'ground_phrase': {'phrase': 'effusion'},
               'segment_pneumothorax': {}, 'measure_ctr': {}, 'classify_roi_attributes': {'roi_mask_name': 'Heart'},
               'classify_view': {}, 'assess_image_quality': {}, 'localize_devices': {}}
ANATOMY = ['segment_lungs', 'segment_heart', 'segment_mediastinum', 'segment_shoulder_bones']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be new')
    args.out.mkdir(parents=True, exist_ok=True)
    reg = build_a_registry()
    specs = {s['name']: s for s in reg.describe_all()}
    rows = []

    def expect(tool, case, result, refused=True):
        ok = isinstance(result, ToolRefusal) == refused
        rows.append({'tool': tool, 'case': case, 'expected_refusal': refused, 'ok': ok,
                     'reason': result.reason if isinstance(result, ToolRefusal) else None})

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img8 = tmp / 'img8.png'; Image.new('L', (40, 30), 80).save(img8)
        img16 = tmp / 'img16.png'; Image.fromarray(np.full((30, 40), 3000, np.uint16)).save(img16)
        exif = Image.Exif(); exif[0x0112] = 6
        imgexif = tmp / 'rot.jpg'; Image.new('L', (40, 30), 80).save(imgexif, exif=exif)

        def session(path=img8):
            s = EvidenceSession(image_ref=str(path))
            right = np.zeros((30, 40), bool); right[5:25, 3:15] = True
            left = np.zeros((30, 40), bool); left[5:25, 25:37] = True
            heart = np.zeros((30, 40), bool); heart[12:24, 12:28] = True
            s.attach_masks({'Right_Lung': right, 'Left_Lung': left, 'Heart': heart})
            return s

        for name, spec in specs.items():
            Draft202012Validator.check_schema(spec['input_schema'])
            Draft202012Validator.check_schema(spec['output_schema'])
            rows.append({'tool': name, 'case': 'schema valid (Draft 2020-12)', 'expected_refusal': False, 'ok': True})
            expect(name, 'unknown parameter', reg.run(name, session(), output_dir=str(tmp / 'u' / name), bogus=1))
            required = spec['input_schema'].get('required', [])
            if required:
                expect(name, f'missing required {required}', reg.run(name, session()))
        for name, params in MASK_PARAMS.items():
            s = session()
            expect(name, 'probe mask absent', reg.run(name, s, **params))
            probe = np.zeros((30, 40), bool); probe[8:14, 4:10] = True
            s.attach_masks({'X': probe}, {'X': {'fov_truncated_sides': ['top']}})
            expect(name, 'probe mask FOV-truncated', reg.run(name, s, **params))
            s.attach_masks({'X': np.where(probe, 0.7, 0.0)}, {'X': {}})
            expect(name, 'probe mask non-binary', reg.run(name, s, **params))
            s.attach_masks({'X': probe}, {'X': {}})
            expect(name, 'valid probe mask', reg.run(name, s, **params), refused=False)
        for name, params in MODEL_TOOLS.items():
            for label, path in (('16-bit image', img16), ('EXIF-rotated image', imgexif)):
                expect(name, label, reg.run(name, session(path), output_dir=str(tmp / 'm' / label[:5] / name), device='cpu', **params))
            expect(name, 'non-string device', reg.run(name, session(), output_dir=str(tmp / 'd' / name), device=4, **params))
        for name in ANATOMY:
            expect(name, 'both masks_dir and output_dir', reg.run(name, session(), masks_dir=str(tmp), output_dir=str(tmp)))
            expect(name, 'cached mode without cache_record', reg.run(name, session(), masks_dir=str(tmp)))
            expect(name, '16-bit image (live)', reg.run(name, session(img16), output_dir=str(tmp / 'p' / name)))
        expect('measure_ctr', 'no A2 Heart mask', reg.run('measure_ctr', EvidenceSession(image_ref=str(img8)), output_dir=str(tmp / 'c')))
        expect('compute_ctr_from_borders', 'heart outside thorax', reg.run('compute_ctr_from_borders', session(),
               heart_borders_x=[0, 30], thoracic_inner_borders_x=[5, 20]))
        expect('measure_lung_symmetry', 'non half-integer midline', reg.run('measure_lung_symmetry', session(), midline_x=10.3))
        overlap = session(); overlap.masks['Left_Lung'][5:25, 10:20] = True
        for name, params in [('compare_lung_geometry', {}), ('assign_lung_zones', {'region_mask_name': 'Heart'}),
                             ('summarize_bilateral_distribution', {'lesion_mask_names': ['Heart']})]:
            expect(name, 'overlapping left/right lungs', reg.run(name, overlap, **params))
    by_tool = {}
    for r in rows:
        t = by_tool.setdefault(r['tool'], {'checks': 0, 'failed': []})
        t['checks'] += 1
        if not r['ok']:
            t['failed'].append(r)
    summary = {'interfaces': len(specs), 'capabilities': sorted(a_capabilities(reg), key=lambda c: int(c[1:])),
               'checks': len(rows), 'failed': sum(1 for r in rows if not r['ok']), 'by_tool': by_tool}
    (args.out / 'v1_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (args.out / 'v1_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(summary['interfaces'], 'interfaces;', summary['checks'], 'checks;', summary['failed'], 'failed')
    for name, t in by_tool.items():
        for f in t['failed']:
            print('FAIL', name, f['case'], f['reason'])


if __name__ == '__main__':
    main()
