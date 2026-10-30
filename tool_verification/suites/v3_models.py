"""V3 for model backends: our interfaces vs the authors' official usage on core40.

A5 HybridGNet, A8 pneumothorax, A26 lines/tubes, CXAS (A12 auto / A25), CheXOne (A6/A7/A18/A24).
Run: .venv/bin/python tool_verification/suites/v3_models.py --out tool_verification/runs/<date>/v3_models
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from visevi import backends
from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import build_a_registry
from visevi.perception import structures

ROOT = Path(__file__).resolve().parents[2]
TV = ROOT / 'tool_verification'
ORACLE_PY = str(ROOT / '.venv-oracle/bin/python')
W = ROOT / 'tools/weights'


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def roi_box(w, h):
    m = np.zeros((h, w), bool)
    m[int(h * .35):int(h * .65), int(w * .15):int(w * .45)] = True
    return m


def run_ours(cases, out, device):
    reg = build_a_registry()
    ours, requests = {}, []
    for case in cases:
        stem = Path(case['image_path']).stem
        image = str((ROOT / 'original_data' / case['image_path']).resolve())
        s = EvidenceSession(image_ref=image)
        w, h = case['size_wh']
        s.attach_masks({'ROI': roi_box(w, h)})
        d = out / 'ours' / stem
        calls = {'a5': ('locate_lung_landmarks', {}), 'a8': ('segment_pneumothorax', {}), 'a26': ('localize_devices', {}),
                 'cxas': ('assess_image_quality', {}), 'a24': ('classify_view', {}),
                 'a6': ('detect_abnormalities', {'findings': ['cardiomegaly', 'pleural effusion']}),
                 'a7': ('ground_phrase', {'phrase': 'pleural effusion'}),
                 'a18': ('classify_roi_attributes', {'roi_mask_name': 'ROI', 'attributes': ['consolidation', 'pleural effusion']})}
        rec = {}
        for key, (tool, params) in calls.items():
            r = reg.run(tool, s, output_dir=str(d / key), device=device, **params)
            if isinstance(r, ToolRefusal):
                rec[key] = {'error': r.reason}
                continue
            rec[key] = json.loads((d / key / 'cache_record.json').read_text())
            if key in ('a24', 'a6', 'a7', 'a18'):
                for n, (spec, ans) in enumerate(zip(rec[key]['binding']['params']['requests'], rec[key]['result']['answers'])):
                    requests.append({'key': f'{stem}/{key}/{n}', 'image_path': image, 'content': spec['content'],
                                     'max_new_tokens': spec.get('max_new_tokens', 256), 'option_letters': spec.get('option_letters')})
        ours[stem] = {'case': case, 'records': rec, 'dir': str(d)}
    return ours, requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:4')
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('--out must be new')
    args.out.mkdir(parents=True)
    tol = json.loads((TV / 'cases/tolerances.json').read_text())['V3']
    cases_file = TV / 'cases/core40.json'
    cases = json.loads(cases_file.read_text())['cases']
    ours, requests = run_ours(cases, args.out, args.device)
    backends.shutdown_workers()
    (args.out / 'chexone_requests.json').write_text(json.dumps(requests))
    off = args.out / 'official'
    common = ['--cases', str(cases_file), '--data-root', str(ROOT / 'original_data')]
    src = lambda name: str(next((ROOT / 'tools/src' / name).glob('*-*')))
    jobs = [
        [ORACLE_PY, str(TV / 'oracles/hybridgnet_official.py'), *common, '--weights', str(W / 'chexmask-u'),
         '--source', src('chexmask-u') + '/HybridGNet', '--out', str(off / 'a5'), '--device', 'cuda:5'],
        [ORACLE_PY, str(TV / 'oracles/pneumothorax_official.py'), *common, '--weights', str(W / 'pneumothorax'),
         '--out', str(off / 'a8'), '--device', 'cuda:5'],
        [ORACLE_PY, str(TV / 'oracles/lines_tubes_official.py'), *common, '--weights', str(W / 'lines-tubes'),
         '--out', str(off / 'a26'), '--device', 'cuda:6'],
        [ORACLE_PY, str(TV / 'oracles/cxas_official.py'), *common, '--weights', str(W / 'cxas'),
         '--source', src('cxas') + '/cxas', '--out', str(off / 'cxas'), '--gpu', '6'],
        [ORACLE_PY, str(TV / 'oracles/chexone_official.py'), '--requests', str(args.out / 'chexone_requests.json'),
         '--weights', str(W / 'chexone'), '--out', str(off / 'chexone.json'), '--device', 'cuda:7'],
    ]
    off.mkdir()
    procs = [subprocess.Popen(j, stdout=open(off / f'log_{i}.txt', 'w'), stderr=subprocess.STDOUT) for i, j in enumerate(jobs)]
    codes = [pr.wait() for pr in procs]
    if any(codes):
        raise SystemExit(f'oracle failures {codes}; see {off}/log_*.txt')
    rows, chex = [], {r['key']: r for r in json.loads((off / 'chexone.json').read_text())}
    for stem, o in ours.items():
        rec, row = o['records'], {'case': stem, 'source': o['case']['source'], 'mode': o['case']['mode']}
        # A5
        r = rec['a5']['result']; fr = r['frame']
        mine = (np.asarray(r['nodes_xy']) + [fr['pad_left'], fr['pad_top']]) * 1024 / fr['square_side']
        ref = np.load(off / 'a5' / f'{stem}.npz')
        row['a5_max_node_dev_px1024'] = float(np.abs(mine - ref['mean']).max())
        row['a5_author_sampling_max_dev_px1024'] = float(np.abs(ref['sampled'] - ref['mean']).max())
        # A8
        r = rec['a8']; ref = np.load(off / 'a8' / f'{stem}.npz')
        prob = np.load(Path(o['dir']) / 'a8/probability_model_frame.npy').astype(np.float32)
        row['a8_case_score_abs'] = abs(r['result']['case_score'] - float(ref['cls']))
        row['a8_prob_max_abs_fp16_artifact'] = float(np.abs(prob - ref['prob']).max())
        row['a8_mask512_mismatch_px'] = int(((prob > 0.5) != (ref['prob'] > 0.5)).sum())
        # A26
        r = rec['a26']['result']; ref = json.loads((off / 'a26' / f'{stem}.json').read_text())
        row['a26_presence_max_abs'] = max(abs(r['devices'][k]['presence_raw_score'] - ref['presence'][k]) for k in ref['presence'])
        row['a26_classification_max_abs'] = max(abs(r['classification_raw'][k] - v) for k, v in ref['classification'].items())
        row['a26_ett_tip_equal'] = r['devices']['ETT']['distal_tip_xy'] == ref['landmarks']['ett_tip_xy']
        row['a26_carina_equal'] = r['carina_xy'] == ref['landmarks']['carina_xy']
        row['a26_fold_ids_equal'] = r['fold_ids'] == ref['fold_ids']
        # CXAS
        rr = rec['cxas']; w, h = o['case']['size_wh']
        labels = list(rr['result']['labels'].values())
        mine = structures.load_labels(rr, str(Path(o['dir']) / 'cxas'), labels, w, h)
        d = np.load(off / 'cxas' / f'{stem}.npz')
        full = np.unpackbits(d['packed'], axis=-1)[..., :w].astype(bool)
        ious = [iou(mine[l], full[int(k)]) for k, l in rr['result']['labels'].items()]
        row['cxas_min_label_iou'] = float(min(ious)); row['cxas_labels'] = len(ious)
        # CheXOne
        same_tokens, prob_abs, n = 0, 0.0, 0
        for key in ('a24', 'a6', 'a7', 'a18'):
            for i, ans in enumerate(rec[key]['result']['answers']):
                ref = chex[f'{stem}/{key}/{i}']
                n += 1
                same_tokens += ans['token_ids'] == ref['token_ids']
                for l, v in (ans.get('first_token_option_probabilities') or {}).items():
                    prob_abs = max(prob_abs, abs(v - ref['option_probs'][l]))
        row['chexone_identical_token_sequences'] = f'{same_tokens}/{n}'
        row['chexone_option_prob_max_abs'] = prob_abs
        rows.append(row)
    checks = {
        'A5 max node deviation <= %.1f px (1024 frame)' % tol['hybridgnet_max_node_deviation_px_1024']:
            all(r['a5_max_node_dev_px1024'] <= tol['hybridgnet_max_node_deviation_px_1024'] for r in rows),
        'A8 case score |d| <= %g' % tol['pneumothorax_prob_max_abs']: all(r['a8_case_score_abs'] <= tol['pneumothorax_prob_max_abs'] for r in rows),
        'A8 512-frame mask identical': all(r['a8_mask512_mismatch_px'] == 0 for r in rows),
        'A26 presence |d| <= %g' % tol['lines_tubes_score_abs']: all(r['a26_presence_max_abs'] <= tol['lines_tubes_score_abs'] for r in rows),
        'A26 classification |d| <= %g' % tol['lines_tubes_score_abs']: all(r['a26_classification_max_abs'] <= tol['lines_tubes_score_abs'] for r in rows),
        'A26 ETT tip and carina identical': all(r['a26_ett_tip_equal'] and r['a26_carina_equal'] and r['a26_fold_ids_equal'] for r in rows),
        'CXAS label IoU >= %g' % tol['cxas_official_vs_adapter_mask_iou']: all(r['cxas_min_label_iou'] >= tol['cxas_official_vs_adapter_mask_iou'] for r in rows),
        'CheXOne identical token sequences': all(r['chexone_identical_token_sequences'].split('/')[0] == r['chexone_identical_token_sequences'].split('/')[1] for r in rows),
        'CheXOne option prob |d| <= %g' % tol['chexone_option_prob_abs']: all(r['chexone_option_prob_max_abs'] <= tol['chexone_option_prob_abs'] for r in rows),
    }
    summary = {'cases': len(rows), 'checks': checks, 'all_passed': all(checks.values()),
               'worst': {k: max(r[k] for r in rows) for k in ('a5_max_node_dev_px1024', 'a5_author_sampling_max_dev_px1024',
                                                              'a8_case_score_abs', 'a8_prob_max_abs_fp16_artifact', 'a8_mask512_mismatch_px',
                                                              'a26_presence_max_abs', 'a26_classification_max_abs', 'chexone_option_prob_max_abs')}
               | {'cxas_min_label_iou': min(r['cxas_min_label_iou'] for r in rows)}}
    (args.out / 'v3_models_summary.json').write_text(json.dumps(summary, indent=1))
    (args.out / 'v3_models_rows.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
