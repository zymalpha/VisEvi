"""Development batch runner. Predicted measurements are not clinical GT."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ..core.session import EvidenceSession
from ..core.units import AnswerVerdict, Demand, EvidenceUnit, fov_refusal
from ..perception.anatomy import SegmentAnatomy
from ..perception.anatomy.pspnet import file_sha256
from ..perception import geometry
from ..perception.quality import AssessExposure, AssessInspiration, AssessRotation
from ..construction.teachers import ComposeEvidenceCTR, ComposeEvidenceSymmetry
from ..serve import evidence_card
from .runner import build_registry


def load_manifest(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row['sample_key'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Manifest contains duplicate sample_key values')
    if not rows:
        raise ValueError('Manifest is empty')
    return rows


def run_batch(manifest: str | Path, data_root: str | Path, out_dir: str | Path, predictor) -> dict:
    manifest, data_root, out_dir = Path(manifest), Path(data_root).resolve(), Path(out_dir)
    rows = load_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / 'predictions.jsonl'
    if results_path.exists():
        raise FileExistsError('Use a new output directory to preserve the previous run')
    schema = build_registry().describe_all()
    (out_dir / 'tool_schemas.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    results = []
    started = time.perf_counter()
    with results_path.open('w', encoding='utf-8') as result_file:
        for row in rows:
            sample_key = row['sample_key']
            folder = 'sample-' + hashlib.sha256(sample_key.encode()).hexdigest()[:16]
            sample_dir = out_dir / folder
            sample_dir.mkdir()
            result = {'sample_key': sample_key, 'dataset': row['dataset'],
                'image_path': row['image_path'], 'sample_output': folder, 'status': 'error',
                'clinical_gt_available': False}
            t0 = time.perf_counter()
            try:
                image_path = (data_root / row['image_path']).resolve()
                if not image_path.is_relative_to(data_root):
                    raise ValueError('Image path leaves data root')
                if file_sha256(image_path) != row['sha256']:
                    raise ValueError('Image checksum differs from manifest')
                cache = predictor.predict(image_path, sample_key, sample_dir / 'masks')
                if cache['binding']['image_sha256'] != row['sha256']:
                    raise ValueError('Predictor cache is bound to a different image')
                session = EvidenceSession(image_ref=str(image_path), demand=Demand(
                    proposition='开发试验：几何与质量代理、证据链可执行性'))
                registry = build_registry()
                segmentation = registry.run('segment_anatomy', session,
                    masks_dir=str(sample_dir / 'masks'), image_ref=str(image_path))
                truncated = (session.fov_truncated(['Heart', 'Right_Lung', 'Left_Lung'])
                             if isinstance(segmentation, list) else {})
                if not isinstance(segmentation, list):
                    result.update(status='refusal', refusal=segmentation.to_dict())
                elif truncated:
                    result.update(status='refusal', refusal=fov_refusal('development_batch', truncated).to_dict())
                else:
                    for unit in segmentation:
                        unit.provenance = ['cache_record.json:' + cache['binding']['image_sha256']]
                        unit.validity_conditions.append('automatically predicted mask; not independent GT')
                    if row.get('view_position'):
                        session.add_observation(EvidenceUnit(id=session.next_id('obs'),
                            category='observation', evidence_type='global_condition',
                            finding_or_anatomy='view_position', value={'view': row['view_position'],
                                'assumed': False, 'source': 'dataset metadata; not a model prediction'},
                            source_tool='dataset_metadata'))
                    for tool in ['assess_inspiration', 'assess_rotation', 'assess_exposure',
                                 'compose_evidence_symmetry']:
                        registry.run(tool, session)
                    refusal = registry.run('compose_evidence_ctr', session)
                    session.verdict = AnswerVerdict(answerable=False,
                        abstain_reason='独立关键点来源未生成；未运行双来源 CTR 证据链',
                        evidence_refs=[u.id for u in session.graph.by_category('claim')])
                    masks = session.masks
                    heart, union = masks['Heart'], masks['Right_Lung'] | masks['Left_Lung']
                    result.update(status='ok', measurements={
                        'ctr_max_row_proxy': geometry.ctr_from_masks(heart, union),
                        'ctr_danzer_proxy': geometry.ctr_danzer(heart, union),
                        'quality': {u.finding_or_anatomy: u.value for u in session.graph.by_category('observation')
                                    if u.source_tool.startswith('assess_')}},
                        ctr_teacher_status='refused_missing_independent_landmarks',
                        ctr_teacher_refusal=refusal.to_dict())
                session.save(sample_dir / 'session.json')
                (sample_dir / 'evidence_card.md').write_text(evidence_card(session.to_dict()), encoding='utf-8')
                (sample_dir / 'program_traces.json').write_text(json.dumps(session.program_traces, ensure_ascii=False, indent=2))
            except Exception as exc:
                result.update(status='error', error_type=type(exc).__name__, error=str(exc))
            result['elapsed_seconds'] = round(time.perf_counter() - t0, 4)
            (sample_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            result_file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            result_file.flush()
            results.append(result)
            print(json.dumps({'sample_key': sample_key, 'status': result['status']}), flush=True)
    counts = Counter(r['status'] for r in results)
    report = {'run_type': 'development_pipeline_validation',
        'finished_at': datetime.now(timezone.utc).isoformat(), 'samples': len(rows),
        'statuses': dict(counts), 'dataset_counts': dict(Counter(r['dataset'] for r in rows)),
        'manifest': str(manifest.resolve()), 'manifest_sha256': file_sha256(manifest),
        'data_root': str(data_root), 'predictor': predictor.identity,
        'code_sha256': {str(p.relative_to(Path(__file__).resolve().parents[3])): file_sha256(p)
            for p in sorted(Path(__file__).resolve().parents[1].rglob('*.py'))},
        'elapsed_seconds': round(time.perf_counter() - started, 4),
        'clinical_accuracy': None, 'clinical_accuracy_status': 'not_evaluated_no_independent_ground_truth',
        'ctr_teacher_completed': 0, 'quality_thresholds': 'uncalibrated placeholders',
        'limits': ['Predicted masks are not anatomical GT.',
            'Metrics are automated proxies, not manual rib-inner-margin CTR or clinical quality grades.',
            'Known exact duplicates were excluded; patient and pretraining independence remain unverified.',
            'Independent landmark pipeline is unavailable; dual-source CTR teacher is refused.']}
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report
