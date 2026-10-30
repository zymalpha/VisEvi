"""Reproduce A-tool engineering checks on one annotated development image.

This is an implementation check, not an independent medical accuracy study.
Run with the project interpreter and a fresh --out directory.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import correlate
from scipy.spatial import ConvexHull, cKDTree
import torch

from visevi.core.session import EvidenceSession
from visevi.core.units import EvidenceUnit, ToolRefusal
from visevi.eval.runner import build_a_registry
from visevi.perception.anatomy.pspnet import file_sha256
from visevi.perception.appearance import load_grayscale


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--model-device', default='cuda:4',
                        help='device for isolated model workers (A5-A8, A12 auto, A18, A24-A26)')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('--out must be empty or new; previous evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    image = ROOT / 'original_data/chestx-det/images/test/36212.png'
    annotations = ROOT / 'original_data/chestx-det/ChestX_Det_test.json'
    rows = json.loads(annotations.read_text())
    index, annotation = next((i, row) for i, row in enumerate(rows)
                             if row['file_name'] == image.name)
    gray, decoding = load_grayscale(image)
    h, w = gray.shape
    registry = build_a_registry()
    session = EvidenceSession(image_ref=str(image))
    calls, checks, intermediates = [], [], {}
    limits = [
        'Development image already used for inspection; not an untouched test sample.',
        'PSPNet predictions have no independent organ GT here. Training independence is not established.',
        'Dataset polygons are supplied inputs to geometry tools, not A6 detector predictions.',
        'Rasterized polygon borders use Pillow ImageDraw pixel inclusion; original polygons are retained.',
        'Patient orientation, pixel spacing and clinical measurement protocols are not independently verified.',
        'No independent thoracic inner boundaries: real-image clinical CTR is not computed.',
        'A20/A23 use geometric lung thirds/overlap, not anatomical lobes or disease severity.',
    ]
    write_json(out / 'inputs.json', {
        'image': str(image), 'image_sha256': file_sha256(image),
        'shape_hw': [h, w], 'decoding': decoding,
        'annotation_file': str(annotations), 'annotation_sha256': file_sha256(annotations),
        'annotation_row_index': index, 'annotation': annotation,
        'device': args.device, 'threads': args.threads, 'limitations': limits,
    })
    write_json(out / 'tool_menu.json', registry.describe_all())
    write_json(out / 'environment.json', {
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch', 'torchxrayvision', 'numpy', 'scipy', 'pillow', 'shapely', 'opencv-python']},
        'script_sha256': file_sha256(__file__),
        'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p)
                          for p in sorted((ROOT / 'src/visevi').rglob('*.py'))},
    })

    def check(name, actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        passed = a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-10, atol=1e-8))
        checks.append({'name': name, 'passed': passed,
                       'actual': a.tolist(), 'expected': b.tolist()})

    def invoke(name, *, target_session=None, expected_refusal=False, **params):
        target = target_session or session
        result = registry.run(name, target, **params)
        refused = isinstance(result, ToolRefusal)
        calls.append({'tool': name, 'input_kind': 'synthetic_arithmetic' if target is not session else 'real_development_image',
                      'parameters': params, 'status': 'refusal' if refused else 'ok',
                      'expected_refusal': expected_refusal,
                      'output': result.to_dict() if refused else [unit.to_dict() for unit in result]})
        checks.append({'name': name + ':call_status', 'passed': refused == expected_refusal})
        with (out / 'calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(calls[-1], ensure_ascii=False, allow_nan=False) + '\n')
        session.save(out / 'session.json')
        print(name, calls[-1]['status'], flush=True)
        return None if refused else result[0].value

    # One fresh forward pass; the other A interfaces share the image-bound output.
    for name in ['segment_lungs', 'segment_heart', 'segment_mediastinum', 'segment_shoulder_bones']:
        value = invoke(name, output_dir=str(out / 'pspnet_masks'), device=args.device)
        if value is not None:
            check(name + ':cache_reuse', value['prediction_cache_hit'], name != 'segment_lungs')

    supplied_names = []
    for i, (label, polygon) in enumerate(zip(annotation['syms'], annotation['polygons'])):
        name = f'Supplied_{i}_{label}'
        raster = Image.new('1', (w, h))
        ImageDraw.Draw(raster).polygon([tuple(point) for point in polygon], fill=1)
        mask = np.asarray(raster, dtype=bool)
        source = {'kind': 'supplied_dataset_polygon', 'not_detector_output': True,
                  'annotation_file': str(annotations), 'annotation_row': index,
                  'polygon_index': i, 'label': label, 'image_sha256': file_sha256(image),
                  'coordinate_frame': 'original_image_pixels',
                  'rasterization': 'Pillow ImageDraw.polygon; integer pixel inclusion'}
        session.attach_masks({name: mask}, {name: source})
        session.add_observation(EvidenceUnit(id=session.next_id('obs'), category='observation',
            evidence_type='region', finding_or_anatomy=name, value=source,
            source_tool='supplied_annotation_loader', provenance=[f'{annotations}#row={index},polygon={i}'],
            validity_conditions=['supplied annotation input; no model prediction or accuracy score']))
        supplied_names.append(name)
    first, second = supplied_names[:2]  # Consolidation and Pneumothorax; not the Cardiomegaly region.
    a, b = session.masks[first], session.masks[second]
    np.savez_compressed(out / 'original_frame_masks.npz', **session.masks)
    values = {}
    values['A9'] = invoke('measure_region_extent', mask_name='Heart')
    values['A10'] = invoke('measure_region_area', mask_name=first)
    values['A11'] = invoke('measure_region_distance', mask_a=first, mask_b=second)
    # Preserve an expected real-image refusal instead of substituting lung width.
    invoke('compute_ctr_from_borders', expected_refusal=True)
    synthetic = EvidenceSession(image_ref='synthetic:known_70_over_100')
    toy_ctr = invoke('compute_ctr_from_borders', target_session=synthetic,
                     heart_borders_x=[15, 85], thoracic_inner_borders_x=[0, 100])
    check('A12:hand_arithmetic_70_over_100', toy_ctr['CTR'], 0.70)
    synthetic.save(out / 'synthetic_ctr_session.json')
    values['A13'] = invoke('compare_lung_geometry')
    values['A14'] = invoke('describe_region_shape', mask_name=first)
    values['A15'] = invoke('summarize_roi_intensity', mask_name=first)
    values['A16'] = invoke('extract_roi_texture', mask_name=first)
    values['A17'] = invoke('measure_roi_sharpness', mask_name=first)
    values['A19'] = invoke('measure_lung_symmetry')
    lungs_present = all(name in session.masks for name in ['Right_Lung', 'Left_Lung'])
    lung_overlap = (int((session.masks['Right_Lung'] & session.masks['Left_Lung']).sum())
                    if lungs_present else None)
    values['A20'] = invoke('assign_lung_zones', region_mask_name=first,
                           expected_refusal=bool(lung_overlap))
    values['A21'] = invoke('relate_lesion_to_anatomy', lesion_mask_name=first, anatomy_mask_name='Right_Lung')
    values['A22'] = invoke('relate_multiple_regions', region_mask_names=[first, second, 'Heart'])
    values['A23'] = invoke('summarize_bilateral_distribution', lesion_mask_names=[first, second],
                           expected_refusal=bool(lung_overlap))
    invoke('measure_region_area', mask_name='AbsentMask', expected_refusal=True)

    # Recompute intermediate arithmetic directly, without calling geometry helpers.
    if values['A9']:
        heart = session.masks['Heart']
        yy, xx = np.nonzero(heart)
        row_widths = [int(np.ptp(np.flatnonzero(row)) + 1) if row.any() else 0 for row in heart]
        check('A9:bbox_dimensions', [values['A9']['bbox_width_px'], values['A9']['bbox_height_px']],
              [int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)])
        check('A9:max_row_width', values['A9']['max_row_width_px'], max(row_widths))
        # Feret over pixel-cell corners via SciPy Qhull (the tool uses OpenCV hulls).
        corners = np.concatenate([np.stack([xx + dx, yy + dy], 1)
                                  for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])
        hull_points = corners[ConvexHull(corners).vertices]
        feret = float(np.sqrt(((hull_points[:, None] - hull_points[None]) ** 2).sum(-1)).max())
        check('A9:feret_pixel_cell_qhull', values['A9']['feret_max_px'], feret)
        np.save(out / 'heart_row_widths.npy', np.array(row_widths))
    if values['A10']:
        check('A10:foreground_count', values['A10']['area_px'], int(np.count_nonzero(a)))
    if values['A11']:
        coords_a, coords_b = np.argwhere(a), np.argwhere(b)
        distances, neighbors = cKDTree(coords_b).query(coords_a)
        closest = int(distances.argmin())
        intermediates['A11_independent_kdtree'] = {
            'point_a_yx': coords_a[closest].tolist(), 'point_b_yx': coords_b[neighbors[closest]].tolist(),
            'minimum_distance_px': float(distances[closest]), 'reference_method': 'cKDTree, not EDT'}
        check('A11:independent_nearest_points', values['A11']['pixel_center_distance_px'], float(distances.min()))
    if values['A13']:
        check('A13:lung_areas', [values['A13']['right_area_px'], values['A13']['left_area_px']],
              [int(session.masks['Right_Lung'].sum()), int(session.masks['Left_Lung'].sum())])
    if values['A14']:
        padded = np.pad(a.astype(np.int8), 1)
        perimeter = int(np.count_nonzero(np.diff(padded, axis=0)) + np.count_nonzero(np.diff(padded, axis=1)))
        check('A14:perimeter_by_exposed_pixel_edges', sum(c['perimeter_px'] for c in values['A14']['components']), perimeter)
        intermediates['A14_exposed_pixel_edge_count'] = perimeter
    if values['A15']:
        samples = gray[a]
        total, squares = float(samples.sum()), float((samples * samples).sum())
        mean = total / len(samples)
        variance = squares / len(samples) - mean * mean
        check('A15:sum_over_count', values['A15']['mean'], mean)
        check('A15:second_moment_variance', values['A15']['variance_population'], variance)
        intermediates['A15_sums'] = {'count': len(samples), 'sum': total, 'sum_of_squares': squares,
                                     'mean': mean, 'variance_population': variance}
    if values['A16']:
        pair_counts = {}
        for name, (dy, dx) in values['A16']['directions'].items():
            y0, y1, x0, x1 = max(0, -dy), min(h, h-dy), max(0, -dx), min(w, w-dx)
            count = int((a[y0:y1, x0:x1] & a[y0+dy:y1+dy, x0+dx:x1+dx]).sum())
            pair_counts[name] = count
            check('A16:ROI_neighbor_count:' + name, values['A16']['directional_features'][name]['valid_pair_count'], count)
        intermediates['A16_pair_counts'] = pair_counts
    if values['A17']:
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8
        gx, gy = correlate(gray, kernel, mode='reflect'), correlate(gray, kernel.T, mode='reflect')
        energy = float((gx[a] ** 2 + gy[a] ** 2).mean())
        check('A17:independent_scipy_convolution', values['A17']['tenengrad_mean'], energy)
        intermediates['A17_scipy_energy'] = energy
    if values['A19']:
        right, left = session.masks['Right_Lung'], session.masks['Left_Lung']
        midline = values['A19']['midline_x']
        yy, xx = np.nonzero(right)
        reflected_x = np.rint(2 * midline - xx).astype(int)
        valid = (reflected_x >= 0) & (reflected_x < w)
        intersection = int(left[yy[valid], reflected_x[valid]].sum())
        denominator = int(right.sum() + left.sum())
        check('A19:explicit_point_reflection', values['A19']['mirror_dice'], 2 * intersection / denominator)
        intermediates['A19_mirror_arithmetic'] = {'midline_x': midline, 'intersection_px': intersection,
            'denominator_original_mask_areas': denominator, 'out_of_frame_px': int((~valid).sum())}
    if values['A20']:
        for side in ['right', 'left']:
            lung = session.masks[side.title() + '_Lung']
            rows_lung = np.flatnonzero(lung.any(axis=1))
            fractions = (np.arange(h) - rows_lung.min() + 0.5) / (rows_lung.max() - rows_lung.min() + 1)
            bands = np.floor(3 * fractions).astype(int)
            expected = [int((a & lung & (bands[:, None] == band)).sum()) for band in range(3)]
            check('A20:pixel_center_row_thirds:' + side,
                  list(values['A20'][side]['zone_overlap_px'].values()), expected)
    if values['A21']:
        check('A21:direct_intersection', values['A21']['intersection_px'], int((a & session.masks['Right_Lung']).sum()))
    if values['A22'] and values['A11']:
        relation = values['A22']['relations'][0]
        check('A22:distance_consistency', relation['pixel_center_distance_px'], values['A11']['pixel_center_distance_px'])
        check('A22:direct_IoU', relation['iou'], int((a & b).sum()) / int((a | b).sum()))
        check('A22:gap_consistency', relation['gap_distance_px'], values['A11']['gap_distance_px'])
    if values['A23']:
        check('A23:deduplicated_candidate_union', values['A23']['union_region_area_px'], int((a | b).sum()))
        for side in ['right', 'left']:
            check('A23:union_overlap:' + side, values['A23']['union_' + side + '_overlap_px'],
                  int(((a | b) & session.masks[side.title() + '_Lung']).sum()))

    intermediates['predicted_lung_overlap_px'] = lung_overlap
    model_values = verify_model_tools(invoke, check, session, out, args.model_device, gray, first, annotation,
                                      intermediates)
    intermediates['mask_pixel_counts'] = {name: int(mask.sum()) for name, mask in session.masks.items()}
    write_json(out / 'intermediates.json', intermediates)
    write_json(out / 'checks.json', checks)
    session.save(out / 'session.json')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
    panels = [(['Right_Lung', 'Left_Lung'], ['cyan', 'yellow'], 'A1: predicted lungs'),
              (['Heart', 'Mediastinum'], ['red', 'lime'], 'A2/A3: predicted heart / mediastinum'),
              (['Left_Clavicle', 'Right_Clavicle', 'Left_Scapula', 'Right_Scapula'],
               ['cyan', 'yellow', 'red', 'lime'], 'A4: predicted clavicles / scapulae'),
              (supplied_names, ['magenta', 'orange', 'lime'], 'Supplied polygons: NOT detector predictions')]
    for ax, (names, colors, title) in zip(axes.flat, panels):
        ax.imshow(gray, cmap='gray')
        handles = []
        for name, color in zip(names, colors):
            if name in session.masks:
                ax.contour(session.masks[name], levels=[0.5], colors=[color], linewidths=0.7)
                handles.append(Line2D([], [], color=color, label=name))
        ax.legend(handles=handles, fontsize=6, loc='upper right')
        ax.set_title(title, fontsize=10)
    axes[1, 1].imshow(gray, cmap='gray')
    for side, color in [('Right', 'cyan'), ('Left', 'yellow')]:
        if side + '_Lung' in session.masks:
            lung = session.masks[side + '_Lung']
            yy, xx = np.nonzero(lung)
            for fraction in [1/3, 2/3]:
                y = yy.min() - 0.5 + fraction * (yy.max() - yy.min() + 1)
                axes[1, 1].plot([xx.min(), xx.max()], [y, y], color=color, linestyle='--')
    axes[1, 1].contour(a, levels=[0.5], colors=['magenta'], linewidths=0.8)
    axes[1, 1].set_title('A20: geometric thirds; supplied consolidation ROI', fontsize=10)
    if session.mask_metadata.get('Right_Lung'):
        x0, y0, x1, y1 = session.mask_metadata['Right_Lung']['valid_fov_bbox_xyxy_exclusive']
        axes[0, 0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='white', linestyle=':'))
    axes[1, 2].axis('off')
    lines = ['ENGINEERING CHECK ONLY', f'Image: {image.name}, {w} x {h}',
             f'A interfaces: {len(registry.describe_all())}', f'Checks passed: {sum(c["passed"] for c in checks)}/{len(checks)}',
             f'Predicted lung overlap: {lung_overlap} px', '', 'Real-image CTR: not available',
             'Synthetic ratio: 70 / 100 = 0.70', '', 'Organ masks: model predictions',
             'Regions: supplied dataset polygons', 'No independent medical accuracy result',
             'Laterality follows model target names']
    axes[1, 2].text(0, 1, '\n'.join(lines), va='top', fontsize=11)
    for ax in axes.flat:
        if ax is not axes[1, 2]:
            ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
    fig.suptitle('A-tool implementation verification on a development image', fontsize=14)
    fig.savefig(out / 'overlay.png', dpi=120)
    plt.close(fig)
    plot_model_overlay(out / 'overlay_models.png', gray, session, model_values, annotation)
    summary = {'implemented_a_interfaces': len(registry.describe_all()), 'tool_call_count': len(calls),
               'successful_calls': sum(c['status'] == 'ok' for c in calls),
               'refused_calls': [c['tool'] for c in calls if c['status'] == 'refusal'],
               'checks_passed': sum(c['passed'] for c in checks), 'checks_total': len(checks),
               'unexpected_failures': [c for c in checks if not c['passed']],
               'medical_accuracy_evaluated': False, 'limitations': limits,
               'overlay': str(out / 'overlay.png')}
    write_json(out / 'verification_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['limitations', 'unexpected_failures']}, ensure_ascii=False), flush=True)
    return 1 if summary['unexpected_failures'] else 0


def verify_model_tools(invoke, check, session, out, device, gray, roi_name, annotation, intermediates):
    """A5-A8, A12 auto, A18, A24-A26 through isolated workers, with independent recomputation."""
    from visevi.perception import structures, vlm
    from visevi.perception.landmarks import SEMANTIC_NODES

    h, w = gray.shape
    values = {}
    values['A5'] = invoke('locate_lung_landmarks', output_dir=str(out / 'models/a5'), device=device,
                          uncertainty_samples=10, seed=0)
    if values['A5']:
        nodes = np.asarray(session.landmarks['lm_px'])
        units = {u.finding_or_anatomy: u.value for u in session.graph.nodes.values()
                 if u.source_tool == 'locate_lung_landmarks' and 'point_xy' in u.value}
        check('A5:points_equal_frozen_nodes', [units[k]['point_xy'] for k in SEMANTIC_NODES],
              [nodes[i].tolist() for i in SEMANTIC_NODES.values()])
        check('A5:apex_is_topmost_node_of_each_lung',
              [int(nodes[0:44, 1].argmin()), int(nodes[44:94, 1].argmin()) + 44], [0, 44])
        check('A5:right_lung_on_image_left', bool(nodes[0:44, 0].mean() < nodes[44:94, 0].mean()), True)
        values['A5_nodes'] = nodes
    values['A6'] = invoke('detect_abnormalities', output_dir=str(out / 'models/chexone_a6'), device=device,
                          findings=['cardiomegaly', 'consolidation', 'pneumothorax', 'nodule'])
    a6 = [u.value for u in session.graph.nodes.values() if u.source_tool == 'detect_abnormalities']
    for item in a6:
        name = 'Candidate_' + item['finding'].replace(' ', '_')
        if item['boxes_xyxy_px']:
            expected = np.zeros((h, w), bool)
            for x0, y0, x1, y1 in item['boxes_xyxy_px']:
                expected[int(np.ceil(y0 - 0.5)):int(np.ceil(y1 - 0.5)), int(np.ceil(x0 - 0.5)):int(np.ceil(x1 - 0.5))] = True
            check(f'A6:box_raster_independent:{item["finding"]}', int(session.masks[name].sum()), int(expected.sum()))
    values['A6_items'] = a6
    values['A7'] = invoke('ground_phrase', output_dir=str(out / 'models/chexone_a7'), device=device,
                          phrase='enlarged cardiac silhouette')
    values['A8'] = invoke('segment_pneumothorax', output_dir=str(out / 'models/a8'), device=device)
    if values['A8'] and 'Pneumothorax_Candidate' in session.masks:
        saved = np.asarray(Image.open(out / 'models/a8/pneumothorax_mask.png')) > 0
        check('A8:attached_mask_equals_artifact', int((session.masks['Pneumothorax_Candidate'] ^ saved).sum()), 0)
    values['A12'] = invoke('measure_ctr', output_dir=str(out / 'models/cxas'), device=device)
    if values['A12']:
        v = values['A12']
        hb, tb = v['heart_borders_x'], v['thoracic_inner_borders_x']
        check('A12:ratio_from_reported_borders', v['CTR'], (hb[1] - hb[0]) / (tb[1] - tb[0]))
        heart = session.masks['Heart']
        xs = np.flatnonzero(heart.any(axis=0))
        check('A12:heart_borders_are_mask_cell_edges', hb, [xs[0] - 0.5, xs[-1] + 0.5])
        record = json.loads((out / 'models/cxas/cache_record.json').read_text())
        data = np.load(out / 'models/cxas/labels_model_frame.npz')
        bits = np.unpackbits(data['packed'], axis=-1)[..., :512].astype(bool)
        index = {n: int(k) for k, n in record['result']['labels'].items()}
        y = v['thoracic_inner_margin']['row_y']
        src_y = int(np.floor(y * 512 / h))
        right = np.logical_or.reduce([bits[index[n]][src_y] for n in structures.RIB_LABELS['right']])
        left = np.logical_or.reduce([bits[index[n]][src_y] for n in structures.RIB_LABELS['left']])
        col = lambda x: min(int(np.floor(x * 512 / w)), 511)
        rx, lx = tb[0] - 0.5, tb[1] + 0.5      # last rib pixel / first non-rib medial pixel
        check('A12:thoracic_margin_is_a_rib_edge_in_raw_cxas_bits',
              [bool(right[col(rx)]), bool(left[col(lx)])], [True, True])
        intermediates['A12_thoracic_margin'] = v['thoracic_inner_margin']
    values['A18'] = invoke('classify_roi_attributes', output_dir=str(out / 'models/chexone_a18'), device=device,
                           roi_mask_name=roi_name, attributes=['consolidation', 'pneumothorax', 'cavitation'])
    values['A24'] = invoke('classify_view', output_dir=str(out / 'models/chexone_a24'), device=device)
    values['A25'] = invoke('assess_image_quality', output_dir=str(out / 'models/cxas'), device=device)
    quality = {u.finding_or_anatomy: u.value for u in session.graph.nodes.values()
               if u.source_tool == 'assess_image_quality'}
    if quality.get('rotation', {}).get('status') == 'ok':
        r = quality['rotation']
        check('A25:asymmetry_index_from_distances', r['asymmetry_index'],
              (r['distance_right_px'] - r['distance_left_px']) / (r['distance_right_px'] + r['distance_left_px']))
    values['A25_items'] = quality
    values['A26'] = invoke('localize_devices', output_dir=str(out / 'models/a26'), device=device)
    devices = {u.finding_or_anatomy: u.value for u in session.graph.nodes.values() if u.source_tool == 'localize_devices'}
    for key, item in devices.items():
        if item.get('status') == 'present':
            saved = np.asarray(Image.open(out / 'models/a26' / f'line_{item["device_family"].replace(" ", "_")}.png')) > 0
            check(f'A26:asserted_mask_equals_artifact:{key}', int((session.masks[f'Device_{key}'] ^ saved).sum()), 0)
    values['A26_items'] = devices
    intermediates['model_tool_values'] = {k: v for k, v in values.items() if k.startswith('A') and '_' not in k}
    return values


def plot_model_overlay(path, gray, session, values, annotation):
    h, w = gray.shape
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), layout='constrained')
    for ax in axes.flat:
        ax.imshow(gray, cmap='gray'); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
    nodes = values.get('A5_nodes')
    if nodes is not None:
        for (a, b), color in [((0, 44), 'cyan'), ((44, 94), 'yellow'), ((94, 120), 'red')]:
            axes[0, 0].plot(*np.vstack([nodes[a:b], nodes[a:a + 1]]).T, color=color, lw=0.8)
        for u in session.graph.nodes.values():
            if u.source_tool == 'locate_lung_landmarks' and u.value.get('point_xy'):
                axes[0, 0].plot(*u.value['point_xy'], 'o', color='lime', ms=5)
                axes[0, 0].annotate(u.finding_or_anatomy.replace('_', ' '), u.value['point_xy'], color='lime', fontsize=6)
    axes[0, 0].set_title('A5 landmarks (CheXmask-U); green = frozen semantic nodes')
    for item, color in zip(values.get('A6_items', []), ['red', 'orange', 'cyan', 'magenta']):
        for x0, y0, x1, y1 in item['boxes_xyxy_px']:
            axes[0, 1].add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, color=color))
            axes[0, 1].annotate(item['finding'], (x0, y0), color=color, fontsize=7)
    for sym, box in zip(annotation['syms'], annotation['boxes']):
        axes[0, 1].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                                       color='white', linestyle=':'))
        axes[0, 1].annotate('ref ' + sym, (box[0], box[3]), color='white', fontsize=6)
    if 'Phrase_Region' in session.masks:
        axes[0, 1].contour(session.masks['Phrase_Region'], [0.5], colors=['lime'], linewidths=0.8)
    axes[0, 1].set_title('A6 boxes (colour), A7 phrase (green), dataset boxes (white dotted)')
    if 'Pneumothorax_Candidate' in session.masks:
        axes[0, 2].contour(session.masks['Pneumothorax_Candidate'], [0.5], colors=['red'])
    for poly, sym in zip(annotation['polygons'], annotation['syms']):
        if sym == 'Pneumothorax':
            axes[0, 2].plot(*np.vstack([poly, poly[:1]]).T, color='white', linestyle=':')
    axes[0, 2].set_title('A8 pneumothorax mask (red); dataset polygon (white dotted)')
    a12 = values.get('A12')
    if a12:
        y = a12['thoracic_inner_margin']['row_y']
        axes[1, 0].plot(a12['thoracic_inner_borders_x'], [y, y], color='cyan', lw=1.5)
        hb = a12['heart_borders_x']
        axes[1, 0].axvline(hb[0], color='red', lw=0.8); axes[1, 0].axvline(hb[1], color='red', lw=0.8)
        axes[1, 0].contour(session.masks['Heart'], [0.5], colors=['red'], linewidths=0.6)
        axes[1, 0].axhline(a12['right_hemidiaphragm_dome_y'], color='orange', linestyle='--', lw=0.8)
        axes[1, 0].set_title(f'A12 CTR = {a12["CTR"]:.3f} (cyan: rib inner margins; orange: R dome)')
    rot = values.get('A25_items', {}).get('rotation', {})
    if rot.get('status') == 'ok':
        for key, color in [('right_medial_end_xy', 'cyan'), ('left_medial_end_xy', 'yellow')]:
            axes[1, 1].plot(*rot[key], 'o', color=color)
        for x, (_, yy) in zip(rot['spine_midline_x_at_rows'], [rot['right_medial_end_xy'], rot['left_medial_end_xy']]):
            axes[1, 1].plot([x], [yy], 'x', color='lime')
        insp = values['A25_items'].get('inspiration', {})
        if insp.get('status') == 'ok':
            axes[1, 1].plot(*insp['right_hemidiaphragm_dome_xy'], 's', color='orange')
        axes[1, 1].set_title(f'A25 rotation index {rot["asymmetry_index"]:.3f}; posterior ribs above dome '
                             f'{insp.get("posterior", {}).get("highest_number_above")}')
    for key, item in values.get('A26_items', {}).items():
        if item.get('device_family'):
            mask_name = f'Device_{key}'
            if mask_name in session.masks:
                axes[1, 2].contour(session.masks[mask_name], [0.5], colors=['lime'])
            if item.get('distal_tip_xy'):
                axes[1, 2].plot(*item['distal_tip_xy'], '^', color='lime' if item['status'] == 'present' else 'grey')
                axes[1, 2].annotate(f'{item["device_family"]} {item["presence_raw_score"]:.2f}', item['distal_tip_xy'],
                                    color='white', fontsize=6)
    carina = values.get('A26_items', {}).get('carina', {}).get('point_xy')
    if carina:
        axes[1, 2].plot(*carina, '*', color='red', ms=10)
    axes[1, 2].set_title('A26 devices (green = asserted, grey tip = below screen), carina (red star)')
    fig.suptitle('Model-backed A tools on a development image: predictions, not ground truth')
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
