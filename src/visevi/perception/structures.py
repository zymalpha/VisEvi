"""CXAS-backed structure measurements: A12 automatic CTR and A25 image quality.

The CXAS worker returns 159 PAX-Ray++ label masks in its 512x512 model frame.
All geometric rules below run here, deterministically, on masks mapped back to
the original pixel frame. Every output is a raw quantity with its definition;
no clinical grade or threshold is applied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backends import Backend
from ..core.units import ToolRefusal, fov_refusal
from . import geometry
from .model_tools import ModelTool, Output, image_size, unit

CXAS_BACKEND = Backend('cxas', 'cxas', {'adapter': 1}, source='cxas')
ORDINALS = {1: '1st', 2: '2nd', 3: '3rd'}


def rib_label(kind: str, number: int, side: str) -> str:
    """CXAS label, e.g. rib_label('posterior', 9, 'right') -> 'posterior 9th rib right'."""
    return f'{kind} {ORDINALS.get(number, f"{number}th")} rib {side}'


# PAX-Ray++ numbers anterior ribs 1-11 and posterior ribs 1-12.
RIB_LABELS = {side: [rib_label('anterior', n, side) for n in range(1, 12)] +
                    [rib_label('posterior', n, side) for n in range(1, 13)]
              for side in ('right', 'left')}


def load_labels(run: dict, output_dir: str, names, width: int, height: int) -> dict[str, np.ndarray]:
    """Map CXAS model-frame masks to the original frame (torch 'nearest' indexing)."""
    if 'labels_model_frame.npz' not in run['artifacts']:
        raise ValueError('CXAS artifact missing')
    data = np.load(Path(output_dir) / 'labels_model_frame.npz')
    shape = tuple(int(v) for v in data['shape'])
    masks = np.unpackbits(data['packed'], axis=-1)[..., :shape[2]].astype(bool)
    index = {name: int(key) for key, name in run['result']['labels'].items()}
    rows = np.minimum(np.floor(np.arange(height) * shape[1] / height).astype(int), shape[1] - 1)
    cols = np.minimum(np.floor(np.arange(width) * shape[2] / width).astype(int), shape[2] - 1)
    out = {}
    for name in names:
        if name not in index:
            raise KeyError(f'unknown CXAS label {name}')
        out[name] = masks[index[name]][np.ix_(rows, cols)]
    return out


def union(masks: dict, names) -> np.ndarray:
    return np.logical_or.reduce([masks[name] for name in names if name in masks])


def spine_midline_x(spine: np.ndarray, y: int) -> float | None:
    xs = np.flatnonzero(spine[y])
    return float(xs.mean()) if xs.size else None


def thoracic_inner_margin(right_ribs: np.ndarray, left_ribs: np.ndarray, midline: float,
                          max_row: int, min_rows: int = 5) -> dict:
    """Widest internal thoracic diameter between inner edges of the lateral rib arcs.

    On each row the outermost rib run on each side is followed medially to its
    last rib pixel; that pixel's medial cell edge is the inner margin. Rows above
    ``max_row`` (inclusive) where both sides have a margin are candidates; the
    widest one is reported. Right ribs are on the image left.
    """
    rows = []
    for y in range(0, min(max_row, right_ribs.shape[0] - 1) + 1):
        right = np.flatnonzero(right_ribs[y, :int(np.floor(midline))])
        left_x0 = int(np.ceil(midline))
        left = np.flatnonzero(left_ribs[y, left_x0:]) + left_x0
        if not right.size or not left.size:
            continue
        r = right[0]
        while r + 1 < right_ribs.shape[1] and right_ribs[y, r + 1]:
            r += 1
        l_ = left[-1]
        while l_ - 1 >= 0 and left_ribs[y, l_ - 1]:
            l_ -= 1
        inner_right, inner_left = r + 0.5, l_ - 0.5
        if inner_left > inner_right:
            rows.append((y, inner_right, inner_left))
    if len(rows) < min_rows:
        raise ValueError(f'only {len(rows)} rows have rib inner margins on both sides (need {min_rows})')
    y, right_x, left_x = max(rows, key=lambda item: item[2] - item[1])
    return {'row_y': int(y), 'right_inner_x': float(right_x), 'left_inner_x': float(left_x),
            'width_px': float(left_x - right_x), 'candidate_rows': len(rows),
            'definition': 'medial cell edge of the outermost rib run per side; widest row at or above '
                          'the right hemidiaphragm dome'}


def dome(mask: np.ndarray) -> tuple[int, float]:
    """Top row of a hemidiaphragm mask and the mean x of that row."""
    ys = np.flatnonzero(mask.any(axis=1))
    if not ys.size:
        raise ValueError('hemidiaphragm not segmented')
    y = int(ys[0])
    return y, float(np.flatnonzero(mask[y]).mean())


def ribs_above(masks: dict, kind: str, side: str, x: float, y_limit: int, half_window: int) -> dict:
    """Ribs whose centre within a column window around ``x`` lies above ``y_limit``."""
    x0, x1 = max(0, int(round(x)) - half_window), int(round(x)) + half_window + 1
    above, found = [], {}
    for n in range(1, 13):
        name = rib_label(kind, n, side)
        if name not in masks:
            continue
        ys = np.nonzero(masks[name][:, x0:x1])[0]
        if ys.size:
            found[n] = float(ys.mean())
            if ys.mean() < y_limit:
                above.append(n)
    return {'count_above': len(above), 'ribs_above': above, 'highest_number_above': max(above) if above else None,
            'rib_centre_y_at_column': found,
            'column_window_x': [x0, x1 - 1]}


def rotation(clavicle_right: np.ndarray, clavicle_left: np.ndarray, spine: np.ndarray) -> dict:
    """Medial clavicle ends vs. the thoracic-spine midline on the same row."""
    ry, rx = np.nonzero(clavicle_right)
    ly, lx = np.nonzero(clavicle_left)
    if not rx.size or not lx.size:
        raise ValueError('clavicle not segmented')
    r_index, l_index = int(rx.argmax()), int(lx.argmin())        # medial ends (right = image left)
    r_point, l_point = (float(rx[r_index]), int(ry[r_index])), (float(lx[l_index]), int(ly[l_index]))
    r_mid, l_mid = spine_midline_x(spine, r_point[1]), spine_midline_x(spine, l_point[1])
    if r_mid is None or l_mid is None:
        raise ValueError('spine not segmented at the clavicle rows')
    d_right, d_left = r_mid - (r_point[0] + 0.5), (l_point[0] - 0.5) - l_mid
    if d_right <= 0 or d_left <= 0:
        raise ValueError('a medial clavicle end lies at or beyond the spine midline; rotation index undefined')
    total = d_right + d_left
    return {'right_medial_end_xy': [r_point[0], r_point[1]], 'left_medial_end_xy': [l_point[0], l_point[1]],
            'spine_midline_x_at_rows': [r_mid, l_mid], 'distance_right_px': d_right, 'distance_left_px': d_left,
            'asymmetry_index': (d_right - d_left) / total,
            'definition': '(d_right - d_left) / (d_right + d_left); d = horizontal distance from the medial '
                          'clavicle cell edge to the mean x of the thoracic-spine mask on that row'}


def penetration(gray: np.ndarray, heart: np.ndarray, spine: np.ndarray) -> dict:
    behind = heart & spine
    beside = heart & ~spine
    if behind.sum() < 50 or beside.sum() < 50:
        raise ValueError('too few heart/spine pixels for a retrocardiac contrast')
    a, b = float(gray[behind].mean()), float(gray[beside].mean())
    return {'retrocardiac_spine_mean': a, 'heart_without_spine_mean': b,
            'difference': a - b, 'pixels': [int(behind.sum()), int(beside.sum())],
            'definition': 'display-grey contrast of the thoracic spine behind the heart vs the rest of the heart; '
                          'a visibility proxy, not an exposure dose'}


def field_of_view(masks: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, mask in masks.items():
        result[name] = {'touches_top': bool(mask[0].any()), 'touches_bottom': bool(mask[-1].any()),
                        'touches_left': bool(mask[:, 0].any()), 'touches_right': bool(mask[:, -1].any()),
                        'segmented': bool(mask.any())}
    return result


def _structure_condition():
    return ['CXAS UNet-ResNet50 support structures (PAX-Ray++ labels); predictions, not ground truth',
            'patient right = image left (CXAS label convention, consistent with PSPNet)']


class MeasureCTR(ModelTool):
    name = 'measure_ctr'
    capability = 'A12'
    description = ('automatic CTR: heart width from a named heart mask (default A2 "Heart") over the '
                   'widest internal thoracic diameter between CXAS rib inner margins at or above the '
                   'right hemidiaphragm dome; projection (AP/PA) is not assessed')
    backend = CXAS_BACKEND
    extra_properties = {'heart_mask_name': {'type': 'string'}}

    def backend_params(self, session, params):
        name = params.get('heart_mask_name', 'Heart')
        if not isinstance(name, str) or session.masks.get(name) is None:
            return self.refuse('heart mask not available; run segment_heart (A2) first',
                               missing=[f'mask:{name}'])
        truncated = session.fov_truncated([name])
        if truncated:
            return fov_refusal(self.name, truncated)
        return {}

    def convert(self, session, run, params):
        width, height = image_size(session)
        heart_name = params.get('heart_mask_name', 'Heart')
        heart = geometry._nonempty(session.masks[heart_name], heart_name)
        if heart.shape != (height, width):
            raise ValueError('heart mask is not in the session image frame')
        names = RIB_LABELS['right'] + RIB_LABELS['left'] + ['right hemidiaphragm', 'thoracic spine']
        masks = load_labels(run, params['output_dir'], names, width, height)
        dome_y, _ = dome(masks['right hemidiaphragm'])
        spine_x = np.nonzero(masks['thoracic spine'])[1]
        if not spine_x.size:
            raise ValueError('thoracic spine not segmented; midline unavailable')
        midline = float(np.median(spine_x))
        thorax = thoracic_inner_margin(union(masks, RIB_LABELS['right']), union(masks, RIB_LABELS['left']),
                                       midline, dome_y)
        bounds = geometry.bbox(heart)
        try:
            ctr = geometry.ctr_from_borders((bounds['x0'] - 0.5, bounds['x1'] + 0.5),
                                            (thorax['right_inner_x'], thorax['left_inner_x']))
        except ValueError as exc:
            return self.refuse(f'inconsistent borders: {exc}')
        value = {**ctr, 'thoracic_inner_margin': thorax, 'right_hemidiaphragm_dome_y': dome_y,
                 'midline_x_spine_median': midline, 'heart_mask_name': heart_name,
                 'heart_boundary_source': f'{heart_name} mask pixel-cell extent (Danzer transverse diameter)',
                 'thoracic_boundary_source': 'CXAS rib masks (this tool)',
                 'scope': 'automatic ratio; AP/PA magnification and patient positioning not assessed'}
        item = unit('cardiothoracic_ratio', 'measurement', value,
                    provenance=session.mask_provenance([heart_name]) + [f'cached:CXAS:{run["binding"]["image_sha256"][:16]}'],
                    validity_conditions=_structure_condition() + [
                        'thoracic margin from predicted ribs; rib-inner-margin accuracy not validated'])
        return Output(units=[item])


class AssessImageQuality(ModelTool):
    name = 'assess_image_quality'
    capability = 'A25'
    description = ('raw technical-quality measurements from CXAS structures: clavicle–spine rotation '
                   'asymmetry, ribs above the right hemidiaphragm (inspiration), retrocardiac spine '
                   'contrast (penetration proxy) and field-of-view border contact; no grades')
    backend = CXAS_BACKEND

    def convert(self, session, run, params):
        from .appearance import load_grayscale

        width, height = image_size(session)
        names = (RIB_LABELS['right'] + RIB_LABELS['left'] + ['right hemidiaphragm', 'left hemidiaphragm', 'thoracic spine',
                                        'clavicle right', 'clavicle left', 'heart', 'right lung', 'left lung'])
        masks = load_labels(run, params['output_dir'], names, width, height)
        provenance = [f'cached:CXAS:{run["binding"]["image_sha256"][:16]}']
        units = []

        def item(finding, compute, extra_conditions=()):
            try:
                value, status = compute(), 'ok'
            except ValueError as exc:
                value, status = {'reason': str(exc)}, 'unknown'
            value.update(status=status, grade=None)
            units.append(unit(finding, 'global_condition', value, provenance=list(provenance),
                              validity_conditions=_structure_condition() + list(extra_conditions)))

        item('rotation', lambda: rotation(masks['clavicle right'], masks['clavicle left'], masks['thoracic spine']),
             ['spine midline approximates the spinous-process line'])

        def inspiration():
            y, x = dome(masks['right hemidiaphragm'])
            window = max(2, width // 100)
            return {'right_hemidiaphragm_dome_xy': [x, y],
                    'posterior': ribs_above(masks, 'posterior', 'right', x, y, window),
                    'anterior': ribs_above(masks, 'anterior', 'right', x, y, window),
                    'definition': 'right ribs whose segmented centre in a narrow column at the dome x lies above '
                                  'the dome; a rib-count proxy of inspiration'}
        item('inspiration', inspiration, ['rib numbering comes from the CXAS labels'])

        gray, decoding = load_grayscale(session.image_ref)
        item('penetration', lambda: {**penetration(gray, masks['heart'], masks['thoracic spine']),
                                     'image_decoding': decoding},
             ['display grey levels depend on post-processing; not comparable across sources without calibration'])
        item('field_of_view', lambda: field_of_view({k: masks[k] for k in
                                                     ('right lung', 'left lung', 'right hemidiaphragm',
                                                      'left hemidiaphragm')} | {
                                                     'ribs': union(masks, RIB_LABELS['right'] + RIB_LABELS['left'])}),
             ['border contact suggests clipped anatomy; a structure may also be missing from the segmentation'])
        return Output(units=units)
