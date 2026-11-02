"""Evidence programs for image-quality functions Q1--Q3.

The outputs are raw research measurements, not clinical quality grades.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Real
from pathlib import Path

import numpy as np

from ..backends import inspect_image
from ..construction.evidence import add_relation


def _units(outputs, tool, target=None):
    found = [u for values in outputs.values() for u in values
             if u.source_tool == tool and (target is None or u.finding_or_anatomy == target)]
    if len(found) != 1:
        raise ValueError(f"expected exactly one {tool} observation for {target}")
    return found[0]


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _point(point, frame, label):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"{label} must be a point")
    x, y = [_number(x, label) for x in point]
    if not (-0.5 <= x <= frame['width'] - 0.5 and -0.5 <= y <= frame['height'] - 0.5):
        raise ValueError(f"{label} is outside the image")
    return [x, y]


def _frontal(outputs):
    view = _units(outputs, 'classify_view', 'view_position')
    if view.value.get('status') != 'ok' or view.value.get('assumed') is not False:
        raise ValueError('an observed frontal projection is required')
    if view.value.get('view') not in ('PA', 'AP'):
        raise ValueError('a PA or AP projection is required; lateral/unknown cannot be interpreted')
    return view


def _quality(outputs, name):
    node = _units(outputs, 'assess_image_quality', name)
    if node.value.get('status') != 'ok':
        raise ValueError(f'{name} quality observation is missing or unknown')
    if node.value.get('grade') is not None:
        raise ValueError(f'{name} clinical grade must not be asserted')
    return node


def _finish(session, spec, proposition, value, nodes):
    support = list(dict.fromkeys(u.id for u in nodes))
    relation = add_relation(session, proposition, value, input_ids=support,
                            rule_id=f'{spec.function_id}:raw_evidence:{spec.version}')
    return value, relation, support


def q1_program(session, spec, outputs, inputs):
    """Combine lung-base landmarks and raw rib/dome proxy on a frontal image."""
    view = _frontal(outputs)
    frame = inspect_image(session.image_ref)
    nodes, landmarks = [], {}
    for side in ('right', 'left'):
        for structure in ('costophrenic_angle', 'cardiophrenic_angle'):
            name = f'{side}_{structure}'
            node = _units(outputs, 'locate_lung_landmarks', name)
            if node.value.get('status') != 'ok':
                raise ValueError(f'{name} is not localized')
            landmarks[name] = _point(node.value.get('point_xy'), frame, name)
            nodes.append(node)
        name = f'{side}_lung_base_contour'
        node = _units(outputs, 'locate_lung_landmarks', name)
        points = node.value.get('polyline_xy')
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f'{name} must contain at least two points')
        landmarks[name] = [_point(p, frame, name) for p in points]
        nodes.append(node)
    quality = _quality(outputs, 'inspiration')
    q = quality.value
    dome = _point(q.get('right_hemidiaphragm_dome_xy'), frame, 'right hemidiaphragm dome')
    counts = {}
    for name, maximum in (('posterior', 12), ('anterior', 11)):
        data = q.get(name)
        count = data.get('count_above') if isinstance(data, dict) else None
        if type(count) is not int or not 0 <= count <= maximum:
            raise ValueError(f'invalid {name} rib count')
        ribs = data.get('ribs_above')
        if (not isinstance(ribs, list) or len(ribs) != count
                or any(type(n) is not int or not 1 <= n <= maximum for n in ribs)
                or len(set(ribs)) != len(ribs)):
            raise ValueError(f'{name} rib count disagrees with rib labels')
        counts[name] = data
    value = {'lung_base_landmarks': landmarks,
             'diaphragm_geometry': {'right_dome_xy_px': dome,
                                     'source': 'CXAS hemidiaphragm mask; not A5 contour'},
             'rib_counts': counts, 'raw_quality': q,
             'applicability': {'projection': view.value['view'], 'clinical_grade': None,
                               'clinical_interpretation': 'not_assessed',
                               'limitations': ['Predicted lung-base contour and CXAS dome are different structures',
                                               'Rib counts are unvalidated proxies, not an inspiration grade']}}
    return _finish(session, spec, 'inspiration_raw_evidence', value, [view, *nodes, quality])


def q2_program(session, spec, outputs, inputs):
    """Keep PSPNet shoulder context separate from CXAS rotation geometry."""
    view = _frontal(outputs)
    frame = inspect_image(session.image_ref)
    nodes, bones = [], {}
    for name in ('Right_Clavicle', 'Left_Clavicle', 'Right_Scapula', 'Left_Scapula'):
        node = _units(outputs, 'segment_shoulder_bones', name)
        mask = np.asarray(session.masks.get(name), dtype=bool)
        if (session.mask_metadata.get(name, {}).get('observation_id') != node.id
                or mask.shape != (frame['height'], frame['width']) or not mask.any()):
            raise ValueError(f'{name} mask missing or not bound to current observation')
        if session.fov_truncated([name]):
            raise ValueError(f'{name} is truncated by model field of view')
        bones[name] = {'observation_id': node.id, 'area_px': int(mask.sum()),
                       'role': 'context_only_not_medial_endpoint'}
        nodes.append(node)
    quality = _quality(outputs, 'rotation')
    q = quality.value
    right = _point(q.get('right_medial_end_xy'), frame, 'right medial clavicle')
    left = _point(q.get('left_medial_end_xy'), frame, 'left medial clavicle')
    mids = q.get('spine_midline_x_at_rows')
    if not isinstance(mids, (list, tuple)) or len(mids) != 2:
        raise ValueError('two spine midline coordinates are required')
    rm, lm = [_number(x, 'spine midline') for x in mids]
    if not all(-0.5 <= x <= frame['width'] - 0.5 for x in (rm, lm)):
        raise ValueError('spine midline is outside image')
    dr = _number(q.get('distance_right_px'), 'right distance')
    dl = _number(q.get('distance_left_px'), 'left distance')
    index = _number(q.get('asymmetry_index'), 'rotation index')
    if (dr <= 0 or dl <= 0 or abs(index) > 1
            or not math.isclose(dr, rm - (right[0] + .5), rel_tol=1e-6, abs_tol=1e-6)
            or not math.isclose(dl, (left[0] - .5) - lm, rel_tol=1e-6, abs_tol=1e-6)
            or not math.isclose(index, (dr - dl) / (dr + dl), rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError('rotation endpoints, distances, or asymmetry are inconsistent')
    value = {'clavicle_spine_geometry': {'right_medial_end_xy': right,
                                        'left_medial_end_xy': left,
                                        'spine_midline_x_at_rows': [rm, lm],
                                        'distance_right_px': dr, 'distance_left_px': dl,
                                        'source': 'CXAS predicted structures', 'shoulder_context': bones},
             'rotation_index': index, 'raw_quality': q,
             'applicability': {'projection': view.value['view'], 'clinical_grade': None,
                               'clinical_interpretation': 'not_assessed',
                               'limitations': ['PSPNet shoulder masks are context only; endpoints come from CXAS',
                                               'A25 asymmetry is not a validated rotation grade']}}
    optional = [u for vals in outputs.values() for u in vals if u.source_tool == 'measure_region_extent']
    if optional:
        for unit in optional:
            name = unit.value.get('parameters', {}).get('mask_name')
            if name not in bones or bones[name]['observation_id'] not in unit.provenance:
                raise ValueError('optional shoulder extent must be bound to a current A4 mask')
        value['applicability']['optional_bone_extent'] = [u.value for u in optional]
    return _finish(session, spec, 'rotation_raw_evidence', value, [view, *nodes, quality, *optional])


def q3_program(session, spec, outputs, inputs):
    """Combine whole-image display intensity, sharpness and retrocardiac proxy."""
    intensity = _units(outputs, 'summarize_roi_intensity')
    sharpness = _units(outputs, 'measure_roi_sharpness')
    quality = _quality(outputs, 'penetration')
    frame = inspect_image(session.image_ref)
    digest = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
    for unit in (intensity, sharpness):
        if unit.value.get('parameters', {}).get('mask_name') is not None:
            raise ValueError('Q3 requires whole-image appearance measurements')
        if unit.value.get('image_sha256') != digest or unit.value.get('pixel_count') != frame['width'] * frame['height']:
            raise ValueError('appearance measurement is not bound to the full current image')
    iv, sv, q = intensity.value, sharpness.value, quality.value
    low, high, mean = [_number(iv.get(k), k) for k in ('minimum', 'maximum', 'mean')]
    if not low <= mean <= high:
        raise ValueError('grayscale statistics are inconsistent')
    for key in ('std_population', 'variance_population'):
        if _number(iv.get(key), key) < 0:
            raise ValueError(f'{key} must be nonnegative')
    if _number(sv.get('tenengrad_mean'), 'tenengrad') < 0:
        raise ValueError('sharpness must be nonnegative')
    behind, beside, difference = [_number(q.get(k), k) for k in
                                  ('retrocardiac_spine_mean', 'heart_without_spine_mean', 'difference')]
    if not math.isclose(difference, behind - beside, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('retrocardiac contrast disagrees with its component means')
    value = {'grayscale_statistics': iv, 'sharpness_proxy': sv, 'retrocardiac_contrast': q,
             'applicability': {'clinical_grade': None, 'clinical_interpretation': 'not_assessed',
                               'physical_exposure_dose': None,
                               'limitations': ['Display gray and visibility are not physical exposure',
                                               'No quality grade or exposure threshold is validated']}}
    return _finish(session, spec, 'display_quality_raw_evidence', value, [intensity, sharpness, quality])


Q_PROGRAMS = {'Q1': q1_program, 'Q2': q2_program, 'Q3': q3_program}
