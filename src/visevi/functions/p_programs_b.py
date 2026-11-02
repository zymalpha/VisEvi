"""Conservative P7/P8/P10/P11/P12 evidence programs.

These programs expose measurable evidence only; they do not grade disease.
Caller-supplied ROIs for P8/P10 require explicit image-bound provenance.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from ..backends import inspect_image
from ..construction.evidence import add_relation
from ..perception.vlm import box_mask


def _one(outputs, step, tool):
    units = outputs.get(step, [])
    if len(units) != 1 or units[0].source_tool != tool:
        raise ValueError(f"{step} requires exactly one {tool} observation")
    return units[0]


def _mask(session, name):
    frame = inspect_image(session.image_ref)
    raw = session.masks.get(name)
    if raw is None:
        raise ValueError(f"required mask {name} is missing")
    mask = np.asarray(raw)
    if mask.shape != (frame['height'], frame['width']) or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"{name} mask is invalid or outside the original image frame")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError(f"{name} mask is empty")
    if session.fov_truncated([name]):
        raise ValueError(f"{name} mask is truncated by model field of view")
    meta = session.mask_metadata.get(name, {})
    owner = meta.get('observation_id')
    if not owner or owner not in session.graph.nodes or session.graph.nodes[owner].category != 'observation':
        raise ValueError(f"{name} mask lacks an observation owner")
    if meta.get('coordinate_frame') != 'original_image_pixels':
        raise ValueError(f"{name} mask coordinate frame is not verified")
    if not isinstance(meta.get('fov_truncated_sides'), list):
        raise ValueError(f"{name} mask lacks a field-of-view completeness record")
    return mask, meta, owner


def _external_roi(session, name, anatomy):
    mask, meta, owner = _mask(session, name)
    image_sha = hashlib.sha256(Path(session.image_ref).read_bytes()).hexdigest()
    if (meta.get('image_sha256') != image_sha or meta.get('anatomy') != anatomy
            or not isinstance(meta.get('source'), str) or not meta['source'].strip()
            or session.graph.nodes[owner].finding_or_anatomy != anatomy
            or not session.graph.nodes[owner].source_tool):
        raise ValueError(f"{name} requires explicit {anatomy} ROI source and current image SHA256")
    return mask, meta, owner


def _metric(outputs, session, step, tool, masks, parameters=None):
    unit = _one(outputs, step, tool)
    for name in masks:
        mask, meta, owner = _mask(session, name)
        record = unit.value.get('input_masks', {}).get(name, {})
        digest = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        if (owner not in unit.provenance or record.get('binary_sha256') != digest
                or record.get('source') != meta):
            raise ValueError(f"{step} is not bound to current {name} mask and owner")
    for key, expected in (parameters or {}).items():
        if unit.value.get('parameters', {}).get(key) != expected:
            raise ValueError(f"{step} parameter {key} disagrees with intended evidence")
    return unit


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finish(session, spec, label, value, units):
    ids = list(dict.fromkeys(u.id for u in units))
    relation = add_relation(session, label, value, input_ids=ids,
                            rule_id=f'{spec.function_id}:evidence:{spec.version}')
    return value, relation, ids


def p7_program(session, spec, outputs, inputs):
    mediastinum = _one(outputs, 'mediastinum', 'segment_mediastinum')
    mask, _, owner = _mask(session, 'Mediastinum')
    if owner != mediastinum.id:
        raise ValueError('mediastinum mask is not bound to current segmenter output')
    extent = _metric(outputs, session, 'extent', 'measure_region_extent', ['Mediastinum'],
                     {'mask_name': 'Mediastinum'})
    width = _number(extent.value.get('bbox_width_px'), 'bbox_width_px', positive=True)
    ys, xs = np.nonzero(mask)
    bbox = {'x0': int(xs.min()), 'y0': int(ys.min()), 'x1': int(xs.max()), 'y1': int(ys.max()),
            'w': int(xs.max() - xs.min() + 1), 'h': int(ys.max() - ys.min() + 1)}
    if width != bbox['w'] or extent.value.get('bbox_inclusive_px') != bbox:
        raise ValueError('mediastinal width or bounding box contradicts current mask bounds')
    value = {'mediastinal_boundary': {'mask_name': 'Mediastinum', 'observation_id': owner,
                                      'bbox_inclusive_px': bbox},
             'width_px': width,
             'measurement_convention': 'inclusive bounding-box width, pixel count; not a row-specific clinical width',
             'applicability': {'clinical_interpretation': 'not_assessed',
                               'physical_size_available': False,
                               'projection_assessed': False}}
    return _finish(session, spec, 'mediastinal_bbox_extent', value, [mediastinum, extent])


def p8_program(session, spec, outputs, inputs):
    right, left = inputs.get('right_hilum_mask_name'), inputs.get('left_hilum_mask_name')
    if not isinstance(right, str) or not isinstance(left, str) or right == left:
        raise ValueError('P8 requires distinct right/left hilar mask names')
    if inputs.get('hilum_mask_names') != [right, left]:
        raise ValueError('hilum_mask_names must match right then left hilar mask')
    rmask, rmeta, rowner = _external_roi(session, right, 'right_hilum')
    lmask, lmeta, lowner = _external_roi(session, left, 'left_hilum')
    if np.any(rmask & lmask):
        raise ValueError('right and left hilar ROIs overlap')
    lung_units = outputs.get('lungs', [])
    lung_owners = []
    for name in ('Right_Lung', 'Left_Lung'):
        _, _, owner = _mask(session, name)
        if not any(unit.id == owner and unit.source_tool == 'segment_lungs' for unit in lung_units):
            raise ValueError(f'{name} not bound to current segment_lungs observation')
        lung_owners.append(owner)
    values = {}
    metrics = []
    for side, name in (('right', right), ('left', left)):
        intensity = _metric(outputs, session, f'{side}_intensity', 'summarize_roi_intensity', [name], {'mask_name': name})
        texture = _metric(outputs, session, f'{side}_texture', 'extract_roi_texture', [name], {'mask_name': name})
        _number(intensity.value.get('mean'), f'{side} mean')
        values[side] = {'intensity': intensity.value, 'texture': texture.value}
        metrics.extend((intensity, texture))
    dist = _metric(outputs, session, 'distribution', 'summarize_bilateral_distribution',
                   ['Right_Lung', 'Left_Lung', right, left], {'lesion_mask_names': [right, left]})
    value = {'hilar_roi_sources': {'right': {'mask_name': right, 'source': rmeta['source'], 'observation_id': rowner},
                                   'left': {'mask_name': left, 'source': lmeta['source'], 'observation_id': lowner}},
             'bilateral_appearance': values, 'bilateral_distribution': dist.value,
             'grade_evidence': {'status': 'not_assessed', 'enlargement': None, 'density_grade': None,
                                'limitations': ['caller-provided ROI identity is not independently verified',
                                                'no validated hilar grade threshold',
                                                'lung overlap is 2D geometry only']}}
    support = [session.graph.nodes[i] for i in [rowner, lowner, *lung_owners]] + metrics + [dist]
    return _finish(session, spec, 'hilar_roi_appearance', value, support)


def p10_program(session, spec, outputs, inputs):
    name = inputs.get('aortic_knob_mask_name')
    if not isinstance(name, str) or not name:
        raise ValueError('P10 requires an aortic knob ROI mask name')
    mask, meta, owner = _external_roi(session, name, 'aortic_knob')
    extent = _metric(outputs, session, 'extent', 'measure_region_extent', [name], {'mask_name': name})
    intensity = _metric(outputs, session, 'intensity', 'summarize_roi_intensity', [name], {'mask_name': name})
    width = _number(extent.value.get('bbox_width_px'), 'aortic knob bbox width', positive=True)
    xs = np.nonzero(mask)[1]
    if width != int(xs.max() - xs.min() + 1):
        raise ValueError('aortic knob width contradicts current ROI')
    _number(intensity.value.get('mean'), 'aortic knob mean')
    attributes = outputs.get('attributes', [])
    used_attributes = []
    calcification = {'status': 'not_assessed'}
    if attributes:
        if len(attributes) == 1:
            attr = attributes[0]
            bound = (attr.source_tool == 'classify_roi_attributes'
                     and attr.value.get('attribute') == 'calcification'
                     and attr.value.get('roi_mask_name') == name
                     and owner in attr.provenance
                     and attr.value.get('roi_mask_sha256') == hashlib.sha256(
                         np.ascontiguousarray(mask).tobytes()).hexdigest())
            if bound and attr.value.get('answer') in {'present', 'absent', 'uncertain'}:
                calcification = {'status': 'model_output', 'value': attr.value, 'calibrated': False}
                used_attributes.append(attr)
            elif not bound:
                calcification = {'status': 'not_assessed', 'reason': 'optional attribute evidence failed binding checks'}
            else:
                calcification = {'status': 'not_assessed', 'reason': 'optional attribute answer is unparsed'}
        else:
            calcification = {'status': 'not_assessed', 'reason': 'optional attribute output is ambiguous'}
    value = {'aortic_roi_source': {'mask_name': name, 'source': meta['source'], 'observation_id': owner},
             'width_px': width, 'grayscale_statistics': intensity.value,
             'calcification_candidate': calcification,
             'applicability': {'clinical_interpretation': 'not_assessed',
                               'width_convention': 'ROI inclusive bounding-box width in pixels',
                               'physical_size_available': False,
                               'roi_identity': 'caller_supplied_not_independently_verified'}}
    return _finish(session, spec, 'aortic_knob_roi_measurement', value,
                   [session.graph.nodes[owner], extent, intensity, *used_attributes])


def _candidate(session, unit, name, label_key, label):
    if unit.value.get(label_key) != label or unit.value.get('status') != 'detected':
        raise ValueError(f'{label} candidate is missing or unparsed')
    boxes = unit.value.get('boxes_xyxy_px')
    if not isinstance(boxes, list) or not boxes:
        raise ValueError('candidate boxes are missing')
    frame = inspect_image(session.image_ref)
    normalized = []
    for box in boxes:
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError('candidate box must have four coordinates')
        x0, y0, x1, y1 = [_number(v, 'box coordinate') for v in box]
        if not (0 <= x0 < x1 <= frame['width'] and 0 <= y0 < y1 <= frame['height']):
            raise ValueError('candidate box out of bounds or empty')
        normalized.append([x0, y0, x1, y1])
    mask, _, owner = _mask(session, name)
    if owner != unit.id or not np.array_equal(mask, box_mask(normalized, frame['width'], frame['height'])):
        raise ValueError('candidate mask disagrees with boxes or observation owner')
    return normalized


def p11_program(session, spec, outputs, inputs):
    candidate = _one(outputs, 'candidates', 'detect_abnormalities')
    boxes = _candidate(session, candidate, 'Candidate_fracture', 'finding', 'fracture')
    sharpness = _metric(outputs, session, 'sharpness', 'measure_roi_sharpness',
                        ['Candidate_fracture'], {'mask_name': 'Candidate_fracture'})
    _number(sharpness.value.get('tenengrad_mean'), 'sharpness')
    bones = outputs.get('bones', [])
    context = {}
    candidate_mask = np.asarray(session.masks['Candidate_fracture'], dtype=bool)
    for name in ('Left_Clavicle', 'Right_Clavicle', 'Left_Scapula', 'Right_Scapula'):
        mask, _, owner = _mask(session, name)
        if not any(unit.id == owner and unit.source_tool == 'segment_shoulder_bones' for unit in bones):
            raise ValueError(f'{name} not bound to current shoulder-bone observation')
        context[name] = {'overlap_px': int(np.count_nonzero(mask & candidate_mask)), 'observation_id': owner}
    value = {'candidate_boxes': boxes, 'bone_context': context, 'roi_sharpness': sharpness.value,
             'case_evidence': {'status': 'candidate_only', 'disease_absence_established': False,
                               'clinical_interpretation': 'not_assessed',
                               'limitations': ['A4 covers shoulder bones, not all ribs',
                                               'box and overlap are predictions, not fracture confirmation',
                                               'sharpness is not fracture evidence']}}
    return _finish(session, spec, 'fracture_candidate_context', value, [*bones, candidate, sharpness])


def p12_program(session, spec, outputs, inputs):
    candidate = _one(outputs, 'candidates', 'ground_phrase')
    boxes = _candidate(session, candidate, 'Phrase_Region', 'phrase', 'subcutaneous emphysema')
    intensity = _metric(outputs, session, 'intensity', 'summarize_roi_intensity',
                        ['Phrase_Region'], {'mask_name': 'Phrase_Region'})
    _number(intensity.value.get('mean'), 'candidate intensity')
    lungs = outputs.get('lungs', [])
    relations = {}
    spatial_units = []
    for step, name in (('spatial', 'Right_Lung'), ('spatial_left', 'Left_Lung')):
        _, _, owner = _mask(session, name)
        if not any(unit.id == owner and unit.source_tool == 'segment_lungs' for unit in lungs):
            raise ValueError(f'{name} not bound to current segment_lungs observation')
        unit = _metric(outputs, session, step, 'relate_lesion_to_anatomy', ['Phrase_Region', name],
                       {'lesion_mask_name': 'Phrase_Region', 'anatomy_mask_name': name})
        spatial_units.append(unit)
        relations[name] = unit.value
    value = {'candidate_boxes': boxes, 'appearance': intensity.value,
             'projected_lung_relations': relations,
             'case_evidence': {'status': 'candidate_only', 'clinical_interpretation': 'not_assessed',
                               'disease_absence_established': False,
                               'limitations': ['box outside lung is not evidence of subcutaneous gas',
                                               'no validated independent attribute channel']}}
    return _finish(session, spec, 'subcutaneous_candidate_context', value,
                   [*lungs, candidate, intensity, *spatial_units])


P_PROGRAMS_B = {'P7': p7_program, 'P8': p8_program, 'P10': p10_program,
                'P11': p11_program, 'P12': p12_program}
