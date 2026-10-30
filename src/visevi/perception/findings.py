"""A8 pneumothorax segmentation and A26 line/tube localization (model-backed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..backends import Backend
from . import geometry
from .model_tools import ModelTool, Output, image_size, inside_image, unit


def load_mask(output_dir: str, name: str, run: dict, width: int, height: int) -> np.ndarray:
    if name not in run['artifacts']:
        raise ValueError(f'backend did not record artifact {name}')
    with Image.open(Path(output_dir) / name) as image:
        mask = geometry._mask(np.asarray(image), name)
    if mask.shape != (height, width):
        raise ValueError(f'{name} is not in the original image frame')
    return mask


def mask_metadata(run: dict, source: str, artifact: str) -> dict:
    return {'source': source, 'artifact': artifact, 'artifact_sha256': run['artifacts'][artifact],
            'image_sha256': run['binding']['image_sha256'], 'coordinate_frame': 'original_image_pixels',
            'full_image_observed': True, 'fov_truncated_sides': [],
            'is_ground_truth': False}


class SegmentPneumothorax(ModelTool):
    name = 'segment_pneumothorax'
    capability = 'A8'
    description = ('pneumothorax probability mask and case score (ianpan/pneumothorax-cxr, '
                   'EfficientNetV2-S U-Net); mask threshold 0.5 on the model probability; '
                   'training data and licence not documented by the author')
    backend = Backend('pneumothorax', 'pneumothorax', {'adapter': 1})

    def convert(self, session, run, params):
        width, height = image_size(session)
        result = run['result']
        mask = load_mask(params['output_dir'], 'pneumothorax_mask.png', run, width, height)
        value = {'case_raw_score': result['case_score'], 'mask_threshold': result['mask_threshold'],
                 'max_pixel_probability': result['max_pixel_probability'], 'area_px': int(mask.sum()),
                 'frame': result['frame'],
                 'status': 'region_above_threshold' if mask.any() else 'no_region_above_threshold',
                 'score_meaning': 'raw sigmoid outputs; not calibrated probabilities'}
        conditions = ['model provenance incomplete: training data and licence not stated on the model card',
                      'pixel threshold 0.5 is the default, not a validated operating point']
        region = None
        if mask.any():
            bounds = geometry.bbox(mask)
            value['bbox_inclusive_px'] = bounds
            region = {'x0': bounds['x0'], 'y0': bounds['y0'], 'x1': bounds['x1'] + 1, 'y1': bounds['y1'] + 1}
        item = unit('pneumothorax_candidate', 'region', value, region=region,
                    provenance=[f'cached:pneumothorax-cxr:{run["binding"]["image_sha256"][:16]}'],
                    validity_conditions=conditions)
        masks = [('Pneumothorax_Candidate', mask,
                  mask_metadata(run, 'A8 segment_pneumothorax', 'pneumothorax_mask.png'), item)] if mask.any() else []
        return Output(units=[item], masks=masks)


DEVICE_KEYS = {'ETT': 'endotracheal_tube', 'NGT': 'nasogastric_tube',
               'CVC': 'central_venous_catheter', 'Swan Ganz': 'swan_ganz_catheter'}


class LocalizeDevices(ModelTool):
    name = 'localize_devices'
    capability = 'A26'
    description = ('ETT, NGT, CVC and Swan-Ganz line masks, distal-tip points and carina '
                   '(ianpan/cxr-lines-and-tubes, release profile max); chest drains, pacemakers '
                   'and other implants are NOT covered')
    backend = Backend('lines-tubes', 'lines_tubes', {'adapter': 1, 'profile': 'max'})

    def convert(self, session, run, params):
        width, height = image_size(session)
        result = run['result']
        threshold = run['worker_runtime']['screening_threshold'] if run.get('worker_runtime') else 0.5
        units, masks = [], []
        for family, item in result['devices'].items():
            key = DEVICE_KEYS[family]
            mask = load_mask(params['output_dir'], item['line_mask_file'], run, width, height)
            screened_in = item['presence_raw_score'] >= threshold
            tip = item['distal_tip_xy']
            value = {'device_family': family, 'presence_raw_score': item['presence_raw_score'],
                     'screening_threshold': threshold,
                     'status': 'present' if screened_in else 'below_screening_threshold',
                     'line_mask_area_px': int(mask.sum()),
                     'distal_tip_xy': tip if inside_image(tip, width, height) else None,
                     'distal_tip_peak_probability': item['distal_tip_peak_probability'],
                     'score_meaning': 'raw sigmoid scores; 0.5 is the author’s uncalibrated screening policy'}
            conditions = ['the distal tip is the model’s clinically distal endpoint; for an incompletely '
                          'imaged NGT it is the lowest visible point, not the true tip']
            if not screened_in:
                conditions.append('presence score below screening threshold: device presence is not asserted; '
                                  'line area and tip are raw model outputs only and no mask is attached')
            owner = unit(key, 'region', value,
                         provenance=[f'cached:cxr-lines-and-tubes:{run["binding"]["image_sha256"][:16]}'],
                         validity_conditions=conditions)
            units.append(owner)
            if screened_in and mask.any():
                masks.append((f'Device_{key}', mask, mask_metadata(run, 'A26 localize_devices', item['line_mask_file']), owner))
        carina = result['carina_xy']
        units.append(unit('carina', 'landmark',
                          {'point_xy': carina if inside_image(carina, width, height) else None,
                           'heatmap_peak': result['carina_heatmap_peak'],
                           'status': 'ok' if inside_image(carina, width, height) else 'outside_image'},
                          provenance=[f'cached:cxr-lines-and-tubes:{run["binding"]["image_sha256"][:16]}'],
                          validity_conditions=['carina from a heatmap model; reported for every image']))
        units.append(unit('device_position_classification', 'attribute',
                          {'raw_scores': result['classification_raw'], 'fold_ids': result['fold_ids'],
                           'label_source': 'RANZCR CLiP label set (Normal/Borderline/Abnormal per family)'},
                          provenance=[f'cached:cxr-lines-and-tubes:{run["binding"]["image_sha256"][:16]}'],
                          validity_conditions=['competition-style position classes; not calibrated probabilities']))
        return Output(units=units, masks=masks)
