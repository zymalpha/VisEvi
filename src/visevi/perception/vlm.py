"""CheXOne-backed tools: A24 view, A6 abnormality boxes, A7 phrase grounding, A18 ROI attributes.

Prompts and parsers are versioned and part of the cache binding. Box outputs use
the CheXinstruct format ``<|ref|>name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>``.
Unparseable answers are reported as ``unparsed`` with the raw text, never coerced.
"""
from __future__ import annotations

import re

import numpy as np

from ..backends import Backend
from . import geometry
from .model_tools import ModelTool, Output, image_size, unit

PROMPT_VERSION = 1
# CheXinstruct boxes are integers on a 0-100 grid over the input image (verified
# 2026-10-30 against ChestX-Det reference boxes; see the project status record).
BOX_GRID = 100
BACKEND = Backend('chexone', 'chexone', {'adapter': 1, 'prompts': PROMPT_VERSION, 'box_grid': BOX_GRID})

FINDINGS = ('atelectasis', 'calcification', 'cardiomegaly', 'consolidation', 'pleural effusion',
            'emphysema', 'fracture', 'mass', 'nodule', 'pleural thickening', 'pneumothorax',
            'pulmonary fibrosis', 'aortic enlargement', 'interstitial lung disease', 'lung opacity')
ATTRIBUTES = ('consolidation', 'nodule or mass', 'pleural effusion', 'pneumothorax', 'atelectasis',
              'calcification', 'cavitation', 'reticular opacity', 'air bronchogram', 'ground-glass opacity')
VIEW_OPTIONS = {'A': 'PA', 'B': 'AP', 'C': 'lateral'}
FORMAT_SPEC = ('\nOutput specification:\nIf a target is detected, output:\n'
               '<|ref|>target name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>\n'
               'Example: <|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|>\n'
               'If no target is detected, output:\nNo target name detected.\n'
               'Example: No pneumothorax detected.')

BOX_RE = re.compile(r'<\|box\|>\s*\((\-?\d+(?:\.\d+)?)\s*,\s*(\-?\d+(?:\.\d+)?)\)\s*,\s*'
                    r'\((\-?\d+(?:\.\d+)?)\s*,\s*(\-?\d+(?:\.\d+)?)\)\s*<\|/box\|>')


def parse_boxes(text: str) -> dict:
    """Return {'status', 'boxes_grid'} from a CheXinstruct-style grounding answer."""
    boxes = [tuple(float(v) for v in match.groups()) for match in BOX_RE.finditer(text)]
    if boxes:
        valid = [b for b in boxes if 0 <= min(b) and max(b) <= BOX_GRID and b[0] < b[2] and b[1] < b[3]]
        return {'status': 'detected' if valid else 'unparsed', 'boxes_grid': valid,
                'rejected_boxes_grid': [b for b in boxes if b not in valid]}
    if re.search(r'\bno\b.*\b(detected|found|seen|present)\b', text, re.I | re.S):
        return {'status': 'not_detected', 'boxes_grid': [], 'rejected_boxes_grid': []}
    return {'status': 'unparsed', 'boxes_grid': [], 'rejected_boxes_grid': []}


def parse_option(text: str, options) -> str | None:
    """Option letter from '(A)', 'A', 'A.' or 'Answer: A'; ``None`` if absent or ambiguous."""
    letters = set(re.findall(r'(?<![A-Za-z])\(?([A-Z])\)?(?![A-Za-z])', text.strip()))
    letters &= set(options)
    return letters.pop() if len(letters) == 1 else None


def grid_to_pixels(box, width: int, height: int, crop=None) -> list[float]:
    """Map a 0-100 grid box to original-image edge coordinates (image spans [0, width])."""
    x0, y0, x1, y1 = box
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    w, h = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (width, height)
    return [ox + x0 / BOX_GRID * w, oy + y0 / BOX_GRID * h, ox + x1 / BOX_GRID * w, oy + y1 / BOX_GRID * h]


def box_mask(boxes_px, width: int, height: int) -> np.ndarray:
    """Union of pixels whose centres (i + 0.5 in edge coordinates) lie in any half-open box."""
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in boxes_px:
        cols = (np.arange(width) >= x0 - 0.5) & (np.arange(width) < x1 - 0.5)
        rows = (np.arange(height) >= y0 - 0.5) & (np.arange(height) < y1 - 0.5)
        mask |= rows[:, None] & cols[None, :]
    return mask


def _vocabulary(params, key, allowed, default):
    values = params.get(key, list(default))
    if (not isinstance(values, list) or not values or len(set(values)) != len(values)
            or any(v not in allowed for v in values)):
        raise ValueError(f'{key} must be a non-empty unique subset of {list(allowed)}')
    return values


CONDITIONS = ['CheXOne (Qwen2.5-VL-3B) generated answer; greedy decoding; outputs are not calibrated',
              'CheXOne training data include ChestX-ray14, NLM-TB, SIIM, VinDr-CXR and MS-CXR; '
              'these sources cannot serve as independent tests']


class ClassifyView(ModelTool):
    name = 'classify_view'
    capability = 'A24'
    description = 'chest X-ray projection (PA / AP / lateral) from CheXOne multiple-choice VQA'
    backend = BACKEND
    prompt = ('What is the view position of this chest X-ray?\nOptions:\n(A) PA\n(B) AP\n(C) Lateral\n'
              'Answer with the option letter only.')

    def backend_params(self, session, params):
        return {'requests': [{'content': [{'type': 'image'}, {'type': 'text', 'text': self.prompt}],
                              'option_letters': list(VIEW_OPTIONS), 'max_new_tokens': 16}]}

    def convert(self, session, run, params):
        answer = run['result']['answers'][0]
        letter = parse_option(answer['text'], VIEW_OPTIONS)
        value = {'view': VIEW_OPTIONS[letter] if letter else 'unknown',
                 'status': 'ok' if letter else 'unparsed', 'raw_text': answer['text'],
                 'option_raw_probabilities': answer.get('first_token_option_probabilities'),
                 'assumed': False, 'prompt_version': PROMPT_VERSION}
        return Output(units=[unit('view_position', 'global_condition', value, validity_conditions=list(CONDITIONS))])


class DetectAbnormalities(ModelTool):
    name = 'detect_abnormalities'
    capability = 'A6'
    description = ('abnormality candidate boxes, one CheXOne grounding query per finding of a frozen '
                   'vocabulary; candidate masks are box approximations attached as Candidate_<finding>')
    backend = BACKEND
    extra_properties = {'findings': {'type': 'array', 'items': {'type': 'string', 'enum': list(FINDINGS)}}}

    def backend_params(self, session, params):
        findings = _vocabulary(params, 'findings', FINDINGS, FINDINGS)
        return {'findings': findings, 'requests': [
            {'content': [{'type': 'image'}, {'type': 'text', 'text': f'Locate the {f} in this chest X-ray.' + FORMAT_SPEC}],
             'max_new_tokens': 128} for f in findings]}

    def convert(self, session, run, params):
        width, height = image_size(session)
        units, masks = [], []
        for finding, answer in zip(run['result']['request_params']['findings'], run['result']['answers']):
            parsed = parse_boxes(answer['text'])
            boxes = [grid_to_pixels(b, width, height) for b in parsed['boxes_grid']]
            value = {'finding': finding, 'status': parsed['status'], 'boxes_xyxy_px': boxes,
                     'boxes_grid_0_100': parsed['boxes_grid'], 'rejected_boxes_grid': parsed['rejected_boxes_grid'],
                     'raw_text': answer['text'],
                     'box_convention': 'edge coordinates: image spans [0, width] x [0, height], pixel (i, j) '
                                       'covers [i, i+1) x [j, j+1) (same as EvidenceUnit.region)',
                     'prompt_version': PROMPT_VERSION}
            owner = unit(f'candidate_{finding.replace(" ", "_")}', 'region', value,
                         validity_conditions=list(CONDITIONS) + ['boxes, not segmentations; area is a box approximation'])
            units.append(owner)
            if boxes:
                mask = box_mask(boxes, width, height)
                if mask.any():
                    masks.append((f'Candidate_{finding.replace(" ", "_")}', mask,
                                  {'source': 'A6 detect_abnormalities', 'shape': 'box union',
                                   'coordinate_frame': 'original_image_pixels', 'full_image_observed': True,
                                   'fov_truncated_sides': [], 'is_ground_truth': False}, owner))
        return Output(units=units, masks=masks)


class GroundPhrase(ModelTool):
    name = 'ground_phrase'
    capability = 'A7'
    description = 'region box(es) for a free-text finding phrase (CheXOne phrase grounding) or not detected'
    backend = BACKEND
    extra_properties = {'phrase': {'type': 'string', 'maxLength': 200}}
    extra_required = ('phrase',)

    def backend_params(self, session, params):
        phrase = params.get('phrase')
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 200 or '<|' in phrase:
            raise ValueError('phrase must be 1-200 characters of plain text')
        return {'requests': [{'content': [{'type': 'image'}, {'type': 'text', 'text':
                 f'Please locate the following finding in this chest X-ray: {phrase.strip()}' + FORMAT_SPEC}],
                 'max_new_tokens': 128}]}

    def convert(self, session, run, params):
        width, height = image_size(session)
        answer = run['result']['answers'][0]
        parsed = parse_boxes(answer['text'])
        boxes = [grid_to_pixels(b, width, height) for b in parsed['boxes_grid']]
        value = {'phrase': params['phrase'].strip(), 'status': parsed['status'], 'boxes_xyxy_px': boxes,
                 'boxes_grid_0_100': parsed['boxes_grid'], 'raw_text': answer['text'],
                 'prompt_version': PROMPT_VERSION}
        owner = unit('phrase_region', 'region', value, validity_conditions=list(CONDITIONS))
        masks = []
        if boxes:
            masks.append(('Phrase_Region', box_mask(boxes, width, height),
                          {'source': 'A7 ground_phrase', 'phrase': value['phrase'], 'shape': 'box union',
                           'coordinate_frame': 'original_image_pixels', 'full_image_observed': True,
                           'fov_truncated_sides': [], 'is_ground_truth': False}, owner))
        return Output(units=[owner], masks=masks)


class ClassifyRoiAttributes(ModelTool):
    name = 'classify_roi_attributes'
    capability = 'A18'
    description = ('present / absent / uncertain for frozen attributes inside a named ROI mask, from '
                   'CheXOne given the full image and an enlarged crop of the ROI bounding box')
    backend = BACKEND
    extra_properties = {'roi_mask_name': {'type': 'string'},
                        'attributes': {'type': 'array', 'items': {'type': 'string', 'enum': list(ATTRIBUTES)}}}
    extra_required = ('roi_mask_name',)
    options = {'A': 'present', 'B': 'absent', 'C': 'uncertain'}

    def backend_params(self, session, params):
        name = params.get('roi_mask_name')
        if not isinstance(name, str) or session.masks.get(name) is None:
            return self.refuse('ROI mask not available', missing=[f'mask:{name}'])
        width, height = image_size(session)
        roi = geometry._nonempty(session.masks[name], name)
        if roi.shape != (height, width):
            raise ValueError('ROI mask is not in the session image frame')
        bounds = geometry.bbox(roi)
        margin_x, margin_y = max(8, bounds['w'] // 10), max(8, bounds['h'] // 10)
        crop = [max(0, bounds['x0'] - margin_x), max(0, bounds['y0'] - margin_y),
                min(width, bounds['x1'] + 1 + margin_x), min(height, bounds['y1'] + 1 + margin_y)]
        attributes = _vocabulary(params, 'attributes', ATTRIBUTES, ATTRIBUTES)
        return {'roi_mask_name': name, 'roi_mask_sha256': __import__('hashlib').sha256(roi.tobytes()).hexdigest(),
                'crop_xyxy': crop, 'attributes': attributes, 'requests': [
            {'content': [{'type': 'image'}, {'type': 'image', 'crop': crop}, {'type': 'text', 'text':
             f'The second image is an enlarged region of the first chest X-ray. Is there {a} in this region?\n'
             'Options:\n(A) Present\n(B) Absent\n(C) Uncertain\nAnswer with the option letter only.'}],
             'option_letters': list(self.options), 'max_new_tokens': 16} for a in attributes]}

    def convert(self, session, run, params):
        request = run['result']['request_params']
        units = []
        for attribute, answer in zip(request['attributes'], run['result']['answers']):
            letter = parse_option(answer['text'], self.options)
            units.append(unit(f'roi_attribute_{attribute.replace(" ", "_")}', 'attribute',
                              {'attribute': attribute, 'answer': self.options[letter] if letter else 'unparsed',
                               'raw_text': answer['text'],
                               'option_raw_probabilities': answer.get('first_token_option_probabilities'),
                               'roi_mask_name': request['roi_mask_name'],
                               'roi_mask_sha256': request['roi_mask_sha256'],
                               'crop_xyxy': request['crop_xyxy'],
                               'prompt_version': PROMPT_VERSION},
                              provenance=session.mask_provenance([request['roi_mask_name']]),
                              validity_conditions=list(CONDITIONS) + [
                                  'ROI given as an enlarged bounding-box crop; pixels outside the mask are visible to the model']))
        return Output(units=units)
