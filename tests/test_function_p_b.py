"""Synthetic end-to-end integration checks for conditional P evidence pilots."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.session import EvidenceSession
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit
from visevi.functions import runner
from visevi.perception.metrics import build_metric_tools
from visevi.perception.vlm import box_mask


class FakeModel:
    def __init__(self, name, harness):
        self.name, self.h = name, harness

    def describe(self):
        properties = {'output_dir': {'type': 'string'}, 'device': {'type': 'string'}}
        if self.name == 'detect_abnormalities':
            properties['findings'] = {'type': 'array'}
        if self.name == 'ground_phrase':
            properties['phrase'] = {'type': 'string'}
        if self.name == 'classify_roi_attributes':
            properties.update({'roi_mask_name': {'type': 'string'},
                               'attributes': {'type': 'array', 'items': {'type': 'string'}}})
        return ToolSpec(self.name, 'perception', 'synthetic model',
                        parameter_schema(properties, ('output_dir',)), evidence_result_schema())

    def run(self, session, **params):
        if self.name == 'classify_roi_attributes' and self.h.attribute_malformed:
            return []
        return getattr(self.h, self.name)(session, params)


class Harness:
    def __init__(self, path):
        self.path = path
        self.session = EvidenceSession(image_ref=str(path))
        self.registry = ToolRegistry()
        self.candidate_status = 'detected'
        self.bad_mask = False
        self.attribute_answer = 'present'
        self.attribute_malformed = False
        for tool in build_metric_tools():
            self.registry.register(tool)
        for name in ('segment_mediastinum', 'segment_lungs', 'segment_shoulder_bones',
                     'detect_abnormalities', 'ground_phrase', 'classify_roi_attributes'):
            self.registry.register(FakeModel(name, self))

    @staticmethod
    def emit(session, tool, name, value, provenance=()):
        unit = EvidenceUnit(session.next_id('obs'), 'observation', 'region', name, value,
                            source_tool=tool, provenance=list(provenance),
                            validity_conditions=['synthetic only'])
        session.add_observation(unit)
        return unit

    @staticmethod
    def attach(session, name, mask, owner, **meta):
        session.attach_masks({name: mask}, {name: {'observation_id': owner.id,
            'coordinate_frame': 'original_image_pixels', 'fov_truncated_sides': [], **meta}})

    def segment_mediastinum(self, session, params):
        mask = np.zeros((24, 30), bool)
        mask[4:20, 11:19] = True
        unit = self.emit(session, 'segment_mediastinum', 'Mediastinum', {})
        self.attach(session, 'Mediastinum', mask, unit)
        return [unit]

    def segment_lungs(self, session, params):
        out = []
        for name, x in [('Right_Lung', slice(1, 13)), ('Left_Lung', slice(17, 29))]:
            mask = np.zeros((24, 30), bool)
            mask[2:22, x] = True
            unit = self.emit(session, 'segment_lungs', name, {})
            self.attach(session, name, mask, unit)
            out.append(unit)
        return out

    def segment_shoulder_bones(self, session, params):
        out = []
        for i, name in enumerate(('Left_Clavicle', 'Right_Clavicle', 'Left_Scapula', 'Right_Scapula')):
            mask = np.zeros((24, 30), bool)
            mask[2+i:4+i, 1+5*i:4+5*i] = True
            unit = self.emit(session, 'segment_shoulder_bones', name, {})
            self.attach(session, name, mask, unit)
            out.append(unit)
        return out

    def _candidate(self, session, tool, key, label, name):
        boxes = [[4, 5, 8, 9]] if self.candidate_status == 'detected' else []
        unit = self.emit(session, tool, name, {key: label, 'status': self.candidate_status,
                                               'boxes_xyxy_px': boxes})
        if boxes:
            mask = box_mask(boxes, 30, 24)
            if self.bad_mask:
                mask = np.roll(mask, 1, axis=1)
            self.attach(session, name, mask, unit)
        return [unit]

    def detect_abnormalities(self, session, params):
        return self._candidate(session, 'detect_abnormalities', 'finding', params['findings'][0], 'Candidate_fracture')

    def ground_phrase(self, session, params):
        return self._candidate(session, 'ground_phrase', 'phrase', params['phrase'], 'Phrase_Region')

    def classify_roi_attributes(self, session, params):
        name = params['roi_mask_name']
        owner = session.mask_metadata[name]['observation_id']
        digest = hashlib.sha256(np.ascontiguousarray(session.masks[name], dtype=bool).tobytes()).hexdigest()
        return [self.emit(session, 'classify_roi_attributes', 'roi_attribute_' + attribute,
                          {'attribute': attribute, 'answer': self.attribute_answer,
                           'roi_mask_name': name, 'roi_mask_sha256': digest}, [owner])
                for attribute in params['attributes']]

    def roi(self, name, anatomy, x):
        mask = np.zeros((24, 30), bool)
        mask[7:11, x:x+4] = True
        owner = self.emit(self.session, 'manual_annotation', anatomy, {'status': 'provided'})
        self.attach(self.session, name, mask, owner, anatomy=anatomy, source='synthetic expert ROI',
                    image_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def run(self, fid, tmp_path, inputs=None, **kwargs):
        return runner.run_function_episode(fid, session=self.session, registry=self.registry,
                                           inputs=inputs, out_dir=tmp_path / 'run', **kwargs)


@pytest.fixture
def harness(tmp_path):
    image = tmp_path / 'image.png'
    Image.fromarray(np.tile(np.arange(30, dtype=np.uint8), (24, 1)) * 5).save(image)
    return Harness(image)


def test_p7_bbox_width_is_not_clinical_width(harness, tmp_path):
    result = harness.run('P7', tmp_path)
    assert result.status == 'ok'
    out = result.functions[0].outputs
    assert out['width_px'] == 8
    assert 'not a row-specific clinical width' in out['measurement_convention']
    assert out['applicability']['clinical_interpretation'] == 'not_assessed'
    assert not validate_graph(result.session)


@pytest.mark.parametrize('fid', ['P11', 'P12'])
def test_candidate_programs_preserve_limitations_and_provenance(harness, tmp_path, fid):
    result = harness.run(fid, tmp_path)
    assert result.status == 'ok'
    out = result.functions[0].outputs
    assert out['candidate_boxes'] == [[4, 5, 8, 9]]
    assert out['case_evidence']['clinical_interpretation'] == 'not_assessed'
    assert out['case_evidence']['disease_absence_established'] is False
    assert not validate_graph(result.session)
    relation = result.session.graph.nodes[result.functions[0].evidence_refs.relation_ids[0]]
    assert set(relation.provenance) == set(result.functions[0].evidence_refs.observation_ids)


@pytest.mark.parametrize('fid', ['P11', 'P12'])
@pytest.mark.parametrize('status', ['not_detected', 'unparsed'])
def test_candidate_negative_or_unparsed_never_becomes_claim(harness, tmp_path, fid, status):
    harness.candidate_status = status
    result = harness.run(fid, tmp_path)
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


@pytest.mark.parametrize('fid', ['P11', 'P12'])
def test_candidate_mask_mismatch_is_refused(harness, tmp_path, fid):
    harness.bad_mask = True
    result = harness.run(fid, tmp_path)
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


def test_p8_two_explicit_bound_rois(harness, tmp_path):
    harness.roi('RH', 'right_hilum', 4)
    harness.roi('LH', 'left_hilum', 20)
    result = harness.run('P8', tmp_path, {'right_hilum_mask_name': 'RH', 'left_hilum_mask_name': 'LH',
                                          'hilum_mask_names': ['RH', 'LH']})
    assert result.status == 'ok'
    output = result.functions[0].outputs
    assert output['grade_evidence']['status'] == 'not_assessed'
    assert output['bilateral_distribution']['pattern'] == 'bilateral'
    assert not validate_graph(result.session)


def test_p8_undocumented_hilum_source_refuses(harness, tmp_path):
    harness.roi('RH', 'wrong_anatomy', 4)
    harness.roi('LH', 'left_hilum', 20)
    result = harness.run('P8', tmp_path, {'right_hilum_mask_name': 'RH', 'left_hilum_mask_name': 'LH',
                                          'hilum_mask_names': ['RH', 'LH']})
    assert result.status == 'refused'
    assert 'right_hilum' in result.functions[0].refusal.reason
    assert not result.session.graph.by_category('claim')


def test_p10_bound_knob_roi_without_optional_attribute(harness, tmp_path):
    harness.roi('KNOB', 'aortic_knob', 4)
    result = harness.run('P10', tmp_path, {'aortic_knob_mask_name': 'KNOB'})
    assert result.status == 'ok'
    output = result.functions[0].outputs
    assert output['width_px'] == 4
    assert output['calcification_candidate']['status'] == 'not_assessed'
    assert output['applicability']['clinical_interpretation'] == 'not_assessed'
    assert not validate_graph(result.session)


@pytest.mark.parametrize('answer', ['present', 'absent', 'uncertain', 'unparsed'])
def test_p10_optional_unparsed_attribute_preserves_primary_measurement(harness, tmp_path, answer):
    harness.roi('KNOB', 'aortic_knob', 4)
    harness.attribute_answer = answer
    result = harness.run('P10', tmp_path, {'aortic_knob_mask_name': 'KNOB'}, include_optional=True)
    assert result.status == 'ok', result.functions[0].refusal
    output = result.functions[0].outputs
    assert output['width_px'] == 4
    if answer == 'unparsed':
        assert output['calcification_candidate']['status'] == 'not_assessed'
        assert 'unparsed' in output['calcification_candidate']['reason']
    else:
        assert output['calcification_candidate']['status'] == 'model_output'
    assert not validate_graph(result.session)


def test_p10_malformed_optional_attribute_preserves_primary_measurement(harness, tmp_path):
    harness.roi('KNOB', 'aortic_knob', 4)
    harness.attribute_malformed = True
    result = harness.run('P10', tmp_path, {'aortic_knob_mask_name': 'KNOB'}, include_optional=True)
    assert result.status == 'ok', result.functions[0].refusal
    assert result.functions[0].outputs['width_px'] == 4
    assert 'attributes' in result.functions[0].missing_optional
    assert result.functions[0].outputs['calcification_candidate']['status'] == 'not_assessed'


def test_p10_roi_from_different_image_refuses(harness, tmp_path):
    harness.roi('KNOB', 'aortic_knob', 4)
    harness.session.mask_metadata['KNOB']['image_sha256'] = '0' * 64
    result = harness.run('P10', tmp_path, {'aortic_knob_mask_name': 'KNOB'})
    assert result.status == 'refused'
    assert 'SHA256' in result.functions[0].refusal.reason
    assert not result.session.graph.by_category('claim')


@pytest.mark.parametrize('field', ['observation_id', 'coordinate_frame', 'fov_truncated_sides', 'source'])
def test_p10_roi_with_missing_source_metadata_refuses(harness, tmp_path, field):
    harness.roi('KNOB', 'aortic_knob', 4)
    harness.session.mask_metadata['KNOB'].pop(field)
    result = harness.run('P10', tmp_path, {'aortic_knob_mask_name': 'KNOB'})
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


def test_p10_roi_cannot_point_at_wrong_anatomical_observation(harness, tmp_path):
    harness.roi('KNOB', 'aortic_knob', 4)
    owner = harness.session.mask_metadata['KNOB']['observation_id']
    harness.session.graph.nodes[owner].finding_or_anatomy = 'whole_aorta'
    result = harness.run('P10', tmp_path, {'aortic_knob_mask_name': 'KNOB'})
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


def test_p8_refuses_swapped_hilar_mask_list(harness, tmp_path):
    harness.roi('RH', 'right_hilum', 4)
    harness.roi('LH', 'left_hilum', 20)
    result = harness.run('P8', tmp_path, {'right_hilum_mask_name': 'RH', 'left_hilum_mask_name': 'LH',
                                          'hilum_mask_names': ['LH', 'RH']})
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')
