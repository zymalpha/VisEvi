"""Q1-Q3 software integration on synthetic models and hand-checkable data.

These tests do not establish medical accuracy or a clinical quality grade.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.spec import ToolSpec, parameter_schema, evidence_result_schema
from visevi.core.units import EvidenceUnit, ToolRefusal
from visevi.functions import runner
from visevi.functions.specs import ToolStep, load_function_specs
from visevi.perception.metrics import build_metric_tools


class Model:
    def __init__(self, name, h):
        self.name, self.h = name, h

    def describe(self):
        return ToolSpec(self.name, 'perception', 'synthetic test model',
                        parameter_schema({'output_dir': {'type': 'string'}}, ('output_dir',)),
                        evidence_result_schema())

    def run(self, session, **params):
        if self.name in self.h.refuse:
            return ToolRefusal(self.name, 'test refusal')
        return getattr(self.h, self.name)(session)


class Harness:
    def __init__(self, image):
        self.image = str(image)
        self.view = 'PA'
        self.landmark_status = 'ok'
        self.quality_status = 'ok'
        self.rotation_index = 0.0
        self.truncate_bone = False
        self.refuse = set()
        self.registry = ToolRegistry()
        for tool in build_metric_tools():
            self.registry.register(tool)
        for tool in ('classify_view', 'locate_lung_landmarks', 'assess_image_quality',
                     'segment_shoulder_bones'):
            self.registry.register(Model(tool, self))

    @staticmethod
    def emit(session, source, target, value):
        unit = EvidenceUnit(session.next_id('obs'), 'observation', 'attribute', target, value,
                            source_tool=source, validity_conditions=['synthetic prediction'])
        session.add_observation(unit)
        return unit

    def classify_view(self, session):
        return [self.emit(session, 'classify_view', 'view_position',
                          {'status': 'ok', 'assumed': False, 'view': self.view})]

    def locate_lung_landmarks(self, session):
        units = []
        for side, x in (('right', 4), ('left', 14)):
            for label in ('costophrenic_angle', 'cardiophrenic_angle'):
                units.append(self.emit(session, 'locate_lung_landmarks', f'{side}_{label}',
                                       {'status': self.landmark_status, 'point_xy': [x, 11]}))
            units.append(self.emit(session, 'locate_lung_landmarks', f'{side}_lung_base_contour',
                                   {'polyline_xy': [[x, 11], [x + 1, 10]]}))
        return units

    def segment_shoulder_bones(self, session):
        units = []
        for i, name in enumerate(('Right_Clavicle', 'Left_Clavicle', 'Right_Scapula', 'Left_Scapula')):
            unit = self.emit(session, 'segment_shoulder_bones', name, {'area_px': 2})
            mask = np.zeros((16, 20), bool)
            mask[1:3, 2 + i * 4] = True
            session.attach_masks({name: mask}, {name: {
                'observation_id': unit.id, 'coordinate_frame': 'original_image_pixels',
                'fov_truncated_sides': ['top'] if self.truncate_bone else []}})
            units.append(unit)
        return units

    def assess_image_quality(self, session):
        values = {
            'inspiration': {'right_hemidiaphragm_dome_xy': [5, 10],
                            'posterior': {'count_above': 2, 'ribs_above': [1, 2]},
                            'anterior': {'count_above': 1, 'ribs_above': [1]}},
            'rotation': {'right_medial_end_xy': [4, 4], 'left_medial_end_xy': [15, 4],
                         'spine_midline_x_at_rows': [9.5, 9.5], 'distance_right_px': 5,
                         'distance_left_px': 5, 'asymmetry_index': self.rotation_index},
            'penetration': {'retrocardiac_spine_mean': 90, 'heart_without_spine_mean': 80,
                            'difference': 10, 'pixels': [60, 60]},
        }
        return [self.emit(session, 'assess_image_quality', name,
                          {**value, 'status': self.quality_status, 'grade': None})
                for name, value in values.items()]

    def run(self, fid, **kwargs):
        return runner.run_function_episode(fid, image_ref=self.image, registry=self.registry, **kwargs)


@pytest.fixture
def harness(tmp_path):
    image = tmp_path / 'image.png'
    Image.fromarray(np.arange(320, dtype=np.uint16).reshape(16, 20).astype('uint8')).save(image)
    return Harness(image)


def specs_with_projection():
    specs = load_function_specs()
    for fid in ('Q1', 'Q2'):
        spec = specs[fid]
        if not any(step.tool == 'classify_view' for step in spec.steps):
            specs[fid] = replace(spec, steps=(ToolStep('view', 'classify_view'), *spec.steps))
    return specs


@pytest.mark.parametrize('fid', ['Q1', 'Q2', 'Q3'])
def test_q_programs_create_a_sourced_relation_and_ungraded_claim(harness, tmp_path, fid):
    result = harness.run(fid, specs=specs_with_projection(), out_dir=tmp_path / fid)
    assert result.status == 'ok', result.functions[0].refusal
    run = result.functions[0]
    assert run.outputs['applicability']['clinical_grade'] is None
    assert run.outputs['applicability']['clinical_interpretation'] == 'not_assessed'
    relation = result.session.graph.nodes[run.evidence_refs.relation_ids[0]]
    assert set(relation.provenance) == set(run.evidence_refs.observation_ids)
    assert validate_graph(result.session) == []


def test_q1_keeps_cxas_dome_separate_from_hybridgnet_base(harness, tmp_path):
    result = harness.run('Q1', specs=specs_with_projection(), out_dir=tmp_path / 'q1')
    output = result.functions[0].outputs
    assert output['rib_counts']['posterior']['count_above'] == 2
    assert output['diaphragm_geometry']['right_dome_xy_px'] == [5.0, 10.0]
    assert output['lung_base_landmarks']['right_costophrenic_angle'] == [4.0, 11.0]


def test_q2_exposes_only_cxas_rotation_not_a_pspnet_endpoint(harness, tmp_path):
    result = harness.run('Q2', specs=specs_with_projection(), out_dir=tmp_path / 'q2')
    output = result.functions[0].outputs
    assert output['rotation_index'] == 0
    assert output['clavicle_spine_geometry']['source'] == 'CXAS predicted structures'
    assert all(item['role'] == 'context_only_not_medial_endpoint'
               for item in output['clavicle_spine_geometry']['shoulder_context'].values())


def test_q2_optional_clavicle_extent_is_bound_to_the_current_a4_observation(harness, tmp_path):
    result = harness.run('Q2', specs=specs_with_projection(), include_optional=True,
                         inputs={'clavicle_mask_name': 'Right_Clavicle'}, out_dir=tmp_path / 'q2')
    assert result.status == 'ok', result.functions[0].refusal
    result_value = result.functions[0].outputs
    optional = result_value['applicability']['optional_bone_extent'][0]
    assert optional['parameters']['mask_name'] == 'Right_Clavicle'
    assert optional['bbox_width_px'] == 1


def test_q2_malformed_optional_extent_does_not_block_rotation_evidence(harness, tmp_path):
    metric = harness.registry._tools['measure_region_extent']
    metric.run = lambda session, **params: []
    result = harness.run('Q2', specs=specs_with_projection(), include_optional=True,
                         inputs={'clavicle_mask_name': 'Right_Clavicle'}, out_dir=tmp_path / 'q2')
    assert result.status == 'ok', result.functions[0].refusal
    assert 'bone_extent' in result.functions[0].missing_optional
    assert result.functions[0].outputs['rotation_index'] == 0
    assert not validate_graph(result.session)


def test_q3_reports_display_values_not_exposure_dose(harness, tmp_path):
    result = harness.run('Q3', specs=specs_with_projection(), out_dir=tmp_path / 'q3')
    out = result.functions[0].outputs
    assert out['grayscale_statistics']['pixel_count'] == 320
    assert out['retrocardiac_contrast']['difference'] == 10
    assert out['applicability']['physical_exposure_dose'] is None


@pytest.mark.parametrize('fid', ['Q1', 'Q2'])
@pytest.mark.parametrize('view', ['lateral', 'unknown'])
def test_nonfrontal_projection_refuses_quality_measurements(harness, tmp_path, fid, view):
    harness.view = view
    result = harness.run(fid, specs=specs_with_projection(), out_dir=tmp_path / fid)
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


@pytest.mark.parametrize('fid', ['Q1', 'Q2', 'Q3'])
def test_unknown_quality_cannot_create_claim(harness, tmp_path, fid):
    harness.quality_status = 'unknown'
    result = harness.run(fid, specs=specs_with_projection(), out_dir=tmp_path / fid)
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


def test_q1_invalid_landmark_refuses(harness, tmp_path):
    harness.landmark_status = 'outside_image'
    result = harness.run('Q1', specs=specs_with_projection(), out_dir=tmp_path / 'q1')
    assert result.status == 'refused'


@pytest.mark.parametrize('mode', ['truncated', 'wrong_index'])
def test_q2_refuses_truncated_bone_or_inconsistent_rotation(harness, tmp_path, mode):
    if mode == 'truncated':
        harness.truncate_bone = True
    else:
        harness.rotation_index = 0.7
    result = harness.run('Q2', specs=specs_with_projection(), out_dir=tmp_path / 'q2')
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


def test_q3_refuses_required_model_failure(harness, tmp_path):
    harness.refuse.add('assess_image_quality')
    result = harness.run('Q3', specs=specs_with_projection(), out_dir=tmp_path / 'q3')
    assert result.status == 'refused'
    assert result.functions[0].refusal.tool == 'assess_image_quality'
