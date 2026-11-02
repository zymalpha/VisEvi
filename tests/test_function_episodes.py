"""Function integration contracts with synthetic images and real spatial A21.

Model callbacks supply fixed observations; no weights/GPU/clinical labels are
used. Expected laterality comes from hand-placed regions, rather than another
call to the implementation under test.
"""
from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest

from visevi.construction.evidence import validate_graph
from visevi.core.registry import ToolRegistry
from visevi.core.session import EvidenceSession
from visevi.core.spec import ToolSpec, evidence_result_schema, parameter_schema
from visevi.core.units import EvidenceUnit, ToolRefusal
from visevi.functions.runner import run_function_episode
from visevi.functions import runner
from visevi.functions.specs import ToolStep, load_function_specs
from visevi.perception.metrics import build_metric_tools


class FakeModelTool:
    def __init__(self, name, harness, properties=None):
        self.name, self.harness = name, harness
        self.properties = {'output_dir': {'type': 'string'}, 'device': {'type': 'string'},
                           **(properties or {})}

    def describe(self):
        return ToolSpec(self.name, 'perception', 'test model',
                        parameter_schema(self.properties, ('output_dir',)), evidence_result_schema())

    def run(self, session, **params):
        h = self.harness
        h.calls.append((self.name, params))
        if self.name in h.refuse:
            return ToolRefusal(self.name, 'synthetic required evidence unavailable')
        return getattr(h, self.name)(session, params)


class Harness:
    def __init__(self, image):
        self.image = image
        self.calls = []
        self.refuse = set()
        self.view = 'PA'
        self.view_status = 'ok'
        self.view_assumed = False
        self.ctr = 0.5
        self.heart_width = 4.0
        self.thorax_width = 8.0
        self.quality_bad = None
        self.candidate_status = 'detected'
        self.truncated_heart = False
        self.candidate = np.zeros((12, 16), bool)
        self.candidate[5:7, 2:4] = True
        self.registry = ToolRegistry()
        for tool in build_metric_tools():
            self.registry.register(tool)
        for name, properties in {
            'classify_view': {'marker': {'type': 'string'}},
            'assess_image_quality': {},
            'segment_heart': {},
            'measure_ctr': {'heart_mask_name': {'type': 'string'}},
            'segment_lungs': {},
            'detect_abnormalities': {'findings': {'type': 'array'}},
        }.items():
            self.registry.register(FakeModelTool(name, self, properties))

    def emit(self, session, tool, target, value, *, provenance=()):
        node = EvidenceUnit(session.next_id('obs'), 'observation', 'attribute', target,
                            value, source_tool=tool, provenance=list(provenance),
                            validity_conditions=['synthetic prediction; raw scores are uncalibrated'])
        session.add_observation(node)
        return node

    def attach(self, session, name, mask, owner, **extra):
        session.attach_masks({name: mask}, {name: {'observation_id': owner.id,
            'coordinate_frame': 'original_image_pixels', 'fov_truncated_sides': [], **extra}})

    def classify_view(self, session, params):
        return [self.emit(session, 'classify_view', 'view_position', {
            'view': self.view, 'status': self.view_status, 'assumed': self.view_assumed,
            'raw_text': 'A', 'option_raw_probabilities': {'A': 0.91, 'B': 0.08, 'C': 0.01}})]

    def assess_image_quality(self, session, params):
        values = {'rotation': {'asymmetry_index': 0.1},
                  'inspiration': {'posterior': {'count_above': 9}},
                  'penetration': {'difference': 8},
                  'field_of_view': {'right lung': {'segmented': True,
                      **{f'touches_{side}': False for side in ('top', 'bottom', 'left', 'right')}}}}
        for value in values.values():
            value['status'] = 'ok'
        if self.quality_bad == 'unknown':
            values['rotation']['status'] = 'unknown'
        elif self.quality_bad == 'border':
            values['field_of_view']['right lung']['touches_top'] = True
        elif self.quality_bad == 'missing_structure':
            values['field_of_view']['right lung']['segmented'] = False
        return [self.emit(session, 'assess_image_quality', name, value,
                          provenance=['cached:CXAS:synthetic']) for name, value in values.items()]

    def segment_heart(self, session, params):
        mask = np.zeros((12, 16), bool)
        mask[4:10, 6:10] = True
        unit = self.emit(session, 'segment_heart', 'Heart', {'area_px': 24})
        self.attach(session, 'Heart', mask, unit,
                    fov_truncated_sides=['top'] if self.truncated_heart else [])
        return [unit]

    def measure_ctr(self, session, params):
        assert 'Heart' in session.masks, 'dependency planner must run segment_heart first'
        return [self.emit(session, 'measure_ctr', 'cardiothoracic_ratio', {
            'CTR': self.ctr, 'heart_width_px': self.heart_width,
            'thoracic_inner_width_px': self.thorax_width,
            'heart_mask_name': params['heart_mask_name']},
            provenance=session.mask_provenance(['Heart']) + ['cached:CXAS:synthetic'])]

    def segment_lungs(self, session, params):
        units = []
        for name, columns in [('Right_Lung', slice(1, 6)), ('Left_Lung', slice(10, 15))]:
            mask = np.zeros((12, 16), bool)
            mask[2:10, columns] = True
            owner = self.emit(session, 'segment_lungs', name, {'area_px': 40})
            self.attach(session, name, mask, owner)
            units.append(owner)
        return units

    def detect_abnormalities(self, session, params):
        finding = params['findings'][0]
        unit = self.emit(session, 'detect_abnormalities', 'candidate_' + finding, {
            'finding': finding, 'status': self.candidate_status,
            'boxes_xyxy_px': [[2, 5, 4, 7]] if self.candidate_status == 'detected' else []})
        if self.candidate_status == 'detected':
            self.attach(session, 'Candidate_' + finding.replace(' ', '_'), self.candidate.copy(), unit)
        return [unit]

    def run(self, function_id=None, **kwargs):
        return run_function_episode(function_id, image_ref=str(self.image),
                                    registry=self.registry, **kwargs)

    def names(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def harness(tmp_path):
    image = tmp_path / 'synthetic.png'
    Image.new('L', (16, 12), 80).save(image)
    return Harness(image)


def test_q4_builds_three_layers_without_turning_raw_score_into_confidence(harness, tmp_path):
    result = harness.run('Q4', out_dir=tmp_path / 'run')
    assert result.status == 'ok'
    nodes = list(result.session.graph.nodes.values())
    assert [node.category for node in nodes] == ['observation', 'relation', 'claim']
    assert all(node.confidence is None for node in nodes)
    assert result.functions[0].outputs['probability_calibrated'] is False
    assert result.functions[0].outputs['option_raw_probabilities']['A'] == 0.91
    assert result.session.verdict.evidence_refs == [nodes[-1].id]
    assert validate_graph(result.session) == []


@pytest.mark.parametrize('field,value', [('view_status', 'unparsed'), ('view', 'unknown'),
                                         ('view_assumed', True)])
def test_q4_cannot_claim_unknown_or_assumed_projection(harness, tmp_path, field, value):
    setattr(harness, field, value)
    result = harness.run('Q4', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')
    assert result.session.verdict.answerable is False


def test_p9_reports_conditioned_measurement_and_retains_sources(harness, tmp_path):
    result = harness.run('P9', out_dir=tmp_path / 'run')
    assert result.status == 'ok'
    output = result.functions[0].outputs
    assert output['CTR'] == 0.5 and output['unit'] == 'ratio'
    assert output['projection'] == 'PA'
    assert output['enlargement'] is None and output['clinical_interpretation'] == 'not_assessed'
    assert harness.names().index('segment_heart') < harness.names().index('measure_ctr')
    relation = result.session.graph.nodes[result.functions[0].evidence_refs.relation_ids[0]]
    assert {result.session.graph.nodes[i].source_tool for i in relation.provenance} == {
        'classify_view', 'assess_image_quality', 'segment_heart', 'measure_ctr'}
    assert validate_graph(result.session) == []


@pytest.mark.parametrize('projection', ['AP', 'lateral'])
def test_p9_non_pa_stops_before_downstream_models(harness, tmp_path, projection):
    harness.view = projection
    result = harness.run('P9', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert 'requires PA' in result.functions[0].refusal.reason
    assert harness.names() == ['classify_view']
    assert not result.session.graph.by_category('claim')


@pytest.mark.parametrize('condition', ['unknown', 'border', 'missing_structure'])
def test_p9_unknown_quality_or_incomplete_fov_does_not_pass(harness, tmp_path, condition):
    harness.quality_bad = condition
    result = harness.run('P9', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert 'segment_heart' not in harness.names()
    assert not result.session.graph.by_category('claim')


@pytest.mark.parametrize('ctr', [0, 1.1, True, float('nan')])
def test_p9_rejects_invalid_ctr_as_structured_refusal(harness, tmp_path, ctr):
    harness.ctr = ctr
    result = harness.run('P9', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


def test_p9_truncated_heart_cannot_create_measurement_claim(harness, tmp_path):
    harness.truncated_heart = True
    result = harness.run('P9', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert 'truncated' in result.functions[0].refusal.reason


@pytest.mark.parametrize('heart,thorax,ctr', [(4.0, 0.0, 0.5), (4.0, 8.0, 0.7),
                                               (-1.0, 8.0, 0.5)])
def test_p9_widths_must_be_positive_and_reproduce_ctr(harness, tmp_path, heart, thorax, ctr):
    harness.heart_width, harness.thorax_width, harness.ctr = heart, thorax, ctr
    result = harness.run('P9', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert not result.session.graph.by_category('claim')


def test_p9_ctr_cannot_be_rebound_to_an_unrelated_mask(harness, tmp_path):
    result = harness.run('P9', out_dir=tmp_path / 'run',
                         tool_parameters={'P9.ctr': {'heart_mask_name': 'Other'}})
    assert result.status == 'refused'
    assert 'current Heart mask' in result.functions[0].refusal.reason
    assert not result.session.graph.by_category('claim')


def test_g1_real_a21_distinguishes_image_side_from_patient_side(harness, tmp_path):
    result = harness.run('G1', inputs={'finding': 'nodule'}, out_dir=tmp_path / 'run')
    assert result.status == 'ok'
    output = result.functions[0].outputs
    assert output['image_sides'] == ['image_left']
    assert output['patient_sides'] is None
    assert output['lung_overlap'] == {'Right_Lung': 1.0, 'Left_Lung': 0.0}
    assert result.executed_tools.count('relate_lesion_to_anatomy') == 2
    assert validate_graph(result.session) == []


@pytest.mark.parametrize('right_is_image_left,expected', [(True, 'patient_right'), (False, 'patient_left')])
def test_g1_patient_side_requires_explicit_verified_orientation(harness, tmp_path,
                                                              right_is_image_left, expected):
    result = harness.run('G1', inputs={'finding': 'nodule', 'orientation_verified': True,
        'patient_right_is_image_left': right_is_image_left, 'orientation_source': 'checked image marker'},
        out_dir=tmp_path / 'run')
    assert result.status == 'ok'
    assert result.functions[0].outputs['patient_sides'] == [expected]
    assert result.functions[0].outputs['orientation_source'] == 'checked image marker'
    orientation = [node for node in result.session.graph.by_category('observation')
                   if node.source_tool == 'verified_orientation_input']
    assert len(orientation) == 1
    relation_id = result.functions[0].evidence_refs.relation_ids[0]
    assert orientation[0].id in result.session.graph.nodes[relation_id].provenance


def test_g1_rejects_a_candidate_mask_name_unbound_to_the_requested_finding(harness, tmp_path):
    with pytest.raises(ValueError, match='must match the requested finding'):
        harness.run('G1', inputs={'finding': 'nodule', 'lesion_mask_name': 'Right_Lung'},
                    out_dir=tmp_path / 'run')
    assert not harness.calls


def test_g1_bilateral_box_union_is_preserved(harness, tmp_path):
    harness.candidate[5:7, 11:13] = True
    result = harness.run('G1', inputs={'finding': 'nodule'}, out_dir=tmp_path / 'run')
    assert result.status == 'ok'
    assert result.functions[0].outputs['image_sides'] == ['image_left', 'image_right']
    assert result.functions[0].outputs['lung_overlap'] == {'Right_Lung': 0.5, 'Left_Lung': 0.5}


def test_g1_no_candidate_skips_measurement_and_does_not_claim_disease_absence(harness, tmp_path):
    harness.candidate_status = 'not_detected'
    result = harness.run('G1', inputs={'finding': 'nodule'}, out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert 'disease absence is not established' in result.functions[0].refusal.reason
    assert not result.session.graph.by_category('claim')
    assert 'relate_lesion_to_anatomy' not in result.executed_tools


def test_g1_unparsed_candidate_is_not_treated_as_negative(harness, tmp_path):
    harness.candidate_status = 'unparsed'
    result = harness.run('G1', inputs={'finding': 'nodule'}, out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert 'unparsed' in result.functions[0].refusal.reason
    assert 'relate_lesion_to_anatomy' not in result.executed_tools


def test_g1_missing_slot_is_refused_before_tools_execute(harness, tmp_path):
    result = harness.run('G1', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert 'missing function input' in result.functions[0].refusal.reason
    assert not harness.calls


def test_required_tool_refusal_is_retained_and_stops_claims(harness, tmp_path):
    harness.refuse.add('measure_ctr')
    result = harness.run('P9', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert result.functions[0].refusal.tool == 'measure_ctr'
    assert result.session.trace[-1]['result'] == 'refusal'
    assert not result.evidence_refs


def test_failed_function_program_rolls_back_its_relation_and_edges(harness, tmp_path, monkeypatch):
    original = runner._PROGRAMS['Q4']

    def malformed(session, spec, outputs, inputs):
        value, relation, support = original(session, spec, outputs, inputs)
        return {}, relation, support

    monkeypatch.setitem(runner._PROGRAMS, 'Q4', malformed)
    result = harness.run('Q4', out_dir=tmp_path / 'run')
    assert result.status == 'refused'
    assert result.session.graph.by_category('observation')
    assert not result.session.graph.by_category('relation')
    assert not result.session.graph.by_category('claim')
    assert not result.session.graph.edges


def test_optional_tool_refusal_is_recorded_but_does_not_block_q4(harness, tmp_path):
    catalog = load_function_specs()
    spec = catalog['Q4']
    catalog['Q4'] = replace(spec, steps=spec.steps + (ToolStep('quality', 'assess_image_quality', required=False),))
    harness.refuse.add('assess_image_quality')
    result = harness.run('Q4', specs=catalog, include_optional=True, out_dir=tmp_path / 'run')
    assert result.status == 'ok'
    assert result.functions[0].missing_optional == ['quality']
    assert result.functions[0].steps[-1]['status'] == 'refusal'


def test_unknown_planned_and_ambiguous_requests_do_not_execute_tools(harness, tmp_path):
    unknown = harness.run('P99')
    assert unknown.status == 'unknown' and not unknown.functions
    planned = harness.run('T4')
    assert planned.status == 'refused'
    assert 'drain_reference' in planned.functions[0].refusal.reason
    ambiguous = harness.run(query='请检查投照和心胸比')
    assert ambiguous.status == 'ambiguous' and not ambiguous.functions
    assert not harness.calls


def test_multi_function_episodes_share_a_observations(harness, tmp_path):
    result = harness.run(['Q4', 'P9'], out_dir=tmp_path / 'run')
    assert result.status == 'ok'
    assert result.function_ids == ('Q4', 'P9')
    assert harness.names().count('classify_view') == 1
    assert result.reused_tools == ['classify_view']
    q4_view = result.functions[0].evidence_refs.observation_ids[0]
    assert q4_view in result.functions[1].evidence_refs.observation_ids
    assert len(result.evidence_refs) == 2
    assert validate_graph(result.session) == []


def test_repeat_session_reuses_a_calls_but_parameter_change_invalidates_them(harness, tmp_path):
    first = harness.run('Q4', out_dir=tmp_path / 'first')
    second = harness.run('Q4', session=first.session, out_dir=tmp_path / 'second')
    assert second.status == 'ok' and second.reused_tools == ['classify_view']
    assert harness.names().count('classify_view') == 1
    third = harness.run('Q4', session=first.session, out_dir=tmp_path / 'third',
                        tool_parameters={'classify_view': {'marker': 'changed'}})
    assert third.status == 'ok' and third.executed_tools == ['classify_view']
    assert harness.names().count('classify_view') == 2


def test_overwritten_output_mask_invalidates_its_producer_cache(harness, tmp_path):
    first = harness.run('P9', out_dir=tmp_path / 'first')
    assert first.status == 'ok'
    first.session.masks['Heart'][4, 6] = False
    second = harness.run('P9', session=first.session, out_dir=tmp_path / 'second')
    assert second.status == 'ok'
    assert harness.names().count('segment_heart') == 2
    assert harness.names().count('measure_ctr') == 2
    assert harness.names().count('classify_view') == 1
    assert first.session.masks['Heart'][4, 6]


def test_existing_out_dir_is_never_overwritten(harness, tmp_path):
    out = tmp_path / 'existing'
    out.mkdir()
    sentinel = out / 'session.json'
    sentinel.write_text('preserve this result')
    with pytest.raises(ValueError, match='new directory'):
        harness.run('Q4', out_dir=out)
    assert sentinel.read_text() == 'preserve this result'
    assert not harness.calls


def test_saved_episode_and_session_reload_without_runtime_mask_cache(harness, tmp_path):
    out = tmp_path / 'run'
    result = harness.run('P9', out_dir=out)
    payload = json.loads((out / 'episode.json').read_text())
    assert payload['status'] == 'ok'
    loaded = EvidenceSession.load(out / 'session.json')
    assert loaded.to_dict() == result.session.to_dict()
    assert not loaded.masks and validate_graph(loaded) == []
    replay = harness.run('P9', session=loaded, out_dir=tmp_path / 'replay')
    assert replay.status == 'ok'
    assert not replay.reused_tools
    assert harness.names().count('segment_heart') == 2
