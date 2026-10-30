"""Model-backed A tools: result conversion, refusals and session effects (stubbed backends)."""
import numpy as np
import pytest
from PIL import Image

from visevi import backends
from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.perception import landmarks, findings, structures


def fake_run(result, artifacts=(), runtime=None, cache_hit=False):
    return {'result': result, 'artifacts': {name: 'sha' for name in artifacts}, 'cache_hit': cache_hit,
            'binding': {'image_sha256': 'a' * 64, 'weights': {'revision': 'rev'}},
            'worker_runtime': runtime or {}}


@pytest.fixture
def session(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('L', (40, 30), color=50).save(image)
    return EvidenceSession(image_ref=str(image))


def stub(monkeypatch, run):
    calls = []

    def run_cached(backend, image_path, output_dir, device, params):
        calls.append((backend.name, device, params))
        return run
    monkeypatch.setattr(backends, 'run_cached', run_cached)
    return calls


def test_a5_semantic_nodes_units_and_landmarks(session, monkeypatch, tmp_path):
    nodes = np.column_stack([np.linspace(1, 38, 120), np.linspace(1, 28, 120)])
    nodes[21] = [-5, 10]                                   # costophrenic node outside the image
    calls = stub(monkeypatch, fake_run({'nodes_xy': nodes.tolist(), 'frame': {}, 'coordinate_convention': 'px'}))
    out = landmarks.LocateLungLandmarks().run(session, output_dir=str(tmp_path / 'o'), device='cpu',
                                             uncertainty_samples=5, seed=3)
    assert calls == [('chexmask-u', 'cpu', {'uncertainty_samples': 5, 'seed': 3})]
    by_name = {u.finding_or_anatomy: u for u in out}
    assert by_name['right_lung_apex'].value['point_xy'] == nodes[0].tolist()
    assert by_name['right_costophrenic_angle'].value['status'] == 'outside_image'
    assert by_name['left_costophrenic_angle'].value['node_index'] == 65
    assert len({u.id for u in out}) == len(out) == 9
    assert all(u.confidence is None and u.source_tool == 'locate_lung_landmarks' for u in out)
    assert session.landmarks['semantic_nodes']['left_lung_apex'] == 44


def test_a5_parameter_validation(session, monkeypatch, tmp_path):
    stub(monkeypatch, None)
    assert isinstance(landmarks.LocateLungLandmarks().run(session), ToolRefusal)
    bad = landmarks.LocateLungLandmarks().run(session, output_dir=str(tmp_path), uncertainty_samples=500)
    assert isinstance(bad, ToolRefusal) and 'uncertainty_samples' in bad.reason


def test_a8_mask_is_attached_with_provenance(session, monkeypatch, tmp_path):
    out_dir = tmp_path / 'ptx'
    out_dir.mkdir()
    mask = np.zeros((30, 40), np.uint8)
    mask[5:10, 3:8] = 255
    Image.fromarray(mask).save(out_dir / 'pneumothorax_mask.png')
    stub(monkeypatch, fake_run({'case_score': 0.7, 'mask_threshold': 0.5, 'max_pixel_probability': 0.9,
                                'frame': {}}, ['pneumothorax_mask.png']))
    out = findings.SegmentPneumothorax().run(session, output_dir=str(out_dir))
    assert out[0].value['area_px'] == 25 and out[0].value['status'] == 'region_above_threshold'
    assert session.mask_metadata['Pneumothorax_Candidate']['observation_id'] == out[0].id
    assert session.mask_provenance(['Pneumothorax_Candidate']) == [out[0].id]


def test_a26_screening_policy_controls_assertion_not_raw_outputs(session, monkeypatch, tmp_path):
    out_dir = tmp_path / 'dev'
    out_dir.mkdir()
    devices = {}
    for name, score in [('ETT', 0.9), ('NGT', 0.1), ('CVC', 0.2), ('Swan Ganz', 0.0)]:
        file_name = f'line_{name.replace(" ", "_")}.png'
        line = np.zeros((30, 40), np.uint8)
        line[2:20, 20] = 255
        Image.fromarray(line).save(out_dir / file_name)
        devices[name] = {'presence_raw_score': score, 'line_mask_file': file_name,
                         'distal_tip_xy': [20.0, 19.0], 'distal_tip_peak_probability': 0.4}
    stub(monkeypatch, fake_run({'devices': devices, 'carina_xy': [20.0, 8.0], 'carina_heatmap_peak': 0.6,
                                'classification_raw': {}, 'fold_ids': {}},
                               [d['line_mask_file'] for d in devices.values()], {'screening_threshold': 0.5}))
    out = {u.finding_or_anatomy: u for u in findings.LocalizeDevices().run(session, output_dir=str(out_dir))}
    assert out['endotracheal_tube'].value['status'] == 'present'
    assert out['nasogastric_tube'].value['status'] == 'below_screening_threshold'
    assert out['nasogastric_tube'].value['distal_tip_xy'] == [20.0, 19.0]      # raw output kept
    assert sorted(session.masks) == ['Device_endotracheal_tube']             # only asserted devices attach
    assert out['carina'].value['point_xy'] == [20.0, 8.0]


def test_a12_requires_a2_heart_and_respects_fov(session, monkeypatch, tmp_path):
    stub(monkeypatch, None)
    tool = structures.MeasureCTR()
    refusal = tool.run(session, output_dir=str(tmp_path / 'c'))
    assert isinstance(refusal, ToolRefusal) and refusal.missing == ['mask:Heart']
    heart = np.zeros((30, 40), bool)
    heart[10:20, 10:30] = True
    session.attach_masks({'Heart': heart}, {'Heart': {'fov_truncated_sides': ['left']}})
    refusal = tool.run(session, output_dir=str(tmp_path / 'c'))
    assert isinstance(refusal, ToolRefusal) and refusal.missing == ['fov_complete:Heart']


def test_backend_failures_become_refusals(session, monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise backends.WorkerError('worker exited')
    monkeypatch.setattr(backends, 'run_cached', broken)
    out = findings.SegmentPneumothorax().run(session, output_dir=str(tmp_path / 'x'))
    assert isinstance(out, ToolRefusal) and 'worker exited' in out.reason
    assert not session.graph.nodes
