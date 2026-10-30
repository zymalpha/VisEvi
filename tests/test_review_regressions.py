"""Independent examples for issues found in the uploaded source snapshot."""
import json

import numpy as np
import pytest
from PIL import Image
from jsonschema import Draft202012Validator, validate

from visevi.construction.teachers import ComposeEvidenceCTR
from visevi.core.session import EvidenceSession
from visevi.eval.runner import build_registry
from visevi.perception.anatomy import DetectLandmarks, LM_GROUPS
from visevi.perception.anatomy.pspnet import file_sha256
from visevi.perception.geometry import mirror_dice


@pytest.mark.parametrize('heart_bounds,direction', [((15, 85), 'present'), ((40, 60), 'absent')])
def test_ctr_verdict_agrees_with_claim(heart_bounds, direction):
    session = EvidenceSession(image_ref='synthetic')
    right, left, heart = [np.zeros((100, 100), bool) for _ in range(3)]
    right[5:95, 5:40] = True
    left[5:95, 60:95] = True
    low, high = heart_bounds
    heart[40:80, low:high] = True
    session.attach_masks({'Right_Lung': right, 'Left_Lung': left, 'Heart': heart})
    points = np.zeros((120, 2))
    for group, (a, b) in LM_GROUPS.items():
        lo, hi = {'right_lung': (5, 40), 'left_lung': (60, 94), 'heart': heart_bounds}[group]
        points[a:b, 0] = np.linspace(lo, hi, b - a)
        points[a:b, 1] = 50
    session.attach_landmarks({'lm_px': points, 'groups': LM_GROUPS})
    claim = ComposeEvidenceCTR().run(session)[0]
    assert claim.value['supported_direction'] == direction
    assert session.verdict.answerable
    assert f'支持方向 {direction}' in session.verdict.answer
    assert ('未检出' in session.verdict.answer) == (direction == 'absent')

@pytest.mark.parametrize('size,expected', [((600, 1000), [300, 500]), ((1000, 600), [500, 300]), ((1000, 1000), [500, 500])])
def test_letterbox_center_mapping(tmp_path, size, expected):
    image = tmp_path / 'image.png'
    landmarks = tmp_path / 'landmarks.json'
    Image.new('L', size).save(image)
    landmarks.write_text(json.dumps({'landmarks_norm': [[0.5, 0.5]] * 120,
                                     'image_sha256': file_sha256(image)}))
    session = EvidenceSession(image_ref=str(image))
    units = DetectLandmarks().run(session, image_ref=str(image), landmarks_json=str(landmarks))
    np.testing.assert_array_equal(session.landmarks['lm_px'][0], expected)
    assert all(unit.confidence is None for unit in units)


@pytest.mark.parametrize('payload', [{'landmarks_norm': [[0.5, 0.5]] * 120}, {}, []])
def test_unbound_or_malformed_landmarks_are_refusals(tmp_path, payload):
    from visevi.core.units import ToolRefusal

    image = tmp_path / 'image.png'
    landmarks = tmp_path / 'landmarks.json'
    Image.new('L', (10, 10)).save(image)
    landmarks.write_text(json.dumps(payload))
    session = EvidenceSession(image_ref=str(image))
    result = build_registry().run('detect_landmarks', session, image_ref=str(image),
                                  landmarks_json=str(landmarks))
    assert isinstance(result, ToolRefusal)
    assert not session.landmarks

def test_real_registry_schemas_and_refusals_are_json():
    registry = build_registry()
    specs = json.loads(json.dumps(registry.describe_all()))
    for spec in specs:
        Draft202012Validator.check_schema(spec['input_schema'])
        Draft202012Validator.check_schema(spec['output_schema'])
    spec = next(s for s in specs if s['name'] == 'compose_evidence_ctr')
    refusal = registry.run('compose_evidence_ctr', EvidenceSession(image_ref='missing'))
    validate(refusal.to_dict(), spec['output_schema'])

def test_reflection_does_not_repeat_border_pixels():
    right = np.zeros((3, 6), bool)
    right[:, 0] = True
    left = np.zeros_like(right)
    left[:, 2] = True
    # Center x=1: only column 2 is a reflection of column 0.
    assert mirror_dice(right, left, 1)['mirror_dice'] == 1.0

def test_two_empty_mirrored_masks_are_not_a_valid_measurement():
    empty = np.zeros((3, 6), bool)
    with pytest.raises(ValueError, match='undefined'):
        mirror_dice(empty, empty, 2.5)
