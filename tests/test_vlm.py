"""CheXOne prompt/answer parsing and A6/A7/A18/A24 conversion (stubbed backend)."""
import numpy as np
import pytest
from PIL import Image

from visevi import backends
from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.perception import vlm
from visevi.workers import chexone


def test_chexone_worker_uses_model_card_user_only_conversation():
    content = [{'type': 'text', 'text': 'test'}]
    assert chexone.build_messages(content) == [{'role': 'user', 'content': content}]


@pytest.mark.parametrize('text,status,boxes', [
    ('<|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|>', 'detected', [(12, 60, 18, 87)]),
    ('<|ref|>nodule<|/ref|><|box|>(1,2),(3,4)<|/box|><|box|>(50,50),(60,70)<|/box|>', 'detected',
     [(1, 2, 3, 4), (50, 50, 60, 70)]),
    ('No pleural effusion detected.', 'not_detected', []),
    ('<|box|>(60,10),(20,30)<|/box|>', 'unparsed', []),          # x0 > x1 rejected, not repaired
    ('<|box|>(10,10),(120,30)<|/box|>', 'unparsed', []),         # outside the 0-100 grid
    ('I think there is something.', 'unparsed', []),
])
def test_parse_boxes(text, status, boxes):
    parsed = vlm.parse_boxes(text)
    assert parsed['status'] == status and parsed['boxes_grid'] == boxes


@pytest.mark.parametrize('text,letter', [('A', 'A'), ('(B)', 'B'), ('Answer: C.', 'C'), ('A or B', None), ('PA', None)])
def test_parse_option(text, letter):
    assert vlm.parse_option(text, 'ABC') == letter


def test_grid_mapping_and_box_mask():
    assert vlm.grid_to_pixels((10, 20, 50, 100), width=200, height=100) == [20, 20, 100, 100]
    assert vlm.grid_to_pixels((0, 0, 50, 50), 200, 100, crop=[10, 20, 50, 60]) == [10, 20, 30, 40]
    mask = vlm.box_mask([[2, 1, 5, 3]], width=8, height=5)       # half-open edges -> pixels 2..4 x 1..2
    assert np.argwhere(mask).tolist() == [[1, 2], [1, 3], [1, 4], [2, 2], [2, 3], [2, 4]]


@pytest.fixture
def session(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('L', (200, 100), color=50).save(image)
    return EvidenceSession(image_ref=str(image))


def stub(monkeypatch, answers):
    seen = {}

    def run_cached(backend, image_path, output_dir, device, params):
        seen['params'] = params
        return {'result': {'answers': answers, 'request_params': {k: v for k, v in params.items() if k != 'requests'}},
                'artifacts': {}, 'cache_hit': False,
                'binding': {'image_sha256': 'a' * 64, 'weights': {'revision': 'rev'}}, 'worker_runtime': {}}
    monkeypatch.setattr(backends, 'run_cached', run_cached)
    return seen


def test_view_is_a_prediction_not_an_assumption(session, monkeypatch, tmp_path):
    stub(monkeypatch, [{'text': 'B', 'first_token_option_probabilities': {'A': 0.1, 'B': 0.8, 'C': 0.1}}])
    out = vlm.ClassifyView().run(session, output_dir=str(tmp_path / 'v'))
    assert out[0].value['view'] == 'AP' and out[0].value['assumed'] is False
    stub(monkeypatch, [{'text': 'maybe'}])
    assert vlm.ClassifyView().run(session, output_dir=str(tmp_path / 'w'))[0].value['view'] == 'unknown'


def test_abnormality_boxes_become_candidate_masks(session, monkeypatch, tmp_path):
    seen = stub(monkeypatch, [{'text': '<|ref|>nodule<|/ref|><|box|>(10,20),(20,40)<|/box|>'},
                              {'text': 'No pneumothorax detected.'}])
    out = vlm.DetectAbnormalities().run(session, output_dir=str(tmp_path / 'd'), findings=['nodule', 'pneumothorax'])
    assert [u.value['status'] for u in out] == ['detected', 'not_detected']
    assert out[0].value['boxes_xyxy_px'] == [[20.0, 20.0, 40.0, 40.0]]
    assert int(session.masks['Candidate_nodule'].sum()) == 20 * 20
    assert len(seen['params']['requests']) == 2
    bad = vlm.DetectAbnormalities().run(session, output_dir=str(tmp_path / 'e'), findings=['unicorn'])
    assert isinstance(bad, ToolRefusal)


def test_phrase_grounding_rejects_markup(session, monkeypatch, tmp_path):
    stub(monkeypatch, [{'text': 'No left basal opacity detected.'}])
    out = vlm.GroundPhrase().run(session, output_dir=str(tmp_path / 'p'), phrase='left basal opacity')
    assert out[0].value['status'] == 'not_detected' and 'Phrase_Region' not in session.masks
    assert isinstance(vlm.GroundPhrase().run(session, output_dir=str(tmp_path / 'q'), phrase='<|box|>'), ToolRefusal)


def test_roi_attributes_use_mask_crop_and_provenance(session, monkeypatch, tmp_path):
    roi = np.zeros((100, 200), bool)
    roi[40:60, 100:150] = True
    session.attach_masks({'Lesion': roi}, {'Lesion': {'observation_id': 'obs-0042'}})
    seen = stub(monkeypatch, [{'text': 'A'}, {'text': 'C'}])
    out = vlm.ClassifyRoiAttributes().run(session, output_dir=str(tmp_path / 'r'), roi_mask_name='Lesion',
                                          attributes=['consolidation', 'cavitation'])
    assert [u.value['answer'] for u in out] == ['present', 'uncertain']
    assert seen['params']['crop_xyxy'] == [92, 32, 158, 68]
    assert out[0].provenance == ['obs-0042']
    missing = vlm.ClassifyRoiAttributes().run(session, output_dir=str(tmp_path / 's'), roi_mask_name='None')
    assert isinstance(missing, ToolRefusal) and missing.missing == ['mask:None']
