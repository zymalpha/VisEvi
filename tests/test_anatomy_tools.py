"""A1-A4 interface, frame binding, and all-or-nothing cache loading tests."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

from visevi.core.session import EvidenceSession
from visevi.core.units import ToolRefusal
from visevi.eval.runner import build_a_registry
from visevi.perception.anatomy import ANATOMY_GROUPS, ORGANS, SegmentAnatomy
from visevi.perception.anatomy.pspnet import (DOCUMENTED_WEIGHTS_SHA256, MODEL_NAME,
                                              PREPROCESSING_VERSION, file_sha256, image_frame)


def valid_record(image, hashes):
    """Minimal record accepted for cached artifacts: image, model, weights, preprocessing."""
    return {'binding': {'image_sha256': file_sha256(image), 'preprocessing': image_frame(image),
                        'producer': {'model': MODEL_NAME, 'weights_sha256': DOCUMENTED_WEIGHTS_SHA256,
                                     'preprocessing_version': PREPROCESSING_VERSION}},
            'mask_sha256': hashes}


@pytest.mark.parametrize('mode', ['I;16', 'I', 'F'])
def test_pspnet_refuses_unwindowed_high_precision_rasters(tmp_path, mode):
    from visevi.perception.anatomy.pspnet import prepare_image

    path = tmp_path / 'high_precision.tiff'
    Image.new(mode, (8, 8), color=1024).save(path)
    with pytest.raises(ValueError, match='windowing'):
        prepare_image(path)


def test_malformed_live_cache_is_a_structured_refusal(tmp_path, monkeypatch):
    from visevi.perception.anatomy import tools

    image = tmp_path / 'image.png'
    Image.new('L', (8, 8), color=100).save(image)

    class BrokenCachePredictor:
        def predict(self, *args):
            raise KeyError('binding')

    monkeypatch.setattr(tools, '_predictor', lambda *args: BrokenCachePredictor())
    session = EvidenceSession(image_ref=str(image))
    result = build_a_registry().run('segment_lungs', session, output_dir=str(tmp_path / 'bad_cache'))
    assert isinstance(result, ToolRefusal)
    assert not session.masks and not session.graph.nodes


@pytest.fixture
def anatomy_artifacts(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('L', (8, 12), color=100).save(image)
    masks = tmp_path / 'artifacts'
    masks.mkdir()
    names = {name for _, group in ANATOMY_GROUPS.values() for name in group}
    hashes = {}
    for name in names:
        pixels = np.zeros((512, 512), dtype=np.uint8)
        pixels[64:448, 64:448] = 255
        path = masks / (name + '.png')
        Image.fromarray(pixels).save(path)
        hashes[path.name] = file_sha256(path)
    (masks / 'cache_record.json').write_text(json.dumps(valid_record(image, hashes)))
    return image, masks


def test_legacy_anatomy_default_remains_lungs_and_heart(anatomy_artifacts):
    image, masks = anatomy_artifacts
    session = EvidenceSession(image_ref=str(image))
    result = SegmentAnatomy().run(session, image_ref=str(image), masks_dir=str(masks))
    assert not isinstance(result, ToolRefusal)
    assert {unit.finding_or_anatomy for unit in result} == set(ORGANS)
    assert len(result) == 3


def test_each_anatomy_capability_has_named_interface_and_fov(anatomy_artifacts):
    image, masks = anatomy_artifacts
    session = EvidenceSession(image_ref=str(image))
    registry = build_a_registry()
    for name, (capability, targets) in ANATOMY_GROUPS.items():
        result = registry.run(name, session, masks_dir=str(masks))
        assert not isinstance(result, ToolRefusal)
        assert {unit.finding_or_anatomy for unit in result} == set(targets)
        for unit in result:
            assert unit.value['capability'] == capability
            assert unit.confidence is None
            metadata = unit.value['mask_metadata']
            assert metadata['valid_fov_bbox_xyxy_exclusive'] == [0, 2, 8, 10]
            assert metadata['full_image_observed'] is False
            assert metadata['cache_binding_verified'] is True
            assert metadata['fov_truncated_sides'] == []
            assert session.mask_metadata[unit.finding_or_anatomy]['observation_id'] == unit.id
            assert session.masks[unit.finding_or_anatomy].shape == (12, 8)


def test_failed_multi_mask_load_does_not_partially_change_session(anatomy_artifacts):
    image, masks = anatomy_artifacts
    (masks / 'Right_Scapula.png').unlink()
    session = EvidenceSession(image_ref=str(image))
    result = build_a_registry().run('segment_shoulder_bones', session, masks_dir=str(masks))
    assert isinstance(result, ToolRefusal)
    assert not session.graph.nodes
    assert not session.masks


def test_cache_for_different_image_is_rejected(anatomy_artifacts, tmp_path):
    image, masks = anatomy_artifacts
    other = tmp_path / 'other.png'
    Image.new('L', (8, 12), color=20).save(other)
    session = EvidenceSession(image_ref=str(other))
    result = build_a_registry().run('segment_heart', session, masks_dir=str(masks))
    assert isinstance(result, ToolRefusal)
    assert 'bound' in result.reason
    assert not session.graph.nodes


def test_modified_cached_mask_is_rejected(anatomy_artifacts):
    image, masks = anatomy_artifacts
    Image.new('L', (512, 512), color=255).save(masks / 'Heart.png')
    result = build_a_registry().run('segment_heart', EvidenceSession(image_ref=str(image)), masks_dir=str(masks))
    assert isinstance(result, ToolRefusal)
    assert 'checksum' in result.reason


def test_live_mode_calls_predictor_and_reuses_generated_artifacts(anatomy_artifacts, tmp_path, monkeypatch):
    from visevi.perception.anatomy import tools

    image, source = anatomy_artifacts
    output = tmp_path / 'live_output'
    calls = []

    class PredictorStub:
        def predict(self, image_ref, sample_key, output_dir):
            calls.append((str(image_ref), sample_key))
            if not Path(output_dir).exists():
                shutil.copytree(source, output_dir)

    predictor = PredictorStub()
    monkeypatch.setattr(tools, '_predictor', lambda *args: predictor)
    registry = build_a_registry()
    session = EvidenceSession(image_ref=str(image))
    first = registry.run('segment_lungs', session, output_dir=str(output))
    second = registry.run('segment_heart', session, output_dir=str(output))
    assert not isinstance(first, ToolRefusal)
    assert not isinstance(second, ToolRefusal)
    assert first[0].value['execution_mode'] == 'live_inference'
    assert first[0].value['prediction_cache_hit'] is False
    assert second[0].value['prediction_cache_hit'] is True
    assert len(calls) == 2


def _rewrite_record(masks, change):
    path = masks / 'cache_record.json'
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def test_masks_without_cache_record_are_not_model_output(anatomy_artifacts):
    image, masks = anatomy_artifacts
    (masks / 'cache_record.json').unlink()
    session = EvidenceSession(image_ref=str(image))
    result = build_a_registry().run('segment_lungs', session, masks_dir=str(masks))
    assert isinstance(result, ToolRefusal)
    assert 'cache_record.json missing' in result.reason
    assert not session.masks


@pytest.mark.parametrize('field,value,message', [
    ('model', 'other_model', 'producer'),
    ('weights_sha256', '0' * 64, 'weights'),
    ('preprocessing_version', 99, 'preprocessing'),
])
def test_cache_from_other_producer_is_rejected(anatomy_artifacts, field, value, message):
    image, masks = anatomy_artifacts
    _rewrite_record(masks, lambda record: record['binding']['producer'].__setitem__(field, value))
    result = build_a_registry().run('segment_heart', EvidenceSession(image_ref=str(image)), masks_dir=str(masks))
    assert isinstance(result, ToolRefusal)
    assert message in result.reason


def test_cache_with_other_preprocessing_frame_is_rejected(anatomy_artifacts):
    image, masks = anatomy_artifacts
    _rewrite_record(masks, lambda record: record['binding']['preprocessing'].__setitem__('crop_offset_xy', [0, 0]))
    result = build_a_registry().run('segment_heart', EvidenceSession(image_ref=str(image)), masks_dir=str(masks))
    assert isinstance(result, ToolRefusal)
    assert 'preprocessing' in result.reason


def test_mask_without_recorded_checksum_is_rejected(anatomy_artifacts):
    image, masks = anatomy_artifacts
    _rewrite_record(masks, lambda record: record['mask_sha256'].pop('Heart.png'))
    result = build_a_registry().run('segment_heart', EvidenceSession(image_ref=str(image)), masks_dir=str(masks))
    assert isinstance(result, ToolRefusal)
    assert 'no checksum' in result.reason


def test_cached_mode_rejects_live_only_parameters(anatomy_artifacts):
    image, masks = anatomy_artifacts
    result = build_a_registry().run('segment_heart', EvidenceSession(image_ref=str(image)),
                                    masks_dir=str(masks), device='cuda')
    assert isinstance(result, ToolRefusal)


def test_mask_reaching_cropped_fov_edge_is_flagged_and_measurement_refused(tmp_path):
    # 8 wide x 12 high: the model sees rows 2..9; rows 0-1 and 10-11 are unobserved.
    image = tmp_path / 'portrait.png'
    Image.new('L', (8, 12), color=100).save(image)
    masks = tmp_path / 'artifacts'
    masks.mkdir()
    hashes = {}
    for name in ('Right_Lung', 'Left_Lung'):
        pixels = np.zeros((512, 512), dtype=np.uint8)
        columns = slice(64, 192) if name == 'Right_Lung' else slice(320, 448)
        pixels[0:448, columns] = 255   # Right_Lung and Left_Lung both reach the top crop edge
        Image.fromarray(pixels).save(masks / f'{name}.png')
        hashes[f'{name}.png'] = file_sha256(masks / f'{name}.png')
    (masks / 'cache_record.json').write_text(json.dumps(valid_record(image, hashes)))
    session = EvidenceSession(image_ref=str(image))
    registry = build_a_registry()
    lungs = registry.run('segment_lungs', session, masks_dir=str(masks))
    assert not isinstance(lungs, ToolRefusal)
    assert all(unit.value['mask_metadata']['fov_truncated_sides'] == ['top'] for unit in lungs)
    refusal = registry.run('compare_lung_geometry', session)
    assert isinstance(refusal, ToolRefusal)
    assert set(refusal.missing) == {'fov_complete:Right_Lung', 'fov_complete:Left_Lung'}


def test_exif_rotated_images_are_refused(tmp_path):
    from visevi.perception.anatomy.pspnet import prepare_image
    from visevi.perception.appearance import load_grayscale

    path = tmp_path / 'rotated.jpg'
    exif = Image.Exif()
    exif[0x0112] = 6  # display rotated 90 degrees
    Image.new('L', (8, 12), color=100).save(path, exif=exif)
    with pytest.raises(ValueError, match='EXIF'):
        prepare_image(path)
    with pytest.raises(ValueError, match='EXIF'):
        load_grayscale(path)
