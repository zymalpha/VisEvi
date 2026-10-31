"""Offline integrity checks using synthetic images and annotations only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_mimic_batch.py'
spec = importlib.util.spec_from_file_location('verify_mimic_batch_test', SCRIPT)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

RELATIVE = 'files/p10/p10000032/s50414267/' + 'a' * 8 + ('-' + 'b' * 8) * 4 + '.jpg'


@pytest.fixture
def sample(tmp_path):
    images = tmp_path / 'images'
    data = tmp_path / 'data'
    image = images / RELATIVE
    image.parent.mkdir(parents=True)
    Image.new('L', (8, 6), 70).save(image, format='JPEG')
    masks = {}
    for kind in ('heart', 'lungs'):
        mask = data / 'masks' / (kind + '.png')
        mask.parent.mkdir(parents=True, exist_ok=True)
        array = np.zeros((6, 8), dtype='uint8')
        array[1:4, 2:6] = 255
        Image.fromarray(array).save(mask)
        masks[kind] = str(mask.relative_to(data))
    manifest = {'files': [{'relative_path': RELATIVE, 'rows': '6', 'columns': '8',
                           'subject_id': 'must-not-be-output',
                           'sources': [{'dataset': 'heart-lung-segmentations', 'masks': masks}]}]}
    receipt = verify._receipt_path(images, RELATIVE)
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({'path': RELATIVE, 'size': image.stat().st_size,
                                   'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}))
    return {'images': images, 'data': data, 'image': image,
            'receipt': receipt, 'manifest': manifest, 'masks': masks}


def run(sample):
    return verify.verify_manifest(sample['manifest'], sample['images'], sample['data'])


def test_complete_batch_has_only_aggregate_information(sample, tmp_path):
    result = run(sample)
    assert result['status'] == 'complete'
    assert result['images']['dimensions_match'] == 1
    assert result['images']['receipt_matched'] == 1
    assert result['manual_masks']['binary'] == 2
    assert result['image_mask_size_relation']['decoded_image_same_dimensions'] == 2
    assert RELATIVE not in json.dumps(result)
    assert 'must-not-be-output' not in json.dumps(result)
    output = tmp_path / 'report.json'
    verify._write_exclusive(output, result)
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        verify._write_exclusive(output, result)


def test_missing_image_is_not_hidden_by_present_masks(sample):
    sample['image'].unlink()
    result = run(sample)
    assert result['status'] == 'issues_or_incomplete'
    assert result['images']['missing'] == 1
    assert result['manual_masks']['decoded'] == 2
    assert result['image_mask_size_relation']['metadata_only_same_dimensions'] == 2
    assert result['image_mask_size_relation']['decoded_image_same_dimensions'] == 0


def test_invalid_image_and_receipt_mismatch_are_both_reported(sample):
    sample['image'].write_bytes(b'invalid')
    result = run(sample)
    assert result['images']['decode_failed'] == 1
    assert result['images']['receipt_mismatch'] == 1
    assert result['status'] == 'issues_or_incomplete'


def test_metadata_dimensions_mismatch_reported_without_modifying_image(sample):
    sample['manifest']['files'][0]['columns'] = '7'
    before = sample['image'].read_bytes()
    result = run(sample)
    assert result['images']['dimensions_differ'] == 1
    assert sample['image'].read_bytes() == before


def test_nonbinary_and_wrong_size_mask_are_reported_without_resizing(sample):
    mask = sample['data'] / sample['masks']['heart']
    Image.fromarray(np.full((2, 3), 127, dtype='uint8')).save(mask)
    before = mask.read_bytes()
    result = run(sample)
    assert result['manual_masks']['nonbinary'] == 1
    assert result['image_mask_size_relation']['decoded_image_different_dimensions_unresolved'] == 1
    assert mask.read_bytes() == before


def test_missing_or_corrupt_receipt_is_explicit(sample):
    sample['receipt'].unlink()
    assert run(sample)['images']['receipt_missing'] == 1
    sample['receipt'].write_text('{')
    assert run(sample)['images']['receipt_invalid'] == 1


def test_missing_or_corrupt_mask_is_reported(sample):
    heart = sample['data'] / sample['masks']['heart']
    lungs = sample['data'] / sample['masks']['lungs']
    heart.unlink()
    lungs.write_bytes(b'invalid')
    result = run(sample)
    assert result['manual_masks']['missing'] == 1
    assert result['manual_masks']['decode_failed'] == 1


def test_rejects_duplicate_and_unsafe_manifest_paths(sample):
    row = sample['manifest']['files'][0]
    sample['manifest']['files'].append(row)
    with pytest.raises(ValueError, match='duplicate'):
        run(sample)
    sample['manifest']['files'].pop()
    row['sources'][0]['masks']['heart'] = '../secret'
    with pytest.raises(ValueError, match='invalid relative path'):
        run(sample)
