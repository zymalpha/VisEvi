"""Offline receipt-generation tests using synthetic JPEGs only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/record_verified_wget_batch.py'
SPEC = importlib.util.spec_from_file_location('record_verified_wget_batch_test', SCRIPT)
record = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record)

FIRST = 'files/p10/p10000032/s50414267/' + 'a' * 8 + ('-' + 'b' * 8) * 4 + '.jpg'
SECOND = 'files/p10/p10000033/s50414268/' + 'c' * 8 + ('-' + 'd' * 8) * 4 + '.jpg'


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    paths = (FIRST, SECOND)
    for index, relative in enumerate(paths):
        image = root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('L', (8, 6), 50 + index).save(image, format='JPEG')
    index_path = tmp_path / 'IMAGE_FILENAMES'
    index_path.write_text('\n'.join(paths) + '\n', encoding='utf-8')
    manifest = {'files': [{'relative_path': path, 'rows': '6', 'columns': '8'} for path in paths]}
    return root, index_path, manifest


def receipt_for(root, relative):
    return root / '_download_receipts' / (hashlib.sha256(relative.encode()).hexdigest() + '.json')


def receipts(root):
    return list((root / '_download_receipts').glob('*.json'))


def run(batch, manifest=None):
    root, index_path, original = batch
    return record.record_batch(original if manifest is None else manifest, root, index_path)


def test_records_only_aggregate_results_and_reuses_matching_receipts(batch):
    root, _, _ = batch
    result = run(batch)
    serialized = json.dumps(result)
    assert FIRST not in serialized and SECOND not in serialized
    for relative in (FIRST, SECOND):
        image, receipt = root / relative, receipt_for(root, relative)
        payload = json.loads(receipt.read_text())
        assert payload['path'] == relative
        assert payload['size'] == image.stat().st_size
        assert payload['sha256'] == hashlib.sha256(image.read_bytes()).hexdigest()
        assert payload['hash_type'] == 'locally_computed_not_official'
        assert receipt.stat().st_mode & 0o777 == 0o600
    original = [receipt_for(root, p).read_bytes() for p in (FIRST, SECOND)]
    run(batch)
    assert [receipt_for(root, p).read_bytes() for p in (FIRST, SECOND)] == original


@pytest.mark.parametrize('failure', ['missing', 'invalid_jpeg', 'dimensions', 'unindexed', 'unsafe', 'duplicate'])
def test_preflight_failure_writes_no_receipts(batch, failure):
    root, index_path, original = batch
    manifest = deepcopy(original)
    if failure == 'missing':
        (root / SECOND).unlink()
    elif failure == 'invalid_jpeg':
        (root / SECOND).write_bytes(b'not-an-image')
    elif failure == 'dimensions':
        manifest['files'][1]['columns'] = '7'
    elif failure == 'unindexed':
        index_path.write_text(FIRST + '\n', encoding='utf-8')
    elif failure == 'unsafe':
        manifest['files'][1]['relative_path'] = '../outside.jpg'
    else:
        manifest['files'][1]['relative_path'] = FIRST
    with pytest.raises((ValueError, OSError)):
        run(batch, manifest)
    assert receipts(root) == []


def test_existing_bad_receipt_is_neither_overwritten_nor_accepted(batch):
    root, _, _ = batch
    receipt = receipt_for(root, SECOND)
    receipt.parent.mkdir()
    receipt.write_text('{"sha256": "bad"}')
    before = receipt.read_bytes()
    with pytest.raises((ValueError, OSError)):
        run(batch)
    assert receipt.read_bytes() == before
    assert not receipt_for(root, FIRST).exists()


def test_interrupted_recording_can_be_rerun_without_overwriting(batch, monkeypatch):
    root, _, _ = batch
    original = record._write_receipt
    calls = 0

    def fail_on_second(directory, path, entry):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('simulated disk interruption')
        return original(directory, path, entry)

    monkeypatch.setattr(record, '_write_receipt', fail_on_second)
    with pytest.raises(OSError):
        run(batch)
    first_receipt = receipt_for(root, FIRST)
    assert first_receipt.exists()
    before = first_receipt.read_bytes()
    assert not receipt_for(root, SECOND).exists()
    monkeypatch.setattr(record, '_write_receipt', original)
    run(batch)
    assert first_receipt.read_bytes() == before
    assert receipt_for(root, SECOND).exists()


def test_rejects_symbolic_link_to_image_outside_batch(batch, tmp_path):
    root, _, _ = batch
    image = root / SECOND
    image.unlink()
    outside = tmp_path / 'outside.jpg'
    Image.new('L', (8, 6), 70).save(outside, format='JPEG')
    image.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        run(batch)
    assert receipts(root) == []
