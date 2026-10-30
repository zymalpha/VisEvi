"""Isolated-worker protocol and bound-cache tests (no model weights, no GPU)."""
import json

import pytest
from PIL import Image

from visevi import backends
from visevi.backends import Backend, WorkerError, run_cached


@pytest.fixture
def echo_backend(tmp_path):
    root = tmp_path / 'weights'
    (root / 'echo').mkdir(parents=True)
    (root / 'echo/w.bin').write_bytes(b'weights')
    (root / 'echo/provenance.json').write_text(json.dumps({'revision': 'r1', 'files': [
        {'path': 'w.bin', 'sha256': backends.file_sha256(root / 'echo/w.bin')}]}))
    image = tmp_path / 'image.png'
    Image.new('L', (8, 6), color=7).save(image)
    yield Backend('echo', '_echo', {'adapter': 1}, startup_timeout=60, request_timeout=60,
                  weights_root=root), image
    backends.shutdown_workers()


def test_cache_binds_and_reuses_without_calling_the_worker(echo_backend, tmp_path):
    backend, image = echo_backend
    first = run_cached(backend, image, tmp_path / 'out', 'cpu', {'value': 1})
    assert not first['cache_hit'] and first['result']['echo'] == {'value': 1}
    backends.shutdown_workers()
    second = run_cached(backend, image, tmp_path / 'out', 'cpu', {'value': 1})
    assert second['cache_hit'] and second['result'] == first['result']
    assert not backends._WORKERS          # served from the record, no worker started


@pytest.mark.parametrize('change', ['params', 'device', 'image', 'weights', 'artifact'])
def test_any_identity_change_or_tampering_is_refused(echo_backend, tmp_path, change):
    backend, image = echo_backend
    out = tmp_path / 'out'
    run_cached(backend, image, out, 'cpu', {'value': 1})
    params, device = {'value': 1}, 'cpu'
    if change == 'params':
        params = {'value': 2}
    elif change == 'device':
        device = 'cuda:9'
    elif change == 'image':
        Image.new('L', (8, 6), color=8).save(image)
    elif change == 'weights':
        (backend.weights_dir / 'w.bin').write_bytes(b'other')
    else:
        (out / 'echo.txt').write_text('tampered')
    with pytest.raises(ValueError):
        run_cached(backend, image, out, device, params)


def test_nonempty_directory_without_record_is_not_overwritten(echo_backend, tmp_path):
    backend, image = echo_backend
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out/other.txt').write_text('x')
    with pytest.raises(FileExistsError):
        run_cached(backend, image, tmp_path / 'out', 'cpu', {})


def test_worker_errors_and_crashes_are_reported(echo_backend, tmp_path):
    backend, image = echo_backend
    with pytest.raises(WorkerError, match='requested failure'):
        run_cached(backend, image, tmp_path / 'a', 'cpu', {'fail': True})
    with pytest.raises(WorkerError, match='exited'):
        run_cached(backend, image, tmp_path / 'b', 'cpu', {'crash': True})
    assert not (tmp_path / 'a/cache_record.json').exists()
    # a fresh worker is started after the crash
    assert run_cached(backend, image, tmp_path / 'c', 'cpu', {'value': 3})['result']['echo'] == {'value': 3}


@pytest.mark.parametrize('mode', ['I;16', 'F'])
def test_high_bit_depth_inputs_are_refused_before_inference(echo_backend, tmp_path, mode):
    backend, _ = echo_backend
    path = tmp_path / 'deep.tiff'
    Image.new(mode, (4, 4)).save(path)
    with pytest.raises(ValueError, match='8-bit'):
        run_cached(backend, path, tmp_path / 'out', 'cpu', {})
