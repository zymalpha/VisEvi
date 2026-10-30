"""Isolated model backends: verified assets, worker processes, bound caches.

Model code runs in ``.venv-models`` as a long-lived worker subprocess that
speaks one JSON object per line on stdin/stdout. The main process never imports
the model stack. Every result is cached in a caller-chosen output directory
together with a binding record; a cache is reused only if the binding (image,
weights, worker code, environment, parameters, device) is identical and every
artifact still matches its checksum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
MODEL_PYTHON = ROOT / '.venv-models/bin/python'
MODEL_LOCK = ROOT / 'tools/requirements-models.lock'
WORKER_DIR = Path(__file__).resolve().parents[1] / 'workers'
DEFAULT_DEVICE_ENV = 'VISEVI_MODEL_DEVICE'
EXIF_ORIENTATION = 0x0112


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=256)
def _cached_sha256(path: str, size: int, mtime_ns: int) -> str:
    return file_sha256(path)


def stable_sha256(path: Path) -> str:
    """SHA256 cached per (path, size, mtime); large weights are hashed once per process."""
    stat = path.stat()
    return _cached_sha256(str(path), stat.st_size, stat.st_mtime_ns)


def default_device() -> str:
    return os.environ.get(DEFAULT_DEVICE_ENV, 'cpu')


def inspect_image(path: str | Path) -> dict:
    """Header-only frame record; refuses rasters that need an unrecorded conversion."""
    with Image.open(path) as image:
        if image.mode in ('I', 'F') or image.mode.startswith('I;16'):
            raise ValueError('model backends expect an 8-bit display raster; high-bit-depth input '
                             'requires an explicit recorded windowing conversion')
        if image.getexif().get(EXIF_ORIENTATION, 1) != 1:
            raise ValueError('EXIF orientation is not identity; transpose the image explicitly first')
        return {'width': image.size[0], 'height': image.size[1], 'mode': image.mode}


@dataclass(frozen=True)
class Backend:
    name: str                      # tools/weights/<name>/provenance.json
    worker: str                    # module in visevi.workers
    version: dict = field(default_factory=dict)   # adapter/prompt/config versions
    startup_timeout: float = 900.0
    request_timeout: float = 900.0
    weights_root: Path | None = None   # override for tests; default tools/weights
    source: str | None = None          # vendored code in tools/src/<source> imported by the worker

    @property
    def weights_dir(self) -> Path:
        return (self.weights_root or ROOT / 'tools/weights') / self.name

    def verified_weights(self) -> dict[str, str]:
        """Re-hash every provenance-listed file; any difference is a hard error."""
        record_path = self.weights_dir / 'provenance.json'
        if not record_path.is_file():
            raise FileNotFoundError(f'{self.name}: missing {record_path}; run scripts/fetch_model_assets.py')
        record = json.loads(record_path.read_text())
        hashes = {}
        for item in record['files']:
            path = self.weights_dir / item['path']
            if not path.is_file() or stable_sha256(path) != item['sha256']:
                raise ValueError(f'{self.name}: asset differs from provenance: {item["path"]}')
            hashes[item['path']] = item['sha256']
        return {'revision': record.get('revision'), 'files': hashes}

    def verified_source(self) -> dict | None:
        """Recompute the vendored source tree digest and compare with tools/src/provenance.json."""
        if self.source is None:
            return None
        records = json.loads((ROOT / 'tools/src/provenance.json').read_text())['sources']
        record = next(r for r in records if r['name'] == self.source)
        digest = source_tree_sha256(Path(record['extracted_path']) if Path(record['extracted_path']).is_absolute()
                                    else ROOT / 'tools/src' / record['extracted_path'])
        if digest != record['tree_sha256']:
            raise ValueError(f'{self.source}: vendored source differs from provenance')
        return {'repo': record['repo'], 'commit': record['commit'], 'tree_sha256': digest}

    def worker_sha256(self) -> str:
        digest = hashlib.sha256()
        for path in (WORKER_DIR / '_runtime.py', WORKER_DIR / f'{self.worker}.py'):
            digest.update(path.read_bytes())
        return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(stable_sha256(path)))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def model_environment() -> dict:
    """Interpreter and package versions of the model environment (queried once)."""
    if not MODEL_PYTHON.is_file():
        raise FileNotFoundError(f'model environment missing: {MODEL_PYTHON}')
    probe = ('import json,importlib.metadata as m,sys\n'
             'names=["torch","torchvision","numpy","opencv-python","pillow","timm","transformers",'
             '"torch-geometric","safetensors","scikit-image"]\n'
             'v={}\n'
             'for n in names:\n'
             '    try: v[n]=m.version(n)\n'
             '    except m.PackageNotFoundError: v[n]=None\n'
             'print(json.dumps({"python":sys.version.split()[0],"base_prefix":sys.base_prefix,"packages":v}))')
    out = subprocess.run([str(MODEL_PYTHON), '-c', probe], capture_output=True, text=True,
                         timeout=120, check=True)
    info = json.loads(out.stdout.strip().splitlines()[-1])
    info['lock_sha256'] = file_sha256(MODEL_LOCK)
    return info


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    """One persistent worker process per (backend, device)."""

    def __init__(self, backend: Backend, device: str):
        self.backend, self.device = backend, device
        self.log = tempfile.NamedTemporaryFile(prefix=f'visevi-{backend.name}-', suffix='.log',
                                               delete=False, mode='w+')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1')
        env.pop('PYTHONHOME', None)
        self.proc = subprocess.Popen(
            [str(MODEL_PYTHON), '-m', f'visevi.workers.{backend.worker}',
             '--weights-dir', str(backend.weights_dir), '--device', device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env, cwd=str(ROOT))
        self.ready = self._read(backend.startup_timeout)
        if not self.ready.get('ready'):
            raise WorkerError(f'{backend.name} worker failed to start: {self.ready}')

    def _stderr_tail(self) -> str:
        self.log.flush()
        return Path(self.log.name).read_text(errors='replace')[-2000:]

    def _read(self, timeout: float) -> dict:
        selector = selectors.DefaultSelector()
        selector.register(self.proc.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise WorkerError(f'{self.backend.name} worker timed out\n{self._stderr_tail()}')
            if selector.select(remaining):
                line = self.proc.stdout.readline()
                if not line:
                    raise WorkerError(f'{self.backend.name} worker exited '
                                      f'(code {self.proc.poll()})\n{self._stderr_tail()}')
                return json.loads(line)

    def call(self, request: dict) -> dict:
        if self.proc.poll() is not None:
            raise WorkerError(f'{self.backend.name} worker is not running\n{self._stderr_tail()}')
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()
        response = self._read(self.backend.request_timeout)
        if not response.get('ok'):
            raise WorkerError(f'{self.backend.name}: {response.get("error")}')
        return response

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


_WORKERS: dict[tuple[str, str], WorkerClient] = {}


def worker(backend: Backend, device: str) -> WorkerClient:
    key = (backend.name, device)
    client = _WORKERS.get(key)
    if client is None or client.proc.poll() is not None:
        client = _WORKERS[key] = WorkerClient(backend, device)
    return client


def shutdown_workers() -> None:
    for client in _WORKERS.values():
        client.close()
    _WORKERS.clear()


def expected_binding(backend: Backend, image_path: Path, device: str, params: dict) -> dict:
    return {'backend': backend.name, 'worker': backend.worker, 'version': backend.version,
            'weights': backend.verified_weights(), 'source': backend.verified_source(),
            'worker_sha256': backend.worker_sha256(),
            'environment': model_environment(), 'device': device,
            'image_sha256': file_sha256(image_path), 'image_frame': inspect_image(image_path),
            'params': params}


def run_cached(backend: Backend, image_path: str | Path, output_dir: str | Path,
               device: str, params: dict) -> dict:
    """Return ``{'result', 'cache_hit', 'binding', 'artifacts'}`` for one image.

    A non-empty output directory is reused only when its record carries the
    identical binding and all artifact checksums match; otherwise it is refused
    rather than overwritten.
    """
    image_path, output_dir = Path(image_path).resolve(), Path(output_dir)
    binding = expected_binding(backend, image_path, device, params)
    record_path = output_dir / 'cache_record.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get('binding') != binding:
            raise ValueError('cache identity mismatch (image, weights, worker, environment, '
                             'parameters or device differ); use a new output directory')
        for name, digest in record['artifacts'].items():
            if file_sha256(output_dir / name) != digest:
                raise ValueError(f'cached artifact checksum mismatch: {name}')
        return {**record, 'cache_hit': True}
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('non-empty output directory has no cache record; use a new directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    response = worker(backend, device).call({'image_path': str(image_path),
                                             'output_dir': str(output_dir.resolve()),
                                             'params': params})
    artifacts = {name: file_sha256(output_dir / name) for name in response.get('artifacts', [])}
    record = {'binding': binding, 'result': response['result'], 'artifacts': artifacts,
              'image_path': str(image_path), 'is_ground_truth': False,
              'worker_runtime': worker(backend, device).ready.get('runtime')}
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return {**record, 'cache_hit': False}
