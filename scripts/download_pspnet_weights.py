"""Download the official release asset with resumable, validated byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'tools/weights'
SIZE = 272988989
EXPECTED_SHA256 = '019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc'
BLOCK = 1024 * 1024
API = 'https://api.github.com/repos/mlmed/torchxrayvision/releases/assets/89760024'
RELEASE = 'https://github.com/mlmed/torchxrayvision/releases/download/v1/pspnet_chestxray_best_model_4.pth'
DOWNLOAD = None
PARTS = WEIGHTS / 'validated_ranges'
PARTS.mkdir(parents=True, exist_ok=True)


def path_for(start):
    return PARTS / f'{start:012d}.bin'


def salvage(path, offset):
    if not path.exists():
        return
    with path.open('rb') as source:
        while True:
            data = source.read(BLOCK)
            if len(data) != min(BLOCK, SIZE - offset):
                break
            target = path_for(offset)
            if not target.exists():
                target.write_bytes(data)
            offset += len(data)
            if offset >= SIZE:
                break


def fetch(start):
    end = min(SIZE, start + BLOCK) - 1
    target = path_for(start)
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return 'cached'
    last_error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(DOWNLOAD,
                headers={'User-Agent': 'VisEvi-evaluation', 'Accept': 'application/octet-stream',
                         'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=40) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise ValueError('Unexpected content range')
                data = response.read(expected + 1)
            if len(data) != expected:
                raise ValueError('Incomplete byte range')
            partial = target.with_suffix('.part')
            partial.write_bytes(data)
            partial.replace(target)
            return 'downloaded'
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Failed range {start}: {last_error}')


def main():
    global DOWNLOAD
    target = WEIGHTS / 'pspnet_chestxray_best_model_4.pth'
    if target.is_file() and target.stat().st_size == SIZE:
        if hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_SHA256:
            print('Official checkpoint already present; recorded SHA256 matches.', flush=True)
            return
    # Resolve a fresh release redirect; signed asset URLs expire.
    with urllib.request.urlopen(urllib.request.Request(RELEASE + '?download=1', method='HEAD'), timeout=40) as response:
        DOWNLOAD = response.geturl()
    salvage(WEIGHTS / 'pspnet_chestxray_best_model_4.pth.part', 0)
    for partial in (WEIGHTS / 'pspnet_parts').glob('*'):
        if partial.suffix in {'.bin', '.part'}:
            salvage(partial, int(partial.stem) * 16 * BLOCK)
    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, start) for start in range(0, SIZE, BLOCK)]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 20 == 0:
                print(json.dumps({'completed_ranges': completed, 'total_ranges': len(futures),
                                  'seconds': round(time.time() - started, 1)}), flush=True)
    temporary = target.with_suffix('.assembled')
    digest = hashlib.sha256()
    with temporary.open('wb') as output:
        for start in range(0, SIZE, BLOCK):
            data = path_for(start).read_bytes()
            output.write(data)
            digest.update(data)
    if temporary.stat().st_size != SIZE or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError('Assembled checkpoint differs from the recorded official checkpoint SHA256')
    temporary.replace(target)
    provenance = {'url': 'https://github.com/mlmed/torchxrayvision/releases/download/v1/pspnet_chestxray_best_model_4.pth',
        'asset_api': API, 'asset_id': 89760024, 'path': str(target), 'bytes': SIZE,
        'sha256': digest.hexdigest(), 'all_content_ranges_validated': True,
        'download_seconds': round(time.time() - started, 1)}
    (WEIGHTS / 'pspnet_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
