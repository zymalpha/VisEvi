"""Download pinned Hugging Face model assets and verify every file.

Run: .venv/bin/python scripts/fetch_model_assets.py chexmask-u chexone lines-tubes pneumothorax cxas

Each backend is fetched at a fixed revision into tools/weights/<name>/. Every
file is hashed after download; LFS files must match the hub's recorded SHA256
and, where available, the 2026-10-28 selection snapshot. A provenance.json with
sizes, hashes, revision and licence notes is written only after all checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'tool_verification/runs/_review_records/20261028_selection/primary_source_snapshot.json'

BACKENDS = {
    'chexmask-u': {'repo': 'mcosarinsky/CheXmask-U', 'revision': 'ed012f0fbb84969cc2af95f3e521dfdef6d334bf',
                   'patterns': ['v1_skip/*', 'README.md'], 'snapshot_key': 'chexmask_info',
                   'licence': 'HF model page: MIT; code repository (GitHub): GPL-3.0'},
    'chexone': {'repo': 'StanfordAIMI/CheXOne', 'revision': '0c350e6852ea08f9d9baf3b7595c1a10d4849927',
                'patterns': None, 'snapshot_key': 'chexone_info',
                'licence': 'CC BY-NC 4.0 (non-commercial); accepted by project owner 2026-10-30'},
    'lines-tubes': {'repo': 'ianpan/cxr-lines-and-tubes', 'revision': 'c11c443df4a1010966e4a8de4f3db6ce3569b3fa',
                    'patterns': None, 'snapshot_key': 'lines_info',
                    'licence': 'code MIT; weights research-only (model card); accepted by project owner 2026-10-30'},
    'pneumothorax': {'repo': 'ianpan/pneumothorax-cxr', 'revision': 'f00a7af5cdd019b2a51d18ea585c947b7ecb27eb',
                     'patterns': None, 'snapshot_key': 'ptx_ian_info',
                     'licence': 'not stated on model card; training data not stated; use accepted by project owner 2026-10-30 for research engineering only'},
}

# Google Drive assets have no published hash; the SHA256 below was recorded at
# the first download (2026-10-30) and pins every later refetch to that file.
DRIVE_BACKENDS = {
    'cxas': {'url': 'https://drive.google.com/file/d/1Y9zubvMzkYHoAqz-NvV6vniH5FKAF2iV/view',
             'file': 'UNet_ResNet50_default.pth', 'bytes': 881081391,
             'sha256': '2bdc485da991693860c18b759d63e7404cc6ab01b7a15c987c350212b581e0e2',
             'code': 'ConstantinSeibold/ChestXRayAnatomySegmentation@eb8024d9b570bc43d74b176459931fa8709d77f8 (MIT)',
             'licence': 'code MIT; TorchXRayVision lists CXAS models/weights as CC BY-NC-SA; accepted by project owner 2026-10-30'},
}


def fetch_drive(name: str) -> dict:
    spec = DRIVE_BACKENDS[name]
    target = ROOT / 'tools/weights' / name
    path = target / spec['file']
    if (target / 'provenance.json').exists():
        raise SystemExit(f'{target}/provenance.json exists; remove the directory deliberately to refetch')
    if not path.is_file():
        raise SystemExit(f'download {spec["url"]} to {path} (e.g. with gdown in .venv-models), then rerun')
    digest = sha256(path)
    if path.stat().st_size != spec['bytes'] or digest != spec['sha256']:
        raise ValueError(f'{name}: file differs from the pinned first download')
    record = {'backend': name, 'revision': spec['code'], 'source': spec['url'], 'licence': spec['licence'],
              'integrity': 'no publisher hash; pinned to SHA256 of first download 2026-10-30',
              'fetched_at': datetime.now(timezone.utc).isoformat(),
              'files': [{'path': spec['file'], 'bytes': spec['bytes'], 'sha256': digest,
                         'hub_lfs_sha256_verified': False, 'snapshot_sha256_verified': False}]}
    (target / 'provenance.json').write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return {'backend': name, 'files': 1, 'bytes': spec['bytes']}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot_hashes(key: str) -> dict[str, str]:
    if not SNAPSHOT.is_file():
        return {}
    data = json.loads(SNAPSHOT.read_text())
    info = json.loads(data[key]['content'])
    files = info.get('files') or [{'name': s['rfilename'], 'sha256': (s.get('lfs') or {}).get('sha256')}
                                  for s in info.get('siblings', [])]
    return {f['name']: f['sha256'] for f in files if f.get('sha256')}


def fetch(name: str) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    spec = BACKENDS[name]
    target = ROOT / 'tools/weights' / name
    if (target / 'provenance.json').exists():
        raise SystemExit(f'{target}/provenance.json exists; remove the directory deliberately to refetch')
    info = HfApi().model_info(spec['repo'], revision=spec['revision'], files_metadata=True)
    if info.sha != spec['revision']:
        raise ValueError(f'{name}: hub returned revision {info.sha}')
    hub = {s.rfilename: s for s in info.siblings}
    # Large files may be pre-fetched with a resumable downloader; keep any local
    # file whose size and LFS hash already match the hub and fetch the rest.
    present = [name for name, entry in hub.items()
               if entry.lfs and (target / name).is_file()
               and (target / name).stat().st_size == entry.size
               and sha256(target / name) == entry.lfs.sha256]
    snapshot_download(spec['repo'], revision=spec['revision'], local_dir=target,
                      allow_patterns=spec['patterns'], ignore_patterns=present or None)
    pinned = snapshot_hashes(spec['snapshot_key'])
    files = []
    for path in sorted(p for p in target.rglob('*') if p.is_file() and '.cache' not in p.parts):
        relative = path.relative_to(target).as_posix()
        entry = hub.get(relative)
        if entry is None:
            raise ValueError(f'{name}: unexpected file {relative}')
        digest = sha256(path)
        expected = entry.lfs.sha256 if entry.lfs else None
        if entry.size is not None and path.stat().st_size != entry.size:
            raise ValueError(f'{name}: size mismatch {relative}')
        if expected and digest != expected:
            raise ValueError(f'{name}: hub SHA256 mismatch {relative}')
        if relative in pinned and digest != pinned[relative]:
            raise ValueError(f'{name}: differs from 2026-10-28 snapshot {relative}')
        files.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest,
                      'hub_lfs_sha256_verified': bool(expected),
                      'snapshot_sha256_verified': relative in pinned})
    record = {'backend': name, 'repo': spec['repo'], 'revision': spec['revision'],
              'source': 'https://huggingface.co', 'licence': spec['licence'],
              'fetched_at': datetime.now(timezone.utc).isoformat(), 'files': files}
    (target / 'provenance.json').write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return {'backend': name, 'files': len(files), 'bytes': sum(f['bytes'] for f in files)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backends', nargs='+', choices=sorted({**BACKENDS, **DRIVE_BACKENDS}))
    for name in parser.parse_args().backends:
        print(json.dumps(fetch_drive(name) if name in DRIVE_BACKENDS else fetch(name)), flush=True)


if __name__ == '__main__':
    main()
