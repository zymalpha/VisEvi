#!/usr/bin/env python3
"""Record local checksums for a complete, independently downloaded MIMIC batch.

This is an offline, two-phase operation: every manifest entry must pass index,
path, JPEG decoding, dimension and existing-receipt checks before any new
receipt is published. Local hashes are not authoritative remote checksums.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / 'original_data/mimic-cxr-jpg/2.1.0'
PATTERN = re.compile(r'files/(p[0-9]{2})/(p[0-9]{8})/(s[0-9]{8})/'
                     r'([a-f0-9]{8}(?:-[a-f0-9]{8}){4}\.jpg)\Z')
BLOCK = 1024 * 1024


def _relative(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError('invalid image path')
    match = PATTERN.fullmatch(value)
    if not match or match.group(2)[:3] != match.group(1):
        raise ValueError('invalid image path')
    return value


def _image_path(root: Path, relative: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError('invalid image root')
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('image path contains a symlink')
    return current


def _dimension(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError('invalid metadata dimensions')
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('invalid metadata dimensions') from None
    if number <= 0 or str(number) != str(value).strip():
        raise ValueError('invalid metadata dimensions')
    return number


def _fingerprint(path: Path, width: int, height: int) -> tuple[int, str, tuple]:
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('image is not a regular file')
        with Image.open(stream) as image:
            if image.format != 'JPEG' or image.size != (width, height):
                raise ValueError('image format or dimensions mismatch')
            image.load()
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(BLOCK), b''):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError('image changed during verification')
    return after.st_size, digest.hexdigest(), identity(after)


def _receipt_path(directory: Path, relative: str) -> Path:
    return directory / (hashlib.sha256(relative.encode('utf-8')).hexdigest() + '.json')


def _matches(path: Path, relative: str, size: int, digest: str) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        entry = json.loads(path.read_text(encoding='utf-8'))
        return entry.get('path') == relative and entry.get('size') == size and entry.get('sha256') == digest
    except (OSError, TypeError, ValueError):
        return False


def _write_receipt(directory: Path, path: Path, entry: dict) -> None:
    fd, name = tempfile.mkstemp(prefix='.visevi-receipt-', dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(entry, stream, ensure_ascii=True, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def record_batch(manifest: dict, image_root: Path, official_index: Path) -> dict:
    """Validate an entire manifest, then atomically publish individual receipts."""
    rows = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('manifest must contain image entries')
    prepared = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid image entry')
        relative = _relative(row.get('relative_path'))
        prepared.append((relative, _dimension(row.get('columns')), _dimension(row.get('rows'))))
    wanted = {item[0] for item in prepared}
    if len(wanted) != len(prepared):
        raise ValueError('duplicate manifest path')
    with official_index.open(encoding='utf-8') as stream:
        found = {line.strip() for line in stream if line.strip() in wanted}
    if found != wanted:
        raise ValueError('manifest contains a path outside the official index')

    receipt_dir = image_root / '_download_receipts'
    if receipt_dir.is_symlink() or (receipt_dir.exists() and not receipt_dir.is_dir()):
        raise ValueError('invalid receipt directory')
    snapshots = []
    # No receipt directory or file is created until every entry has passed.
    for relative, width, height in prepared:
        path = _image_path(image_root, relative)
        try:
            size, digest, identity = _fingerprint(path, width, height)
        except (OSError, ValueError) as exc:
            raise ValueError('image missing, unreadable, incomplete or inconsistent with metadata') from exc
        receipt = _receipt_path(receipt_dir, relative)
        if (receipt.exists() or receipt.is_symlink()) and not _matches(receipt, relative, size, digest):
            raise ValueError('existing receipt does not match; not overwritten')
        snapshots.append((relative, width, height, size, digest, identity))

    receipt_dir.mkdir(mode=0o700, exist_ok=True)
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise ValueError('invalid receipt directory')
    if stat.S_IMODE(receipt_dir.stat().st_mode) & 0o077:
        raise ValueError('receipt directory must be private')
    created = existing = total_bytes = 0
    for relative, width, height, size, digest, identity in snapshots:
        path = _image_path(image_root, relative)
        if _fingerprint(path, width, height) != (size, digest, identity):
            raise ValueError('image changed after batch preflight')
        receipt = _receipt_path(receipt_dir, relative)
        if receipt.exists() or receipt.is_symlink():
            if not _matches(receipt, relative, size, digest):
                raise ValueError('existing receipt changed; not overwritten')
            existing += 1
        else:
            _write_receipt(receipt_dir, receipt, {
                'path': relative, 'size': size, 'sha256': digest,
                'source': 'MIMIC-CXR-JPG 2.1.0',
                'hash_type': 'locally_computed_not_official',
                'download_method': 'wget; locally verified after download',
            })
            try:
                if _fingerprint(_image_path(image_root, relative), width, height) != (size, digest, identity):
                    raise ValueError('image changed while recording receipt')
            except BaseException:
                # Only remove the receipt we just created, and only if it still
                # describes this verification. Never remove an image.
                if _matches(receipt, relative, size, digest):
                    receipt.unlink()
                raise
            created += 1
        total_bytes += size
    return {'requested': len(snapshots), 'receipts_created': created,
            'receipts_existing': existing, 'image_bytes': total_bytes,
            'hash_type': 'locally_computed_not_official'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='explicit local JSON manifest')
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGES)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        result = record_batch(manifest, args.image_root, args.image_root / 'IMAGE_FILENAMES')
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError):
        print('batch verification failed; no unverified image was overwritten')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
