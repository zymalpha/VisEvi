#!/usr/bin/env python3
"""Verify a local MIMIC development batch and write aggregate-only results.

No image or mask is modified. Differing image/mask sizes are reported as
unresolved, never resized. The receipt hash is locally computed, not an
official reference checksum.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'original_data'
DEFAULT_IMAGES = DATA / 'mimic-cxr-jpg/2.1.0'
DEFAULT_MANIFEST = DATA / '_manifests/20261031-mimic-integration-development-v1/manifest.json'
IMAGE_PATH = re.compile(r'files/(p[0-9]{2})/(p[0-9]{8})/(s[0-9]{8})/'
                        r'([a-f0-9]{8}(?:-[a-f0-9]{8}){4}\.jpg)\Z')


def _path(root: Path, relative: object, *, image: bool = False) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError('invalid relative path')
    parts = PurePosixPath(relative).parts
    if (PurePosixPath(relative).is_absolute() or any(p in ('.', '..') for p in parts)
            or '\\' in relative or (image and not IMAGE_PATH.fullmatch(relative))):
        raise ValueError('invalid relative path')
    root_resolved = root.resolve()
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root_resolved):
        raise ValueError('path escapes data root')
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _receipt_path(root: Path, relative: str) -> Path:
    name = hashlib.sha256(relative.encode('utf-8')).hexdigest() + '.json'
    return root / '_download_receipts' / name


def _verify_receipt(root: Path, relative: str, image: Path) -> str:
    receipt = _receipt_path(root, relative)
    if not receipt.is_file() or receipt.is_symlink():
        return 'missing'
    try:
        data = json.loads(receipt.read_text(encoding='utf-8'))
        if (data.get('path') != relative or data.get('size') != image.stat().st_size
                or data.get('sha256') != _sha256(image)):
            return 'mismatch'
        return 'matched'
    except (OSError, TypeError, ValueError):
        return 'invalid'


def _dimension(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if number > 0 and str(value).strip() == str(number) else None


def _image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        if image.format != 'JPEG':
            raise ValueError('not a JPEG')
        return image.size


def _mask(path: Path) -> tuple[tuple[int, int], bool, bool]:
    with Image.open(path) as image:
        image.load()
        values = np.asarray(image)
        if values.ndim != 2:
            raise ValueError('mask is not 2D')
        binary = bool(np.isin(values, (0, 1, 255)).all())
        return image.size, binary, bool(np.any(values))


def verify_manifest(manifest: dict, image_root: Path = DEFAULT_IMAGES,
                    data_root: Path = DATA) -> dict:
    rows = manifest.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('manifest files must be a nonempty list')
    images, masks, sizes = Counter(), Counter(), Counter()
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('manifest contains an invalid entry')
        relative = row.get('relative_path')
        path = _path(image_root, relative, image=True)
        if relative in seen:
            raise ValueError('manifest has duplicate image paths')
        seen.add(relative)
        width, height = _dimension(row.get('columns')), _dimension(row.get('rows'))
        if width is None or height is None:
            images['metadata_dimensions_invalid'] += 1
        actual_size = None
        if not path.is_file() or path.is_symlink():
            images['missing'] += 1
        else:
            images['present'] += 1
            images['bytes'] += path.stat().st_size
            images['receipt_' + _verify_receipt(image_root, relative, path)] += 1
            try:
                actual_size = _image(path)
                images['decoded'] += 1
                if width is not None and height is not None:
                    images['dimensions_match' if actual_size == (width, height)
                           else 'dimensions_differ'] += 1
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                images['decode_failed'] += 1
        sources = row.get('sources', [])
        if not isinstance(sources, list):
            raise ValueError('manifest sources must be a list')
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError('manifest contains an invalid source')
            if source.get('dataset') != 'heart-lung-segmentations':
                continue
            source_masks = source.get('masks')
            if not isinstance(source_masks, dict) or set(source_masks) != {'heart', 'lungs'}:
                raise ValueError('heart/lung source must specify both masks')
            for kind in ('heart', 'lungs'):
                masks['expected'] += 1
                target = _path(data_root, source_masks[kind])
                if not target.is_file() or target.is_symlink():
                    masks['missing'] += 1
                    continue
                try:
                    mask_size, binary, nonempty = _mask(target)
                    masks['decoded'] += 1
                    masks['binary' if binary else 'nonbinary'] += 1
                    masks['nonempty' if nonempty else 'empty'] += 1
                    # Distinguish decoded image dimensions from metadata-only checks.
                    if actual_size is not None:
                        key = ('decoded_image_same_dimensions' if mask_size == actual_size
                               else 'decoded_image_different_dimensions_unresolved')
                    elif width is not None and height is not None:
                        key = ('metadata_only_same_dimensions' if mask_size == (width, height)
                               else 'metadata_only_different_dimensions_unresolved')
                    else:
                        key = 'unknown'
                    sizes[key] += 1
                except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                    masks['decode_failed'] += 1
    issue_fields = ('missing', 'decode_failed', 'metadata_dimensions_invalid',
                    'dimensions_differ', 'receipt_missing', 'receipt_mismatch', 'receipt_invalid')
    issue = (any(images[k] for k in issue_fields)
             or any(masks[k] for k in ('missing', 'decode_failed', 'nonbinary', 'empty'))
             or bool(sizes['unknown'] or sizes['decoded_image_different_dimensions_unresolved']
                     or sizes['metadata_only_different_dimensions_unresolved']))
    return {
        'purpose': 'development_batch_integrity_only_not_medical_accuracy',
        'requested_images': len(rows),
        'images': {key: images[key] for key in ('present', 'missing', 'bytes', 'decoded',
                   'decode_failed', 'metadata_dimensions_invalid', 'dimensions_match',
                   'dimensions_differ', 'receipt_matched', 'receipt_missing',
                   'receipt_mismatch', 'receipt_invalid')},
        'manual_masks': {key: masks[key] for key in ('expected', 'missing', 'decoded',
                         'decode_failed', 'binary', 'nonbinary', 'nonempty', 'empty')},
        'image_mask_size_relation': {key: sizes[key] for key in
                                     ('decoded_image_same_dimensions',
                                      'decoded_image_different_dimensions_unresolved',
                                      'metadata_only_same_dimensions',
                                      'metadata_only_different_dimensions_unresolved', 'unknown')},
        'status': 'issues_or_incomplete' if issue else 'complete',
        'limitations': ['local receipt hashes are not official reference hashes',
                        'matching raster dimensions do not establish anatomical alignment',
                        'different raster dimensions are not automatically rescaled',
                        'report contains counts only; image predictions are not evaluated'],
    }


def _write_exclusive(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True)
        stream.write('\n')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out', type=Path, required=True,
                        help='new aggregate-only JSON report path; never overwritten')
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            raise ValueError('manifest must be an object')
        result = verify_manifest(manifest)
        _write_exclusive(args.out, result)
        print(json.dumps(result, ensure_ascii=True))
        return 0 if result['status'] == 'complete' else 1
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print('verification failed: invalid manifest, unsafe paths, unreadable data or output already exists',
              file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
