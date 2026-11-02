#!/usr/bin/env python3
"""Download an officially indexed MIMIC-CXR-JPG subset as the authorized user.

Input is a UTF-8 file with one official IMAGE_FILENAMES relative path per line.
Credentials are requested interactively, remain in memory, and are never logged.
Existing files are only reused when a private prior receipt matches their hash.
Incomplete transfers are discarded and retried from byte zero, not resumed.
If the OS forcibly terminates the process between image and receipt publication,
inspect the unverified file and move it to private quarantine before retrying;
the downloader will never automatically overwrite or delete such existing files.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import http.client
import http.cookiejar
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / 'original_data/mimic-cxr-jpg/2.1.0'
BASE = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/'
PATH_PATTERN = re.compile(
    r'files/(p[0-9]{2})/(p[0-9]{8})/(s[0-9]{8})/'
    r'([a-f0-9]{8}(?:-[a-f0-9]{8}){4}\.jpg)\Z')
CHUNK = 1024 * 1024


class DownloadError(Exception):
    """Sanitized failure without URL, patient identifiers or secrets."""


def valid_path(value: str) -> bool:
    match = PATH_PATTERN.fullmatch(value)
    return bool(match and match.group(2)[:3] == match.group(1))


def load_manifest(path: Path, official_index: Path) -> list[str]:
    """Read a newline-delimited path list, enforcing exact official membership."""
    lines = path.read_text(encoding='utf-8').splitlines()
    paths = [line.strip() for line in lines if line.strip()]
    if not paths or len(set(paths)) != len(paths) or not all(valid_path(p) for p in paths):
        raise DownloadError('manifest is empty, has duplicates, or has invalid paths')
    expected = set(paths)
    with official_index.open(encoding='utf-8') as stream:
        found = {line.strip() for line in stream if line.strip() in expected}
    if found != expected:
        raise DownloadError('one or more manifest paths are absent from the official index')
    return paths


class SameHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if (target.scheme != 'https' or target.hostname != 'physionet.org'
                or target.port not in (None, 443) or target.username or target.password):
            raise DownloadError('redirect to an unapproved host or scheme')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def opener_for(username: str, password: str):
    # PhysioNet sets a test cookie on its 401 challenge, then a session cookie
    # after Basic authentication. Both credentials and cookies stay in memory.
    passwords = urllib.request.HTTPPasswordMgr()
    passwords.add_password('PhysioNet', BASE, username, password)
    opener = urllib.request.build_opener(
        SameHostRedirect(), urllib.request.HTTPBasicAuthHandler(passwords),
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [('User-Agent', 'MedVisEvi-subset-downloader/1.0')]
    return opener


def private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise DownloadError('destination directory is invalid')
    os.chmod(path, 0o700)


def ensure_parents(root: Path, relative: str) -> Path:
    current = root
    for part in Path(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise DownloadError('destination contains a symbolic link')
        current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise DownloadError('destination contains a non-directory component')
    target = root / relative
    if target.is_symlink():
        raise DownloadError('destination is a symbolic link')
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_path(receipts: Path, relative: str) -> Path:
    return receipts / (hashlib.sha256(relative.encode('utf-8')).hexdigest() + '.json')


def verified_existing(target: Path, receipt: Path, relative: str) -> bool:
    if target.is_symlink() or not target.is_file() or receipt.is_symlink() or not receipt.is_file():
        return False
    try:
        entry = json.loads(receipt.read_text(encoding='utf-8'))
        return (entry.get('path') == relative and entry.get('size') == target.stat().st_size
                and entry.get('sha256') == sha256(target))
    except (ValueError, OSError):
        return False


def write_receipt(path: Path, entry: dict) -> None:
    # Atomic receipt publication avoids leaving a partial JSON after interruption.
    fd, name = tempfile.mkstemp(prefix='.visevi-receipt-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=True, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def remove_uncommitted_image(target: Path, temporary: Path, receipt: Path, relative: str) -> None:
    """Roll back only our hard link when no matching complete receipt exists."""
    if verified_existing(target, receipt, relative):
        return
    try:
        source = temporary.stat(follow_symlinks=False)
        output = target.stat(follow_symlinks=False)
        if (source.st_dev, source.st_ino) == (output.st_dev, output.st_ino):
            target.unlink()
    except FileNotFoundError:
        pass


def preflight_existing(root: Path, receipts: Path, paths: list[str], limit: int) -> dict:
    """Account for all existing manifest images before admitting new transfers."""
    stored = 0
    counted = set()
    for relative in paths:
        current = root
        for part in Path(relative).parts:
            current = current / part
            if current.is_symlink():
                raise DownloadError('destination contains a symbolic link')
        receipt = receipt_path(receipts, relative)
        if current.exists():
            if not verified_existing(current, receipt, relative):
                raise DownloadError('existing file has no matching private receipt; review or move it to a private quarantine before retry')
            stored += current.stat().st_size
            counted.add(relative)
        elif receipt.exists() or receipt.is_symlink():
            raise DownloadError('existing receipt has no matching image; not overwritten')
    if stored > limit:
        raise DownloadError('existing manifest images exceed batch storage budget')
    return {'stored': stored, 'counted': counted}


def download_one(opener, root: Path, receipts: Path, relative: str,
                 budget: dict, attempts: int = 3) -> str:
    target = ensure_parents(root, relative)
    receipt = receipt_path(receipts, relative)
    budget.setdefault('stored', 0)
    budget.setdefault('counted', set())
    if target.exists():
        if verified_existing(target, receipt, relative):
            if relative not in budget['counted']:
                size = target.stat().st_size
                if size > budget['limit'] - budget['stored']:
                    raise DownloadError('batch storage budget exceeded')
                budget['stored'] += size
                budget['counted'].add(relative)
            return 'verified_existing'
        raise DownloadError('existing file has no matching private receipt; review or move it to a private quarantine before retry')
    if receipt.exists() or receipt.is_symlink():
        raise DownloadError('existing receipt has no matching image; not overwritten')

    for attempt in range(attempts):
        temp = None
        try:
            response = opener.open(urllib.request.Request(BASE + relative), timeout=60)
            with response:
                if response.getcode() != 200:
                    raise DownloadError('unexpected HTTP status')
                content_type = response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                if content_type not in ('image/jpeg', 'application/octet-stream'):
                    raise DownloadError('server did not return an image content type')
                raw_size = response.headers.get('Content-Length')
                if raw_size is None or not raw_size.isdecimal() or int(raw_size) < 4:
                    raise DownloadError('server did not provide a valid full Content-Length')
                size = int(raw_size)
                if response.headers.get('Content-Range') is not None:
                    raise DownloadError('unexpected partial response')
                if (size > budget['limit'] - budget['stored']
                        or size > budget['limit'] - budget['bytes']):
                    raise DownloadError('batch storage or transfer budget exceeded')
                if shutil.disk_usage(root).free < size + budget['reserve']:
                    raise DownloadError('insufficient free space')
                fd, temp_name = tempfile.mkstemp(prefix='.visevi-part-', dir=target.parent)
                temp = Path(temp_name)
                digest, total, first, tail = hashlib.sha256(), 0, b'', b''
                with os.fdopen(fd, 'wb') as f:
                    while total < size:
                        data = response.read(min(CHUNK, size - total))
                        if not data:
                            raise OSError('short response')
                        if not first:
                            first = data[:2]
                        tail = (tail + data)[-2:]
                        total += len(data)
                        budget['bytes'] += len(data)
                        f.write(data)
                        digest.update(data)
                    f.flush()
                    os.fsync(f.fileno())
                if total != size or first != b'\xff\xd8' or tail != b'\xff\xd9':
                    raise DownloadError('download failed length or JPEG marker check')
                # Atomic hard-link creation fails if any other process created target.
                os.link(temp, target, follow_symlinks=False)
                os.chmod(target, 0o600)
                write_receipt(receipt, {'path': relative, 'size': size,
                                        'sha256': digest.hexdigest(),
                                        'source': 'MIMIC-CXR-JPG 2.1.0',
                                        'hash_type': 'locally_computed_not_official'})
                budget['stored'] += size
                budget['counted'].add(relative)
                return 'downloaded'
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise DownloadError('authorization denied; check credentials, project approval, DUA or network proxy') from None
            if attempt + 1 == attempts:
                raise DownloadError(f'HTTP request failed with status {exc.code}') from None
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            if isinstance(exc, FileExistsError):
                raise DownloadError('destination or receipt already exists; not overwritten') from None
            if attempt + 1 == attempts:
                raise DownloadError('transfer or local file operation failed') from None
        finally:
            if temp is not None:
                remove_uncommitted_image(target, temp, receipt, relative)
                temp.unlink(missing_ok=True)
    raise DownloadError('transfer attempts exhausted')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='local newline-delimited official image paths')
    parser.add_argument('--max-gib', type=float, default=20.0,
                        help='cap on manifest images and aggregate transfer bytes (default: 20 GiB)')
    parser.add_argument('--reserve-gib', type=float, default=1.0,
                        help='minimum free space to retain (default: 1 GiB)')
    args = parser.parse_args(argv)
    if not 0 < args.max_gib <= 20 or not 0 <= args.reserve_gib <= 1024:
        parser.error('--max-gib must be > 0 and <= 20; --reserve-gib must be between 0 and 1024')
    try:
        paths = load_manifest(args.manifest, DEFAULT_DEST / 'IMAGE_FILENAMES')
        private_dir(DEFAULT_DEST / '_download_receipts')
        root = DEFAULT_DEST.resolve(strict=True)
        limit = int(args.max_gib * 1024**3)
        budget = {'limit': limit, 'reserve': int(args.reserve_gib * 1024**3), 'bytes': 0,
                  **preflight_existing(root, root / '_download_receipts', paths, limit)}
        username = input('PhysioNet username: ').strip()
        password = getpass.getpass('PhysioNet password (hidden): ')
        if not username or not password or ':' in username:
            raise DownloadError('username/password are invalid')
        opener = opener_for(username, password)
        del username, password
        counts = {'downloaded': 0, 'verified_existing': 0, 'failed': 0}
        for index, relative in enumerate(paths, 1):
            try:
                status = download_one(opener, root, root / '_download_receipts', relative, budget)
                counts[status] += 1
            except DownloadError as exc:
                counts['failed'] += 1
                print(f'item {index}: {exc}', file=sys.stderr)
                if 'authorization denied' in str(exc) or 'budget exceeded' in str(exc) or 'free space' in str(exc):
                    break
            if index % 10 == 0 or index == len(paths):
                print(f"progress={index}/{len(paths)} downloaded={counts['downloaded']} "
                      f"verified_existing={counts['verified_existing']} failed={counts['failed']} "
                      f"stored_bytes={budget['stored']}", flush=True)
        print(f"downloaded={counts['downloaded']} verified_existing={counts['verified_existing']} "
              f"failed={counts['failed']} requested={len(paths)} transferred_bytes={budget['bytes']} "
              f"manifest_stored_bytes={budget['stored']}")
        return 1 if counts['failed'] else 0
    except (DownloadError, OSError, EOFError, KeyboardInterrupt) as exc:
        print(f'download stopped: {exc if isinstance(exc, DownloadError) else type(exc).__name__}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
