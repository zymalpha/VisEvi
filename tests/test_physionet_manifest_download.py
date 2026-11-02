"""Offline tests for the restricted subset downloader; never contact PhysioNet."""
import importlib.util
import http.client
import io
from email.message import Message
from pathlib import Path
import urllib.error
import urllib.request
import urllib.response

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/download_physionet_manifest.py'
spec = importlib.util.spec_from_file_location('physionet_downloader_test', MODULE_PATH)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)

IMAGE = 'files/p10/p10000032/s50414267/' + 'a' * 8 + ('-' + 'b' * 8) * 4 + '.jpg'
JPEG = b'\xff\xd8synthetic-test\xff\xd9'


class Response(io.BytesIO):
    def __init__(self, data=JPEG, length=None, kind='image/jpeg', status=200, extra=None):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Type': kind, 'Content-Length': str(len(data) if length is None else length)}
        self.headers.update(extra or {})

    def getcode(self):
        return self.status


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        assert request.full_url == downloader.BASE + IMAGE
        return next(self.responses)


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    receipts = root / '_download_receipts'
    downloader.private_dir(receipts)
    return root, receipts


def test_valid_manifest_requires_exact_official_path_and_rejects_duplicates(tmp_path):
    index, manifest = tmp_path / 'index', tmp_path / 'manifest'
    index.write_text(IMAGE + '\n')
    manifest.write_text(IMAGE + '\n')
    assert downloader.load_manifest(manifest, index) == [IMAGE]
    manifest.write_text(IMAGE + '\n' + IMAGE + '\n')
    with pytest.raises(downloader.DownloadError):
        downloader.load_manifest(manifest, index)
    manifest.write_text('../' + IMAGE + '\n')
    with pytest.raises(downloader.DownloadError):
        downloader.load_manifest(manifest, index)
    manifest.write_text(IMAGE.replace('p10/', 'p20/') + '\n')
    with pytest.raises(downloader.DownloadError):
        downloader.load_manifest(manifest, index)


def test_full_image_is_saved_with_private_receipt_and_reused(setup):
    root, receipts = setup
    opener = Opener([Response()])
    budget = {'limit': 1024, 'reserve': 0, 'bytes': 0}
    assert downloader.download_one(opener, root, receipts, IMAGE, budget) == 'downloaded'
    assert (root / IMAGE).read_bytes() == JPEG
    receipt = downloader.receipt_path(receipts, IMAGE)
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert downloader.download_one(opener, root, receipts, IMAGE, budget) == 'verified_existing'
    assert opener.calls == 1
    assert budget['bytes'] == len(JPEG)


def test_existing_unverified_or_tampered_file_is_never_overwritten(setup):
    root, receipts = setup
    target = downloader.ensure_parents(root, IMAGE)
    target.write_bytes(b'user-data')
    with pytest.raises(downloader.DownloadError, match='private quarantine'):
        downloader.download_one(Opener([]), root, receipts, IMAGE,
                                {'limit': 1024, 'reserve': 0, 'bytes': 0})
    assert target.read_bytes() == b'user-data'


def test_truncated_transfer_retries_from_zero_and_tracks_all_bytes(setup):
    root, receipts = setup
    opener = Opener([Response(JPEG[:4], length=len(JPEG)), Response(JPEG)])
    budget = {'limit': 1024, 'reserve': 0, 'bytes': 0}
    assert downloader.download_one(opener, root, receipts, IMAGE, budget) == 'downloaded'
    assert (root / IMAGE).read_bytes() == JPEG
    assert budget['bytes'] == len(JPEG) + 4
    assert not list((root / IMAGE).parent.glob('.visevi-part-*'))


@pytest.mark.parametrize('response', [
    Response(b'<html>error</html>', kind='text/html'),
    Response(JPEG, status=206, extra={'Content-Range': f'bytes 0-{len(JPEG)-1}/{len(JPEG)}'}),
    Response(b'bad-image-data'),
])
def test_invalid_response_does_not_leave_an_image(setup, response):
    root, receipts = setup
    with pytest.raises(downloader.DownloadError):
        downloader.download_one(Opener([response]), root, receipts, IMAGE,
                                {'limit': 1024, 'reserve': 0, 'bytes': 0})
    assert not (root / IMAGE).exists()


def test_budget_is_checked_before_writing(setup):
    root, receipts = setup
    with pytest.raises(downloader.DownloadError, match='budget exceeded'):
        downloader.download_one(Opener([Response()]), root, receipts, IMAGE,
                                {'limit': len(JPEG) - 1, 'reserve': 0, 'bytes': 0})
    assert not (root / IMAGE).exists()


def test_symlink_destination_is_refused(setup, tmp_path):
    root, receipts = setup
    (root / 'files').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(downloader.DownloadError, match='symbolic link'):
        downloader.download_one(Opener([]), root, receipts, IMAGE,
                                {'limit': 1024, 'reserve': 0, 'bytes': 0})


def test_redirect_to_another_host_or_http_is_rejected():
    handler = downloader.SameHostRedirect()
    request = urllib.request.Request(downloader.BASE + IMAGE)
    for location in ('https://example.com/path', 'http://physionet.org/path',
                     'https://physionet.org.evil.example/path'):
        with pytest.raises(downloader.DownloadError, match='unapproved'):
            handler.redirect_request(request, None, 302, 'redirect', {}, location)


def test_corrupted_previously_downloaded_image_is_not_replaced(setup):
    root, receipts = setup
    budget = {'limit': 1024, 'reserve': 0, 'bytes': 0}
    assert downloader.download_one(Opener([Response()]), root, receipts, IMAGE, budget) == 'downloaded'
    target = root / IMAGE
    target.write_bytes(b'changed')
    with pytest.raises(downloader.DownloadError, match='private quarantine'):
        downloader.download_one(Opener([]), root, receipts, IMAGE, budget)
    assert target.read_bytes() == b'changed'


def test_403_is_reported_without_credentials_or_patient_path(setup):
    root, receipts = setup
    class Forbidden:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, 'secret text', {}, None)

    with pytest.raises(downloader.DownloadError, match='authorization denied') as error:
        downloader.download_one(Forbidden(), root, receipts, IMAGE,
                                {'limit': 1024, 'reserve': 0, 'bytes': 0})
    assert IMAGE not in str(error.value)
    assert 'secret text' not in str(error.value)
    assert not (root / IMAGE).exists()


def test_network_interruption_retries_without_publishing_partial_image(setup):
    root, receipts = setup
    class Interrupting(Response):
        def read(self, size=-1):
            raise http.client.IncompleteRead(b'partial')

    opener = Opener([Interrupting(), Response()])
    result = downloader.download_one(opener, root, receipts, IMAGE,
                                     {'limit': 1024, 'reserve': 0, 'bytes': 0})
    assert result == 'downloaded'
    assert opener.calls == 2
    assert (root / IMAGE).read_bytes() == JPEG


def test_preflight_accounts_existing_images_against_whole_manifest_cap(setup):
    root, receipts = setup
    first_budget = {'limit': 1024, 'reserve': 0, 'bytes': 0}
    assert downloader.download_one(Opener([Response()]), root, receipts, IMAGE, first_budget) == 'downloaded'
    with pytest.raises(downloader.DownloadError, match='storage budget'):
        downloader.preflight_existing(root, receipts, [IMAGE], len(JPEG) - 1)
    existing = downloader.preflight_existing(root, receipts, [IMAGE], len(JPEG))
    assert existing['stored'] == len(JPEG)
    budget = {'limit': len(JPEG), 'reserve': 0, 'bytes': 0, **existing}
    assert downloader.download_one(Opener([]), root, receipts, IMAGE, budget) == 'verified_existing'
    assert budget['stored'] == len(JPEG)


def test_preflight_blocks_unverified_existing_images_before_any_download(setup):
    root, receipts = setup
    target = downloader.ensure_parents(root, IMAGE)
    target.write_bytes(b'old user file')
    with pytest.raises(downloader.DownloadError, match='private quarantine'):
        downloader.preflight_existing(root, receipts, [IMAGE], 1024)
    assert target.read_bytes() == b'old user file'


def test_opener_honours_https_proxy_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv('https_proxy', 'http://127.0.0.1:8123')
    opener = downloader.opener_for('test-user', 'test-password')
    proxy_handlers = [handler for handler in opener.handlers
                      if isinstance(handler, urllib.request.ProxyHandler)]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies['https'] == 'http://127.0.0.1:8123'
    assert any(isinstance(handler, downloader.SameHostRedirect) for handler in opener.handlers)
    assert not any(name.lower() == 'authorization' for name, _ in opener.addheaders)


def test_physionet_challenge_and_session_cookie_are_used_in_memory(monkeypatch):
    """Exercise urllib's actual auth/cookie handlers with fake HTTPS responses."""
    seen = []

    def reply(request, status, headers):
        message = Message()
        for key, value in headers:
            message.add_header(key, value)
        response = urllib.response.addinfourl(io.BytesIO(JPEG if status == 200 else b''),
                                              message, request.full_url, status)
        response.msg = 'OK' if status == 200 else 'Unauthorized'
        return response

    def mock_https_open(handler, request):
        auth = request.get_header('Authorization')
        cookie = request.get_header('Cookie', '')
        seen.append((auth is not None, cookie))
        if not auth and 'sessionid=synthetic-session' not in cookie:
            return reply(request, 401, [
                ('WWW-Authenticate', 'Basic realm="PhysioNet", charset="UTF-8"'),
                ('Set-Cookie', 'testcookie=1; Path=/; Secure'),
            ])
        if auth:
            assert auth == 'Basic dGVzdC11c2VyOnRlc3QtcGFzc3dvcmQ='
            assert 'testcookie=1' in cookie
            return reply(request, 200, [('Set-Cookie', 'sessionid=synthetic-session; Path=/; Secure')])
        assert 'sessionid=synthetic-session' in cookie
        return reply(request, 200, [])

    monkeypatch.setattr(urllib.request.HTTPSHandler, 'https_open', mock_https_open)
    opener = downloader.opener_for('test-user', 'test-password')
    with opener.open(downloader.BASE + IMAGE) as response:
        assert response.read() == JPEG
    with opener.open(downloader.BASE + IMAGE) as response:
        assert response.read() == JPEG
    assert len(seen) == 3
    assert seen[0] == (False, '')
    assert seen[1][0] is True
    assert seen[2][0] is False


@pytest.mark.parametrize('failure', [OSError('receipt failure'), KeyboardInterrupt()])
def test_receipt_failure_or_interrupt_rolls_back_only_new_image(setup, monkeypatch, failure):
    root, receipts = setup
    def fail(*args):
        raise failure
    monkeypatch.setattr(downloader, 'write_receipt', fail)
    with pytest.raises((downloader.DownloadError, KeyboardInterrupt)):
        downloader.download_one(Opener([Response(), Response(), Response()]), root, receipts, IMAGE,
                                {'limit': 1024, 'reserve': 0, 'bytes': 0})
    assert not (root / IMAGE).exists()
    assert not downloader.receipt_path(receipts, IMAGE).exists()
    assert not list((root / IMAGE).parent.glob('.visevi-part-*'))


def test_interrupt_after_receipt_publication_preserves_verified_pair(setup, monkeypatch):
    root, receipts = setup
    original = downloader.write_receipt
    def commit_then_interrupt(*args):
        original(*args)
        raise KeyboardInterrupt()
    monkeypatch.setattr(downloader, 'write_receipt', commit_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        downloader.download_one(Opener([Response()]), root, receipts, IMAGE,
                                {'limit': 1024, 'reserve': 0, 'bytes': 0})
    assert downloader.verified_existing(root / IMAGE, downloader.receipt_path(receipts, IMAGE), IMAGE)
