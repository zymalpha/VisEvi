import hashlib

import pytest

from tool_verification.suites.case_manifest import verify_frozen_cases


def test_frozen_case_hash_is_verified(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"frozen image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()

    verify_frozen_cases([{"image_path": "image.png", "sha256": digest}], tmp_path)


def test_frozen_case_hash_mismatch_fails(tmp_path):
    (tmp_path / "image.png").write_bytes(b"replacement image bytes")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_frozen_cases([{"image_path": "image.png", "sha256": "0" * 64}], tmp_path)


def test_frozen_case_without_hash_fails(tmp_path):
    (tmp_path / "image.png").write_bytes(b"image")

    with pytest.raises(ValueError, match="no valid sha256"):
        verify_frozen_cases([{"image_path": "image.png"}], tmp_path)
