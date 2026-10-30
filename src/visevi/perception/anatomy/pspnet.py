"""Live PSPNet inference with image- and configuration-bound cached artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = 'torchxrayvision.baseline_models.chestx_det.PSPNet'
WEIGHTS_FILENAME = 'pspnet_chestxray_best_model_4.pth'
# Release asset recorded in tools/weights/pspnet_provenance.json.
DOCUMENTED_WEIGHTS_SHA256 = '019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc'
# v2 (2026-10-30): official TorchXRayVision XRayCenterCrop + XRayResizer (skimage);
# v1 used torch bilinear resizing and (W-s)//2 offsets, which differed from the official pipeline.
PREPROCESSING_VERSION = 2
EXIF_ORIENTATION = 0x0112


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_cache_record(record, image_path: str | Path, mask_names) -> dict:
    """Check that a cache record belongs to this image, model, weights and preprocessing.

    Returns the verified ``mask_sha256`` mapping; raises ValueError otherwise.
    Used for cached artifacts (``masks_dir``); live inference instead requires
    the complete binding, including adapter hash and thread count, to be equal.
    """
    if not isinstance(record, dict):
        raise ValueError('cache_record.json must contain an object')
    binding, hashes = record.get('binding'), record.get('mask_sha256')
    if not isinstance(binding, dict) or not isinstance(hashes, dict):
        raise ValueError('cache record must contain binding and mask_sha256 objects')
    if binding.get('image_sha256') != file_sha256(image_path):
        raise ValueError('mask cache is not bound to this image')
    producer = binding.get('producer')
    if not isinstance(producer, dict):
        raise ValueError('cache record has no producer identity')
    if producer.get('model') != MODEL_NAME:
        raise ValueError(f'cache producer is not {MODEL_NAME}')
    if producer.get('weights_sha256') != DOCUMENTED_WEIGHTS_SHA256:
        raise ValueError('cache was produced with undocumented PSPNet weights')
    if producer.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('cache preprocessing version differs from the current adapter')
    if binding.get('preprocessing') != image_frame(image_path):
        raise ValueError('cache preprocessing/frame record differs from this image')
    unverified = [name for name in mask_names if name not in hashes]
    if unverified:
        raise ValueError('cache record has no checksum for: ' + ', '.join(unverified))
    return hashes


def image_frame(path: str | Path) -> dict:
    """Preprocessing/frame record for an image, read from the header only."""
    with Image.open(path) as image:
        _check_image(image)
        return _frame_record(image.mode, *image.size)


def _check_image(image: Image.Image) -> None:
    if image.mode in ('I', 'F') or image.mode.startswith('I;16'):
        raise ValueError('PSPNet expects an 8-bit display raster; high-bit-depth/float input requires an explicit recorded windowing conversion')
    if image.getexif().get(EXIF_ORIENTATION, 1) != 1:
        raise ValueError('EXIF orientation is not identity; transpose the image explicitly and record it before inference')


def _frame_record(mode: str, width: int, height: int) -> dict:
    side = min(width, height)
    ox, oy = width // 2 - side // 2, height // 2 - side // 2      # XRayCenterCrop
    return {'original_width': width, 'original_height': height,
        'source_mode': mode, 'crop_side': side, 'crop_offset_xy': [ox, oy],
        'model_frame': [512, 512], 'gray_conversion': 'Pillow.convert(L)',
        'normalization': 'gray/255*2048-1024',
        'resize': 'skimage.transform.resize(mode=constant, preserve_range=True) as in TorchXRayVision XRayResizer(512)',
        'mask_threshold': 'sigmoid(logits) > 0.5',
        'original_frame_transform': 'nearest resize to crop_side, paste at crop_offset_xy'}


def prepare_image(path: str | Path) -> tuple[np.ndarray, dict]:
    with Image.open(path) as image:
        _check_image(image)
        record = _frame_record(image.mode, *image.size)
        gray = image.convert('L')  # resolves palette and normalizes RGBA/L modes
    from skimage.transform import resize

    # Same order as the official TorchXRayVision pipeline: normalize -> centre crop
    # -> XRayResizer (skimage, mode='constant' pads with 0 in normalized units).
    (ox, oy), side = record['crop_offset_xy'], record['crop_side']
    pixels = np.asarray(gray, dtype=np.float32) / 255.0 * 2048.0 - 1024.0
    pixels = pixels[None, oy:oy + side, ox:ox + side]
    pixels = resize(pixels, (1, 512, 512), mode='constant', preserve_range=True).astype(np.float32)
    return pixels[None], record


class PSPNetPredictor:
    def __init__(self, weights_dir: str | Path, device: str = 'cpu'):
        import torch
        from torchxrayvision.baseline_models.chestx_det import PSPNet

        self.device = device
        weights_dir = Path(weights_dir).resolve()
        weights = weights_dir / WEIGHTS_FILENAME
        if not weights.is_file():
            raise FileNotFoundError(f'Download the documented PSPNet checkpoint first: {weights}')
        weights_sha256 = file_sha256(weights)
        if weights_sha256 != DOCUMENTED_WEIGHTS_SHA256:
            raise ValueError(f'PSPNet checkpoint differs from the documented release: {weights}')
        self.identity = {'model': MODEL_NAME,
            'torchxrayvision_version': importlib.metadata.version('torchxrayvision'),
            'torch_version': torch.__version__, 'pillow_version': importlib.metadata.version('pillow'),
            'numpy_version': np.__version__, 'device': device,
            'torch_cpu_threads': torch.get_num_threads(),
            'weights_sha256': weights_sha256,
            'adapter_sha256': file_sha256(__file__), 'preprocessing_version': PREPROCESSING_VERSION}
        self.model = PSPNet(cache_dir=str(weights_dir)).eval().to(device)

    def predict(self, image_path: str | Path, sample_key: str, output_dir: str | Path) -> dict:
        import torch

        image_path, output_dir = Path(image_path).resolve(), Path(output_dir)
        pixels, preprocessing = prepare_image(image_path)
        binding = {'sample_key': sample_key, 'image_sha256': file_sha256(image_path),
                   'producer': self.identity, 'preprocessing': preprocessing}
        record_path = output_dir / 'cache_record.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['binding'] != binding:
                raise ValueError('Cache identity mismatch; use a new output directory')
            for name, digest in record['mask_sha256'].items():
                if file_sha256(output_dir / name) != digest:
                    raise ValueError('Cached mask checksum mismatch: ' + name)
            return record
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError('Nonempty prediction directory has no valid cache record; use a new directory')
        output_dir.mkdir(parents=True, exist_ok=True)
        tensor = torch.from_numpy(pixels).to(self.device)
        with torch.inference_mode():
            masks = (self.model(tensor).sigmoid()[0] > 0.5).cpu().numpy()
        hashes = {}
        for target, mask in zip(self.model.targets, masks):
            name = target.replace(' ', '_') + '.png'
            path = output_dir / name
            Image.fromarray(mask.astype('uint8') * 255).save(path)
            hashes[name] = file_sha256(path)
        record = {'binding': binding, 'image_path': str(image_path), 'mask_sha256': hashes,
                  'is_ground_truth': False, 'mask_coordinate_frame': '512x512 center-crop model frame'}
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        return record
