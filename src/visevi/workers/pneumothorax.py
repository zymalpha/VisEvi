"""Pneumothorax segmentation worker (A8; ianpan/pneumothorax-cxr f00a7af5)."""
from __future__ import annotations

import os
from pathlib import Path

from ._runtime import device_runtime, serve

ROOT = Path(__file__).resolve().parents[3]
MASK_THRESHOLD = 0.5


def load(weights_dir, device):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / 'tools/cache/hf_modules'))
    from transformers import AutoModel

    model, info = AutoModel.from_pretrained(weights_dir, trust_remote_code=True,
                                            output_loading_info=True)
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise RuntimeError(f'checkpoint does not match the model exactly: {problems}')
    model = model.eval().to(device)
    runtime = device_runtime(device)
    runtime.update(input='grayscale 0-255, direct resize to 512x512 (cv2.INTER_LINEAR, aspect not kept)',
                   normalization='model.normalize: [0,255] -> [-1,1]', mask_threshold=MASK_THRESHOLD)
    return model, runtime


def handle(model, request):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(request['image_path']) as image:
        gray = np.asarray(image.convert('L'))
    height, width = gray.shape
    size = tuple(model.img_size)
    resized = cv2.resize(gray, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype('float32'))[None, None].to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model(tensor)
    probability = out['mask'][0, 0].float().cpu().numpy()
    case_score = float(out['cls'][0, 0].float().cpu())
    original = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = original > MASK_THRESHOLD
    out_dir = Path(request['output_dir'])
    np.save(out_dir / 'probability_model_frame.npy', probability.astype('float16'))
    Image.fromarray(mask.astype('uint8') * 255).save(out_dir / 'pneumothorax_mask.png')
    result = {'case_score': case_score, 'mask_area_px': int(mask.sum()),
              'max_pixel_probability': float(probability.max()), 'mask_threshold': MASK_THRESHOLD,
              'frame': {'original_width': width, 'original_height': height, 'model_frame_hw': list(size),
                        'mapping': 'model frame stretched to the full image (no crop, no padding); '
                                   'probability resized bilinearly to the original size, then thresholded'}}
    return result, ['probability_model_frame.npy', 'pneumothorax_mask.png']


if __name__ == '__main__':
    serve(load, handle)
