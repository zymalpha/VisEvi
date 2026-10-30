"""CXAS chest anatomy segmentation worker (159 PAX-Ray++ labels; supports A12 and A25).

Weights: UNet_ResNet50_default (Google Drive, SHA256 pinned at first download).
Only the UNet modules are imported from the author's source; the ``cxas``
package itself is not imported (it needs DICOM extras and writes to $HOME).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ._runtime import device_runtime, serve

ROOT = Path(__file__).resolve().parents[3]
SOURCE = next((ROOT / 'tools/src/cxas').glob('ChestXRayAnatomySegmentation-*/cxas'))
MODEL_SIZE = 512
THRESHOLD = 0.5


def load(weights_dir, device):
    import torch

    sys.path.insert(0, str(SOURCE / 'models'))
    from UNet.backbone_unet import BackboneUNet

    labels = json.loads((SOURCE / 'data/paxray_labels.json').read_text())['label_dict']
    model = BackboneUNet('UNet_ResNet50_default', len(labels))
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(Path(weights_dir) / 'UNet_ResNet50_default.pth',
                                map_location='cpu', weights_only=True)
    state = {key.removeprefix('module.'): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    runtime = device_runtime(device)
    runtime.update(source=str(SOURCE.relative_to(ROOT)), labels=len(labels),
                   input='RGB/255, ImageNet mean/std, F.interpolate to 512x512 (nearest, whole image)',
                   threshold=THRESHOLD, checkpoint_epoch=int(checkpoint.get('epoch', -1)))
    return (model, labels), runtime


def handle(state, request):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model, labels = state
    with Image.open(request['image_path']) as image:
        rgb = np.asarray(image.convert('RGB'))
    height, width = rgb.shape[:2]
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
    tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / \
        torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = F.interpolate(tensor[None], MODEL_SIZE).to(next(model.parameters()).device)
    with torch.inference_mode():
        logits = model._forward(tensor)['logits'][0].float()
    masks = (logits.sigmoid() > THRESHOLD).cpu().numpy()
    out_dir = Path(request['output_dir'])
    np.savez_compressed(out_dir / 'labels_model_frame.npz', packed=np.packbits(masks, axis=-1),
                        shape=np.array(masks.shape))
    result = {'labels': labels, 'model_frame_hw': [MODEL_SIZE, MODEL_SIZE],
              'pixel_counts_model_frame': {labels[str(i)]: int(masks[i].sum()) for i in range(len(labels))},
              'frame': {'original_width': width, 'original_height': height,
                        'mapping': 'whole image squashed to 512x512 (nearest); masks map back with '
                                   'torch nearest indexing src = floor(dst * 512 / size)'},
              'threshold': THRESHOLD}
    return result, ['labels_model_frame.npz']


if __name__ == '__main__':
    serve(load, handle)
