"""CheXmask-U HybridGNet ``v1_skip`` landmark worker (A5)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from ._runtime import device_runtime, serve

ROOT = Path(__file__).resolve().parents[3]
SOURCE = next((ROOT / 'tools/src/chexmask-u').glob('CheXmask-U-*/HybridGNet'))
INPUT_SIZE = 1024


def load(weights_dir, device):
    import torch
    from safetensors.torch import load_file

    sys.path.insert(0, str(SOURCE))
    from models.HybridGNet2IGSC import HybridGNetHF

    config = json.loads((Path(weights_dir) / 'v1_skip/config.json').read_text())
    config.pop('n_nodes', None)
    # The released config samples the latent in eval mode; decode from the mean
    # for the primary (deterministic) prediction and sample only on request.
    config['eval_sampling'] = False
    model = HybridGNetHF(device=device, **config)
    model.load_state_dict(load_file(str(Path(weights_dir) / 'v1_skip/model.safetensors'),
                                    device=device), strict=True)
    model.eval()
    # PyG 2.8 generates a jinja ``propagate`` from each subclass signature; for the
    # author's re-declared ChebConv/Pool this template rejects keyword ``x``. The
    # generic implementation it replaces is functionally identical.
    from torch_geometric.nn import MessagePassing
    for module in model.modules():
        if isinstance(module, MessagePassing) and hasattr(type(module), '_orig_propagate'):
            module.propagate = type(module)._orig_propagate.__get__(module, type(module))
    runtime = device_runtime(device)
    runtime.update(source=str(SOURCE.relative_to(ROOT)), input_size=INPUT_SIZE,
                   primary_output='decoder(latent mean)')
    return model, runtime


def preprocess(image_path):
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        gray = np.asarray(image.convert('L'))
    height, width = gray.shape
    side = max(height, width)
    pad_top, pad_left = (side - height) // 2, (side - width) // 2
    square = np.zeros((side, side), dtype=gray.dtype)
    square[pad_top:pad_top + height, pad_left:pad_left + width] = gray
    resized = cv2.resize(square, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    frame = {'original_width': width, 'original_height': height, 'square_side': side,
             'pad_left': pad_left, 'pad_top': pad_top, 'model_frame': [INPUT_SIZE, INPUT_SIZE],
             'gray_conversion': 'Pillow.convert(L)', 'padding': 'zeros, centred (author DataPreparation)',
             'resize': 'cv2.INTER_LINEAR', 'normalization': 'gray/255',
             'output_mapping': 'x_orig = x_model * side/1024 - pad_left; y_orig = y_model * side/1024 - pad_top'}
    return resized.astype('float32') / 255.0, frame


def handle(model, request):
    import numpy as np
    import torch

    params = request['params']
    samples, seed = int(params.get('uncertainty_samples', 0)), int(params.get('seed', 0))
    pixels, frame = preprocess(request['image_path'])
    scale = frame['square_side'] / INPUT_SIZE
    offset = np.array([frame['pad_left'], frame['pad_top']], dtype=np.float64)
    data = torch.from_numpy(pixels)[None, None].to(model.device)
    with torch.no_grad():  # inference_mode breaks the sparse adjacency ops
        mu, log_var, conv6, conv5 = model.model.encode(data)
        nodes = model.model.decode(mu, conv6, conv5)[0].reshape(-1, 2).double().cpu().numpy()
        result = {'nodes_xy': (nodes * INPUT_SIZE * scale - offset).tolist(), 'frame': frame,
                  'node_groups': {'right_lung': [0, 44], 'left_lung': [44, 94], 'heart': [94, 120]},
                  'coordinate_convention': 'original image pixels, x=column, y=row'}
        if samples:
            torch.manual_seed(seed)
            draws = torch.stack([model.model.decode(model.model.sampling(mu, log_var), conv6, conv5)[0]
                                 .reshape(-1, 2) for _ in range(samples)]).double().cpu().numpy()
            draws = draws * INPUT_SIZE * scale
            result['sampling'] = {'samples': samples, 'seed': seed,
                                  'node_std_xy': draws.std(axis=0).tolist(),
                                  'meaning': 'spread of decoded nodes under latent sampling; not a calibrated error'}
    return result, []


if __name__ == '__main__':
    serve(load, handle)
