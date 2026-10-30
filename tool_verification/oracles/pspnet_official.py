"""Official TorchXRayVision PSPNet pipeline (V3 oracle; no visevi import).

normalize(gray, 255) -> XRayCenterCrop -> XRayResizer(512, skimage) -> PSPNet -> sigmoid > 0.5
Run with .venv: python pspnet_official.py --cases core40.json --data-root original_data --weights tools/weights --out DIR --device cuda:4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision
import torchxrayvision as xrv


def main():
    p = argparse.ArgumentParser()
    for a in ('--cases', '--data-root', '--weights', '--out'):
        p.add_argument(a, type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(args.weights)).eval().to(args.device)
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])
    for case in json.loads(args.cases.read_text())['cases']:
        with Image.open(args.data_root / case['image_path']) as im:
            gray = np.asarray(im.convert('L')).astype(np.float32)   # same decoded grey as the adapter
        img = transform(xrv.datasets.normalize(gray, 255)[None])
        with torch.no_grad():
            masks = (model(torch.from_numpy(img)[None].to(args.device)).sigmoid()[0] > 0.5).cpu().numpy()
        np.savez_compressed(args.out / (Path(case['image_path']).stem + '.npz'), packed=np.packbits(masks, axis=-1),
                            shape=np.array(masks.shape), targets=np.array(model.targets))
    print('done')


if __name__ == '__main__':
    main()
