"""Official pneumothorax-cxr usage via transformers trust_remote_code (V3 oracle; no visevi import).

AutoModel.from_pretrained(dir, trust_remote_code=True); model.preprocess (albumentations Resize 512);
forward on float tensor in [0, 255]. Grey decoded with PIL convert('L') (author gives no decoder for PNG).
"""
import argparse, json, os, tempfile
from pathlib import Path
import numpy as np, torch
from PIL import Image

p = argparse.ArgumentParser()
for a in ('--cases', '--data-root', '--weights', '--out'):
    p.add_argument(a, type=Path, required=True)
p.add_argument('--device', default='cpu')
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['HF_MODULES_CACHE'] = tempfile.mkdtemp()
from transformers import AutoModel
model = AutoModel.from_pretrained(str(args.weights), trust_remote_code=True).eval().to(args.device)
for case in json.loads(args.cases.read_text())['cases']:
    with Image.open(args.data_root / case['image_path']) as im:
        gray = np.asarray(im.convert('L'))
    x = model.preprocess(gray)
    with torch.no_grad():
        out = model(torch.from_numpy(x.astype('float32'))[None, None].to(args.device))
    np.savez(args.out / (Path(case['image_path']).stem + '.npz'), prob=out['mask'][0, 0].float().cpu().numpy(),
             cls=out['cls'][0, 0].float().cpu().numpy())
print('done')
