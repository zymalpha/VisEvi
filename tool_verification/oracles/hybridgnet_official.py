"""Official CheXmask-U HybridGNet usage (V3 oracle; no visevi import).

Author DataPreparation: cv2.imread(GRAYSCALE) -> centred zero pad -> cv2.resize 1024 -> /255;
weights: v1_skip/pytorch_model.bin as loaded by HybridGNetHF.from_pretrained (here weights_only=True).
Outputs nodes in the 1024 model frame from the latent mean, plus one author-default sampled run.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch

p = argparse.ArgumentParser()
for a in ('--cases', '--data-root', '--weights', '--source', '--out'):
    p.add_argument(a, type=Path, required=True)
p.add_argument('--device', default='cpu')
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.source))
from models.HybridGNet2IGSC import HybridGNetHF
from torch_geometric.nn import MessagePassing
cfg = json.loads((args.weights / 'v1_skip/config.json').read_text()); cfg.pop('n_nodes', None)
model = HybridGNetHF(device=args.device, **cfg)
state = torch.load(args.weights / 'v1_skip/pytorch_model.bin', map_location=args.device, weights_only=True)
if not next(iter(state)).startswith('model.'):
    state = {f'model.{k}': v for k, v in state.items()}
model.load_state_dict(state, strict=True); model.eval()
for m in model.modules():   # same PyG 2.8 propagate workaround as needed to run the author's code at all
    if isinstance(m, MessagePassing) and hasattr(type(m), '_orig_propagate'):
        m.propagate = type(m)._orig_propagate.__get__(m, type(m))
for case in json.loads(args.cases.read_text())['cases']:
    img = cv2.imread(str(args.data_root / case['image_path']), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape; L = max(h, w)
    sq = np.zeros((L, L), img.dtype); pt, pl = (L - h) // 2, (L - w) // 2
    sq[pt:pt + h, pl:pl + w] = img
    x = torch.from_numpy(cv2.resize(sq, (1024, 1024)) / 255.0).float()[None, None].to(args.device)
    with torch.no_grad():
        mu, lv, c6, c5 = model.model.encode(x)
        mean = model.model.decode(mu, c6, c5)[0].reshape(-1, 2).double().cpu().numpy() * 1024
        torch.manual_seed(0)
        sampled = model.model.decode(model.model.sampling(mu, lv), c6, c5)[0].reshape(-1, 2).double().cpu().numpy() * 1024
    np.savez(args.out / (Path(case['image_path']).stem + '.npz'), mean=mean, sampled=sampled,
             pad=np.array([pl, pt, L]))
print('done')
