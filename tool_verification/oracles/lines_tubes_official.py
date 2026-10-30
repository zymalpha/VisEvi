"""Official cxr-lines-and-tubes runtime predict(), release profile 'max' (V3 oracle; no visevi import).

The author's _load_state is replaced only by an identical function using torch.load(weights_only=True);
all 25 checkpoints contain tensors only, so the loaded state is the same.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch
from PIL import Image

p = argparse.ArgumentParser()
for a in ('--cases', '--data-root', '--weights', '--out'):
    p.add_argument(a, type=Path, required=True)
p.add_argument('--device', default='cpu')
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('rt', args.weights / 'local_lines_tubes_runtime.py')
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
def load_state(model, path):
    ck = torch.load(path, map_location='cpu', weights_only=True); raw = ck.get('state_dict', ck)
    st = {k.removeprefix('model.'): v for k, v in raw.items() if k.startswith('model.') and not k.startswith('model.criterion.')}
    missing, unexpected = model.load_state_dict(st, strict=False)
    assert not missing and not unexpected
rt._load_state = load_state
runtime = rt.LocalLinesTubesRuntime(args.weights, args.device)
for case in json.loads(args.cases.read_text())['cases']:
    with Image.open(args.data_root / case['image_path']) as im:
        gray = np.asarray(im.convert('L'))
    out = runtime.predict(gray, presence_folds=5, device_folds=5, carina_folds=5, classifier_folds=5, force_full_pipeline=True)
    (args.out / (Path(case['image_path']).stem + '.json')).write_text(json.dumps(
        {'presence': out['presence'], 'classification': out['classification'], 'landmarks': out['landmarks'],
         'fold_ids': out['metadata']['fold_ids']}))
print('done')
