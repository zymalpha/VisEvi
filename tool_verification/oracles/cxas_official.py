"""Official CXAS inference path (V3 oracle; no visevi import).

cxas.file_io.FileLoader.load_image -> BackboneUNet forward (get_results, threshold 0.5) ->
CXAS.resize_to_numpy (nearest to file size). Weights loaded strictly with weights_only=True
(argparse.Namespace allowed) instead of the author's weights_only=False / strict=False.
"""
import argparse, json, os, sys, tempfile
from pathlib import Path
import numpy as np, torch

p = argparse.ArgumentParser()
for a in ('--cases', '--data-root', '--weights', '--source', '--out'):
    p.add_argument(a, type=Path, required=True)
p.add_argument('--gpu', default='4')
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
os.environ['CXAS_PATH'] = tempfile.mkdtemp()            # the package writes a welcome marker here, not $HOME
sys.path.insert(0, str(args.source.parent))
from cxas.file_io import FileLoader
from cxas.segmentor import CXAS
from cxas.models.UNet.backbone_unet import BackboneUNet
from cxas.label_mapper import id2label_dict
import argparse as _ap
model = BackboneUNet('UNet_ResNet50_default', len(id2label_dict))
with torch.serialization.safe_globals([_ap.Namespace]):
    ck = torch.load(args.weights / 'UNet_ResNet50_default.pth', map_location='cpu', weights_only=True)
model.load_state_dict({k.removeprefix('module.'): v for k, v in ck['model'].items()}, strict=True)
model = model.eval().to(f'cuda:{args.gpu}')
loader = FileLoader(args.gpu)
for case in json.loads(args.cases.read_text())['cases']:
    fd = loader.load_file(str(args.data_root / case['image_path']))
    fd['filename'] = [fd['filename']]; fd['file_size'] = [fd['file_size']]
    with torch.no_grad():
        pred = model(fd)
    full = CXAS.resize_to_numpy(None, pred['segmentation_preds'][0], fd['file_size'][0])
    np.savez_compressed(args.out / (Path(case['image_path']).stem + '.npz'), packed=np.packbits(full, axis=-1),
                        shape=np.array(full.shape))
print('done')
