"""Lines/tubes worker (A26; ianpan/cxr-lines-and-tubes c11c443d, release profile 'max')."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from ._runtime import device_runtime, serve

PROFILE = {'presence': 5, 'device': 5, 'carina': 5, 'classifier': 5}   # release 'max'
FAMILIES = ('ETT', 'NGT', 'CVC', 'Swan Ganz')
LINE_THRESHOLD = 0.5


def load(weights_dir, device):
    path = Path(weights_dir) / 'local_lines_tubes_runtime.py'
    spec = importlib.util.spec_from_file_location('_lines_tubes_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def load_state(model, checkpoint_path):
        """Author's _load_state with torch.load restricted to tensors/containers."""
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        raw = checkpoint.get('state_dict', checkpoint)
        state = {key.removeprefix('model.'): value for key, value in raw.items()
                 if key.startswith('model.') and not key.startswith('model.criterion.')}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f'checkpoint mismatch for {checkpoint_path}: {missing}, {unexpected}')

    module._load_state = load_state
    runtime_model = module.LocalLinesTubesRuntime(weights_dir, device)
    if tuple(module.PRESENCE_NAMES) != FAMILIES:
        raise RuntimeError(f'unexpected family order {module.PRESENCE_NAMES}')
    runtime = device_runtime(device)
    runtime.update(profile='max', fold_counts=PROFILE, autocast='bfloat16 on CUDA (author runtime)',
                   screening_threshold=module.PRESENCE_SCREENING_THRESHOLD,
                   classifier_names=list(module.CLASSIFIER_NAMES))
    return (module, runtime_model), runtime


def _peak(probability, shape):
    import numpy as np
    y, x = np.unravel_index(int(np.argmax(probability)), probability.shape)
    height, width = shape
    # same pixel-corner-to-corner scaling as the author runtime (_peak_xy)
    return [float(x / max(probability.shape[1] - 1, 1) * (width - 1)),
            float(y / max(probability.shape[0] - 1, 1) * (height - 1))], float(probability[y, x])


def handle(state, request):
    import cv2
    import numpy as np
    from PIL import Image

    module, model = state
    with Image.open(request['image_path']) as image:
        gray = np.asarray(image.convert('L'))
    shape = gray.shape
    presence, presence_folds = model._presence(gray, PROFILE['presence'])
    agnostic, family, raw_family, device_folds = model._device(gray, PROFILE['device'])
    carina, carina_xy, carina_folds = model._carina(gray, PROFILE['carina'])
    classification, classifier_folds = model._classifier(gray, agnostic, carina, PROFILE['classifier'])
    out_dir = Path(request['output_dir'])
    devices, artifacts = {}, []
    for index, name in enumerate(FAMILIES):
        line = cv2.resize(family[index], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) > LINE_THRESHOLD
        file_name = f'line_{name.replace(" ", "_")}.png'
        Image.fromarray(line.astype('uint8') * 255).save(out_dir / file_name)
        artifacts.append(file_name)
        tip_xy, tip_peak = _peak(raw_family[4 + index] * agnostic[1], shape)
        devices[name] = {'presence_raw_score': presence[name], 'line_mask_file': file_name,
                         'line_mask_area_px': int(line.sum()),
                         'distal_tip_xy': tip_xy, 'distal_tip_peak_probability': tip_peak}
    _, carina_peak = _peak(carina, shape)
    np.savez_compressed(out_dir / 'model_frame_maps.npz', agnostic=agnostic.astype('float16'),
                        family=family.astype('float16'), carina=carina.astype('float16'))
    artifacts.append('model_frame_maps.npz')
    result = {'devices': devices, 'carina_xy': carina_xy, 'carina_heatmap_peak': carina_peak,
              'classification_raw': classification,
              'fold_ids': {'presence': presence_folds, 'device': device_folds,
                           'carina': carina_folds, 'classifier': classifier_folds},
              'frame': {'original_width': shape[1], 'original_height': shape[0],
                        'mapping': 'direct resize of the whole image (1024 devices, 512 carina/classifier, '
                                   '384 presence); coordinates scaled corner-to-corner as in the author runtime'},
              'line_threshold': LINE_THRESHOLD}
    return result, artifacts


if __name__ == '__main__':
    serve(load, handle)
