"""Exercise batch isolation and image decoding without downloading a model."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visevi.eval.batch import run_batch
from visevi.perception.anatomy.pspnet import (DOCUMENTED_WEIGHTS_SHA256, MODEL_NAME,
                                              PREPROCESSING_VERSION, file_sha256, image_frame,
                                              prepare_image)


class FakePredictor:
    identity = {'model': MODEL_NAME, 'weights_sha256': DOCUMENTED_WEIGHTS_SHA256,
                'preprocessing_version': PREPROCESSING_VERSION}

    def predict(self, image_path, sample_key, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        bounds = {'Right_Lung': (40, 40, 210, 470),
                  'Left_Lung': (300, 40, 470, 470),
                  'Heart': (190, 200, 330, 440)}
        for name, (x0, y0, x1, y1) in bounds.items():
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            Image.fromarray(mask).save(output_dir / (name + '.png'))
        record = {'binding': {'sample_key': sample_key, 'preprocessing': image_frame(image_path),
                  'image_sha256': file_sha256(image_path), 'producer': self.identity},
                  'mask_sha256': {name + '.png': file_sha256(output_dir / (name + '.png'))
                                  for name in bounds}}
        (output_dir / 'cache_record.json').write_text(json.dumps(record))
        return record


def test_batch_keeps_failures_and_continues(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('L', (64, 64), color=90).save(image)
    rows = [{'sample_key': name, 'dataset': 'synthetic', 'image_path': image.name,
             'sha256': digest} for name, digest in [
                 ('first', file_sha256(image)), ('bad', '0' * 64),
                 ('last', file_sha256(image))]]
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = run_batch(manifest, tmp_path, tmp_path / 'results', FakePredictor())
    predictions = [json.loads(line) for line in
                   (tmp_path / 'results/predictions.jsonl').read_text().splitlines()]
    assert [row['status'] for row in predictions] == ['ok', 'error', 'ok']
    assert report['statuses'] == {'ok': 2, 'error': 1}
    assert report['clinical_accuracy'] is None
    assert report['ctr_teacher_completed'] == 0
    assert 'checksum' in predictions[1]['error']
    assert predictions[2]['ctr_teacher_status'] == 'refused_missing_independent_landmarks'


def test_palette_is_resolved_before_intensity_normalization(tmp_path):
    image = Image.new('P', (2, 2))
    palette = [0] * 768
    palette[:6] = [255, 255, 255, 0, 0, 0]
    image.putpalette(palette)
    image.putdata([0, 1, 0, 1])
    path = tmp_path / 'palette.png'
    image.save(path)
    pixels, record = prepare_image(path)
    # official pipeline output is 512x512; palette index 0 (white) -> positive, index 1 (black) -> negative
    assert pixels.shape == (1, 1, 512, 512)
    assert pixels[0, 0, 256, 100] > 500 and pixels[0, 0, 256, 400] < -500   # bilinear blend between columns
    assert record['source_mode'] == 'P'
