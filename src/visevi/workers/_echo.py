"""Protocol test worker: echoes parameters and writes one artifact (tests only)."""
from __future__ import annotations

from pathlib import Path
import os

from ._runtime import serve


def load(weights_dir, device):
    return None, {'device': device, 'weights_dir': weights_dir}


def handle(state, request):
    params = request['params']
    if params.get('crash'):
        os._exit(3)
    if params.get('fail'):
        raise ValueError('requested failure')
    (Path(request['output_dir']) / 'echo.txt').write_text(str(params.get('value')))
    return {'echo': params, 'image_path': request['image_path']}, ['echo.txt']


if __name__ == '__main__':
    serve(load, handle)
