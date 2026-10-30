"""Worker loop shared by all model backends (runs inside .venv-models).

stdout carries exactly one JSON object per line: first a ready message, then
one response per request. Anything libraries print is redirected to stderr.
A handler receives ``(image_path, output_dir, params)`` and returns
``(result_dict, [artifact file names written into output_dir])``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('non-finite number in worker result')
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def serve(load, handle) -> None:
    """``load(weights_dir, device) -> (state, runtime_info)``; ``handle(state, request)``."""
    protocol = sys.stdout
    sys.stdout = sys.stderr
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()

    def send(message):
        protocol.write(json.dumps(message, allow_nan=False) + '\n')
        protocol.flush()

    try:
        state, runtime = load(args.weights_dir, args.device)
    except Exception:
        send({'ready': False, 'error': traceback.format_exc()[-4000:]})
        return
    send({'ready': True, 'runtime': runtime})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            result, artifacts = handle(state, request)
            send({'ok': True, 'result': _finite(result), 'artifacts': list(artifacts)})
        except Exception as exc:
            traceback.print_exc()
            send({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})


def device_runtime(device: str) -> dict:
    import torch
    info = {'torch': torch.__version__, 'device': device, 'deterministic_algorithms': False}
    if device.startswith('cuda'):
        info['gpu'] = torch.cuda.get_device_name(torch.device(device))
    return info
