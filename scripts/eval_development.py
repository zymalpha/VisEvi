"""Run: PYTHONPATH=src .venv/bin/python scripts/eval_development.py --help"""
import argparse
import json
from pathlib import Path

from visevi.eval.batch import run_batch
from visevi.perception.anatomy.pspnet import PSPNetPredictor


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=root / 'original_data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, default=root / 'tools/weights')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    predictor = PSPNetPredictor(args.weights_dir, args.device)
    print(json.dumps(run_batch(args.manifest, args.data_root, args.out, predictor),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
