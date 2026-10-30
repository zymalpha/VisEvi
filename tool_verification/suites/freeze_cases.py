"""Freeze verification sample lists (run once; refuses to overwrite)."""
import hashlib, json, random
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'original_data'
OUT = ROOT / 'tool_verification/cases'
SEED = 20261030


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def entry(source, rel):
    p = DATA / rel
    with Image.open(p) as im:
        return {'source': source, 'image_path': rel, 'sha256': sha(p), 'size_wh': list(im.size), 'mode': im.mode}


def pick(source, rels, n, forced=()):
    rng = random.Random(f'{SEED}:{source}')
    rest = [r for r in sorted(rels) if r not in forced]
    chosen = list(forced) + rng.sample(rest, n - len(forced))
    return [entry(source, r) for r in chosen]


def main():
    if any(OUT.glob('*.json')):
        raise SystemExit('cases already frozen')
    m = DATA / '_manifests'
    cxd = [r['image_path'] for r in rows(m / '20261026/chestx-det-source-test-excluding-known-train-duplicates.jsonl')]
    nih = [r['image_path'] for r in rows(m / '20261026/nih-development-excluding-known-chestxdet-duplicates.jsonl')]
    jsrt = [r['image_path'] for r in rows(m / '20261026/jsrt-png-conversion.jsonl')]
    mont = [r['image_path'] for r in rows(m / '20261028/montgomery-lungs.jsonl')]
    def modes(rels, mode):
        out = []
        for r in sorted(rels):
            with Image.open(DATA / r) as im:
                if im.mode == mode:
                    out.append(r)
        return out
    cxd_rgba, nih_rgba = modes(cxd, 'RGBA')[:2], modes(nih, 'RGBA')[:1]
    core = (pick('chestx-det', cxd, 10, cxd_rgba) + pick('nih', nih, 10, nih_rgba)
            + pick('jsrt', jsrt, 10) + pick('montgomery', mont, 10))
    OUT.joinpath('core40.json').write_text(json.dumps({'seed': SEED, 'purpose': 'V3 differential / V4 metamorphic / cross-tool', 'cases': core}, indent=1))
    nih_rows = rows(m / '20261026/nih-development-excluding-known-chestxdet-duplicates.jsonl')
    OUT.joinpath('nih_view292.json').write_text(json.dumps({'purpose': 'V5 A24 (CheXOne training data; implementation check only)',
        'cases': [{'image_path': r['image_path'], 'view_position': r['view_position']} for r in nih_rows]}, indent=1))
    OUT.joinpath('montgomery138.json').write_text(json.dumps({'purpose': 'V5 A1/A5 lungs; leftMask=image-left=patient right',
        'cases': [{'image_path': r['image_path'], 'image_sha256': r['image_sha256']} for r in rows(m / '20261028/montgomery-lungs.jsonl')]}, indent=1))
    scr = rows(m / '20261028/jsrt-scr-mirror-candidates.jsonl')
    OUT.joinpath('jsrt_scr247.json').write_text(json.dumps({'purpose': 'V5 A1/A2/A4 (third-party mirror; implementation check only)',
        'cases': [{'image_path': r['image_path'], 'image_sha256': r['image_sha256'], 'masks': {k: v['path'] for k, v in r['masks'].items()}} for r in scr]}, indent=1))
    print({p.name: len(json.loads(p.read_text())['cases']) for p in OUT.glob('*.json')})
    print([(c['source'], c['mode'], tuple(c['size_wh'])) for c in core])


if __name__ == '__main__':
    main()
