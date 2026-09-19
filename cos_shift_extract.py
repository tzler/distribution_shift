"""
Extract DINOv2 ViT-L/14 reg4 CLS features (224px, after the final norm) for
  * every MOCHI test image
  * the HIDA training reference bank: ShapeNet renders (12 categories) + ShapeGen,
    white background, with MOCHI's own test objects EXCLUDED

Two arms:
  --ckpt ''         NEUTRAL RULER : the fixed pretrained encoder
  --ckpt <path>     OWN SPACE     : the evaluated fine-tuned model

Reuses build()/feats() from ../L1norm_vs_distshift/ft/extract_ft.py, which already
rebuilds the encoder the way hida-tune does (peft LoRA on ["qkv","proj"], with
assertions that the LoRA tensors actually mapped).
"""
import argparse, os, sys, random
import numpy as np
import pandas as pd

NAV = '/vast/projects/bonnen/naturalistic-navig'
sys.path.insert(0, f'{NAV}/Dist-shift-data/L1norm_vs_distshift/ft')
from extract_ft import build, feats                                   # noqa: E402

MOCHI_IMG = f'{NAV}/MOCHI/images'
SHAPENET = f'{NAV}/Dist-shift-data/shapenet_rendered/white'
SHAPEGEN = f'{NAV}/Dist-shift-data/shapegen/images_white_bg'
MOCHI_IDS = f'{NAV}/Dist-shift-data/mochi/mochi_shapenet_object_ids.txt'


def manifest(n_obj, n_view, n_sg_obj, seed=0):
    rng = random.Random(seed)
    rows = []

    # ---- MOCHI test images ------------------------------------------------
    for f in sorted(os.listdir(MOCHI_IMG)):
        if f.endswith('.png'):
            rows.append((f'{MOCHI_IMG}/{f}', 'mochi', 'mochi', f))

    # ---- HIDA bank: ShapeNet, MOCHI objects excluded ----------------------
    mochi_obj = {l.strip().split('/')[1] for l in open(MOCHI_IDS) if l.strip()}
    n_excl = 0
    for cat in sorted(os.listdir(SHAPENET)):
        d = f'{SHAPENET}/{cat}'
        if not os.path.isdir(d):
            continue
        objs = [o for o in sorted(os.listdir(d)) if o not in mochi_obj]
        n_excl += len(os.listdir(d)) - len(objs)
        for o in rng.sample(objs, min(n_obj, len(objs))):
            ims = sorted(x for x in os.listdir(f'{d}/{o}') if x.endswith('.png'))
            step = max(1, len(ims) // n_view)
            for im in ims[::step][:n_view]:
                rows.append((f'{d}/{o}/{im}', 'bank_shapenet', cat, f'{cat}/{o}/{im}'))
    print(f'  excluded {n_excl} MOCHI objects from the ShapeNet bank', flush=True)

    # ---- HIDA bank: ShapeGen ---------------------------------------------
    sg = []
    for ex in sorted(os.listdir(SHAPEGEN)):
        for sm in sorted(os.listdir(f'{SHAPEGEN}/{ex}')):
            p = f'{SHAPEGEN}/{ex}/{sm}'
            if os.path.isdir(p):
                sg += [f'{p}/{o}' for o in sorted(os.listdir(p))]
    for s in rng.sample(sg, min(n_sg_obj, len(sg))):
        if not os.path.isdir(s):
            continue
        ims = sorted(x for x in os.listdir(s) if x.endswith('.png'))
        step = max(1, len(ims) // n_view)
        for im in ims[::step][:n_view]:
            rows.append((f'{s}/{im}', 'bank_shapegen', 'shapegen',
                         os.path.relpath(f'{s}/{im}', SHAPEGEN)))
    return pd.DataFrame(rows, columns=['path', 'source', 'category', 'key'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--backbone', default='vit_large_patch14_reg4_dinov2')
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=8)
    ap.add_argument('--lora-dropout', type=float, default=0.1)
    ap.add_argument('--n-obj', type=int, default=300)
    ap.add_argument('--n-view', type=int, default=8)
    ap.add_argument('--n-sg-obj', type=int, default=1500)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--out', default=f'{NAV}/Dist-shift-data/geometric_shift/cosfeat')
    a = ap.parse_args()

    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = manifest(a.n_obj, a.n_view, a.n_sg_obj)
    print(f'{a.tag}: {len(idx)} images {idx.source.value_counts().to_dict()}', flush=True)

    vit, tf = build(a.backbone, a.ckpt, a.lora_r, a.lora_alpha, a.lora_dropout, dev)
    X = feats(vit, tf, idx.path.tolist(), dev, bs=a.bs)
    os.makedirs(f'{a.out}/{a.tag}', exist_ok=True)
    np.save(f'{a.out}/{a.tag}/features.npy', X.astype(np.float32))
    idx.to_csv(f'{a.out}/{a.tag}/index.csv', index=False)
    print(f'wrote {a.out}/{a.tag}/features.npy  {X.shape}', flush=True)
