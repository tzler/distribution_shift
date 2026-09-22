"""Track A: per-image features of the training renders (white bg) and MOCHI ShapeNet images
under the pretrained backbone and three fine-tuned category models, using the evaluation
script's own loaders so features match how the margins were computed."""
import os, sys, ast, glob, argparse, numpy as np, pandas as pd, torch
H='/vast/projects/bonnen/naturalistic-navig/Dist-shift/HIDA/hida-tune'; sys.path.insert(0,H); sys.path.insert(0,f'{H}/evaluation')
from ood_distance_analysis import get_pretrained_model, load_checkpoint_model, get_transform, extract_features_batch
NAV='/vast/projects/bonnen/naturalistic-navig'; R=f'{NAV}/Dist-shift-data/shapenet_rendered/white'; MOCHI=f'{NAV}/MOCHI'
LOG=f'{NAV}/Dist-shift/logs'; OUT=f'{NAV}/Dist-shift-data/knockout/eval/encoder_features'; os.makedirs(OUT,exist_ok=True)
BB='vit_large_patch14_reg4_dinov2'
def ckpt(tag):
    d=glob.glob(f'{LOG}/vit_large_patch14_reg4_dinov2_bs32x1_lr1e-06_ep30_multi_similarity_seed42_train:_val:_lora_r16_alpha8_dropout0.1_|{tag}|*')[0]
    return sorted(glob.glob(f'{d}/checkpoints/*.pth'),key=os.path.getmtime)[-1]
MODELS={'pretrained':None,'ft_chair':ckpt(60),'ft_airplane':ckpt(57),'ft_table':ckpt(74)}
ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',default=list(MODELS)); ap.add_argument('--bs',type=int,default=256); a=ap.parse_args()
dev='cuda'; tf=get_transform(224)
# image lists
train=[]; 
for cat in sorted(os.listdir(R)):
    for obj in sorted(os.listdir(f'{R}/{cat}')):
        for f in sorted(os.listdir(f'{R}/{cat}/{obj}')):
            if f.endswith('.png'): train.append((f'{cat}/{obj}/{f[:-4]}',f'{R}/{cat}/{obj}/{f}'))
m=pd.read_csv(f'{MOCHI}/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
test=sorted({f for _,r in mm.iterrows() for f in ast.literal_eval(r['images'])}); test=[(f,f'{MOCHI}/images/{f}') for f in test]
print(f'train images {len(train)}  test images {len(test)}',flush=True)
for name in a.models:
    out=f'{OUT}/{name}.npz'
    if os.path.exists(out): print('skip',name); continue
    model=get_pretrained_model(BB,dev) if MODELS[name] is None else load_checkpoint_model(MODELS[name],BB,dev,lora_r=16,lora_alpha=8,lora_dropout=0.1)
    model.eval(); print(f'[{name}] loaded {MODELS[name]}',flush=True)
    feats={}
    for split,items in [('test',test),('train',train)]:
        X=[]; paths=[p for _,p in items]
        for i in range(0,len(paths),a.bs):
            X.append(extract_features_batch(model,paths[i:i+a.bs],tf,dev,batch_size=a.bs).float().cpu().numpy().astype(np.float16))
            if (i//a.bs)%100==0: print(f'   {name} {split} {i}/{len(paths)}',flush=True)
        feats[split]=(np.array([k for k,_ in items]),np.concatenate(X))
    np.savez(out,train_ids=feats['train'][0],train_X=feats['train'][1],test_ids=feats['test'][0],test_X=feats['test'][1])
    print(f'[{name}] saved {out}  train {feats["train"][1].shape} test {feats["test"][1].shape}',flush=True)
    del model; torch.cuda.empty_cache()
