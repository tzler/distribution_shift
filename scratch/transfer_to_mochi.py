"""Transfer test (lead, 2026-09-21): the search's best measures, applied to images the models never trained on AND from a
different render pipeline — the MOCHI trials (706 single-category ShapeNet trials), scored by the 34 cluster models.
Same within-trial score: does distance to each model's 25 training objects predict that model's margin on the trial?"""
import json, ast, numpy as np, pandas as pd
from scipy import stats
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}; INV={v:k for k,v in SYN.items()}
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet'].copy()
mm['objs']=[sorted({'/'.join(f[:-4].split('_')[:2]) for f in ast.literal_eval(s)}) for s in mm.images]; mm['imgs']=[ast.literal_eval(s) for s in mm.images]
mm['tcat']=[INV.get(o[0].split('/')[0]) for o in mm.objs]; mm=mm[np.array([len({o.split('/')[0] for o in l})==1 for l in mm.objs])&mm.tcat.notna().values]
D=json.load(open(f'{K}/design_clusters_all.json')); models=sorted(f'{c}_c{k}_n25' for c in SYN for k in range(3)); import os
models=[x for x in models if os.path.exists(f'{K}/eval/{x}/ood_analysis_results.csv')]
train={x:[f'{SYN[x.split("_")[0]]}/{o}' for o in D[x.split('_')[0]]['split'][x.split('_c')[1][0]]['train'][:25]] for x in models}
marg={x:pd.read_csv(f'{K}/eval/{x}/ood_analysis_results.csv') for x in models}; pre=marg[models[0]].pretrained_oddity_margin.values
def norm(X): mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd; return Xn/(np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12)
# features: object-level for the MOCHI test objects (test_*.npz) and the bank
feats={}
for nm in ['bbox','voxel16','structure']:
    b=np.load(f'{G}/bank/bank_{nm}.npz',allow_pickle=True); t=np.load(f'{G}/bank/test_{nm}.npz',allow_pickle=True)
    X=np.concatenate([b['X'],t['X']]).astype(np.float32); ids=np.concatenate([b['ids'],t['ids']]); Xn=norm(X); feats[nm]={i:x for i,x in zip(ids,Xn)}
z=np.load(f'{K}/eval/encoder_features/pretrained.npz',allow_pickle=True)
bank_obj={}; 
for i,x in zip(z['train_ids'],z['train_X']): bank_obj.setdefault(f'{SYN[i.split("/")[0]]}/{i.split("/")[1]}',[]).append(x)
bank_obj={k:np.mean(v,axis=0) for k,v in bank_obj.items()}
test_img={i:x for i,x in zip(z['test_ids'],z['test_X'])}      # MOCHI image-level DINOv2 features
allX=np.stack(list(bank_obj.values())+list(test_img.values())); Xn=norm(allX); keys=list(bank_obj)+list(test_img)
feats['dinov2_pretrained']={k:x for k,x in zip(keys,Xn)}
rows=[]
for nm,F in feats.items():
    for est in ['nn1','knn5','knn10']:
        k={'nn1':1,'knn5':5,'knn10':10}[est]; recs=[]
        for x in models:
            S=np.stack([F[o] for o in train[x] if o in F]); mo=marg[x].fine_tuned_oddity_margin.values; cat=x.split('_')[0]
            for ti,r in mm.iterrows():
                if nm=='dinov2_pretrained': q=[F[f] for f in r.imgs if f in F]          # image-level for MOCHI images
                else: q=[F[o] for o in r.objs if o in F]
                if not q: continue
                Dm=1-np.stack(q)@S.T; d=np.sort(Dm,axis=1)[:,:k].mean(1).mean(); recs.append((ti,x,cat==r.tcat,d,mo[ti]))
        L=pd.DataFrame(recs,columns=['trial','model','own','x','ft']); g=L.groupby('trial'); xc=L.x-g.x.transform('mean'); yc=L.ft-g.ft.transform('mean')
        def sc(mask): 
            d=L[mask]; gg=d.groupby('trial'); a=d.x-gg.x.transform('mean'); b=d.ft-gg.ft.transform('mean'); return stats.pearsonr(a,b)[0]
        rows.append((nm,est,len(L),sc(np.ones(len(L),bool)),sc(L.own.values),sc(~L.own.values)))
        print(f'{nm:18s} {est:6s} n={len(L):6d}   all {rows[-1][3]:+.3f}   own-category {rows[-1][4]:+.3f}   other-category {rows[-1][5]:+.3f}',flush=True)
pd.DataFrame(rows,columns=['descriptor','estimator','n','all','own','other']).to_csv(f'{G}/out/transfer_mochi.csv',index=False); print('done')
