"""Search over distance measures, scored against the round-3 margins (34 models × 11,634 trials).
For each descriptor × estimator: per trial × model, the distance from the trial's objects to the model's 25 training
objects; then the within-trial correlation with the margin, fitted on odd trials and REPORTED on even trials.
Three scores: all 34 models (the category step dominates); the trial's own-category models only (graded, within);
other-category models only (does the measure order far training sets at all?)."""
import json, ast, os, numpy as np, pandas as pd
from scipy import stats
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}; INV={v:k for k,v in SYN.items()}
L=pd.read_csv(f'{G}/out/all_categories_long.csv'); L=L[L.kind=='cluster'].copy()
T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv'); T['objs']=[sorted({f'{SYN[c]}/'+n[len(c)+1:-8] for n in ast.literal_eval(s)}) for c,s in zip(T.dataset,T.images)]
D=json.load(open(f'{K}/design_clusters_all.json')); models=sorted(L.model.unique())
train={m:[f'{SYN[m.split("_")[0]]}/{o}' for o in D[m.split('_')[0]]['split'][m.split('_c')[1][0]]['train'][:25]] for m in models}
need=set(o for l in T.objs for o in l)|set(o for l in train.values() for o in l)
# ---- descriptors, object level, keyed by 'synset/id'
def objlevel(ids,X,key):
    d={}
    for i,k in zip(ids,X): d.setdefault(key(i),[]).append(k)
    return {k:np.mean(v,axis=0) for k,v in d.items()}
banks={}
for nm in ['voxel8','voxel16','voxel32','bbox','structure','volatility','multiview']:
    z=np.load(f'{G}/bank/bank_{nm}.npz',allow_pickle=True); banks[nm]=dict(zip(z['ids'],z['X'].astype(np.float32)))
z=np.load(f'{G}/bank/bank3d_shapenet.npz',allow_pickle=True); banks['d57']=dict(zip(z['ids'],z['X'].astype(np.float32)))
z=np.load(f'{G}/bank/bank_viewdepth.npz',allow_pickle=True); ids=z['ids']; keep=np.array([i.rsplit('/',1)[0] in need for i in ids]); banks['viewdepth']=objlevel(ids[keep],z['X'][keep].astype(np.float32),lambda i:i.rsplit('/',1)[0])
z=np.load(f'{K}/eval/encoder_features/pretrained.npz',allow_pickle=True); ids=z['train_ids']; key=lambda i:f'{SYN[i.split("/")[0]]}/{i.split("/")[1]}'; keep=np.array([key(i) in need for i in ids]); banks['dinov2_pretrained']=objlevel(ids[keep],z['train_X'][keep].astype(np.float32),key)
print('descriptors:',{k:(len(v),next(iter(v.values())).shape[0]) for k,v in banks.items()})
# ---- normalisation: z-score with the WHOLE bank of that descriptor (as before), L2-normalise, cosine
def prep(bank):
    ks=list(bank); X=np.stack([bank[k] for k in ks]); mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd; Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12; return {k:x for k,x in zip(ks,Xn)}
ESTS={'nn1':lambda Dm:np.sort(Dm,axis=1)[:,0],'knn5':lambda Dm:np.sort(Dm,axis=1)[:,:5].mean(1),'knn10':lambda Dm:np.sort(Dm,axis=1)[:,:10].mean(1),'mean25':lambda Dm:Dm.mean(1)}
rows=[]; Lidx=L.set_index(['model','trial'])
odd=T.index.values%2==1
for nm,bank in banks.items():
    B=prep(bank); miss=sum(o not in B for o in need)
    Tt=[np.stack([B[o] for o in l]) if all(o in B for o in l) else None for l in T.objs]
    for est,f in ESTS.items():
        dist=np.full((len(models),len(T)),np.nan)
        for mi,m in enumerate(models):
            S=np.stack([B[o] for o in train[m] if o in B])
            for ti,t in enumerate(Tt):
                if t is None: continue
                dist[mi,ti]=f(1-t@S.T).mean()
        # assemble aligned with L
        Lm=L.copy(); Lm['x']=[dist[models.index(m),t] for m,t in zip(Lm.model,Lm.trial)]; Lm=Lm.dropna(subset=['x'])
        def score(d,half):
            d=d[np.isin(d.trial,np.where(half)[0])]; g=d.groupby('trial'); xc=d.x-g.x.transform('mean'); yc=d.ft-g.ft.transform('mean')
            return stats.pearsonr(xc,yc)[0] if xc.std()>0 else np.nan
        own=Lm[Lm.train_cat==Lm.test_cat]; oth=Lm[Lm.train_cat!=Lm.test_cat]
        rows.append((nm,est,miss,score(Lm,odd),score(Lm,~odd),score(own,odd),score(own,~odd),score(oth,odd),score(oth,~odd)))
        print(f'{nm:18s} {est:6s} all {rows[-1][3]:+.3f}/{rows[-1][4]:+.3f}  own-category {rows[-1][5]:+.3f}/{rows[-1][6]:+.3f}  other-category {rows[-1][7]:+.3f}/{rows[-1][8]:+.3f}   (fit/held-out)',flush=True)
R=pd.DataFrame(rows,columns=['descriptor','estimator','missing','all_fit','all_test','own_fit','own_test','oth_fit','oth_test']); R.to_csv(f'{G}/out/distance_search.csv',index=False)
print('\nheld-out, sorted by the own-category (within) score:'); print(R.sort_values('own_test')[['descriptor','estimator','all_test','own_test','oth_test']].round(3).to_string(index=False))
