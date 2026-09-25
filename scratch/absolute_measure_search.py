"""Search for a distance measure that is ABSOLUTE across categories.
The earlier search (D36) ranked measures by within-trial prediction — a relative criterion.
Here the criterion is the one the reviewers' question implies: does one function map distance
to margin everywhere? Fit on eleven categories, predict the twelfth.
Crucially, most variants here are NOT normalised: z-scoring per dimension and L2-normalising
rows makes cosine scale-free, which throws away exactly the magnitude an absolute measure needs.
Encoder-free (voxels, bounding box) and encoder-based (frozen DINOv2) variants are kept apart."""
from _repo import G, K
import json, ast, numpy as np, pandas as pd
from scipy import stats
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster'].reset_index(drop=True)
T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv')
T['objs']=[sorted({f'{SYN[c]}/'+n[len(c)+1:-8] for n in ast.literal_eval(s)}) for c,s in zip(T.dataset,T.images)]
D=json.load(open(f'{K}/design_clusters_all.json')); models=sorted(L.model.unique())
train={m:[f'{SYN[m.split("_")[0]]}/{o}' for o in D[m.split('_')[0]]['split'][m.split('_c')[1][0]]['train'][:25]] for m in models}
need=sorted(set(o for l in T.objs for o in l)|set(o for l in train.values() for o in l)); nix={o:i for i,o in enumerate(need)}
def load_voxel():
    z=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); d=dict(zip(z['ids'],z['X'].astype(np.float32)))
    return np.stack([d[o] for o in need])
def load_bbox():
    z=np.load(f'{G}/bank/bank_bbox.npz',allow_pickle=True); d=dict(zip(z['ids'],z['X'].astype(np.float32)))
    return np.stack([d[o] for o in need])
def load_dino():
    z=np.load(f'{K}/eval/encoder_features/pretrained.npz',allow_pickle=True); acc={}
    for i,x in zip(z['train_ids'],z['train_X']):
        k=f'{SYN[i.split("/")[0]]}/{i.split("/")[1]}'
        if k in nix: acc.setdefault(k,[]).append(x)
    return np.stack([np.mean(acc[o],axis=0) for o in need])
RAW={'voxel16':load_voxel(),'bbox':load_bbox(),'dinov2':load_dino()}
print({k:v.shape for k,v in RAW.items()})
def zrows(X): mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1; return (X-mu)/sd
def l2n(X): return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
VARIANTS={}   # name -> (matrix, metric)  metric in {'cos','l2','l1','iou'}
VARIANTS['voxel16 · cosine, z-scored + unit rows (used so far)']=(l2n(zrows(RAW['voxel16'])),'cos')
VARIANTS['voxel16 · RAW Euclidean (keeps magnitude)']=(RAW['voxel16'],'l2')
VARIANTS['voxel16 · 1 − IoU (overlap of the two shapes)']=(RAW['voxel16'],'iou')
VARIANTS['voxel16 · cosine on RAW occupancy (no z-scoring)']=(l2n(RAW['voxel16']),'cos')
VARIANTS['bbox · RAW Euclidean (physical proportions)']=(RAW['bbox'],'l2')
VARIANTS['bbox · cosine, z-scored + unit rows']=(l2n(zrows(RAW['bbox'])),'cos')
VARIANTS['dinov2 · cosine, z-scored + unit rows (search winner)']=(l2n(zrows(RAW['dinov2'])),'cos')
VARIANTS['dinov2 · RAW Euclidean (keeps feature magnitude)']=(RAW['dinov2'],'l2')
VARIANTS['dinov2 · cosine on RAW features']=(l2n(RAW['dinov2']),'cos')
def dists(X,metric,A,B):
    if metric=='cos': return 1-A@B.T
    if metric=='l2':  return np.sqrt(np.maximum(((A**2).sum(1)[:,None]+(B**2).sum(1)[None,:]-2*A@B.T),0))
    if metric=='l1':  return np.abs(A[:,None,:]-B[None,:,:]).sum(-1)
    if metric=='iou':
        inter=A@B.T; union=A.sum(1)[:,None]+B.sum(1)[None,:]-inter; return 1-inter/np.maximum(union,1e-9)
rows_idx=np.array([[nix[o] for o in l] for l in T.objs])
mi={m:i for i,m in enumerate(models)}; Lm=L.model.map(mi).values; Lt=L.trial.values
def r2(y,p): return 1-np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
out=[]
for name,(X,metric) in VARIANTS.items():
    dmat=np.full((len(models),len(T)),np.nan)
    for m in models:
        S=X[[nix[o] for o in train[m]]]
        near=np.sort(dists(X,metric,X,S),axis=1)[:,0]
        dmat[mi[m]]=near[rows_idx].mean(1)
    x=dmat[Lm,Lt]; y=L.ft.values; ok=np.isfinite(x)
    x,y2=x[ok],y[ok]; cat=L.test_cat.values[ok]; tr=Lt[ok]
    b=np.polyfit(x,y2,1); pooled=r2(y2,np.polyval(b,x))
    loo=[]
    for c in np.unique(cat):
        m1=cat!=c; bb=np.polyfit(x[m1],y2[m1],1); loo.append(r2(y2[~m1],np.polyval(bb,x[~m1])))
    dfx=pd.DataFrame({'x':x,'y':y2,'t':tr}); g=dfx.groupby('t'); wr=stats.pearsonr(dfx.x-g.x.transform('mean'),dfx.y-g.y.transform('mean'))[0]
    # spread across categories at a matched distance (middle tertile of this measure's own range)
    q1,q2=np.quantile(x,[0.4,0.6]); band=pd.DataFrame({'c':cat,'y':y2})[(x>q1)&(x<q2)]
    spread=band.groupby('c').y.mean().max()-band.groupby('c').y.mean().min()
    swing=abs(b[0])*(np.quantile(x,0.95)-np.quantile(x,0.05))
    out.append((name,pooled,np.mean(loo),min(loo),wr,spread,swing,spread/max(swing,1e-9)))
    print(f'{name:58s} pooled R² {pooled:+.3f} | LOO-category R² {np.mean(loo):+.3f} (worst {min(loo):+.3f}) | within-trial r {wr:+.3f} | category spread {spread:.3f} vs distance swing {swing:.3f}',flush=True)
R=pd.DataFrame(out,columns=['measure','pooled_R2','loo_R2','loo_worst','within_trial_r','cat_spread','distance_swing','spread_over_swing'])
R.to_csv(f'{G}/data/absolute_measure_search.csv',index=False)
print('\nBest by the ABSOLUTE criterion (leave-one-category-out R²):'); print(R.sort_values('loo_R2',ascending=False)[['measure','loo_R2','spread_over_swing','within_trial_r']].round(3).to_string(index=False))
