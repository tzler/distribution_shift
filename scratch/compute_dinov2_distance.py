"""Save the frozen-network distance for every (held-out trial x cluster model) pair, so the
absolute-vs-relative tests use the measure the search and the transfer test both picked."""
from _repo import G, K
import json, ast, numpy as np, pandas as pd
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv')
T['objs']=[sorted({f'{SYN[c]}/'+n[len(c)+1:-8] for n in ast.literal_eval(s)}) for c,s in zip(T.dataset,T.images)]
D=json.load(open(f'{K}/design_clusters_all.json')); models=sorted(L.model.unique())
train={m:[f'{SYN[m.split("_")[0]]}/{o}' for o in D[m.split('_')[0]]['split'][m.split('_c')[1][0]]['train'][:25]] for m in models}
need=set(o for l in T.objs for o in l)|set(o for l in train.values() for o in l)
z=np.load(f'{K}/eval/encoder_features/pretrained.npz',allow_pickle=True); acc={}
for i,x in zip(z['train_ids'],z['train_X']):
    k=f'{SYN[i.split("/")[0]]}/{i.split("/")[1]}'
    if k in need: acc.setdefault(k,[]).append(x)
ks=list(acc); X=np.stack([np.mean(acc[k],axis=0) for k in ks]); mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1
X=(X-mu)/sd; X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-12; ix={k:i for i,k in enumerate(ks)}
rows=np.array([[ix[o] for o in l] for l in T.objs])          # (nT, 2)
out=np.full((len(models),len(T)),np.nan)
for mi,m in enumerate(models):
    S=X[[ix[o] for o in train[m] if o in ix]]
    Dm=1-X[rows.ravel()]@S.T                                   # (2nT, 25)
    out[mi]=np.sort(Dm,axis=1)[:,0].reshape(len(T),2).mean(1)  # nearest, averaged over the trial's objects
d=pd.DataFrame({'model':np.repeat(models,len(T)),'trial':np.tile(np.arange(len(T)),len(models)),'dinov2':out.ravel()})
d.to_csv(f'{G}/data/dinov2_distance_long.csv.gz',index=False,compression={'method':'gzip','compresslevel':9})
print(f'{len(models)} models x {len(T)} trials -> data/dinov2_distance_long.csv.gz ; range {np.nanmin(out):.3f}-{np.nanmax(out):.3f}')
