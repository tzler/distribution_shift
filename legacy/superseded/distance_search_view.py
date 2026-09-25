"""View-specific geometric distance (lead's request): each test image is a bank render at a known view, so its
depth+silhouette descriptor at THAT view is in bank_viewdepth. Distance from the trial to a model's training set =
per test image, distance to the model's training IMAGES (25 objects × 15 views); variants: all views / same view only.
Scored exactly as distance_search.py (within-trial r, fit on odd trials, reported on even)."""
import json, ast, numpy as np, pandas as pd
from scipy import stats
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
L=pd.read_csv(f'{G}/out/all_categories_long.csv' if __import__('os').path.exists(f'{G}/out/all_categories_long.csv') else f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster'].copy()
T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv')
T['imgs']=[[f'{SYN[c]}/'+n[len(c)+1:-8]+'/'+n[-7:-4] for n in ast.literal_eval(s)] for c,s in zip(T.dataset,T.images)]   # synset/id/view
D=json.load(open(f'{K}/design_clusters_all.json')); models=sorted(L.model.unique())
train={m:[f'{SYN[m.split("_")[0]]}/{o}' for o in D[m.split('_')[0]]['split'][m.split('_c')[1][0]]['train'][:25]] for m in models}
need_obj=set(i.rsplit('/',1)[0] for l in T.imgs for i in l)|set(o for l in train.values() for o in l)
z=np.load(f'{G}/bank/bank_viewdepth.npz',allow_pickle=True); ids=z['ids']; keep=np.array([i.rsplit('/',1)[0] in need_obj for i in ids])
X=z['X'][keep].astype(np.float32); ids=ids[keep]; mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1; X=(X-mu)/sd; X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-12
ix={i:k for k,i in enumerate(ids)}; print('view descriptors kept:',len(ids),flush=True)
def img_rows(obj): return [ix[f'{obj}/{v:03d}'] for v in range(15) if f'{obj}/{v:03d}' in ix]
S={m:np.stack([X[r] for o in train[m] for r in img_rows(o)]) for m in models}; Sview={m:np.array([int(ids[r][-3:]) for o in train[m] for r in img_rows(o)]) for m in models}
odd=T.index.values%2==1; rows=[]
VARIANTS={'view_nn1_allviews':lambda Dm,v:np.sort(Dm)[0],'view_knn5_allviews':lambda Dm,v:np.sort(Dm)[:5].mean(),'view_knn10_allviews':lambda Dm,v:np.sort(Dm)[:10].mean(),
          'view_nn1_sameview':lambda Dm,v:np.sort(Dm[v])[0] if v.any() else np.nan,'view_knn5_sameview':lambda Dm,v:np.sort(Dm[v])[:5].mean() if v.sum()>=5 else np.nan}
tests=[[ix.get(i) for i in l] for l in T.imgs]; tviews=[[int(i[-3:]) for i in l] for l in T.imgs]
for name,f in VARIANTS.items():
    dist=np.full((len(models),len(T)),np.nan)
    for mi,m in enumerate(models):
        Sm=S[m]; sv=Sview[m]
        for ti,(rs,vs) in enumerate(zip(tests,tviews)):
            if any(r is None for r in rs): continue
            vals=[]
            for r,v in zip(rs,vs):
                Dm=1-X[r]@Sm.T; vals.append(f(Dm,sv==v))
            dist[mi,ti]=np.nanmean(vals)
    Lm=L.copy(); Lm['x']=[dist[models.index(m),t] for m,t in zip(Lm.model,Lm.trial)]; Lm=Lm.dropna(subset=['x'])
    def score(d,half):
        d=d[np.isin(d.trial,np.where(half)[0])]; g=d.groupby('trial'); xc=d.x-g.x.transform('mean'); yc=d.ft-g.ft.transform('mean'); return stats.pearsonr(xc,yc)[0]
    own=Lm[Lm.train_cat==Lm.test_cat]; oth=Lm[Lm.train_cat!=Lm.test_cat]
    rows.append(('viewdepth_image',name,0,score(Lm,odd),score(Lm,~odd),score(own,odd),score(own,~odd),score(oth,odd),score(oth,~odd)))
    print(f'{name:22s} all {rows[-1][3]:+.3f}/{rows[-1][4]:+.3f}  own-category {rows[-1][5]:+.3f}/{rows[-1][6]:+.3f}  other-category {rows[-1][7]:+.3f}/{rows[-1][8]:+.3f}   (fit/held-out)',flush=True)
pd.DataFrame(rows,columns=['descriptor','estimator','missing','all_fit','all_test','own_fit','own_test','oth_fit','oth_test']).to_csv(f'{G}/out/distance_search_view.csv',index=False); print('done')
