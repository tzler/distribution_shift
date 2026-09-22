"""Vectorised search over distance measures against the round-3 margins (34 models × 11,634 trials).
Features: 9 object-level descriptors + DINOv2 pretrained (object mean) + view-specific depth maps (image level).
Comparisons: nearest / 5-nearest / 10-nearest / mean over all / centroid / coverage at three radii; for the
view-specific features also same-view-only. Score: within-trial r, fit on odd trials, reported on even trials,
for all models / own-category models / other-category models."""
import json, ast, numpy as np, pandas as pd
from scipy import stats
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
L=pd.read_csv(f'{G}/out/all_categories_long.csv' if __import__('os').path.exists(f'{G}/out/all_categories_long.csv') else f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster'].copy(); models=sorted(L.model.unique()); mix={m:i for i,m in enumerate(models)}
T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv'); nT=len(T)
T['imgs']=[[f'{SYN[c]}/'+n[len(c)+1:-8]+'/'+n[-7:-4] for n in ast.literal_eval(s)] for c,s in zip(T.dataset,T.images)]
T['objs']=T.imgs.apply(lambda l: sorted({i.rsplit('/',1)[0] for i in l}))
D=json.load(open(f'{K}/design_clusters_all.json')); train={m:[f'{SYN[m.split("_")[0]]}/{o}' for o in D[m.split('_')[0]]['split'][m.split('_c')[1][0]]['train'][:25]] for m in models}
need=set(o for l in T.objs for o in l)|set(o for l in train.values() for o in l)
# index arrays: trial -> its unique objects (2), trial -> its images (3)
uobj=sorted(need); oix={o:i for i,o in enumerate(uobj)}
tri_obj=np.array([[oix[l[0]],oix[l[-1]]] for l in T.objs])          # (nT,2)
Lm=L[['model','trial','ft','train_cat','test_cat']].copy(); Lm['mi']=Lm.model.map(mix); odd=(Lm.trial.values%2==1)
own=(Lm.train_cat==Lm.test_cat).values
def score(x):
    d=Lm.assign(x=x); out=[]
    for mask in [np.ones(len(d),bool),own,~own]:
        for half in [odd,~odd]:
            dd=d[mask&half]; g=dd.groupby('trial'); xc=dd.x-g.x.transform('mean'); yc=dd.ft-g.ft.transform('mean'); out.append(stats.pearsonr(xc,yc)[0])
    return out
def norm(X): mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd; return Xn/(np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12)
rows=[]
def run_object_level(name,bank):
    ks=[k for k in uobj if k in bank]; miss=len(uobj)-len(ks)
    Xall=norm(np.stack([bank[k] for k in bank]))                     # normalise on the whole bank of this descriptor
    bkeys={k:i for i,k in enumerate(bank)}; X=Xall[[bkeys[k] for k in ks]]; kix={k:i for i,k in enumerate(ks)}
    rng=np.random.default_rng(0); samp=Xall[rng.choice(len(Xall),min(3000,len(Xall)),replace=False)]; pdist=1-samp@samp.T; pdist=pdist[np.triu_indices(len(samp),1)]; eps={q:np.quantile(pdist,q) for q in [0.02,0.05,0.10]}
    per_obj={}   # estimator -> (nmodels, nobj)
    ests=['nn1','knn5','knn10','mean25','centroid','cov_q02','cov_q05','cov_q10']
    for e in ests: per_obj[e]=np.full((len(models),len(ks)),np.nan)
    for m in models:
        S=X[[kix[o] for o in train[m] if o in kix]]; Dm=1-X@S.T; srt=np.sort(Dm,axis=1); c=S.mean(0); c/=np.linalg.norm(c)+1e-12; i=mix[m]
        per_obj['nn1'][i]=srt[:,0]; per_obj['knn5'][i]=srt[:,:5].mean(1); per_obj['knn10'][i]=srt[:,:10].mean(1); per_obj['mean25'][i]=Dm.mean(1); per_obj['centroid'][i]=1-X@c
        for q,e in [(0.02,'cov_q02'),(0.05,'cov_q05'),(0.10,'cov_q10')]: per_obj[e][i]=-np.log1p((Dm<eps[q]).sum(1))
    to=np.array([[kix.get(uobj[a],-1),kix.get(uobj[b],-1)] for a,b in tri_obj])
    valid=(to>=0).all(1)
    for e in ests:
        P=per_obj[e]; x=np.full(len(Lm),np.nan); tt=Lm.trial.values; mi=Lm.mi.values; ok=valid[tt]
        x[ok]=(P[mi[ok],to[tt[ok],0]]+P[mi[ok],to[tt[ok],1]])/2
        sc=score(x[~np.isnan(x)]) if np.isnan(x).any() else score(x)
        if np.isnan(x).any():
            keep=~np.isnan(x); d=Lm[keep].assign(x=x[keep]); 
            def sc2(mask,half):
                dd=d[mask[keep]&half[keep]]; g=dd.groupby('trial'); xc=dd.x-g.x.transform('mean'); yc=dd.ft-g.ft.transform('mean'); return stats.pearsonr(xc,yc)[0]
            sc=[sc2(np.ones(len(Lm),bool),odd),sc2(np.ones(len(Lm),bool),~odd),sc2(own,odd),sc2(own,~odd),sc2(~own,odd),sc2(~own,~odd)]
        rows.append((name,e,miss,*sc)); print(f'{name:18s} {e:8s} all {sc[0]:+.3f}/{sc[1]:+.3f}  own {sc[2]:+.3f}/{sc[3]:+.3f}  other {sc[4]:+.3f}/{sc[5]:+.3f}',flush=True)
def objlevel(ids,X,key):
    d={}
    for i,k in zip(ids,X): d.setdefault(key(i),[]).append(k)
    return {k:np.mean(v,axis=0) for k,v in d.items()}
for nm in ['voxel16','bbox','structure','volatility','multiview']:
    z=np.load(f'{G}/bank/bank_{nm}.npz',allow_pickle=True); run_object_level(nm,dict(zip(z['ids'],z['X'].astype(np.float32))))
z=np.load(f'{G}/bank/bank3d_shapenet.npz',allow_pickle=True); run_object_level('d57',dict(zip(z['ids'],z['X'].astype(np.float32))))
z=np.load(f'{K}/eval/encoder_features/pretrained.npz',allow_pickle=True); ids=z['train_ids']; key=lambda i:f'{SYN[i.split("/")[0]]}/{i.split("/")[1]}'; keep=np.array([key(i) in need for i in ids]); run_object_level('dinov2_pretrained',objlevel(ids[keep],z['train_X'][keep].astype(np.float32),key))
z=np.load(f'{G}/bank/bank_viewdepth.npz',allow_pickle=True); ids=z['ids']; keep=np.array([i.rsplit('/',1)[0] in need for i in ids]); Xv=z['X'][keep].astype(np.float32); idv=ids[keep]
run_object_level('viewdepth_objmean',objlevel(idv,Xv,lambda i:i.rsplit('/',1)[0]))
# ---- view-specific (image level)
Xv=norm(Xv); vix={i:k for k,i in enumerate(idv)}; views=np.array([int(i[-3:]) for i in idv])
tri_img=np.array([[vix.get(i,-1) for i in l] for l in T.imgs]); valid=(tri_img>=0).all(1)
ests=['view_nn1','view_knn5','view_knn10','view_nn1_sameview','view_knn5_sameview']; per_img={e:np.full((len(models),len(idv)),np.nan) for e in ests}
for m in models:
    rowsS=[vix[f'{o}/{v:03d}'] for o in train[m] for v in range(15) if f'{o}/{v:03d}' in vix]; S=Xv[rowsS]; sv=views[rowsS]; i=mix[m]
    Dm=1-Xv@S.T; srt=np.sort(Dm,axis=1); per_img['view_nn1'][i]=srt[:,0]; per_img['view_knn5'][i]=srt[:,:5].mean(1); per_img['view_knn10'][i]=srt[:,:10].mean(1)
    same=(views[:,None]==sv[None,:]); Ds=np.where(same,Dm,np.inf); srt2=np.sort(Ds,axis=1); per_img['view_nn1_sameview'][i]=srt2[:,0]; k5=srt2[:,:5]; per_img['view_knn5_sameview'][i]=np.where(np.isinf(k5).any(1),np.nan,k5.mean(1))
for e in ests:
    P=per_img[e]; tt=Lm.trial.values; mi=Lm.mi.values; ok=valid[tt]; x=np.full(len(Lm),np.nan)
    x[ok]=np.nanmean(np.stack([P[mi[ok],tri_img[tt[ok],j]] for j in range(3)]),axis=0)
    keep=~np.isnan(x); d=Lm[keep].assign(x=x[keep])
    def sc2(mask,half):
        dd=d[mask[keep]&half[keep]]; g=dd.groupby('trial'); xc=dd.x-g.x.transform('mean'); yc=dd.ft-g.ft.transform('mean'); return stats.pearsonr(xc,yc)[0]
    sc=[sc2(np.ones(len(Lm),bool),odd),sc2(np.ones(len(Lm),bool),~odd),sc2(own,odd),sc2(own,~odd),sc2(~own,odd),sc2(~own,~odd)]
    rows.append(('viewdepth_image',e,int((~keep).sum()),*sc)); print(f'{"viewdepth_image":18s} {e:18s} all {sc[0]:+.3f}/{sc[1]:+.3f}  own {sc[2]:+.3f}/{sc[3]:+.3f}  other {sc[4]:+.3f}/{sc[5]:+.3f}',flush=True)
R=pd.DataFrame(rows,columns=['descriptor','estimator','missing','all_fit','all_test','own_fit','own_test','oth_fit','oth_test']); R.to_csv(f'{G}/out/distance_search.csv',index=False)
print('\nHELD-OUT, sorted by own-category score:'); print(R.sort_values('own_test')[['descriptor','estimator','all_test','own_test','oth_test']].round(3).to_string(index=False)); print('done')
