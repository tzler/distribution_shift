"""Density-based shift estimates, compared against knn_mean on the same battery.

  knn_mean      mean cosine distance to the 50 nearest bank objects   (the current metric)
  kde_logdens   -log( (1/n_C) sum_i exp(-D_i^2 / 2h^2) ),  h = median bank-internal NN distance
                a proper kernel density under category C's training set, normalised by n_C
  coverage      log(1 + #bank objects within eps),  eps = global median bank-internal NN dist
                raw training MASS near the test point, not distance to the nearest one
  dens_ratio    kde_logdens(C) - kde_logdens(ALL 12 categories pooled)
                category-specific coverage with object atypicality divided out
All are oddity-blind: per object, then averaged over the trial's images.
"""
import ast, sys, numpy as np, pandas as pd
from scipy import stats
NAV='/vast/projects/bonnen/naturalistic-navig'
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343',
 'chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459',
 'sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
INV={v:k for k,v in SYN.items()}
REP=sys.argv[1] if len(sys.argv)>1 else 'voxel16'
bank={'d57':'bank3d_shapenet_trained.npz'}.get(REP,f'bank_{REP}.npz')
test={'d57':'test3d_shapenet.npz'}.get(REP,f'test_{REP}.npz')

bz=np.load(f'{G}/bank/{bank}',allow_pickle=True); tz=np.load(f'{G}/bank/{test}',allow_pickle=True)
B,T=bz['X'].astype(float),tz['X'].astype(float)
mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12
TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-12
bsyn=np.array([i.split('/')[0] for i in bz['ids']]); tix={k:i for i,k in enumerate(tz['ids'])}
D=1.0-TN@BN.T                                            # 897 x 20885 cosine distances
print(f'[{REP}] D {D.shape}')

# bank-internal NN distance (for bandwidth / eps), computed on a 4000-object subsample for speed
rng=np.random.default_rng(0); sub=rng.choice(len(BN),min(4000,len(BN)),replace=False)
DB=1.0-BN[sub]@BN.T; DB[np.arange(len(sub)),sub]=np.inf
nn_int=DB.min(1); eps_global=np.median(nn_int)
h_cat={}
for cat,s in SYN.items():
    m=bsyn[sub]==s; h_cat[cat]=np.median(nn_int[m]) if m.sum()>20 else eps_global
print(f'  global eps = {eps_global:.4f};  per-category h: '+' '.join(f'{c[:4]}={h:.3f}' for c,h in h_cat.items()))

def kde(Dsub,h):  # -log mean exp(-d^2/2h^2), stable
    e=-(Dsub**2)/(2*h*h); m=e.max(1,keepdims=True)
    return -(m[:,0]+np.log(np.exp(e-m).mean(1)))
kde_all=kde(D,np.median(list(h_cat.values())))

per={}
for cat,s in SYN.items():
    sel=bsyn==s; Dc=D[:,sel]
    part=np.sort(np.partition(Dc,49,axis=1)[:,:50],axis=1)
    per[cat]=dict(knn_mean=part.mean(1),
                  kde_logdens=kde(Dc,h_cat[cat]),
                  coverage=-np.log1p((Dc<eps_global).sum(1)),   # sign flipped: larger = LESS coverage
                  dens_ratio=kde(Dc,h_cat[cat])-kde_all)
EST=['knn_mean','kde_logdens','coverage','dens_ratio']

m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
rows=[]
for cat in SYN:
    for _,r in mm.iterrows():
        keys=['/'.join(f[:-4].split('_')[:2]) for f in ast.literal_eval(r['images'])]
        if not all(k in tix for k in keys): continue
        ix=[tix[k] for k in keys]
        rows.append(dict(trial=r['trial'],category=cat,**{e:float(per[cat][e][ix].mean()) for e in EST}))
P=pd.DataFrame(rows)
Y=[]
for cat in SYN:
    o=pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv'); o['trial']=m['trial'].values
    o['category']=cat; o['ft']=o['fine_tuned_oddity_margin']; o['pre']=o['pretrained_oddity_margin']
    Y.append(o[['trial','category','ft','pre']])
d=P.merge(pd.concat(Y),on=['trial','category'])
own={}
for _,r in mm.iterrows():
    s={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s)==1: own[r['trial']]=INV.get(list(s)[0])
d['own']=d.trial.map(own); d=d.dropna(subset=['own']); d=d[d.groupby('trial').trial.transform('size')==12]
w=d[d.category==d.own]
print(f'  n={len(d)} trials={d.trial.nunique()}\n')
print(f'{"estimator":12s} | {"WITHIN-TRIAL":^22s} | {"POOLED (raw)":^22s} | {"WITHIN-CATEGORY":^22s} | {"ranks 2-12":^12s}')
print(f'{"":12s} | {"r":>8s} {"%neg":>6s} {"":>5s} | {"r(ft)":>8s} {"control":>9s}    | {"r(ft)":>8s} {"control":>9s}    | {"r":>8s}')
out=[]
for e in EST:
    xc=d[e]-d.groupby('trial')[e].transform('mean'); yc=d.ft-d.groupby('trial').ft.transform('mean')
    rw=stats.pearsonr(xc,yc)[0]
    sl=np.array([stats.linregress(s[e],s.ft).slope for _,s in d.groupby('trial') if s[e].std()>0])
    rp=stats.pearsonr(d[e],d.ft)[0]; rpc=stats.pearsonr(d[e],d.pre)[0]
    rc=stats.pearsonr(w[e],w.ft)[0]; rcc=stats.pearsonr(w[e],w.pre)[0]
    off=d[d.category!=d.own].copy(); xo=off[e]-off.groupby('trial')[e].transform('mean'); yo=off.ft-off.groupby('trial').ft.transform('mean')
    ro=stats.pearsonr(xo,yo)[0]
    print(f'{e:12s} | {rw:+8.3f} {100*(sl<0).mean():5.1f}% {"":5s} | {rp:+8.3f} {rpc:+9.3f}    | {rc:+8.3f} {rcc:+9.3f}    | {ro:+8.3f}')
    out.append(dict(rep=REP,estimator=e,within_trial_r=rw,pct_neg=100*(sl<0).mean(),pooled_r=rp,pooled_control=rpc,
                    within_cat_r=rc,within_cat_control=rcc,offcat_within_trial_r=ro))
pd.DataFrame(out).to_csv(f'{G}/out/density_estimators_{REP}.csv',index=False)
d.to_csv(f'{G}/out/density_long_{REP}.csv',index=False)
