"""Density/coverage variants, evaluated on the rows that matter — including the on-category
row AFTER category centring, which is the test of a within-category claim."""
import ast, numpy as np, pandas as pd
from scipy import stats
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117',
 'lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
INV={v:k for k,v in SYN.items()}; EPS=0.12
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); tz=np.load(f'{G}/bank/test_voxel16.npz',allow_pickle=True)
B,T=bz['X'].astype(float),tz['X'].astype(float); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12; TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-12
bsyn=np.array([i.split('/')[0] for i in bz['ids']]); tix={k:i for i,k in enumerate(tz['ids'])}; D=1.0-TN@BN.T
rng=np.random.default_rng(0)
per={}
for c,s in SYN.items():
    sel=bsyn==s; Dc=D[:,sel]; n=sel.sum()
    cnt=(Dc<EPS).sum(1)
    # bank-internal coverage for calibration: leave-one-out count within eps, on a subsample
    sub=rng.choice(np.where(sel)[0],min(1500,n),replace=False); Db=1.0-BN[sub]@BN[sel].T
    bank_cnt=(Db<EPS).sum(1)-1
    pct=np.searchsorted(np.sort(bank_cnt),cnt)/len(bank_cnt)          # where the test object sits in the bank's own coverage distribution
    per[c]=dict(
        knn=np.sort(np.partition(Dc,49,axis=1)[:,:50],axis=1).mean(1),
        cov=-np.log1p(cnt),                                            # hard coverage (mass)
        cov_frac=-np.log1p(cnt/n*1000),                                # density: count per 1000 bank objects (removes bank-size)
        cov_soft=-np.log1p(np.exp(-Dc/EPS).sum(1)),                    # soft coverage WITH a ceiling (the +1)
        cov_soft2=-np.log1p(np.exp(-(Dc/EPS)**2).sum(1)),              # gaussian soft coverage with ceiling
        cov_pct=-pct,                                                  # category-calibrated: percentile within own bank's coverage
        cov_multi=-(np.log1p((Dc<.06).sum(1))+np.log1p((Dc<.12).sum(1))+np.log1p((Dc<.2).sum(1)))/3,   # multi-scale
    )
EST=list(per['chair'].keys())
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']; rows=[]
for c in SYN:
    for _,r in mm.iterrows():
        keys=['/'.join(f[:-4].split('_')[:2]) for f in ast.literal_eval(r['images'])]
        if all(k in tix for k in keys): ix=[tix[k] for k in keys]; rows.append(dict(trial=r['trial'],category=c,**{e:float(per[c][e][ix].mean()) for e in EST}))
P=pd.DataFrame(rows); Y=[]
for c in SYN:
    o=pd.read_csv(f'{S}/{c}/ood_analysis_results.csv'); o['trial']=m['trial'].values; o['category']=c; o['ft']=o['fine_tuned_oddity_margin']; o['pre']=o['pretrained_oddity_margin']; Y.append(o[['trial','category','ft','pre']])
d=P.merge(pd.concat(Y),on=['trial','category']); own={}
for _,r in mm.iterrows():
    s={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s)==1: own[r['trial']]=INV.get(list(s)[0])
d['own']=d.trial.map(own); d=d.dropna(subset=['own']); d=d[d.groupby('trial').trial.transform('size')==12].copy(); w=d[d.category==d.own].copy()
def cen(v,g): return v-v.groupby(g).transform('mean')
print(f'{"estimator":10s} | {"within-trial":>12s} | {"pooled r / ctrl":>16s} | {"on-cat r / ctrl":>16s} | {"ON-CAT, CATEGORY-CENTRED r / ctrl":>34s}')
for e in EST:
    xc=cen(d[e],d.trial); yc=cen(d.ft,d.trial); rw=stats.pearsonr(xc,yc)[0]
    rp,rpc=stats.pearsonr(d[e],d.ft)[0],stats.pearsonr(d[e],d.pre)[0]
    ro,roc=stats.pearsonr(w[e],w.ft)[0],stats.pearsonr(w[e],w.pre)[0]
    xo=cen(w[e],w.own); yo=cen(w.ft,w.own); po=cen(w.pre,w.own); rcc,pcc=stats.pearsonr(xo,yo); rccc=stats.pearsonr(xo,po)[0]
    print(f'{e:10s} | {rw:+12.3f} | {rp:+7.3f} / {rpc:+6.3f} | {ro:+7.3f} / {roc:+6.3f} | {rcc:+8.3f} (p={pcc:.2f}) / {rccc:+6.3f}')
d.to_csv(f'{G}/out/coverage_variants_voxel16.csv',index=False)
