"""Coverage (training MASS near the test object) as a function of radius, vs knn_mean."""
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
REP=sys.argv[1]
bank={'d57':'bank3d_shapenet_trained.npz'}.get(REP,f'bank_{REP}.npz'); test={'d57':'test3d_shapenet.npz'}.get(REP,f'test_{REP}.npz')
bz=np.load(f'{G}/bank/{bank}',allow_pickle=True); tz=np.load(f'{G}/bank/{test}',allow_pickle=True)
B,T=bz['X'].astype(float),tz['X'].astype(float); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12
TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-12
bsyn=np.array([i.split('/')[0] for i in bz['ids']]); tix={k:i for i,k in enumerate(tz['ids'])}
D=1.0-TN@BN.T
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
Y=[]
for cat in SYN:
    o=pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv'); o['trial']=m['trial'].values
    o['category']=cat; o['ft']=o['fine_tuned_oddity_margin']; o['pre']=o['pretrained_oddity_margin']; Y.append(o[['trial','category','ft','pre']])
Y=pd.concat(Y)
own={}
for _,r in mm.iterrows():
    s={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s)==1: own[r['trial']]=INV.get(list(s)[0])
trial_ix=[]
for _,r in mm.iterrows():
    keys=['/'.join(f[:-4].split('_')[:2]) for f in ast.literal_eval(r['images'])]
    if all(k in tix for k in keys): trial_ix.append((r['trial'],[tix[k] for k in keys]))

def evaluate(scores):     # scores: dict cat -> per-test-object array (larger = more shift)
    rows=[dict(trial=t,category=c,x=float(scores[c][ix].mean())) for c in SYN for t,ix in trial_ix]
    d=pd.DataFrame(rows).merge(Y,on=['trial','category']); d['own']=d.trial.map(own); d=d.dropna(subset=['own'])
    d=d[d.groupby('trial').trial.transform('size')==12]
    xc=d.x-d.groupby('trial').x.transform('mean'); yc=d.ft-d.groupby('trial').ft.transform('mean')
    bc=(d.category==d.own).astype(float); bc=bc-bc.groupby(d.trial).transform('mean')
    Z=np.column_stack([np.ones(len(d)),bc]); rx=xc-Z@np.linalg.lstsq(Z,xc,rcond=None)[0]; ry=yc-Z@np.linalg.lstsq(Z,yc,rcond=None)[0]
    sl=np.array([stats.linregress(s.x,s.ft).slope for _,s in d.groupby('trial') if s.x.std()>0])
    w=d[d.category==d.own]
    return dict(within_r=stats.pearsonr(xc,yc)[0], pct_neg=100*(sl<0).mean(),
                partial_beyond_binary=stats.pearsonr(rx,ry)[0],
                pooled_r=stats.pearsonr(d.x,d.ft)[0], pooled_ctrl=stats.pearsonr(d.x,d.pre)[0],
                wcat_r=stats.pearsonr(w.x,w.ft)[0], wcat_ctrl=stats.pearsonr(w.x,w.pre)[0])

res=[]
part={c:np.sort(np.partition(D[:,bsyn==s],49,axis=1)[:,:50],axis=1).mean(1) for c,s in SYN.items()}
res.append(dict(rep=REP,est='knn_mean k=50',eps=np.nan,**evaluate(part)))
for eps in [0.03,0.06,0.10,0.15,0.20,0.30,0.45]:
    hard={c:-np.log1p((D[:,bsyn==s]<eps).sum(1)) for c,s in SYN.items()}
    res.append(dict(rep=REP,est='hard coverage',eps=eps,**evaluate(hard)))
    soft={c:-np.log(np.exp(-D[:,bsyn==s]/eps).sum(1)) for c,s in SYN.items()}   # exponential-kernel mass, global bandwidth
    res.append(dict(rep=REP,est='soft coverage',eps=eps,**evaluate(soft)))
R=pd.DataFrame(res); R.to_csv(f'{G}/out/coverage_sweep_{REP}.csv',index=False)
print(f'\n[{REP}]  within-trial r | %neg | partial beyond on-cat binary | pooled r / CONTROL | within-cat r / control')
for _,r in R.iterrows():
    print(f'  {r.est:14s} eps={r.eps if r.eps==r.eps else "-":>5}  {r.within_r:+.3f}  {r.pct_neg:5.1f}%   {r.partial_beyond_binary:+.3f}       '
          f'{r.pooled_r:+.3f} / {r.pooled_ctrl:+.3f}      {r.wcat_r:+.3f} / {r.wcat_ctrl:+.3f}')
