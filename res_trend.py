import ast, sys, numpy as np, pandas as pd
from scipy import stats
NAV='/vast/projects/bonnen/naturalistic-navig'; G=f'{NAV}/Dist-shift-data/geometric_shift'
sys.path.insert(0,G)
from metric_space import build, behaviour, SYN
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
INV={v:k for k,v in SYN.items()}; own={}
for _,r in mm.iterrows():
    s={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s)==1: own[r['trial']]=INV.get(list(s)[0])
REPS=[('57-number (rot-invariant)',f'{G}/bank/bank3d_shapenet_trained.npz',f'{G}/bank/test3d_shapenet.npz',57),
      ('voxel 8^3',f'{G}/bank/bank_voxel8.npz',f'{G}/bank/test_voxel8.npz',512),
      ('voxel 16^3',f'{G}/bank/bank_voxel16.npz',f'{G}/bank/test_voxel16.npz',4096),
      ('voxel 32^3',f'{G}/bank/bank_voxel32.npz',f'{G}/bank/test_voxel32.npz',32768)]
print("WITHIN-CATEGORY effect vs representation resolution   (own-category model, n=706)")
print(f"  {'representation':28s}{'dims':>7s}{'ft margin':>11s}{'PRE margin':>12s}"
      f"{'partial ft|pre':>16s}{'all-12 within':>15s}")
rows=[]
for nm,bf,tf,dims in REPS:
    d=build(bf,tf).merge(behaviour(),on=['trial','category'])
    d['own']=d.trial.map(own); d=d.dropna(subset=['own']); oc=d[d.category==d.own]
    Z=np.column_stack([np.ones(len(oc)),oc.pre])
    rx=oc.both_mean-Z@np.linalg.lstsq(Z,oc.both_mean,rcond=None)[0]
    ry=oc.ft-Z@np.linalg.lstsq(Z,oc.ft,rcond=None)[0]
    pr=stats.pearsonr(rx,ry)
    w=d.copy()
    for c in ['both_mean','ft']:
        w[c]=d[c]-d.groupby('trial')[c].transform('mean')-d.groupby('category')[c].transform('mean')+d[c].mean()
    r_all=stats.pearsonr(w.both_mean,w.ft)[0]
    rf=stats.pearsonr(oc.both_mean,oc.ft); rp=stats.pearsonr(oc.both_mean,oc.pre)
    print(f"  {nm:28s}{dims:>7d}{rf[0]:>+11.3f}{rp[0]:>+12.3f}{pr[0]:>+11.3f} (p={pr[1]:.3f}){r_all:>+15.3f}")
    rows.append((nm,dims,rf[0],rp[0],pr[0],pr[1],r_all))
pd.DataFrame(rows,columns=['rep','dims','ft','pre','partial','p_partial','all12']).to_csv(
    f'{G}/out/resolution_trend.csv',index=False)
print(f"\nwrote {G}/out/resolution_trend.csv")
