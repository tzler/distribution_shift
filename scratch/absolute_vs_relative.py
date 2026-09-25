"""Is the margin an ABSOLUTE measure of distribution shift, or only a RELATIVE one?
Absolute = one function margin = g(distance) holds everywhere: the same distance implies the
same margin whichever category or trial it comes from. Relative = the relationship only holds
after each category's (or each trial's) own level is removed.
Test: fit on eleven categories, predict the twelfth; compare with fitting that category itself."""
from _repo import G
import numpy as np, pandas as pd
from scipy import stats
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
DV=pd.read_csv(f'{G}/data/dinov2_distance_long.csv.gz')
L=L.merge(DV,on=['model','trial'],how='left'); L=L.dropna(subset=['dinov2'])
print(f'{len(L):,} (trial x model) pairs, {L.test_cat.nunique()} test categories, {L.model.nunique()} models\n')
def fit(x,y): b=np.polyfit(x,y,1); return b
def r2(y,p): return 1-np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
for xcol,name in [('dinov2','frozen-network distance'),('dist','16³ voxel distance')]:
    print(f'=== {name}')
    x,y=L[xcol].values,L.ft.values
    b=fit(x,y); print(f'  one function for everything:        R² = {r2(y,np.polyval(b,x)):+.4f}   slope {b[0]:+.4f}')
    # per category
    sl,ic,r2s=[],[],[]
    for c,d in L.groupby('test_cat'):
        bb=fit(d[xcol],d.ft); sl.append(bb[0]); ic.append(bb[1]); r2s.append(r2(d.ft.values,np.polyval(bb,d[xcol])))
    print(f'  per-category fits:                  slope {np.mean(sl):+.4f} ± {np.std(sl):.4f} (range {min(sl):+.4f} to {max(sl):+.4f})')
    print(f'                                      intercept {np.mean(ic):+.3f} ± {np.std(ic):.3f}   mean within-category R² = {np.mean(r2s):+.4f}')
    # leave-one-category-out: the honest absolute test
    loo=[]
    for c,d in L.groupby('test_cat'):
        tr=L[L.test_cat!=c]; bb=fit(tr[xcol],tr.ft); loo.append((c,r2(d.ft.values,np.polyval(bb,d[xcol])),np.polyval(bb,d[xcol]).mean()-d.ft.mean()))
    lo=pd.DataFrame(loo,columns=['cat','R2','bias'])
    print(f'  fit on 11 categories, predict the 12th:  R² = {lo.R2.mean():+.4f} (worst {lo.R2.min():+.3f}, best {lo.R2.max():+.3f});  mean |bias| {lo.bias.abs().mean():.3f} of margin')
    # relative versions
    g=L.groupby('trial'); xc=L[xcol]-g[xcol].transform('mean'); yc=L.ft-g.ft.transform('mean')
    print(f'  same trial, different training sets:     r = {stats.pearsonr(xc,yc)[0]:+.3f}   R² = {stats.pearsonr(xc,yc)[0]**2:.4f}')
    gc=L.groupby('test_cat'); xk=L[xcol]-gc[xcol].transform('mean'); yk=L.ft-gc.ft.transform('mean')
    print(f'  same category, level removed:            r = {stats.pearsonr(xk,yk)[0]:+.3f}')
    print(f'  how much of the margin is the trial:     {100*g.ft.transform("mean").var()/L.ft.var():.0f}% of its variance\n')
# what a fixed distance means in different categories
print('=== the same distance, different categories (frozen-network distance 0.5–0.7, n per cell in brackets)')
band=L[(L.dinov2>0.5)&(L.dinov2<0.7)]
t=band.groupby('test_cat').agg(margin=('ft','mean'),n=('ft','size'),pre=('pre','mean')).round(3).sort_values('margin')
print(t.to_string()); print(f'  spread of the margin across categories at the same distance: {t.margin.max()-t.margin.min():.3f}  (the whole effect of distance across its range is ~{abs(np.polyfit(L.dinov2,L.ft,1)[0])*0.9:.3f})')
