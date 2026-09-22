"""What is inside the on-category coverage curve: bin composition by category, the curve
after category-centring, and the curve inside every category separately."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

G=_RESOLVED_G
TEAL,GREY,SURF,INK,INK2,HL='#1baf7a','#8a8884','#fcfcfb','#0b0b0b','#52514e','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,
 'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':10.5})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
d=pd.read_csv(f'{G}/out/coverage_shapenet_voxel16_percat.csv'); w=d[d.category==d.own].copy()
CB=[-0.5,0.5,2.5,5.5,10.5,20.5,50.5,100.5,200.5,1e9]; CL=['0','1–2','3–5','6–10','11–20','21–50','51–100','101–200','>200']
w['bin']=pd.cut(w['count'],CB,labels=False); CATS=sorted(w.own.unique())
cmap=plt.get_cmap('tab20'); CC={c:cmap(i/12) for i,c in enumerate(CATS)}
w['ftc']=w.ft-w.groupby('own').ft.transform('mean')

fig=plt.figure(figsize=(21,11)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.05],left=.045,right=.99,top=.80,bottom=.07,hspace=.5,wspace=.25)
# ---- A: the curve as shown
a=fig.add_subplot(gs[0,0]); style(a); g=w.groupby('bin').ft.agg(['mean','sem','size'])
a.errorbar(g.index,g['mean'],yerr=g['sem'],fmt='o-',color=TEAL,ms=8,mfc=TEAL,mec=SURF,mew=1.4,lw=2,ecolor='#d8d7d2',zorder=4)
a.set_xticks(range(9)); a.set_xticklabels(CL,fontsize=8.5); a.invert_xaxis(); a.set_ylabel('fine-tuned oddity margin')
a.set_title(f'A  The curve as shown in fig 54\nbinned r on 9 bin means = {-stats.pearsonr(g.index,g["mean"])[0]:+.2f};  point-level r on 706 trials = {stats.pearsonr(w.coverage,w.ft)[0]:+.3f}',loc='left')
# ---- B: composition of each bin
a=fig.add_subplot(gs[0,1]); style(a); comp=pd.crosstab(w.bin,w.own).reindex(range(9)).fillna(0)
bottom=np.zeros(9)
for c in CATS:
    v=comp[c].values if c in comp else np.zeros(9); a.bar(range(9),v,bottom=bottom,color=CC[c],width=.72,label=c,zorder=3,edgecolor=SURF,lw=.5); bottom+=v
a.set_xticks(range(9)); a.set_xticklabels(CL,fontsize=8.5); a.invert_xaxis(); a.set_ylabel('number of trials in the bin'); a.legend(fontsize=7.6,ncol=2,loc='upper left')
a.set_title('B  Who is in each bin\nthe high-coverage bins are six homogeneous categories; the low bins are the other six',loc='left')
# ---- C: category mean margin vs category mean coverage (12 points) — the thing the curve is actually showing
a=fig.add_subplot(gs[0,2]); style(a); cm=w.groupby('own').agg(medcov=('count','median'),ft=('ft','mean'),se=('ft','sem'),n=('ft','size'))
for c,r in cm.iterrows(): a.errorbar(np.log1p(r.medcov),r.ft,yerr=r.se,fmt='o',color=CC[c],ms=11,mec=SURF,mew=1.4,ecolor='#d8d7d2',zorder=4); a.text(np.log1p(r.medcov)+.08,r.ft,c,fontsize=8.3,va='center')
rr=stats.pearsonr(np.log1p(cm.medcov),cm.ft)
a.set_xlabel('median # own-category training objects within ε   (log scale, log(1+n))'); a.set_ylabel('mean fine-tuned margin for that category')
a.set_title(f'C  The same relationship at the category level: 12 points\nr = {rr[0]:+.2f}, p = {rr[1]:.2f}   — this is what panel A is made of',loc='left')
# ---- D: the curve after removing each category's mean margin
a=fig.add_subplot(gs[1,0]); style(a); g2=w.groupby('bin').ftc.agg(['mean','sem'])
a.errorbar(g2.index,g2['mean'],yerr=g2['sem'],fmt='o-',color=TEAL,ms=8,mfc=TEAL,mec=SURF,mew=1.4,lw=2,ecolor='#d8d7d2',zorder=4); a.axhline(0,color='#dcdad5',lw=1.2,zorder=1)
a.set_xticks(range(9)); a.set_xticklabels(CL,fontsize=8.5); a.invert_xaxis(); a.set_ylabel('margin minus its category\'s mean margin')
xo=w.coverage-w.groupby('own').coverage.transform('mean'); rc=stats.pearsonr(xo,w.ftc)
a.set_title(f'D  Same curve, margin centred within category\npoint-level r = {rc[0]:+.3f}, p = {rc[1]:.2f}',loc='left')
# ---- E: inside every category separately (small multiples as scatter + fit)
a=fig.add_subplot(gs[1,1:]); style(a)
xs_all=[]; 
for i,c in enumerate(CATS):
    s=w[w.own==c]; x=np.log1p(s['count']); y=s.ft
    if s['count'].std()>0 and len(s)>=10:
        r=stats.pearsonr(x,y)[0]; b1,b0=np.polyfit(x,y,1); xx=np.linspace(x.min(),x.max(),20)
        a.scatter(x,y,s=14,color=CC[c],alpha=.45,linewidths=0,zorder=3); a.plot(xx,b1*xx+b0,color=CC[c],lw=2.2,zorder=4,label=f'{c}  r={r:+.2f} (n={len(s)})')
a.legend(fontsize=7.8,ncol=3,loc='upper right'); a.set_xlabel('log(1 + # own-category training objects within ε)'); a.set_ylabel('fine-tuned oddity margin')
pc=[stats.pearsonr(np.log1p(s['count']),s.ft)[0] for c,s in w.groupby('own') if s['count'].std()>0 and len(s)>=10]
a.set_title(f'E  Inside each category separately: one line per category\nmean within-category r = {np.mean(pc):+.3f};  {sum(np.array(pc)<0)} of {len(pc)} negative  (sign test p = {stats.binomtest(int(sum(np.array(pc)<0)),len(pc)).pvalue:.2f})',loc='left')
fig.suptitle('Anatomy of the on-category coverage curve',fontsize=15.5,x=.045,ha='left',y=.965)
fig.text(.045,.905,'Panel A is the green panel of fig 54. Every other panel takes it apart. B: the bins are sorted by category, not by anything within a category. C: the curve is a 12-point relationship between\n'
         'category homogeneity and category margin. D: remove each category\'s mean margin and the curve is flat. E: inside every single category, coverage and margin are unrelated.\n'
         'The pretrained control cannot catch this — it rules out "hard for every model", not "this category\'s model is good at this category".',fontsize=9.3,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig57_oncat_anatomy.png',dpi=300); print('[fig] 57'); print(f'category-level r (12 pts) = {rr[0]:+.3f} p={rr[1]:.3f};  within-category rs: '+' '.join(f'{v:+.2f}' for v in pc))
