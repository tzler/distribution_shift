"""Three shift metrics side by side — coverage, knn_mean, and the manuscript's original
DINOv2-l1 distance — in three views: on-category row, two-way-centred (fig 7 style),
and pooled. The pretrained control always gets its OWN axis."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
GEOM,HL,GREY,SURF,INK,INK2,TEAL='#2a78d6','#eb6834','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans',
 'text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':10.5})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
EMP='trial_distance_(L1_not_normalized)'
d=pd.read_csv(f'{G}/out/coverage_shapenet_voxel16_percat.csv')           # trial, category, coverage, count, knn, ft, pre, fc, pc, own
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); rows=[]
for cat in d.category.unique():
    o=pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv'); o['trial']=m['trial'].values; o['category']=cat; rows.append(o[['trial','category',EMP]])
d=d.merge(pd.concat(rows),on=['trial','category']).rename(columns={EMP:'orig'})
METRICS=[('coverage','coverage','training mass within ε\n(model-free, count-based)',TEAL),
         ('knn','knn_mean','distance to 50 nearest\n(model-free, oddity-blind)',GEOM),
         ('orig','original','DINOv2 ℓ₁ distance\n(the manuscript\'s metric)',HL)]
CB=[-0.5,0.5,2.5,5.5,10.5,20.5,50.5,100.5,200.5,1e9]; CL=['0','1–2','3–5','6–10','11–20','21–50','51–100','101–200','>200']

def bins_for(col,x,y,nb=15,cnt=None):
    """coverage → count bins (labels), others → quantile bins."""
    if col=='coverage':
        q=pd.cut(cnt,CB,labels=False); g=pd.DataFrame({'q':q,'y':y}).groupby('q'); return np.arange(len(g)),g.y.mean().values,g.y.sem().values,g.size().values
    q=pd.qcut(x.rank(method='first'),nb,labels=False); g=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return g.x.mean().values,g.y.mean().values,g.y.sem().values,g.size().values
def cov_axis(a): a.set_xticks(range(len(CL))); a.set_xticklabels(CL,fontsize=8.2); a.invert_xaxis()
def draw(a,col,x,y,color,cnt=None,nb=15,fit=False):
    bx,by,be,n=bins_for(col,x,y,nb,cnt); style(a)
    a.errorbar(bx,by,yerr=be,fmt='o'+('' if fit else '-'),color=color,ms=7.5,mfc=color,mec=SURF,mew=1.3,lw=1.9,ecolor='#d8d7d2',elinewidth=1.3,zorder=4)
    if fit:
        b1,b0=np.polyfit(bx,by,1); xs=np.linspace(bx.min(),bx.max(),50); a.plot(xs,b1*xs+b0,color=color,lw=2.4,alpha=.5,zorder=3); a.axhline(0,color='#e6e4df',lw=1,zorder=1)
    if col=='coverage': cov_axis(a)
    rb=stats.pearsonr(bx,by)[0]
    return -rb if col=='coverage' else rb    # coverage bins run low→high count; flip so negative = margin falls with shift

# ======================= A: on-category row
w=d[d.category==d.own].copy()
fig,ax=plt.subplots(2,3,figsize=(19,9.6),gridspec_kw=dict(height_ratios=[1.35,1])); fig.subplots_adjust(left=.05,right=.99,top=.80,bottom=.07,hspace=.5,wspace=.25)
for j,(col,name,desc,color) in enumerate(METRICS):
    r=stats.pearsonr(w[col],w.ft)[0]; rc=stats.pearsonr(w[col],w.pre)[0]
    rb=draw(ax[0,j],col,w[col],w.ft,color,cnt=w['count']); ax[0,j].set_title(f'{name}  —  {desc}\nfine-tuned:  r = {r:+.3f}   binned r = {rb:+.2f}',loc='left'); ax[0,j].set_ylabel('fine-tuned oddity margin')
    rbc=draw(ax[1,j],col,w[col],w.pre,GREY,cnt=w['count']); ax[1,j].set_title(f'control: pretrained DINOv2  (own axis)\nr = {rc:+.3f}   binned r = {rbc:+.2f}',loc='left'); ax[1,j].set_ylabel('pretrained oddity margin')
    ax[1,j].set_xlabel('# own-category training objects within ε' if col=='coverage' else f'{name} shift to the trial\'s OWN category training set')
fig.suptitle('The on-category row: three shift metrics, one point per trial  (n = 706)',fontsize=15,x=.05,ha='left',y=.97)
fig.text(.05,.905,'Keep only the model that was fine-tuned on each trial\'s own category. This is the row that looked weak under distance. Top: the fine-tuned margin. Bottom: the pretrained control on its own axis —\n'
         'it never trained on these sets, so a shift measure should leave it flat. Coverage is binned on the raw count (bin sizes shown); the other two on 15 rank-quantile bins.',fontsize=9.3,color=INK2,va='top')
for j in range(3):
    if METRICS[j][0]=='coverage':
        _,_,_,n=bins_for('coverage',w.coverage,w.ft,cnt=w['count'])
        for k,nn in enumerate(n): ax[0,j].text(k,ax[0,j].get_ylim()[0]+.01*np.diff(ax[0,j].get_ylim())[0],f'{nn}',ha='center',fontsize=7.2,color=INK2)
fig.savefig(f'{G}/out/figures/fig54_oncat_three_metrics.png',dpi=300); plt.close(fig); print('[fig] 54')

# ======================= B: fig-7 style, two-way centred, three metrics (single row: the control is identically 0 here)
def within(df,cols):
    o=df.copy()
    for c in cols: o[c]=df[c]-df.groupby('trial')[c].transform('mean')-df.groupby('category')[c].transform('mean')+df[c].mean()
    return o
fig,ax=plt.subplots(1,3,figsize=(19,6.4)); fig.subplots_adjust(left=.05,right=.99,top=.72,bottom=.15,wspace=.25)
for j,(col,name,desc,color) in enumerate(METRICS):
    v=within(d,[col,'ft']); x,y=v[col].values,v['ft'].values
    q=pd.qcut(pd.Series(x).rank(method='first'),30,labels=False); g=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); bx,by,be=g.x.mean().values,g.y.mean().values,g.y.sem().values
    a=ax[j]; style(a); a.errorbar(bx,by,yerr=be,fmt='o',color=color,ms=8,mfc=color,mec=SURF,mew=1.8,ecolor='#d8d7d2',zorder=4)
    b1,b0=np.polyfit(bx,by,1); xs=np.linspace(bx.min(),bx.max(),50); a.plot(xs,b1*xs+b0,color=color,lw=2.5,alpha=.55,zorder=3); a.axhline(0,color='#e6e4df',lw=1,zorder=1)
    rb=stats.pearsonr(bx,by); rt=stats.pearsonr(x,y); rc=stats.pearsonr(x,within(d,[col,'pre'])['pre'].values)[0]
    a.set_title(f'{name}  —  {desc}\nbinned r = {rb[0]:+.3f}   trial level r = {rt[0]:+.3f}   control = {rc:+.0e}',loc='left')
    a.set_ylabel('fine-tuned oddity margin\n(centred within trial & training set)'); a.set_xlabel(f'{name} shift  (centred within trial & training set)')
fig.suptitle('The oddity margin recovers distance-to-training, with trial difficulty removed by design — three metrics',fontsize=15,x=.05,ha='left',y=.965)
fig.text(.05,.875,'fig 7\'s template. Each of 706 trials appears 12 times, once per training category; x and y are centred within trial and within training set, which is what makes the values negative:\n'
         'they are relative to each trial\'s own average across the 12 training sets. 30 rank-quantile bins on all 8,472 observations. The pretrained margin is trial-constant, so in this view\n'
         'its correlation with any metric is exactly zero — there is no control row to draw. All three metrics work here; this is the view in which the original metric is at its best.',fontsize=9.3,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig55_centred_three_metrics.png',dpi=300); plt.close(fig); print('[fig] 55')

# ======================= C: pooled row, three metrics, control on own axis
fig,ax=plt.subplots(2,3,figsize=(19,9.6),gridspec_kw=dict(height_ratios=[1.35,1])); fig.subplots_adjust(left=.05,right=.99,top=.80,bottom=.07,hspace=.5,wspace=.25)
for j,(col,name,desc,color) in enumerate(METRICS):
    r=stats.pearsonr(d[col],d.ft)[0]; rc=stats.pearsonr(d[col],d.pre)[0]
    rb=draw(ax[0,j],col,d[col],d.ft,color,cnt=d['count'],nb=12); ax[0,j].set_title(f'{name}  —  {desc}\nfine-tuned:  r = {r:+.3f}   binned r = {rb:+.2f}',loc='left'); ax[0,j].set_ylabel('fine-tuned oddity margin')
    rbc=draw(ax[1,j],col,d[col],d.pre,GREY,cnt=d['count'],nb=12); ax[1,j].set_title(f'control: pretrained DINOv2  (own axis)\nr = {rc:+.3f}   binned r = {rbc:+.2f}   {"← passes" if abs(rc)<.05 else "← fails"}',loc='left'); ax[1,j].set_ylabel('pretrained oddity margin')
    ax[1,j].set_xlabel('# training objects of that category within ε' if col=='coverage' else f'{name} shift to that model\'s training set')
fig.suptitle('The pooled row: every trial × every model, three shift metrics  (n = 8,472)',fontsize=15,x=.05,ha='left',y=.97)
fig.text(.05,.905,'Raw x, nothing centred. The pretrained control is on its own axis in the bottom row. A metric that tracks the training set and not the stimulus should leave the bottom row flat:\n'
         'coverage does; the two distance metrics do not, because objects far from every training set are also hard for the untouched encoder.',fontsize=9.3,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig56_pooled_three_metrics.png',dpi=300); plt.close(fig); print('[fig] 56')

# numbers for the write-up
print('\nON-CATEGORY (n=706):'); 
for col,name,_,_ in METRICS: print(f'  {name:10s} r(ft) {stats.pearsonr(w[col],w.ft)[0]:+.3f}   r(pre) {stats.pearsonr(w[col],w.pre)[0]:+.3f}')
print('POOLED (n=8472):')
for col,name,_,_ in METRICS: print(f'  {name:10s} r(ft) {stats.pearsonr(d[col],d.ft)[0]:+.3f}   r(pre) {stats.pearsonr(d[col],d.pre)[0]:+.3f}')
print('TWO-WAY CENTRED trial-level r:')
for col,name,_,_ in METRICS:
    v=within(d,[col,'ft']); print(f'  {name:10s} {stats.pearsonr(v[col],v.ft)[0]:+.3f}')
# on-category coverage: is it graded or a step?  compare >100 vs <=100
hi=w[w['count']>100]; lo=w[w['count']<=100]
print(f'\nON-CAT coverage shape: >100 neighbours n={len(hi)} margin {hi.ft.mean():.3f};  <=100 n={len(lo)} margin {lo.ft.mean():.3f};  within <=100 r = {stats.pearsonr(lo.coverage,lo.ft)[0]:+.3f} (p={stats.pearsonr(lo.coverage,lo.ft)[1]:.2f})')
