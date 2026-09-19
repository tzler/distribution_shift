import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import validate_margin as V
G='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
GEOM,HL,GREY,SURF,INK,INK2='#2a78d6','#eb6834','#8a8884','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans',
 'text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y',color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)

# ---------------- fig49: the entanglement, made visible
m=V.build()
fig,ax=plt.subplots(1,2,figsize=(13,6.2)); fig.subplots_adjust(left=.07,right=.985,top=.70,bottom=.30,wspace=.26)
for k,(ds,mode) in enumerate([('shapegen','2d'),('shapenet','3d')]):
    g=pd.read_csv(f'{G}/out/shift{mode}_{ds}.csv'); b=pd.read_csv(f'{G}/out/blindshift_{ds}.csv')
    d=g.merge(b,on='trial').merge(m,on='trial',validate='1:1'); sh=1-d['margin_dinov2-large']
    items=[(f'geom_dAB_{mode}','d(A,B) alone\n(object dissimilarity)',HL),(f'geom_shift_{mode}','entangled shift\nmin ½[dA+dB]',HL),
           (V.INC,'incumbent\n(DINOv2 ℓ₁)',GREY),('blind_k50','oddity-blind\nkNN-50 cosine',GEOM)]
    rb=[abs(V.binned_r(d[c].values,sh.values,30)[0]) for c,_,_ in items]; rd=[stats.pearsonr(d[c],d[f'geom_dAB_{mode}'])[0] for c,_,_ in items]
    a=ax[k]; style(a); x=np.arange(len(items))
    a.bar(x,rb,color=[c for _,_,c in items],width=.6,zorder=3,edgecolor=SURF)
    for i,(v,q) in enumerate(zip(rb,rd)):
        a.text(i,v+.015,f'|r| = {v:.2f}',ha='center',fontsize=9.5,color=INK)
        a.text(i,-.27,f'r with d(A,B) = {q:+.2f}',ha='center',va='top',fontsize=8.6,color=HL if abs(q)>.9 else INK2,transform=a.get_xaxis_transform(),fontweight='bold' if abs(q)>.9 else 'normal')
    a.set_xticks(x); a.set_xticklabels([l for _,l,_ in items],fontsize=9); a.set_ylim(0,.95)
    a.set_ylabel('|binned r| with ŝ = 1 − m  (30 bins)' if k==0 else '')
    a.set_title(f'{ds}  ·  dinov2-large  ·  n = {len(d)} trials',loc='left')
fig.suptitle('Act 2 revisited: the entangled shift was d(A,B) in disguise',fontsize=14.5,x=.07,ha='left',y=.97)
fig.text(.07,.865,'The shift used in the first encoder comparison is min over training objects of ½[d(A,C)+d(B,C)], which is bounded below by ½·d(A,B).\n'
         'On shapegen it correlates with d(A,B) at r = 0.99 and its "result" is the trivial one: dissimilar objects are easy. The oddity-blind metric\n'
         '(mean over images, never differenced) removes it — and on shapegen nothing remains. On shapenet a modest relationship survives, comparable to the incumbent.',
         fontsize=9.2,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig49_entanglement.png',dpi=300); plt.close(fig); print('[fig] fig49')

# ---------------- fig50: coverage vs knn_mean, eps sweep
fig,ax=plt.subplots(1,3,figsize=(17,6.2)); fig.subplots_adjust(left=.05,right=.99,top=.66,bottom=.14,wspace=.3)
for a in ax: style(a)
for rep,ls,mk in [('voxel16','-','o'),('d57','--','s')]:
    R=pd.read_csv(f'{G}/out/coverage_sweep_{rep}.csv'); H=R[R.est=='hard coverage']; K=R[R.est=='knn_mean k=50'].iloc[0]
    ax[0].plot(H.eps,-H.within_r,ls,marker=mk,color=GEOM,ms=7,mec=SURF,lw=2,label=f'coverage · {rep}')
    ax[0].axhline(-K.within_r,color=GREY,ls=ls,lw=1.6,label=f'knn_mean · {rep}')
    ax[1].plot(H.eps,-H.pooled_r,ls,marker=mk,color=GEOM,ms=7,mec=SURF,lw=2,label=f'coverage effect · {rep}')
    ax[1].plot(H.eps,-H.pooled_ctrl,ls,marker=mk,color=HL,ms=7,mec=SURF,lw=2,label=f'coverage CONTROL · {rep}')
    ax[1].axhline(-K.pooled_ctrl,color=GREY,ls=ls,lw=1.6,label=f'knn_mean control · {rep}')
    ax[2].plot(H.eps,-H.partial_beyond_binary,ls,marker=mk,color=GEOM,ms=7,mec=SURF,lw=2,label=f'coverage · {rep}')
    ax[2].axhline(-K.partial_beyond_binary,color=GREY,ls=ls,lw=1.6,label=f'knn_mean · {rep}')
ax[0].set_title('A  Within-trial −r  (the causal design)',loc='left'); ax[0].set_ylabel('−r'); ax[0].legend(fontsize=8,loc='lower left')
ax[1].set_title('B  Pooled row: effect vs base-DINOv2 control\ncoverage keeps the control near zero; knn_mean does not',loc='left'); ax[1].set_ylabel('−r (pooled, raw x)'); ax[1].legend(fontsize=7.6,loc='center left',bbox_to_anchor=(0.0,0.42)); ax[1].axhline(0,color='#dcdad5',lw=1)
ax[2].set_title('C  Graded signal beyond the on-category binary\n(partial −r, within trial)',loc='left'); ax[2].set_ylabel('partial −r'); ax[2].legend(fontsize=8,loc='lower left')
from matplotlib.ticker import NullFormatter, NullLocator
for a in ax:
    a.set_xlabel('ε  (cosine radius of the coverage ball)'); a.set_xscale('log'); a.set_xticks([.03,.06,.1,.15,.2,.3,.45]); a.set_xticklabels(['.03','.06','.1','.15','.2','.3','.45'])
    a.xaxis.set_minor_locator(NullLocator()); a.xaxis.set_minor_formatter(NullFormatter())
fig.suptitle('Coverage — training mass within ε of the object — vs distance to the nearest neighbours',fontsize=14.5,x=.05,ha='left',y=.97)
fig.text(.05,.87,'coverage = −log(1 + #training objects of category C within cosine distance ε of the test object), averaged over the trial\'s images.  Nothing beyond "uncovered" counts as\n'
         'more shift, which is what stops object atypicality leaking into the pooled row. Split-half (choose ε on half the trials, score the other half, 20 splits): coverage beats\n'
         'knn_mean in 20/20, held-out within-trial −0.414 vs −0.328, held-out pooled control −0.041 vs −0.143. ε chosen 0.10–0.15 every time.',
         fontsize=9.1,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig50_coverage.png',dpi=300); plt.close(fig); print('[fig] fig50')
