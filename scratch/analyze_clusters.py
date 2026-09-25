"""Sub-category calibration (D19). Bank-built chair trials from three held-out cluster halves, scored by:
(a) the 12 original category models  → does the bank test set carry the category effect?
(b) 9 cluster models (3 clusters × N)  → does sub-category membership move the margin, and at what N?"""
import json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _repo import G as _RESOLVED_G

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
T=pd.read_csv(f'{K}/banktrials/banktrials_chair.csv'); T['cluster']=T.condition.str[-1].astype(int)
def load(name):
    o=pd.read_csv(f'{K}/eval_bank/{name}/ood_analysis_results.csv'); assert len(o)==len(T); o['trial']=T.trial.values; o['cluster']=T.cluster.values; return o
# ---------- (a) reference twelve
CATS=['airplane','bench','cabinet','car','chair','display','lamp','loudspeaker','sofa','table','telephone','watercraft']
R={c:load(f'ref_{c}') for c in CATS}; pre=R['chair'].pretrained_oddity_margin
for c in CATS: assert np.allclose(R[c].pretrained_oddity_margin,pre,atol=1e-5)
ref=pd.DataFrame({c:R[c].fine_tuned_oddity_margin for c in CATS}); ref['pre']=pre
print('(a) reference twelve on 882 bank-built chair trials — mean margin:'); print(ref.mean().round(3).sort_values(ascending=False).to_string())
print(f'    pretrained accuracy {R["chair"].pretrained_correct.mean():.3f}; chair model {R["chair"].fine_tuned_correct.mean():.3f}; airplane model {R["airplane"].fine_tuned_correct.mean():.3f}')
# category distance of each model's training set to chairs (voxel16 knn10, from the MOCHI-era per-category table if present) — rank by mean margin instead
# ---------- (b) cluster models
Ns=['all','n50','n25']; M={}
for c in range(3):
    for n in Ns: M[(c,n)]=load(f'chair_c{c}_{n}')
full=load('chair_full')
print('\n(b) cluster models: mean margin on each cluster\'s held-out trials (rows = model trained on cluster, cols = test cluster)')
res=[]
for n in Ns:
    mat=np.zeros((3,3)); 
    for c in range(3):
        for t in range(3): mat[c,t]=M[(c,n)][M[(c,n)].cluster==t].fine_tuned_oddity_margin.mean()
    diag=np.diag(mat).mean(); off=mat[~np.eye(3,dtype=bool)].mean()
    print(f'  N={n}:'); print(pd.DataFrame(mat,index=[f'train c{c}' for c in range(3)],columns=[f'test c{t}' for t in range(3)]).round(3).to_string()); print(f'    diagonal {diag:.3f}  off-diagonal {off:.3f}  difference {diag-off:+.4f}')
    # per-trial: own-cluster model minus mean of the other two, then t-test over trials
    own=np.concatenate([M[(t,n)][M[(t,n)].cluster==t].fine_tuned_oddity_margin.values for t in range(3)])
    oth=np.concatenate([np.mean([M[(c,n)][M[(c,n)].cluster==t].fine_tuned_oddity_margin.values for c in range(3) if c!=t],axis=0) for t in range(3)])
    d=own-oth; print(f'    per trial own − others: {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f}, {100*(d>0).mean():.0f}% positive, n={len(d)}, p={stats.ttest_1samp(d,0).pvalue:.2g}')
    res.append((n,diag,off,d.mean(),d.std(ddof=1)/np.sqrt(len(d)),(d>0).mean()))
print(f'\n  full-bank model (all 2,000 chairs): per cluster {[round(full[full.cluster==t].fine_tuned_oddity_margin.mean(),3) for t in range(3)]}; pretrained per cluster {[round(pre[T.cluster==t].mean(),3) for t in range(3)]}')
pd.DataFrame(res,columns=['N','diag','off','own_minus_others','sem','frac_pos']).to_csv(f'{G}/out/cluster_summary.csv',index=False)
# ---------- fig67: per-trial centred: margin minus the trial's average over the three cluster models (same N)
fig,ax=plt.subplots(1,3,figsize=(15,6.6),sharey=True); fig.subplots_adjust(left=.06,right=.99,top=.68,bottom=.19,wspace=.1)
cols=['#2a78d6','#1baf7a','#eb6834']; names=['tall narrow-backed','wide armchair-like','round-backed / office']
for a,n in zip(ax,Ns):
    style(a); Y=np.stack([M[(c,n)].fine_tuned_oddity_margin.values for c in range(3)]); cl=T.cluster.values
    colm=np.array([Y[:,cl==t].mean() for t in range(3)]); Yi=Y-Y.mean(1,keepdims=True)-colm[cl][None,:]+Y.mean()   # remove model level and cluster difficulty
    mat=np.array([[Y[c][cl==t].mean() for t in range(3)] for c in range(3)]); I=mat-mat.mean(1,keepdims=True)-mat.mean(0,keepdims=True)+mat.mean()
    import itertools; dg=np.trace(I)/3; null=sorted(np.trace(I[list(q),:])/3 for q in itertools.permutations(range(3)))
    for c in range(3):
        rel=Yi[c]
        y=[rel[T.cluster.values==t].mean() for t in range(3)]; e=[rel[T.cluster.values==t].std(ddof=1)/np.sqrt((T.cluster.values==t).sum()) for t in range(3)]
        a.errorbar([0,1,2],y,yerr=e,fmt='o-',color=cols[c],ms=9,mfc=cols[c],mec=SURF,mew=1.3,lw=2.4,ecolor='#d5d3ce',zorder=4,label=f'model trained on cluster {c+1} ({names[c]})')
        a.plot([c],[y[c]],'o',ms=16,mfc='none',mec=cols[c],mew=2,zorder=5)
    a.axhline(0,color=GREY,ls='--',lw=2.2,zorder=3,label='zero: no specialisation')
    nn={'all':'all of its half (87–104 chairs)','n50':'50 chairs','n25':'25 chairs'}[n]
    a.set_title(f'each model trained on {nn}',loc='left',fontsize=11); a.set_xticks([0,1,2]); a.set_xticklabels(['cluster 1\ntrials','cluster 2\ntrials','cluster 3\ntrials'])
    good=dg>=null[-1]-1e-12
    box(a,f'own-cluster advantage (diagonal): {dg:+.3f}\n{"✓" if good else "!"}  the largest of the 6 ways of pairing models with clusters\n    (next best {null[-2]:+.3f}; chance ≈ 0)',OK if good else BAD,'ll')
ax[0].set_ylabel('oddity margin, with each model\'s overall level\nand each cluster\'s difficulty removed')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.06,.79),ncol=2,fontsize=9.5,frameon=False)
fig.suptitle('Sub-categories inside "chair": does a model trained on one cluster do best on that cluster?',fontsize=15,x=.06,ha='left',y=.975)
fig.text(.06,.92,'Three clusters of chairs (tall narrow-backed / wide armchair-like / round-backed & office), each split in half. One model per cluster is fine-tuned on its training half; 882 hard\n'
 'oddity trials are built from the held-out halves; every model is scored on every cluster. Each model\'s overall level (some models are better) and each cluster\'s difficulty (some clusters\n'
 'are easier) are removed, leaving only the pairing. '
 'What to expect if the margin depends on which chairs the model saw: each line above zero at its own cluster (circled), below elsewhere.\n'
 'Left to right: fewer training chairs per model.',fontsize=9.8,color=INK2,va='top')
fig.text(.06,.015,'The pairing matters at every training-set size, and more as the set shrinks (+0.009 → +0.012 → +0.015): fewer chairs, more specialised. Two of the three models specialise clearly (tall\n'
 'narrow-backed, wide armchair-like); the round-backed/office model hardly does — it is the most "typical chair" cluster and helps everywhere. The category effect one level down, small but there.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig67_cluster_matrix.png',dpi=300); print('[fig] 67')
# ---------- fig68: reference twelve rank curve on bank trials
fig,a=plt.subplots(figsize=(11,5.6)); fig.subplots_adjust(left=.08,right=.98,top=.72,bottom=.2); style(a)
order=ref[CATS].mean().sort_values(ascending=False); x=np.arange(12)
a.errorbar(x,order.values,yerr=[ref[c].sem() for c in order.index],fmt='o-',color=BLUE,ms=9,mfc=BLUE,mec=SURF,lw=2.4,ecolor='#bcd0ea',zorder=4,label='fine-tuned on that category')
a.axhline(pre.mean(),color=GREY,ls='--',lw=2.2,label='pretrained model (same on every trial)'); a.set_xticks(x); a.set_xticklabels(order.index,rotation=30,ha='right'); a.set_ylabel('mean margin on 882 chair trials'); a.set_xlabel('the twelve original category models, ordered by margin')
box(a,f'chair model {order.iloc[0]:.3f}  ·  next best {order.iloc[1]:.3f} ({order.index[1]})  ·  pretrained {pre.mean():.3f}\n✓  the model trained on chairs wins on chair trials by a wide margin',OK,'ur')
a.legend(loc='center right',fontsize=9.5,frameon=False)
fig.suptitle('The original twelve category models on bank-built chair trials',fontsize=15,x=.08,ha='left',y=.975)
fig.text(.08,.9,'Check of the new test set (882 hard oddity trials built from held-out bank chairs) using models we already had. Expect: the chair model far above the rest; the pretrained\n'
 'model flat. The chair model has seen these objects in training, so its height alone could be memorisation; the ordering of the other eleven cannot be.',fontsize=9.8,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig68_ref_on_bank.png',dpi=300); print('[fig] 68')
