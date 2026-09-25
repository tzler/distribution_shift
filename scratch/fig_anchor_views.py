"""Three views of the anchor result.
A · the staircase: margin by how related the training set is to the trial's objects.
B · the transfer matrix: every model scored on every model's training objects.
C · the continuum with its left end: memorised objects joined to held-out trials on one distance axis."""
from _repo import G, K
import json, ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE,GREY,SURF,INK,INK2,OK,ORA='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':11.5})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
A=pd.read_csv(f'{G}/data/anchor_long.csv.gz')
T=pd.read_csv(f'{K}/banktrials/traintrials_all.csv'); A['test_cat']=T.dataset.values[A.trial.values]
A['same_cluster']=A.is_owner; A['rel4']=np.where(A.is_owner,'owner',np.where(A.same_cat,'same category,\nother kind','another category'))
pre=A.groupby('trial').pre.first().mean()
# ---------- A: staircase
fig,ax=plt.subplots(1,3,figsize=(18,6.8),gridspec_kw={'width_ratios':[1,1.2,1.3]}); fig.subplots_adjust(left=.055,right=.985,top=.68,bottom=.20,wspace=.42)
a=ax[0]; style(a)
groups=[('the model that trained\non these very objects',A[A.is_owner],ORA),('another model of the\nsame category',A[~A.is_owner&A.same_cat],BLUE),('a model of another\ncategory',A[~A.same_cat],GREY)]
for i,(lab,sub,c) in enumerate(groups):
    a.bar(i,sub.ft.mean(),yerr=sub.ft.sem(),width=.62,color=c,ecolor=INK2,capsize=4,zorder=4)
    a.text(i,sub.ft.mean()+0.006,f'{sub.ft.mean():+.3f}',ha='center',fontsize=10,color=INK)
a.axhline(pre,color=INK,ls='--',lw=1.8,zorder=5); a.text(-0.45,pre+0.004,f'the pretrained model, which saw none of them:  {pre:+.3f}',ha='left',fontsize=9,color=INK2)
a.set_xticks(range(3)); a.set_xticklabels([g[0] for g in groups],fontsize=8.6); a.set_ylabel('oddity margin on the trial'); a.set_title('A · what training on an object is worth',loc='left')
a.set_ylim(0,0.165); a.text(0.02,0.99,'a staircase, not a step: the object itself is worth most,\nits kind next, its category next',transform=a.transAxes,fontsize=9.4,va='top',color=INK2)
# ---------- B: transfer matrix, 35 models x 35 training sets (mean margin), ordered by category
a=ax[1]; models=sorted(A.model.unique()); owner_of=A.groupby('trial').owner.first()
M=A.pivot_table(index='model',columns=A.trial.map(owner_of),values='ft',aggfunc='mean').reindex(index=models,columns=models)
V=M.values; Z=V-np.nanmean(V,axis=1,keepdims=True)-np.nanmean(V,axis=0,keepdims=True)+np.nanmean(V)   # model level and trial-set difficulty removed
im=a.imshow(Z,cmap='RdBu_r',vmin=-0.05,vmax=0.05,aspect='auto')
cats=[m.split('_')[0] for m in models]; bounds=[i for i in range(1,len(cats)) if cats[i]!=cats[i-1]]
for b in bounds: a.axhline(b-.5,color=SURF,lw=.8); a.axvline(b-.5,color=SURF,lw=.8)
ticks=[(bounds+[len(cats)])[i]-( (bounds+[len(cats)])[i]-([0]+bounds)[i])/2 for i in range(len(bounds)+1)]
a.set_xticks(ticks); a.set_xticklabels(sorted(set(cats)),rotation=45,ha='right',fontsize=8); a.set_yticks(ticks); a.set_yticklabels(sorted(set(cats)),fontsize=8)
a.set_xlabel('trials built from THIS model\'s training objects'); a.set_ylabel('margin given by THIS model'); a.set_title('B · every model on every training set',loc='left')
cb=plt.colorbar(im,ax=a,fraction=.042,pad=.02); cb.set_label('margin, model level and set difficulty removed',fontsize=8.5)
a.text(0.0,-0.30,'the diagonal is each model on the objects it memorised; the blocks are categories',transform=a.transAxes,fontsize=9.4,color=INK2)
d=np.diag(Z); off=Z[~np.eye(len(Z),dtype=bool)]
a.text(0.0,-0.38,f'diagonal {np.nanmean(d):+.3f} vs off-diagonal {np.nanmean(off):+.3f}',transform=a.transAxes,fontsize=9.4,color=INK2)
# ---------- C: the continuum, memorised objects joined to held-out trials
a=ax[2]; style(a)
H=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); H=H[(H.kind=='cluster')&(H.train_cat==H.test_cat)]
hg=H.groupby('trial'); hrel=H.ft-hg.ft.transform('mean')
q=pd.qcut(H.dist.rank(method='first'),10,labels=False); hb=H.assign(q=q,rel=hrel).groupby('q')
ag=A.groupby('trial'); arel=A.ft-ag.ft.transform('mean'); own=A[A.is_owner]
qa=pd.qcut(A[~A.is_owner].dinov2.rank(method='first'),6,labels=False); ab=A[~A.is_owner].assign(q=qa,rel=arel[~A.is_owner]).groupby('q')
a.errorbar(ab.dinov2.mean(),ab.rel.mean(),yerr=ab.rel.sem(),fmt='o-',color=BLUE,ms=7,mec=SURF,lw=2,ecolor='#bcd0ea',zorder=4,label='trials built from TRAINING objects, scored by other models')
a.errorbar([own.dinov2.mean()],[arel[A.is_owner].mean()],yerr=[arel[A.is_owner].sem()],fmt='o',color=ORA,ms=15,mec=SURF,mew=1.5,ecolor=ORA,zorder=6,label='… scored by the model that trained on them')

a.set_xlabel('distance from the trial\'s objects to the model\'s training objects\n(frozen pretrained network, nearest)   farther →')
a.set_ylabel('oddity margin,\nrelative to the trial\'s average'); a.set_title('C · distance orders them, with the anchor at zero',loc='left'); a.legend(fontsize=8.8,frameon=False,loc='upper right')
r=stats.pearsonr(A.dinov2,arel)[0]
a.text(0.02,0.04,f'r = {r:+.2f} across all 94,500 points\nthe anchor sits where a distance measure says it must',transform=a.transAxes,fontsize=9.4,va='bottom',color=INK2)
fig.suptitle('Trials built from the objects a model was trained on: three views',fontsize=15,x=.055,ha='left',y=.975)
fig.text(.055,.915,'2,700 oddity trials made from the 25 training objects of each of 35 models, every model scored on all of them. This is the one condition where the distance from a trial to a\n'
 'model\'s training set is known to be zero, so it says what "trained on it" is worth in margin, and it is the left-hand end of the curve every other figure in this project measures.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig83_anchor_views.png',dpi=300); print('[fig] 83')
