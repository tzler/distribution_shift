"""Models trained on different subsets; x = distance from the trial to THAT model's training objects (best measures);
y = what training added (margin − pretrained margin on the same trial). One curve per model. Held-out chair trials."""
import json, ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G
SURF,INK,INK2,GREY='#fcfcfb','#0b0b0b','#52514e','#8a8884'; COLS=['#2a78d6','#1baf7a','#eb6834']; NAMES=['tall narrow-backed','wide armchair-like','round-backed / office']
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':11.5})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def norm(X): mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd; return Xn/(np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12)
T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv'); chair=np.where(T.dataset=='chair')[0]; Tc=T.iloc[chair]
objs=[sorted({'03001627/'+n[len('chair_'):-8] for n in ast.literal_eval(s)}) for s in Tc.images]; imgs=[[ '03001627/'+n[len('chair_'):-8] for n in ast.literal_eval(s)] for s in Tc.images]
D=json.load(open(f'{K}/design_clusters_all.json'))['chair']['split']; train={c:[f'03001627/{o}' for o in D[str(c)]['train'][:25]] for c in range(3)}
need=set(o for l in objs for o in l)|set(o for l in train.values() for o in l)
# features
b=np.load(f'{G}/bank/bank_bbox.npz',allow_pickle=True); Fb=dict(zip(b['ids'],norm(b['X'].astype(np.float32))))
z=np.load(f'{K}/eval/encoder_features/pretrained.npz',allow_pickle=True); acc={}
for i,x in zip(z['train_ids'],z['train_X']):
    k='03001627/'+i.split('/')[1] if i.startswith('chair/') else None
    if k in need: acc.setdefault(k,[]).append(x)
keys=list(acc); Fd=dict(zip(keys,norm(np.stack([np.mean(acc[k],axis=0) for k in keys]))))
def dist(F,c): S=np.stack([F[o] for o in train[c]]); return np.array([np.mean([np.sort(1-F[o]@S.T)[0] for o in l]) for l in objs])   # nearest, mean over the trial's objects
def load(n): o=pd.read_csv(f'{K}/eval_bank_all/{n}/ood_analysis_results.csv').iloc[chair]; return o.fine_tuned_oddity_margin.values, o.pretrained_oddity_margin.values
fig,ax=plt.subplots(2,2,figsize=(14,10.2)); fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.07,hspace=.42,wspace=.22)
for i,(Fname,F) in enumerate([('pretrained DINOv2 features, nearest',Fd),('bounding box (7 numbers), nearest',Fb)]):
    for j,(rung,suffix) in enumerate([('gentle fine-tune (LoRA, lr 10⁻⁶)',''),('full fine-tune (lr 10⁻⁵)','_fullft')]):
        a=ax[i,j]; style(a); allx=[]; ally=[]
        for c in range(3):
            ft,pre=load(f'chair_c{c}_n25{suffix}'); x=dist(F,c); y=ft-pre; allx.append(x); ally.append(y)
            q=pd.qcut(pd.Series(x).rank(method='first'),10,labels=False); g=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q')
            a.errorbar(g.x.mean(),g.y.mean(),yerr=g.y.sem(),fmt='o-',color=COLS[c],ms=7,mec=SURF,lw=2,ecolor='#d5d3ce',zorder=4,label=f'model trained on 25 {NAMES[c]} chairs')
        X=np.concatenate(allx); Y=np.concatenate(ally); r=stats.pearsonr(X,Y)[0]
        # within-trial r (the controlled version) for the box
        L=pd.DataFrame({'t':np.tile(np.arange(len(objs)),3),'x':X,'y':Y}); gg=L.groupby('t'); rw=stats.pearsonr(L.x-gg.x.transform('mean'),L.y-gg.y.transform('mean'))[0]
        a.axhline(0,color=GREY,ls='--',lw=1.8,zorder=3); a.set_title(f'{rung}   ·   x: {Fname}',loc='left'); a.set_xlabel('distance from the trial\'s objects to that model\'s 25 training chairs   farther →'); a.set_ylabel('what training added to the margin\n(fine-tuned − pretrained, same trial)')
        a.text(0.98,0.96,f'pooled r = {r:+.2f}     within-trial r = {rw:+.2f}',transform=a.transAxes,fontsize=9.6,va='top',ha='right',color=INK2)
        if i==0 and j==0: a.legend(fontsize=8.8,frameon=False,loc='lower left')
fig.suptitle('Three models, three training sets: does the margin follow distance along one continuum?',fontsize=15,x=.07,ha='left',y=.975)
fig.text(.07,.925,'882 held-out chair trials. Each model was fine-tuned on 25 chairs of one kind. For every trial × model, x is how far the trial\'s objects are from the 25 that model saw, in the two\n'
 'measures the search and the transfer test picked out; y is what fine-tuning added to that trial\'s margin (the pretrained model scored the same trial, so its level is removed).\n'
 'What to expect if the measure tracks shift: the three curves fall, and lie on one continuum — a trial at a given distance gets the same gain whichever model it comes from.\n'
 'Top row: the gently fine-tuned models. Bottom row: the full-fine-tune rung, where more of the prior is overwritten and the effect of the training data is larger.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig81_gradient.png',dpi=300); print('[fig] 81')
