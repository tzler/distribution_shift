"""Knock-in, random arm: 8 models each fine-tuned on a different random set of 100 chairs.
For every chair trial: does the model whose 100 chairs sit closer to the trial's objects give a bigger margin?
Within-trial comparison across the 8 models (the trial is fixed; only the 100 chairs differ)."""
import json, ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); tz=np.load(f'{G}/bank/test_voxel16.npz',allow_pickle=True)
B,T=bz['X'].astype(float),tz['X'].astype(float); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12; TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-12
bids=np.array(bz['ids']); tids=np.array(tz['ids']); tix={k:i for i,k in enumerate(tids)}; bix={k.split('/')[-1]:i for i,k in enumerate(bids) if k.startswith('03001627/')}
kd=json.load(open(f'{K}/design_knockin.json')); dk=json.load(open(f'{K}/design.json')); m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mi=m.set_index('trial')
def load(path):
    o=pd.read_csv(path); o['trial']=m['trial'].values; return o.set_index('trial')
trials=dk['chair']['trials']; rows=[]
for i in range(8):
    subset=kd['random'][f'random_100_{i}']; Bs=BN[[bix[o] for o in subset]]
    mo=load(f'{K}/eval/chair_random_100_{i}/ood_analysis_results.csv')
    for t in trials:
        objs=['03001627/'+f[:-4].split('_')[1] for f in ast.literal_eval(mi.loc[t,'images'])]
        Tt=TN[[tix[o] for o in set(objs)]]; D=1-Tt@Bs.T
        knn=np.sort(D,axis=1)[:,:10].mean(1).mean()        # mean over the trial's objects of distance to the 10 nearest of the 100
        nn1=np.sort(D,axis=1)[:,0].mean()                  # nearest single one
        cov=np.log1p((D<0.12).sum(1)).mean()               # how many of the 100 within the radius (log count)
        rows.append((i,t,knn,nn1,cov,mo.fine_tuned_oddity_margin[t],mo.pretrained_oddity_margin[t]))
L=pd.DataFrame(rows,columns=['model','trial','knn10','nn1','cov','margin','pre']); L.to_csv(f'{G}/out/knockin_random_long.csv',index=False)
g=L.groupby('trial')
for k in ['knn10','nn1','cov','margin']: L[k+'_c']=L[k]-g[k].transform('mean')
print('within-trial (8 models per trial, 76 trials):')
for k in ['knn10','nn1','cov']:
    r=stats.pearsonr(L[k+'_c'],L['margin_c'])[0]
    sl=L.groupby('trial').apply(lambda x: np.polyfit(x[k+'_c'],x['margin_c'],1)[0] if x[k+'_c'].std()>0 else np.nan)
    print(f'  {k:6s}: r = {r:+.3f}   per-trial slopes {100*(sl<0).mean():.0f}% negative (n={sl.notna().sum()}), mean {sl.mean():+.4f}, p={stats.ttest_1samp(sl.dropna(),0).pvalue:.2g}')
# permutation: shuffle model labels within trial
rng=np.random.default_rng(0); obs=stats.pearsonr(L.knn10_c,L.margin_c)[0]; null=[]
for _ in range(2000):
    s=L.groupby('trial').margin_c.transform(lambda x: rng.permutation(x.values)); null.append(stats.pearsonr(L.knn10_c,s)[0])
print(f'  permutation (shuffle within trial): observed r {obs:+.3f}, null sd {np.std(null):.3f}, p = {(np.sum(np.array(null)<=obs)+1)/2001:.3g}')
# how much does the 100-subset distance vary within a trial?
print(f'  within-trial spread of knn10 across the 8 subsets: mean sd {g.knn10.std().mean():.3f} (between-trial sd {g.knn10.mean().std():.3f}); margin sd {g.margin.std().mean():.4f}')
# fig65
fig,ax=plt.subplots(1,2,figsize=(14,7.2)); fig.subplots_adjust(left=.07,right=.98,top=.70,bottom=.16,wspace=.25)
def bins(x,y,nb=8):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
for a,(xk,t,dsc) in zip(ax,[('knn10','as measured','608 points: 76 chair trials × 8 models'),('knn10_c','each trial\'s own average subtracted','so only which 100 chairs the model saw differs')]):
    style(a); yk='margin' if xk=='knn10' else 'margin_c'; pk='pre' if xk=='knn10' else None
    bx,bf,ef=bins(L[xk],L[yk]); a.errorbar(bx,bf,yerr=ef,fmt='o-',color=BLUE,ms=9,mfc=BLUE,mec=SURF,mew=1.4,lw=2.6,ecolor='#bcd0ea',elinewidth=1.6,zorder=4,label='model fine-tuned on those 100 chairs')
    if pk: _,bp,ep=bins(L[xk],L[pk]); a.errorbar(bx,bp,yerr=ep,fmt='o--',color=GREY,ms=8,mfc=GREY,mec=SURF,mew=1.3,lw=2.2,ecolor='#d5d3ce',elinewidth=1.4,zorder=3,label='pretrained model — saw none of them')
    else: a.axhline(0,color=GREY,ls='--',lw=2.2,zorder=3,label='pretrained model (flat: same model on all 8 points)')
    rf=stats.pearsonr(L[xk],L[yk])[0]; a.set_title(f'{t}\n{dsc}',loc='left',fontsize=11)
    a.set_xlabel('distance from the trial\'s objects to the 10 nearest of the model\'s 100 chairs\n(3-D shape'+(')' if xk=='knn10' else ', relative to the trial\'s average)')); a.set_ylabel('oddity margin'+('' if xk=='knn10' else ', relative to the trial\'s average'))
    if xk=='knn10': rp=stats.pearsonr(L[xk],L['pre'])[0]; box(a,f'fine-tuned r = {rf:+.2f}     pretrained r = {rp:+.2f}\n!  RISES for both: within chairs, an unusual chair is an\n    easier trial for every model. Trial difficulty, not training.',BAD,'ll')
    else: box(a,f'fine-tuned r = {rf:+.2f}   permutation p = 0.011\n✓  falls, weakly: the closer 100 chairs give the bigger\n    margin in 66% of trials (p = 0.003). Small — about\n    0.01 across the range — but the first within-category effect',OK,'ur')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.07,.80),ncol=2,fontsize=10,frameon=False)
fig.suptitle('Eight models, each trained on a different random 100 chairs: does the closer set give the bigger margin?',fontsize=15,x=.07,ha='left',y=.975)
fig.text(.07,.92,'Starting from the pretrained model, eight models were fine-tuned on eight random sets of 100 chairs. Every chair trial is scored by all eight. For each trial × model we measure how\n'
 'far the trial\'s objects are from that model\'s 100 chairs (3-D shape, no network). What to expect if the margin depends on nearby training data within a category: after subtracting each\n'
 'trial\'s own average (right), BLUE should fall — the model whose chairs sit closer to this trial should give it the bigger margin.',fontsize=9.8,color=INK2,va='top')
fig.text(.07,.015,'Left: as measured, the margin rises with distance for the fine-tuned and the pretrained model alike — inside a category, the unusual objects are the easy trials. Right: with the trial\'s own\naverage removed, the model whose 100 chairs sit closer gives the bigger margin. A weak effect at the edge of the run-to-run noise (about ±0.02 per trial), but it is in the predicted direction and survives a permutation test.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig65_knockin_random.png',dpi=300); print('[fig] 65')
