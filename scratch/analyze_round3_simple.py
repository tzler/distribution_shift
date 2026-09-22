"""The analyses that worked in state 3, run on the round-3 models. Every held-out bank trial is scored by every model
(each trained on 25 objects of one kind of one category). x = distance from the trial's objects to that model's 25
training objects (3-D shape); y = margin. Left: as measured. Middle: each trial's mean over the models subtracted
(the pretrained model is then flat by construction). Right: rank each trial's training sets nearest → farthest."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

G=_RESOLVED_G
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
def bins(x,y,nb=12):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
def line(a,x,y,e,c,ls,lab,z): a.errorbar(x,y,yerr=e,fmt='o'+ls,color=c,ms=8,mfc=c,mec=SURF,mew=1.3,lw=2.4,ecolor='#d5d3ce',elinewidth=1.4,zorder=z,label=lab)
L=pd.read_csv(f'{G}/out/all_categories_long.csv' if __import__('os').path.exists(f'{G}/out/all_categories_long.csv') else f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster'].copy()
M=L.model.nunique(); ncat=L.train_cat.nunique(); print(f'{M} models from {ncat} categories × {L.trial.nunique():,} trials = {len(L):,} points')
g=L.groupby('trial'); L['dc']=L.dist-g.dist.transform('mean'); L['mc']=L.ft-g.ft.transform('mean'); L['pc']=L.pre-g.pre.transform('mean'); L['rank']=g.dist.rank(method='first')
assert L.pc.abs().max()<1e-9
r_pool_ft,r_pool_pre=stats.pearsonr(L.dist,L.ft)[0],stats.pearsonr(L.dist,L.pre)[0]; r_wt=stats.pearsonr(L.dc,L.mc)[0]
sl=L.groupby('trial').apply(lambda x: np.polyfit(x.dc,x.mc,1)[0] if x.dc.std()>0 else np.nan); pneg=(sl<0).mean()
rng=np.random.default_rng(0); obs=r_wt; null=[stats.pearsonr(L.dc,g.mc.transform(lambda x: rng.permutation(x.values)))[0] for _ in range(50)]
print(f'pooled: fine-tuned r = {r_pool_ft:+.3f}, pretrained r = {r_pool_pre:+.3f} | within-trial r = {r_wt:+.3f}, {100*pneg:.0f}% of trials slope down, permutation null sd {np.std(null):.4f}')
fig,ax=plt.subplots(1,3,figsize=(17,6.4)); fig.subplots_adjust(left=.05,right=.99,top=.66,bottom=.18,wspace=.25)
a=ax[0]; style(a); bx,bf,ef=bins(L.dist,L.ft); _,bp,ep=bins(L.dist,L.pre)
line(a,bx,bp,ep,GREY,'--','pretrained model — saw none of the training sets',3); line(a,bx,bf,ef,BLUE,'-','model fine-tuned on that training set',4)
a.set_title(f'all trials × all models, as measured\n{len(L):,} points',loc='left',fontsize=11); a.set_xlabel('distance from the trial\'s objects to the model\'s 25 training objects\n(3-D shape)   farther →'); a.set_ylabel('oddity margin')
box(a,f'fine-tuned r = {r_pool_ft:+.2f}     pretrained r = {r_pool_pre:+.2f}\n!  grey falls too: far-from-everything objects are hard\n    for every model — not yet about the training set',BAD,'ur')
a=ax[1]; style(a); bx,bf,ef=bins(L.dc,L.mc); _,bp,ep=bins(L.dc,L.pc)
line(a,bx,bp,ep,GREY,'--','pretrained model',3); line(a,bx,bf,ef,BLUE,'-','fine-tuned model',4)
a.set_title('the same points, each trial\'s own average subtracted\nso only the training set differs between a trial\'s points',loc='left',fontsize=11); a.set_xlabel('distance to the model\'s training objects,\nrelative to the trial\'s average   farther →'); a.set_ylabel('oddity margin, relative to the trial\'s average')
box(a,f'fine-tuned r = {r_wt:+.2f}     pretrained r = +0.00\n✓  grey is flat by construction; blue falls:\n    {100*pneg:.0f}% of trials slope down (null sd {np.std(null):.3f})',OK,'ur')
a=ax[2]; style(a); rk=L.groupby('rank').agg(m=('mc','mean'),e=('mc','sem')); rp=L.groupby('rank').agg(m=('pc','mean'),e=('pc','sem'))
line(a,rp.index,rp.m,rp.e,GREY,'--','pretrained model',3); line(a,rk.index,rk.m,rk.e,BLUE,'-','fine-tuned model',4)
a.set_title('rank each trial\'s training sets, nearest → farthest',loc='left',fontsize=11); a.set_xlabel(f'rank of the model\'s training set for this trial  (1 = nearest of {M})'); a.set_ylabel('oddity margin, relative to the trial\'s average')
box(a,f'nearest: {rk.m.iloc[0]:+.3f}   farthest: {rk.m.iloc[-1]:+.3f}',INK,'ur')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.05,.775),ncol=2,fontsize=10,frameon=False)
fig.suptitle('Moving the training data moves the margin — models trained on 25 objects of one kind',fontsize=15,x=.05,ha='left',y=.975)
fig.text(.05,.92,f'{M} models, each fine-tuned on 25 objects of one kind of one category ({ncat} categories so far), every one scored on all {L.trial.nunique():,} held-out bank trials from twelve categories.\n'
 'For each trial × model we measure how far the trial\'s objects are from the 25 the model saw (3-D shape, no network). Same three panels as the twelve-category result:\n'
 'what to expect if the margin tracks the training set — BLUE falls; GREY, which saw none of the training sets, is flat once the trial is held fixed.',fontsize=9.8,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig76_round3_moving_training.png',dpi=300); print('[fig] 76')
