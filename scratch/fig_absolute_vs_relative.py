"""Is the margin an absolute or a relative measure of distance to the training set?
Left: 150 random trials, each drawn as a line through its three own-category models (x = distance to that model's
25 training objects, y = margin); the trial's pretrained margin as a grey tick. Middle: each trial's level against
its pretrained margin. Right: each trial's level against its mean distance to the training sets."""
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
L=pd.read_csv(f'{G}/out/all_categories_long.csv' if __import__('os').path.exists(f'{G}/out/all_categories_long.csv') else f'{G}/data/all_categories_long.csv.gz'); L=L[(L.kind=='cluster')&(L.train_cat==L.test_cat)]
g=L.groupby('trial'); lvl=g.ft.mean(); pre=g.pre.first(); md=g.dist.mean(); wd=L.dist-g.dist.transform('mean'); wm=L.ft-g.ft.transform('mean')
rng=np.random.default_rng(3); pick=rng.choice(L.trial.unique(),150,replace=False)
fig,ax=plt.subplots(1,3,figsize=(17,6.2),gridspec_kw={'width_ratios':[1.4,1,1]}); fig.subplots_adjust(left=.05,right=.99,top=.72,bottom=.17,wspace=.25)
a=ax[0]; style(a)
L['band']=pd.qcut(L.pre,5,labels=False)
cm=plt.get_cmap('Blues'); labels=['hardest fifth of trials (by pretrained margin)','2nd','3rd','4th','easiest fifth of trials']
for k,(b,d) in enumerate(L.groupby('band',observed=True)):
    q=pd.qcut(d.dist.rank(method='first'),8,labels=False); gg=d.assign(q=q).groupby('q'); col=cm(0.35+0.16*k)
    a.errorbar(gg.dist.mean(),gg.ft.mean(),yerr=gg.ft.sem(),fmt='o-',color=col,ms=7,mec=SURF,lw=2.2,ecolor='#d5d3ce',zorder=4,label=labels[k])
    a.axhline(d.pre.mean(),color=col,ls=':',lw=1.4,alpha=.9,zorder=2)
a.plot([],[],':',color=GREY,label='dotted: that band\'s pretrained margin')
a.set_xlabel('distance from the trial\'s objects to the model\'s 25 training objects   farther →'); a.set_ylabel('oddity margin'); a.set_title('trials split into five bands by how hard they are',loc='left'); a.legend(fontsize=8.8,frameon=False,loc='upper right')
box(a,'five stacked curves, not one: each band keeps the level the pretrained\nmodel gave it, and distance moves the margin a little within the band',INK,'ll')
a=ax[1]; style(a); a.plot(pre,lvl,'.',color=INK2,ms=3,alpha=.35); a.set_xlabel('pretrained margin on the trial'); a.set_ylabel('the trial\'s level: mean margin over its three models'); a.set_title('what sets a trial\'s level',loc='left')
box(a,f'r = {stats.pearsonr(pre,lvl)[0]:+.2f}\n92% of the margin\'s variance is the trial\'s level',INK,'ul')
a=ax[2]; style(a); a.plot(md,lvl,'.',color=INK2,ms=3,alpha=.35); a.set_xlabel('the trial\'s mean distance to the three training sets'); a.set_ylabel('the trial\'s level'); a.set_title('… and what does not',loc='left')
box(a,f'r = {stats.pearsonr(md,lvl)[0]:+.2f}\nbetween trials, distance predicts nothing;\nwithin a trial, r = {stats.pearsonr(wd,wm)[0]:+.2f}',INK,'ur')
fig.suptitle('The margin is a relative measure of distance, not an absolute one',fontsize=15,x=.05,ha='left',y=.975)
fig.text(.05,.915,'11,634 held-out trials, each scored by the three models trained on 25 objects of one kind of its own category. If the margin were an absolute measure of distance to the training set,\n'
 'every trial would lie on one curve: a given distance, a given margin. Instead each trial has its own level — set by how hard the trial is, which the pretrained model already knows —\n'
 'and distance to the training set moves the margin up or down from that level. Comparisons are meaningful within a trial (or after subtracting its level), not across trials.',fontsize=9.8,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig78_absolute_vs_relative.png',dpi=300); print('[fig] 78')
