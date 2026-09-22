"""No normalisation: the margin itself against distance to the model's training objects, fine-tuned (blue) and pretrained
(grey) on the same trials. Left: every model on every trial. Right: only the models of the trial's own category."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

G=_RESOLVED_G
BLUE,GREY,SURF,INK,INK2='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def bins(x,y,nb=30):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
def line(a,x,y,e,c,ls,lab,z): a.errorbar(x,y,yerr=e,fmt='o'+ls,color=c,ms=6,mfc=c,mec=SURF,mew=1.1,lw=1.8,ecolor='#d5d3ce',elinewidth=1.4,zorder=z,label=lab)
L=pd.read_csv(f'{G}/out/all_categories_long.csv' if __import__('os').path.exists(f'{G}/out/all_categories_long.csv') else f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
fig,ax=plt.subplots(1,2,figsize=(14,6.2)); fig.subplots_adjust(left=.07,right=.98,top=.74,bottom=.16,wspace=.22)
for a,(d,t) in zip(ax,[(L,f'every model on every trial   ({len(L):,} points)'),(L[L.train_cat==L.test_cat],f'only the models of the trial\'s own category   ({(L.train_cat==L.test_cat).sum():,} points)')]):
    style(a); bx,bf,ef=bins(d.dist,d.ft); _,bp,ep=bins(d.dist,d.pre)
    line(a,bx,bp,ep,GREY,'--','pretrained model, same trials',3); line(a,bx,bf,ef,BLUE,'-','model fine-tuned on those 25 objects',4)
    a.set_title(t,loc='left',fontsize=11); a.set_xlabel('distance from the trial\'s objects to the model\'s 25 training objects  (3-D shape)   farther →'); a.set_ylabel('oddity margin')
    a.text(0.98,0.96,f'fine-tuned r = {stats.pearsonr(d.dist,d.ft)[0]:+.2f}   pretrained r = {stats.pearsonr(d.dist,d.pre)[0]:+.2f}\ngap (fine-tuned − pretrained): nearest bin {bf.iloc[0]-bp.iloc[0]:+.3f}, farthest bin {bf.iloc[-1]-bp.iloc[-1]:+.3f}',transform=a.transAxes,fontsize=9.4,va='top',ha='right',color=INK2)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.07,.82),ncol=2,fontsize=10,frameon=False)
fig.suptitle('The margin against distance to the training objects — nothing normalised',fontsize=15,x=.07,ha='left',y=.975)
fig.text(.07,.92,'34 models, each fine-tuned on 25 objects of one kind of one category; 11,634 held-out trials. Each point is a trial scored by a model; x is how far the trial\'s objects are\n'
 'from the 25 that model saw. Both models score the same trials, so the gap between blue and grey is what fine-tuning on those 25 objects added — read the gap, not the slope.',fontsize=9.8,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig77_raw_margin_vs_distance.png',dpi=300); print('[fig] 77')
