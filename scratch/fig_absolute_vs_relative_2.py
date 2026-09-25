"""Absolute or relative? Fine bins and the spread, not just the mean."""
from _repo import G
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
BLUE,GREY,SURF,INK,INK2,OK,ORA='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':11.5})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
L=L.merge(pd.read_csv(f'{G}/data/dinov2_distance_long.csv.gz'),on=['model','trial'],how='left').dropna(subset=['dinov2'])
def prof(x,y,nb=40,lo=2,hi=98):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); d=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q')
    return d.x.mean(),d.y.mean(),d.y.sem(),d.y.quantile(lo/100),d.y.quantile(hi/100)
fig,ax=plt.subplots(1,3,figsize=(17.5,6.6)); fig.subplots_adjust(left=.055,right=.985,top=.68,bottom=.19,wspace=.26)
# A · everything, as measured, with the density
a=ax[0]; style(a)
hb=a.hexbin(L.dinov2,L.ft,gridsize=70,cmap='Blues',norm=LogNorm(vmin=1,vmax=3000),mincnt=1,zorder=2,linewidths=0)
bx,by,be,lo,hi=prof(L.dinov2,L.ft)
a.fill_between(bx,lo,hi,color=INK2,alpha=.12,zorder=3,lw=0); a.plot(bx,by,'-',color=INK,lw=2.4,zorder=5)
a.set_xlabel('distance from the trial\'s objects to the model\'s training objects\n(frozen pretrained network, nearest)   farther →'); a.set_ylabel('oddity margin')
a.set_title('A · all 395,556 trial × model pairs',loc='left')
a.text(0.98,0.97,f'one line for everything explains R² = {1-np.sum((L.ft-np.polyval(np.polyfit(L.dinov2,L.ft,1),L.dinov2))**2)/np.sum((L.ft-L.ft.mean())**2):.3f}\nof the margin.  shaded: the middle 96 % of trials',transform=a.transAxes,ha='right',va='top',fontsize=9.4,color=INK2)
plt.colorbar(hb,ax=a,fraction=.04,pad=.02,label='pairs per cell')
# B · one line per category
a=ax[1]; style(a); cm=plt.get_cmap('tab20')
cats=sorted(L.test_cat.unique())
for i,c in enumerate(cats):
    d=L[L.test_cat==c]; bx,by,be,_,_=prof(d.dinov2,d.ft,nb=16)
    a.plot(bx,by,'-',color=cm(i%20),lw=1.9,zorder=4,label=c)
a.axvspan(0.5,0.7,color='#f3e9d9',alpha=.6,lw=0,zorder=1)
band=L[(L.dinov2>0.5)&(L.dinov2<0.7)].groupby('test_cat').ft.mean()
a.annotate('',xy=(0.6,band.min()),xytext=(0.6,band.max()),arrowprops=dict(arrowstyle='<->',color=ORA,lw=2),zorder=6)
a.text(0.63,(band.min()+band.max())/2,f'{band.max()-band.min():.3f}\nat one\ndistance',fontsize=9,color=ORA,va='center')
a.set_xlim(0.15,1.3); a.set_xlabel('the same distance   farther →'); a.set_ylabel('oddity margin'); a.set_title('B · one line per test category',loc='left')
a.legend(fontsize=7.6,frameon=False,ncol=2,loc='upper right',handlelength=1.2,columnspacing=1)
a.text(0.02,0.03,f'every category sits at its own level. Moving across the whole range\nof distance is worth {abs(np.polyfit(L.dinov2,L.ft,1)[0])*0.9:.3f} of margin — a quarter of the spread between\ncategories at a single distance.',transform=a.transAxes,fontsize=9.2,va='bottom',color=INK2)
# C · the same trial, different training sets
a=ax[2]; style(a); g=L.groupby('trial'); xc=L.dinov2-g.dinov2.transform('mean'); yc=L.ft-g.ft.transform('mean')
hb2=a.hexbin(xc,yc,gridsize=70,cmap='Blues',norm=LogNorm(vmin=1,vmax=3000),mincnt=1,zorder=2,linewidths=0)
bx,by,be,lo,hi=prof(xc,yc)
a.fill_between(bx,lo,hi,color=INK2,alpha=.12,zorder=3,lw=0); a.plot(bx,by,'-',color=INK,lw=2.4,zorder=5); a.axhline(0,color=GREY,ls='--',lw=1.6,zorder=4)
a.set_xlabel('distance, relative to the trial\'s own average   farther →'); a.set_ylabel('margin, relative to the trial\'s own average'); a.set_title('C · the same trial, different training sets',loc='left')
a.text(0.98,0.97,f'r = {stats.pearsonr(xc,yc)[0]:+.2f};  the relationship that is stable\nis the one inside a trial',transform=a.transAxes,ha='right',va='top',fontsize=9.4,color=OK)
plt.colorbar(hb2,ax=a,fraction=.04,pad=.02,label='pairs per cell')
fig.suptitle('Is the margin an absolute measure of distribution shift, or a relative one?',fontsize=15,x=.055,ha='left',y=.975)
fig.text(.055,.915,'Absolute would mean one function: the same distance from a model\'s training set implies the same margin, whichever trial or category it comes from. Relative means the relationship\n'
 'only holds once each trial\'s (or category\'s) own level is taken out. 34 models trained on 25 objects each, 11,634 held-out trials, distance in the measure that won the search.\n'
 'What to expect if the margin were absolute: one tight falling band in A, the twelve lines in B lying on top of each other.',fontsize=9.6,color=INK2,va='top')
fig.text(.055,.015,'A: the cloud is wide and the single function explains almost none of the margin. B: every category has its own level — at a fixed distance the margin ranges over 0.084, while moving across the\n'
 'entire range of distance is worth 0.022. C: hold the trial fixed and the relationship is there and stable (r = −0.25). The margin measures shift relative to what that model already had, not on an absolute scale.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig84_absolute_vs_relative.png',dpi=300); print('[fig] 84')
