"""Why the as-measured margin RISES with distance to the training set on bank trials: the trials pair each object with a
near neighbour, and objects far from the training set live in sparse regions, so their neighbours are farther too."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _repo import G as _RESOLVED_G

G=_RESOLVED_G
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
def bins(x,y,nb=10):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
T=pd.read_csv(f'{G}/out/banktrials_dAB.csv')
def partial(x,y,z):
    rxy,rxz,ryz=[stats.pearsonr(a,b)[0] for a,b in [(x,y),(x,z),(y,z)]]; return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))
fig,ax=plt.subplots(1,3,figsize=(16,5.6)); fig.subplots_adjust(left=.06,right=.98,top=.7,bottom=.2,wspace=.28)
a=ax[0]; style(a); bx,by,be=bins(T.dist2000,T.dAB); a.errorbar(bx,by,yerr=be,fmt='o-',color=INK,ms=8,mec=SURF,lw=2.4,ecolor='#d5d3ce',zorder=4)
a.set_xlabel('distance from the trial\'s objects to the training set (all 2,000 chairs)   farther →'); a.set_ylabel('distance between the trial\'s two objects'); a.set_title('far from training = far from everything, including its own partner',loc='left',fontsize=10.5)
box(a,f'r = {stats.pearsonr(T.dist2000,T.dAB)[0]:+.2f}\nthe trial pairs each object with a near neighbour;\nin a sparse region the nearest neighbour is far',INK,'ul')
a=ax[1]; style(a); bx,by,be=bins(T.dAB,T.pre); a.errorbar(bx,by,yerr=be,fmt='o--',color=GREY,ms=8,mec=SURF,lw=2.2,ecolor='#d5d3ce',zorder=3,label='pretrained model'); bx,by,be=bins(T.dAB,T.ft2000); a.errorbar(bx,by,yerr=be,fmt='o-',color=BLUE,ms=8,mec=SURF,lw=2.4,ecolor='#bcd0ea',zorder=4,label='model trained on all 2,000 chairs')
a.set_xlabel('distance between the trial\'s two objects   more different →'); a.set_ylabel('oddity margin'); a.set_title('and a trial whose two objects differ more is easier — for every model',loc='left',fontsize=10.5); a.legend(fontsize=9,frameon=False,loc='lower right')
box(a,f'pretrained r = {stats.pearsonr(T.dAB,T.pre)[0]:+.2f}   fine-tuned r = {stats.pearsonr(T.dAB,T.ft2000)[0]:+.2f}',INK,'ul')
a=ax[2]; style(a)
res_pre=T.pre-np.polyval(np.polyfit(T.dAB,T.pre,1),T.dAB); res_d=T.dist2000-np.polyval(np.polyfit(T.dAB,T.dist2000,1),T.dAB)
bx,by,be=bins(res_d,res_pre); a.errorbar(bx,by,yerr=be,fmt='o--',color=GREY,ms=8,mec=SURF,lw=2.2,ecolor='#d5d3ce',zorder=3,label='pretrained model')
a.axhline(0,color='#d8d7d2',lw=1); a.set_xlabel('distance to the training set, with the pair distance held fixed'); a.set_ylabel('pretrained margin, with the pair distance held fixed'); a.set_title('hold the pair distance fixed: the rise is gone',loc='left',fontsize=10.5)
box(a,f'r = {partial(T.dist2000,T.pre,T.dAB):+.2f}\n✓  flat: nothing about the training set',OK,'ur')
fig.suptitle('Why the margin seemed to rise with distance from the training set',fontsize=15,x=.06,ha='left',y=.975)
fig.text(.06,.9,'On the bank-built trials the "as measured" curves rose with distance to the training set before falling. That is not the training set: it is how the trials were built. Each held-out\n'
 'object is paired with one of its ten nearest held-out neighbours. Objects far from the training set sit in sparse regions of shape space, so their nearest neighbours are also far —\n'
 'the trial\'s two objects differ more, and the trial is easier for any model, trained or not.',fontsize=9.8,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig73_why_margin_rises.png',dpi=300); print('[fig] 73')
