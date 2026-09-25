"""The same claim, made literal: three categories, and one prediction."""
from _repo import G
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SURF,INK,INK2,GREY,OK,BAD='#fcfcfb','#0b0b0b','#52514e','#8a8884','#1baf7a','#eb6834'
C={'chair':'#2a78d6','sofa':'#1baf7a','telephone':'#eb6834'}
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':11.5,'axes.titlesize':13})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
L=L.merge(pd.read_csv(f'{G}/data/dinov2_distance_long.csv.gz'),on=['model','trial'],how='left').dropna(subset=['dinov2'])
L['gain']=L.ft-L.pre
def prof(d,col,nb=14):
    q=pd.qcut(d.dinov2.rank(method='first'),nb,labels=False); g=d.assign(q=q).groupby('q'); return g.dinov2.mean(),g[col].mean(),g[col].sem()
fig,ax=plt.subplots(2,2,figsize=(14.5,11)); fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.07,hspace=.38,wspace=.24)
# ---- row 1: the margin itself
a=ax[0,0]; style(a)
for c in C:
    d=L[L.test_cat==c]; x,y,e=prof(d,'ft'); a.errorbar(x,y,yerr=e,fmt='o-',color=C[c],ms=7,mec=SURF,lw=2.6,ecolor='#d5d3ce',zorder=4)
    a.text(x.iloc[-1]+0.02,y.iloc[-1],c,color=C[c],fontsize=12,va='center',fontweight='medium')
a.axvline(0.6,color=INK2,ls=':',lw=1.6,zorder=3)
vals={c:L[(L.test_cat==c)&(L.dinov2.between(0.55,0.65))].ft.mean() for c in C}
for c,v in vals.items(): a.plot([0.6],[v],'o',ms=13,mfc='none',mec=C[c],mew=2.5,zorder=6)
a.annotate('',xy=(0.6,min(vals.values())),xytext=(0.6,max(vals.values())),arrowprops=dict(arrowstyle='<->',color=INK,lw=2),zorder=7)
a.text(0.63,np.mean(list(vals.values())),f'at the same distance,\nthe margin differs by {max(vals.values())-min(vals.values()):.02f}',fontsize=11,color=INK)
a.set_xlim(0.15,1.35); a.set_ylabel('oddity margin'); a.set_xlabel('distance from the trial to the model\'s training objects   farther →')
a.set_title('1 · the margin: every category sits at its own height',loc='left')
# ---- row 1 right: predict a held-out category, margin
a=ax[0,1]; style(a); held='chair'
tr=L[L.test_cat!=held]; b=np.polyfit(tr.dinov2,tr.ft,1)
x,y,e=prof(L[L.test_cat==held],'ft'); xs=np.linspace(0.2,1.2,50)
a.plot(xs,np.polyval(b,xs),'--',color=INK2,lw=2.6,zorder=4,label='predicted from the other 11 categories')
a.errorbar(x,y,yerr=e,fmt='o-',color=C[held],ms=7,mec=SURF,lw=2.6,ecolor='#d5d3ce',zorder=5,label=f'what {held} actually does')
a.fill_between(x,np.polyval(b,x),y,color=BAD,alpha=.18,zorder=3)
a.set_ylabel('oddity margin'); a.set_xlabel('distance   farther →'); a.set_title(f'2 · predicting a category you did not fit: {held}',loc='left'); a.legend(fontsize=10,frameon=False,loc='lower left')
a.text(0.97,0.95,f'off by {np.mean(np.abs(np.polyval(b,x)-y)):.03f} on average —\nworse than just guessing the category mean',transform=a.transAxes,ha='right',va='top',fontsize=11,color=BAD)
# ---- row 2: the gain
a=ax[1,0]; style(a)
for c in C:
    d=L[L.test_cat==c]; x,y,e=prof(d,'gain'); a.errorbar(x,y,yerr=e,fmt='o-',color=C[c],ms=7,mec=SURF,lw=2.6,ecolor='#d5d3ce',zorder=4)
    a.text(x.iloc[-1]+0.02,y.iloc[-1],c,color=C[c],fontsize=12,va='center',fontweight='medium')
a.axvline(0.6,color=INK2,ls=':',lw=1.6,zorder=3)
vals2={c:L[(L.test_cat==c)&(L.dinov2.between(0.55,0.65))].gain.mean() for c in C}
for c,v in vals2.items(): a.plot([0.6],[v],'o',ms=13,mfc='none',mec=C[c],mew=2.5,zorder=6)
a.text(0.63,np.mean(list(vals2.values()))+0.010,f'the gap shrinks from {max(vals.values())-min(vals.values()):.03f} to {max(vals2.values())-min(vals2.values()):.03f};\nchairs still start higher than the rest',fontsize=11,color=OK)
a.set_xlim(0.15,1.35); a.set_ylabel('what fine-tuning ADDED to the margin\n(margin − the pretrained model on the same trial)'); a.set_xlabel('distance from the trial to the model\'s training objects   farther →')
a.set_title('3 · what training added: much closer, not identical',loc='left')
# ---- row 2 right: predict held-out category, gain
a=ax[1,1]; style(a)
b2=np.polyfit(tr.dinov2,tr.gain,1); x,y,e=prof(L[L.test_cat==held],'gain')
a.plot(xs,np.polyval(b2,xs),'--',color=INK2,lw=2.6,zorder=4,label='predicted from the other 11 categories')
a.errorbar(x,y,yerr=e,fmt='o-',color=C[held],ms=7,mec=SURF,lw=2.6,ecolor='#d5d3ce',zorder=5,label=f'what {held} actually does')
a.fill_between(x,np.polyval(b2,x),y,color=OK,alpha=.18,zorder=3)
a.set_ylabel('what fine-tuning added'); a.set_xlabel('distance   farther →'); a.set_title(f'4 · the same prediction, for what training added',loc='left'); a.legend(fontsize=10,frameon=False,loc='lower left')
a.text(0.97,0.95,f'off by {np.mean(np.abs(np.polyval(b2,x)-y)):.03f} — the curve fitted\nwithout chairs predicts chairs',transform=a.transAxes,ha='right',va='top',fontsize=11,color=OK)
fig.suptitle('Is the margin an absolute measure of shift? Not the margin — but what training added to it',fontsize=16,x=.08,ha='left',y=.965)
fig.text(.08,.905,'Three categories out of twelve, and one prediction, both told twice: once with the margin (top, it fails) and once with what fine-tuning added to the margin (bottom, it works).\n'
 '"What training added" is the model\'s margin minus the pretrained model\'s margin on the very same trial — no knowledge of the training set is needed to compute it.\n'
 'Top left: at one distance the three categories give margins 0.08 apart. Top right: a line fitted to eleven categories misses the twelfth. Bottom: the gap shrinks about fourfold and the prediction lands —\nthough the categories are closer, not identical.',fontsize=11,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig86_absolute_simple.png',dpi=300); print('[fig] 86')
