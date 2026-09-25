"""Can the margin be an absolute measure of shift? Not as it stands — but what fine-tuning ADDED is."""
from _repo import G
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE,GREY,SURF,INK,INK2,OK,ORA='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':11.5})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
L=L.merge(pd.read_csv(f'{G}/data/dinov2_distance_long.csv.gz'),on=['model','trial'],how='left').dropna(subset=['dinov2'])
S=pd.read_csv(f'{G}/data/absolute_measure_search.csv'); S=S[S.loo_R2>-1]
def r2(y,p): return 1-np.sum((y-p)**2)/np.sum((y-np.mean(y))**2)
def loo(x,y,cat):
    o=[]
    for c in np.unique(cat):
        m=cat!=c; b=np.polyfit(x[m],y[m],1); o.append(r2(y[~m],np.polyval(b,x[~m])))
    return np.mean(o)
def prof(x,y,nb=18):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); d=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return d.x.mean(),d.y.mean()
cat=L.test_cat.values
fig,ax=plt.subplots(1,3,figsize=(17.5,6.4)); fig.subplots_adjust(left=.055,right=.985,top=.66,bottom=.19,wspace=.28)
# A · the two criteria disagree, and nothing passes the absolute one
a=ax[0]; style(a)
raw=S.measure.str.contains('RAW'); enc=S.measure.str.startswith('dinov2')
for m_,lab,mk in [(~raw&~enc,'model-free, normalised','o'),(raw&~enc,'model-free, raw magnitude','s'),(~raw&enc,'frozen network, normalised','o'),(raw&enc,'frozen network, raw magnitude','s')]:
    d=S[m_]; a.scatter(-d.within_trial_r,d.loo_R2,s=110,marker=mk,facecolor=BLUE if not lab.startswith('frozen') else ORA,edgecolor=SURF,lw=1.4,zorder=4,label=lab)
a.axhline(0,color=INK,ls='--',lw=1.8,zorder=3); a.text(0.98,0.955,'an absolute measure would be above this line',transform=a.transAxes,fontsize=9.2,color=INK2,ha='right',va='top')
a.set_xlabel('relative criterion: predicts which model wins a trial   better →'); a.set_ylabel('absolute criterion: fit on 11 categories,\npredict the 12th   (R²)')
a.set_title('A · eight distance measures, two criteria',loc='left'); a.legend(fontsize=8.6,frameon=False,loc='lower right')
a.set_ylim(-0.095,0.02); a.text(0.02,0.30,'un-normalising helps a little (squares are raw magnitude)\nbut nothing crosses the line; the measure that predicts best\nwithin a trial is among the worst across categories',transform=a.transAxes,fontsize=9.2,va='top',color=INK2)
# B · the fix is on the other axis
a=ax[1]; style(a); rows=[]
for nm,y in [('the margin itself',L.ft.values),('margin − the pretrained margin\non the same trial',(L.ft-L.pre).values),('margin − the average over\nthe 34 models',(L.ft-L.groupby('trial').ft.transform('mean')).values)]:
    for xn,lab,c in [('dinov2','frozen network',ORA),('dist','model-free voxels',BLUE)]:
        rows.append((nm,lab,c,loo(L[xn].values,np.asarray(y),cat)))
B=pd.DataFrame(rows,columns=['y','x','c','loo']); w=0.36
for i,(xl,c) in enumerate([('frozen network',ORA),('model-free voxels',BLUE)]):
    d=B[B.x==xl]; a.barh(np.arange(3)+(i-.5)*w,d.loo,height=w,color=c,label=xl,zorder=4)
    for j,v in enumerate(d.loo): a.text(v+(0.004 if v>0 else -0.004),j+(i-.5)*w,f'{v:+.3f}',va='center',ha='left' if v>0 else 'right',fontsize=8.8,color=INK2)
a.axvline(0,color=INK,lw=1.6,zorder=5); a.set_yticks(range(3)); a.set_yticklabels(B.y.unique(),fontsize=9)
a.set_xlabel('absolute criterion: fit on 11 categories, predict the 12th   (R²)'); a.set_title('B · what you put on the y axis decides it',loc='left'); a.legend(fontsize=9,frameon=False,loc='lower right'); a.set_xlim(-0.11,0.09)
a.text(0.02,0.30,'the margin carries the model\'s prior;\nthe part fine-tuning added does not —\nand that part transfers across categories',transform=a.transAxes,fontsize=9.2,va='top',color=INK2)
# C · the category lines, before and after
a=ax[2]; style(a); cm=plt.get_cmap('tab20'); cats=sorted(L.test_cat.unique())
for i,c in enumerate(cats):
    d=L[L.test_cat==c]
    bx,by=prof(d.dinov2,d.ft); a.plot(bx,by,'-',color=cm(i%20),lw=1.5,alpha=.45,zorder=3)
    bx,by=prof(d.dinov2,d.ft-d.pre); a.plot(bx,by,'-',color=cm(i%20),lw=2.1,zorder=4)
a.set_xlabel('distance to the model\'s training objects   farther →'); a.set_ylabel('faint: the margin   ·   bold: what training added',fontsize=9.5)
a.set_title('C · the same twelve categories, both ways',loc='left')
sp1=L.groupby('test_cat').ft.mean(); sp2=(L.ft-L.pre).groupby(L.test_cat).mean()
a.text(0.98,0.97,f'spread between categories:\nthe margin  {sp1.max()-sp1.min():.3f}\nwhat training added  {sp2.max()-sp2.min():.3f}',transform=a.transAxes,ha='right',va='top',fontsize=9.4,color=INK2)
fig.suptitle('Can the margin be an absolute measure of distribution shift?',fontsize=15,x=.055,ha='left',y=.975)
fig.text(.055,.90,'Absolute = one function from distance to margin, valid in a category it was not fitted on. The lead\'s two objections, tested: normalising the descriptors (z-scoring dimensions, unit-length rows,\n'
 'cosine) throws away the magnitude an absolute measure needs — so raw Euclidean and raw-overlap variants are included here; and the measure that wins on the relative criterion need not win on this one.',fontsize=9.6,color=INK2,va='top')
fig.text(.055,.015,'Neither objection rescues the margin itself: raw magnitude helps slightly (−0.067 vs −0.071) and no measure of any kind predicts a held-out category better than its mean. The reason is on the other axis — a\n'
 'category\'s mean margin is almost perfectly the pretrained model\'s (r = 0.985). Subtract the pretrained margin on the same trial and the relationship does generalise to an unseen category (R² = +0.01, worst −0.05).',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig85_absolute_fixed.png',dpi=300); print('[fig] 85')
