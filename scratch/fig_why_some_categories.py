"""Why do loudspeaker, display and cabinet show almost no shift–margin relationship?
Not because the measure fails there — because fine-tuning barely moves their margin at all."""
from _repo import G
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SURF,INK,INK2,GREY,OK,BAD,BLUE='#fcfcfb','#0b0b0b','#52514e','#8a8884','#1baf7a','#eb6834','#2a78d6'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':11,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
L=L.merge(pd.read_csv(f'{G}/data/dinov2_distance_long.csv.gz'),on=['model','trial'],how='left').dropna(subset=['dinov2'])
rows=[]
for c,d in L.groupby('test_cat'):
    g=d.groupby('trial'); xc=(d.dinov2-g.dinov2.transform('mean')).values; yc=(d.ft-g.ft.transform('mean')).values
    rows.append((c,d.ft.mean(),d.pre.mean(),np.polyfit(xc,yc,1)[0],stats.pearsonr(xc,yc)[0],(d.ft<0).mean()))
R=pd.DataFrame(rows,columns=['cat','margin','pre','slope','r','frac_neg']).set_index('cat')
FLAG=['loudspeaker','display','cabinet']; col=lambda c: BAD if c in FLAG else BLUE
fig,ax=plt.subplots(1,3,figsize=(16.5,6),gridspec_kw={'width_ratios':[1.15,1,1]}); fig.subplots_adjust(left=.06,right=.985,top=.66,bottom=.215,wspace=.3)
# A · the relationship is weak exactly where the margin is small
a=ax[0]; style(a)
for c in R.index:
    a.scatter(R.margin[c],-R.r[c],s=130,color=col(c),edgecolor=SURF,lw=1.3,zorder=4)
    a.annotate(c,(R.margin[c],-R.r[c]),textcoords='offset points',xytext=(7,-3),fontsize=9.5,color=col(c))
b=np.polyfit(R.margin,-R.r,1); xs=np.linspace(0.03,0.125,10); a.plot(xs,np.polyval(b,xs),'--',color=GREY,lw=1.8,zorder=3)
a.set_xlabel('how big this category\'s margins are (its mean margin)'); a.set_ylabel('strength of the shift–margin relationship\n(within-trial |r|)')
a.set_title('A · weak where the margins are small',loc='left')
a.text(0.03,0.95,f'r = {stats.pearsonr(R.margin,-R.r)[0]:+.2f} across the twelve categories',transform=a.transAxes,fontsize=10.5,va='top',color=INK2)
# B · scaled by each category's own margin, the three are ordinary
a=ax[1]; style(a); R['scaled']=R.slope/R.margin
order=R.sort_values('scaled').index
a.barh(range(12),[-R.scaled[c] for c in order],color=[col(c) for c in order],zorder=4)
a.set_yticks(range(12)); a.set_yticklabels(order,fontsize=10); a.set_xlabel('how much of its OWN margin a category loses\nper unit of distance  (slope ÷ mean margin)')
a.set_title('B · scaled by their own margin: less extreme, still low',loc='left')
a.axvline(-R.scaled.mean(),color=INK,ls='--',lw=1.6,zorder=5); a.text(-R.scaled.mean()+0.015,11.4,'average',fontsize=9.5,color=INK2,ha='left',va='center')
# C · what they have instead: margins piled at zero
a=ax[2]; style(a)
for c,lab in [('chair','chair — margins spread out'),('display','display — margins piled at zero')]:
    d=L[L.test_cat==c].ft
    a.hist(d,bins=80,range=(-0.2,0.35),histtype='step',lw=2.4,color=col(c),density=True,zorder=4,label=lab)
a.axvline(0,color=GREY,ls='--',lw=1.6,zorder=3)
a.set_xlabel('oddity margin on a trial'); a.set_ylabel('density of trials'); a.set_title('C · the reason: no room to move',loc='left'); a.legend(fontsize=10,frameon=False)
a.text(0.97,0.60,f'display: {100*R.frac_neg["display"]:.0f}% of trials\nbelow zero\nchair: {100*R.frac_neg["chair"]:.0f}%',transform=a.transAxes,ha='right',fontsize=10,color=INK2)
fig.suptitle('Why loudspeaker, display and cabinet show almost no shift–margin relationship',fontsize=15.5,x=.06,ha='left',y=.965)
fig.text(.06,.905,'Not because their shapes are uninformative — those three have the MOST diverse shapes of the twelve (mean within-category distance 0.85 against 0.49) and the best-separated clusters.\n'
 'The common factor is that fine-tuning barely moves their margins at all: they are the categories the models are worst at, with margins sitting near zero, a third of trials below it,\n'
 'and therefore little range for the training set to act on. Scaled by their own margins the three are much less extreme, though still below average.',fontsize=10.2,color=INK2,va='top')
fig.text(.06,.015,'So this is a floor, not a failure of the distance measure: the relationship is compressed where the margin itself is compressed. It also warns against reading the absolute size of a slope —\n'
 'a category that is hard for every model will look like a category where shift does not matter.',fontsize=10.5,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig88_why_flat_categories.png',dpi=300); print('[fig] 88')
