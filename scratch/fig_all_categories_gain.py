"""All twelve categories, side by side: the margin and what training added, each with the
prediction made from the other eleven categories. One small panel per category."""
from _repo import G
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SURF,INK,INK2,GREY,OK,BAD='#fcfcfb','#0b0b0b','#52514e','#8a8884','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.7,zorder=0); a.set_axisbelow(True)
L=pd.read_csv(f'{G}/data/all_categories_long.csv.gz'); L=L[L.kind=='cluster']
L=L.merge(pd.read_csv(f'{G}/data/dinov2_distance_long.csv.gz'),on=['model','trial'],how='left').dropna(subset=['dinov2'])
L['gain']=L.ft-L.pre; cats=sorted(L.test_cat.unique())
def prof(d,col,nb=12):
    q=pd.qcut(d.dinov2.rank(method='first'),nb,labels=False); g=d.assign(q=q).groupby('q'); return g.dinov2.mean().values,g[col].mean().values,g[col].sem().values
fig,axes=plt.subplots(3,4,figsize=(16,11),sharex=True,sharey=True); fig.subplots_adjust(left=.07,right=.985,top=.80,bottom=.135,hspace=.30,wspace=.12)
errs=[]
for ax_,c in zip(axes.ravel(),cats):
    style(ax_); d=L[L.test_cat==c]; o=L[L.test_cat!=c]
    bm=np.polyfit(o.dinov2,o.ft,1); bg=np.polyfit(o.dinov2,o.gain,1)
    x,ym,em=prof(d,'ft'); _,yg,eg=prof(d,'gain'); xs=np.linspace(0.2,1.15,40)
    ax_.plot(xs,np.polyval(bm,xs),'--',color=BAD,lw=1.6,alpha=.8,zorder=3)
    ax_.errorbar(x,ym,yerr=em,fmt='o-',color=BAD,ms=4.5,mec=SURF,lw=1.8,ecolor='#f2d3c5',zorder=4)
    ax_.plot(xs,np.polyval(bg,xs),'--',color=OK,lw=1.6,alpha=.8,zorder=3)
    ax_.errorbar(x,yg,yerr=eg,fmt='o-',color=OK,ms=4.5,mec=SURF,lw=1.8,ecolor='#c2e8d8',zorder=4)
    em_=np.mean(np.abs(np.polyval(bm,x)-ym)); eg_=np.mean(np.abs(np.polyval(bg,x)-yg)); errs.append((c,em_,eg_))
    ax_.set_title(c,loc='left',fontsize=12)
    ax_.text(0.97,0.95,f'miss {em_:.03f}',transform=ax_.transAxes,ha='right',va='top',fontsize=9,color=BAD)
    ax_.text(0.97,0.83,f'miss {eg_:.03f}',transform=ax_.transAxes,ha='right',va='top',fontsize=9,color=OK)
for ax_ in axes[-1]: ax_.set_xlabel('distance to the model\'s\ntraining objects   farther →',fontsize=9.5)
for ax_ in axes[:,0]: ax_.set_ylabel('margin  /  what training added',fontsize=9.5)
E=pd.DataFrame(errs,columns=['cat','margin','gain'])
h=[plt.Line2D([],[],color=BAD,marker='o',ms=5,lw=1.8,label='the margin itself'),plt.Line2D([],[],color=OK,marker='o',ms=5,lw=1.8,label='what fine-tuning added (margin − pretrained, same trial)'),plt.Line2D([],[],color=INK2,ls='--',lw=1.6,label='predicted for this category from the other eleven')]
fig.legend(handles=h,loc='upper left',bbox_to_anchor=(.07,.865),ncol=3,fontsize=10.5,frameon=False)
fig.suptitle('All twelve categories, each predicted from the other eleven',fontsize=16,x=.07,ha='left',y=.965)
fig.text(.07,.925,'For every category: the solid line is what it actually does, the dashed line is what a single function fitted to the other eleven categories predicts it will do. Red uses the margin,\n'
 'green uses what fine-tuning added to the margin. "Miss" is the average gap between the two, in units of margin — small means the function transfers to a category it never saw.',fontsize=10.2,color=INK2,va='top')
worst2=E.sort_values('gain',ascending=False).cat.head(2).tolist()
fig.text(.07,.015,f'The margin misses by {E.margin.mean():.03f} of margin on average (worst {E.margin.max():.03f}, {E.cat[E.margin.idxmax()]}); what training added misses by {E.gain.mean():.03f} (worst {E.gain.max():.03f}, {E.cat[E.gain.idxmax()]}) — {E.margin.mean()/E.gain.mean():.1f}× better, and better in all {int((E.gain<E.margin).sum())} categories.\n'
 f'The red lines sit at twelve different heights; the green ones nearly share a height and a slope. {worst2[0].capitalize()} and {worst2[1]} are the two the green function still misses, and red does badly wherever a category\'s own line is not flat.',fontsize=10.5,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig87_all_categories.png',dpi=300); print('[fig] 87'); print(E.round(4).to_string(index=False))
