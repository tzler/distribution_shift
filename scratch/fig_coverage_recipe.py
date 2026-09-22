"""Two figures for the coverage state, both to the state-figure recipe.
fig56: three ways of scoring 'how far is the trial from the training set', each against the fine-tuned
       margin (top) and the pretrained margin on its own axis (bottom). Expect: top falls; bottom flat.
fig58: coverage counted from the object's shape vs from training IMAGES at the test's own viewpoint vs
       at the nearest training viewpoint; within-trial (each trial's average subtracted). Expect: if
       models learn view-specific appearance, the actual-viewpoint count should track the margin best."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
BLUE,GREY,SURF,INK,INK2,OK,BAD,GRN,HL='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def bins(x,y,nb=12):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
def line(a,x,y,e,c,ls,lab,z): a.errorbar(x,y,yerr=e,fmt='o'+ls,color=c,ms=8,mfc=c,mec=SURF,mew=1.3,lw=2.4,ecolor='#d5d3ce',elinewidth=1.4,zorder=z,label=lab)
EMP='trial_distance_(L1_not_normalized)'
d=pd.read_csv(f'{G}/out/coverage_shapenet_voxel16_percat.csv'); m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); rows=[]
for cat in sorted(d.category.unique()):
    o=pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv'); o['trial']=m['trial'].values; o['category']=cat; rows.append(o[['trial','category',EMP]])
d=d.merge(pd.concat(rows),on=['trial','category']).rename(columns={EMP:'orig'})
# coverage = -log(1+count): 0 when nothing is nearby, more negative with more training objects. 85% of pairs sit at 0,
# so bin on the raw COUNT with fixed edges (agent/CHECKS.md), not on quantiles of the score.
CEDGES=[0,1,3,6,11,21,51,101,201,10**9]; CLAB=['0','1–2','3–5','6–10','11–20','21–50','51–100','101–200','>200']
d['cbin']=pd.cut(d['count'],bins=[-0.5,0.5,2.5,5.5,10.5,20.5,50.5,100.5,200.5,1e12],labels=CLAB)
def cbins(y):
    gg=d.groupby('cbin',observed=True)[y]; return np.arange(len(CLAB))[::-1], gg.mean().reindex(CLAB).values, gg.sem().reindex(CLAB).values
d['shift_cov']=d.coverage
# ---------- fig56
fig,ax=plt.subplots(2,3,figsize=(16,10.4)); fig.subplots_adjust(left=.06,right=.98,top=.75,bottom=.13,hspace=.62,wspace=.28)
cols=[('shift_cov','Count the training objects near the trial\'s objects','training objects of that category within a fixed\nradius of the trial\'s objects  (fewer →)','coverage: a count, on 3-D shape'),
      ('knn','Measure how far the nearest training objects are','distance to the fifty nearest training objects  (farther →)','nearest-neighbour distance, on 3-D shape'),
      ('orig','The manuscript\'s estimate','distance measured in the network\'s features  (farther →)','the submitted metric, in the pretrained network\'s features')]
for j,(k,t,xl,sub) in enumerate(cols):
    x=d[k]; rf=stats.pearsonr(x,d.ft)[0]; rp=stats.pearsonr(x,d.pre)[0]
    if k=='shift_cov': bx,bf,ef=cbins('ft'); _,bp,ep=cbins('pre')
    else: bx,bf,ef=bins(x,d.ft); _,bp,ep=bins(x,d.pre)
    a=ax[0,j]; style(a); line(a,bx,bf,ef,BLUE,'-','fine-tuned model — trained on that training set',4)
    a.set_title(f'{t}\n{sub}',loc='left',fontsize=10.5); a.set_ylabel('oddity margin, fine-tuned model'); a.set_xlabel(xl)
    if k=='shift_cov':
        for aa in (ax[0,j],ax[1,j]): aa.set_xticks(np.arange(len(CLAB))[::-1]); aa.set_xticklabels(CLAB,fontsize=8.5)
    box(a,f'r = {rf:+.2f}\n'+('✓  falls' if rf<-0.1 else '!  does not simply fall'),OK if rf<-0.1 else BAD)
    a=ax[1,j]; style(a); line(a,bx,bp,ep,GREY,'--','pretrained model — never saw any training set (own axis)',3)
    a.set_ylabel('oddity margin, pretrained model'); a.set_xlabel(xl)
    passed=abs(rp)<0.08
    box(a,f'r = {rp:+.2f}\n'+('✓  flat: the score is about the training set' if passed else ('!  falls too: partly about how\n    unusual the objects are' if rp<0 else '!  RISES: the score is about the\n    network, not the training set')),OK if passed else BAD,where={'shift_cov':'ur','knn':'ll','orig':'ul'}[k])
    if j==0: a.set_ylim(0.02,0.08)
h1,l1=ax[0,0].get_legend_handles_labels(); h2,l2=ax[1,0].get_legend_handles_labels(); fig.legend(h1+h2,l1+l2,loc='upper left',bbox_to_anchor=(.06,.835),ncol=2,fontsize=10,frameon=False)
fig.suptitle('Three ways of scoring how far a trial is from a model\'s training set',fontsize=16,x=.06,ha='left',y=.975)
fig.text(.06,.925,'All 706 ShapeNet trials × all 12 category models (8,472 points), nothing subtracted — the plot a paper would naturally show, one model at a time. Each column is one way of scoring\n'
 'distance to the training set; more shift is to the right in every panel. What to expect: the fine-tuned model (top, blue) should have a smaller margin as shift grows. The pretrained\n'
 'model (bottom, grey; its own axis, because its margins are small) never saw any of the training sets — if the score is about the training set, the bottom row should be flat.',fontsize=9.8,color=INK2,va='top')
fig.text(.06,.015,'Counting nearby training objects (left) is the first score whose bottom row is flat across all trials. Measuring distance to the nearest ones (middle) falls in the bottom row too: objects far\n'
 'from every training set are unusual, and unusual objects are hard for every model. The manuscript\'s estimate (right) rises in the bottom row: it tracks the network, not the training set.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig56_pooled_three_metrics.png',dpi=300); print('[fig] 56')
# ---------- fig58
v_act=pd.read_csv(f'{G}/out/viewdepth_long_full.csv'); v_near=pd.read_csv(f'{G}/out/viewdepth_long_full_fixed15_nearest.csv')
EPS='img_cov0.3'
w=v_act[['trial','category','obj_cov','ft','pre',EPS]].rename(columns={EPS:'act'}).merge(v_near[['trial','category',EPS]].rename(columns={EPS:'near'}),on=['trial','category'])
g=w.groupby('trial'); 
for k in ['obj_cov','act','near','ft','pre']: w[k+'_c']=w[k]-g[k].transform('mean')
fig,ax=plt.subplots(1,3,figsize=(16,7.6),sharey=True); fig.subplots_adjust(left=.06,right=.98,top=.66,bottom=.19,wspace=.12)
panels=[('obj_cov_c','from the objects\' 3-D shape','training objects with a similar shape'),
        ('near_c','from training images,\nnearest training viewpoint','training images that look like the object\nfrom the closest camera the training set used'),
        ('act_c','from training images,\nthe test\'s actual viewpoint','training images that look like the object as the\ntest shows it (camera recovered from the silhouette)')]
FE=[-9,-1.5,-1,-0.6,-0.3,-0.1,0.1,0.3,0.6,1,1.5,9]
def fbins(x,y):
    q=pd.cut(x,FE); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q',observed=True); return gg.x.mean(),gg.y.mean(),gg.y.sem()
for a,(k,t,sub) in zip(ax,panels):
    style(a); x=w[k]; rf=stats.pearsonr(x,w.ft_c)[0]
    bx,bf,ef=fbins(x,w.ft_c); _,bp,ep=fbins(x,w.pre_c)
    line(a,bx,bp,ep,GREY,'--','pretrained model (flat: same model on all twelve points)',3); line(a,bx,bf,ef,BLUE,'-','fine-tuned model',4)
    a.set_title(f'Counted {t}\n{sub}',loc='left',fontsize=10); a.set_xlabel('fewer training items nearby  →\n(relative to the trial\'s average)')
    strong=rf<-0.35
    box(a,f'fine-tuned r = {rf:+.2f}\n'+('✓  the full effect' if strong else '!  weaker — and not because the\n    camera was recovered badly'),OK if strong else BAD)
ax[0].set_ylabel('oddity margin, relative to the trial\'s average')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.06,.79),ncol=2,fontsize=10,frameon=False)
fig.suptitle('Does it matter which viewpoint the training images showed?',fontsize=16,x=.06,ha='left',y=.975)
fig.text(.06,.92,'Same 706 trials × 12 models, each trial\'s own average subtracted (so only the training set differs between a trial\'s twelve points). Coverage — how much training data sits\n'
 'near the test object — is counted three ways. What to expect: if models learn what objects look like from particular viewpoints, counting training images from the test\'s actual\n'
 'viewpoint (right) should track the margin best. If they learn the object, all three should look alike.',fontsize=9.8,color=INK2,va='top')
fig.text(.06,.015,'Counting from the object\'s shape and from training images at the nearest training viewpoint give the same curve. Only the actual test viewpoint — which falls between the training\n'
 'cameras, typically about 20° off — is weaker, and it stays weaker on the trials where the camera was recovered well. Within the ~25° spacing of the training views, the viewpoint does not matter.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig58_viewdepth_ladder.png',dpi=300); print('[fig] 58')
