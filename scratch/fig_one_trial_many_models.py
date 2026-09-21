"""The direct version: hold one trial fixed; retrain on data at different distances from it; watch its margin.
Each panel is ONE held-out trial. Each point is one of the 34 models (trained on 25 objects of one kind of one
category), placed by how far that model's training objects are from this trial's objects. Grey line: the
pretrained margin on this trial. Last panel: all trials' curves relative to their pretrained margin, overlaid."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
G='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':11})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
L=pd.read_csv(f'{G}/out/all_categories_long.csv'); L=L[L.kind=='cluster']
g=L.groupby('trial'); pre=g.pre.first(); lvl=g.ft.mean()
# pick 8 trials spanning difficulty: quantiles of the pretrained margin, chairs and others mixed
rng=np.random.default_rng(7); qs=np.quantile(pre,[0.05,0.2,0.35,0.5,0.65,0.8,0.9,0.98]); picks=[]
for q in qs:
    cand=pre[(pre-q).abs()<0.005].index; cand=[c for c in cand if c not in picks]; picks.append(rng.choice(cand))
fig,ax=plt.subplots(2,4,figsize=(17,8.6)); fig.subplots_adjust(left=.05,right=.99,top=.8,bottom=.09,hspace=.5,wspace=.22); ax=ax.ravel()
for a,t in zip(ax[:7],picks[:7]):
    style(a); d=L[L.trial==t].sort_values('dist'); own=d[d.train_cat==d.test_cat]; oth=d[d.train_cat!=d.test_cat]
    a.plot(oth.dist,oth.ft,'o',color=BLUE,ms=6,mfc='none',mew=1.4,alpha=.8,zorder=3,label='model of another category'); a.plot(own.dist,own.ft,'o',color=BLUE,ms=8,mec=SURF,zorder=4,label='model of this trial\'s category')
    a.axhline(d.pre.iloc[0],color=GREY,ls='--',lw=2,zorder=2,label='pretrained margin on this trial')
    r=stats.spearmanr(d.dist,d.ft)[0]; cat=d.test_cat.iloc[0]
    a.set_title(f'{cat} trial · pretrained margin {d.pre.iloc[0]:+.2f} · ρ = {r:+.2f}',loc='left'); a.set_xlabel('distance from this trial to the model\'s training objects'); a.set_ylabel('this trial\'s margin')
h,l=ax[0].get_legend_handles_labels()
a=ax[7]; style(a)
for t in rng.choice(L.trial.unique(),300,replace=False):
    d=L[L.trial==t].sort_values('dist'); a.plot(d.dist,d.ft-d.pre.iloc[0],'-',color=BLUE,lw=.5,alpha=.12,zorder=2)
q=pd.qcut(L.dist.rank(method='first'),15,labels=False); gg=L.assign(q=q,rel=L.ft-L.pre).groupby('q'); a.errorbar(gg.dist.mean(),gg.rel.mean(),yerr=gg.rel.sem(),fmt='o-',color=INK,ms=6,mec=SURF,lw=2.2,zorder=5,label='mean over all 11,634 trials')
a.axhline(0,color=GREY,ls='--',lw=2,zorder=3); a.set_title('300 trials overlaid, each relative to its own pretrained margin',loc='left'); a.set_xlabel('distance from the trial to the model\'s training objects'); a.set_ylabel('margin − pretrained margin'); a.legend(fontsize=8.5,frameon=False,loc='upper right'); a.set_ylim(-0.12,0.2)
fig.legend(h,l,loc='upper left',bbox_to_anchor=(.05,.86),ncol=3,fontsize=9.5,frameon=False)
fig.suptitle('Hold one trial fixed; retrain on data at different distances from it; watch its margin',fontsize=15,x=.05,ha='left',y=.975)
fig.text(.05,.925,'Each of the first seven panels is one held-out trial, chosen to span the range of difficulty. Each point is one of 34 models, each trained on 25 objects of one kind of one category,\n'
 'placed by how far its training objects are from this trial\'s objects. Filled points: the three models of the trial\'s own category. The dashed line is where the pretrained model put this trial.\n'
 'What to expect: the margin falls with distance, from above the pretrained line when the training data is near, towards it when the training data is far.',fontsize=9.8,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig79_one_trial_many_models.png',dpi=300); print('[fig] 79', picks)
