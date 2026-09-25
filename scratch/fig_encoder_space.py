import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _repo import G as _RESOLVED_G

K='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout'; G=_RESOLVED_G
GEOM,HL,GREY,TEAL,SURF,INK,INK2='#2a78d6','#eb6834','#8a8884','#1baf7a','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y',color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
R=pd.read_csv(f'{K}/eval/encoder_space_battery.csv'); R=R[(R.level=='image')&(R.estimator=='knn')].set_index('space')
rows=[('geometric\nvoxel16 coverage',dict(within_r=-0.424,pooled_r=-0.268,pooled_ctrl=-0.030,oncat_r=-0.118,oncat_ctrl=-0.047,oncat_catcentred_r=0.013),TEAL),
      ('pretrained\nDINOv2-L space',R.loc['pretrained'],GREY),('chair model\'s\nown space',R.loc['ft_chair'],GEOM),('airplane model\'s\nown space',R.loc['ft_airplane'],GEOM),('table model\'s\nown space',R.loc['ft_table'],GEOM)]
fig,ax=plt.subplots(1,4,figsize=(23,6.6)); fig.subplots_adjust(left=.045,right=.99,top=.72,bottom=.27,wspace=.28)
x=np.arange(len(rows)); labels=[r[0] for r in rows]; cols=[r[2] for r in rows]
def bars(a,key,ttl,ctrl=None):
    style(a); v=[-float(r[1][key]) for r in rows]; a.bar(x-(.2 if ctrl else 0),v,.38 if ctrl else .6,color=cols,zorder=3,edgecolor=SURF,label='fine-tuned margin')
    if ctrl: c=[-float(r[1][ctrl]) for r in rows]; a.bar(x+.2,c,.38,color=cols,alpha=.35,zorder=3,edgecolor=SURF,label='pretrained margin (control)')
    for i,val in enumerate(v): a.text(x[i]-(.2 if ctrl else 0),val+(.01 if val>=0 else -.01),f'{val:+.2f}',ha='center',va='bottom' if val>=0 else 'top',fontsize=8.4)
    if ctrl:
        for i,val in enumerate(c): a.text(x[i]+.2,val+(.01 if val>=0 else -.01),f'{val:+.2f}',ha='center',va='bottom' if val>=0 else 'top',fontsize=8.4,color=INK2)
    a.axhline(0,color=INK2,lw=1); a.set_xticks(x); a.set_xticklabels(labels,fontsize=8.2,rotation=25,ha='right'); a.set_title(ttl,loc='left'); a.set_ylabel('−r')
    if ctrl: a.legend(fontsize=7.6,loc='upper left',bbox_to_anchor=(0,1.0))
bars(ax[0],'within_r','A  Within-trial (the causal design)\nstimulus fixed, training set varied')
bars(ax[1],'pooled_r','B  Pooled row, all 8,472\nmust leave the pretrained margin flat','pooled_ctrl')
bars(ax[2],'oncat_r','C  On-category row, n = 706\nencoder spaces fail the control here','oncat_ctrl')
bars(ax[3],'oncat_catcentred_r','D  On-category, category-centred\nthe within-category signal geometry lacks')
fig.suptitle('The oddity-blind estimate inside the encoders\' own feature spaces, against the same training bank',fontsize=14.5,x=.05,ha='left',y=.975)
fig.text(.05,.875,'Mean cosine distance to the 50 nearest training renders, averaged over the trial\'s images — the same form as the geometric estimate, computed in DINOv2-L features (pretrained, and each fine-tune\'s own).\n'
         'A–B: fixing the form is enough for the pooled control even inside the encoder. C: the on-category row is where encoder spaces leak — distance and atypicality are more entangled there.\n'
         'D: encoder spaces carry a within-category signal (−0.10 to −0.15, p < .01) that no geometric descriptor has (+0.01). In the airplane model\'s own space, on airplane trials: r = −0.42 (control −0.22).',fontsize=9.1,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig62_encoder_space.png',dpi=300); print('[fig] 62')
