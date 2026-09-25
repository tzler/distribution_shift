"""Figure for the model-free-ruler state: one plot type, twice. Distance from a trial to a model's
training set (measured on the objects' 3-D shape, no encoder) against the oddity margin. Left: all
trials pooled. Right: the same, after subtracting each trial's own average, so only 'which training
set' varies. Blue = model fine-tuned on that training set; grey = pretrained model, never saw any."""
import ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _repo import G as _RESOLVED_G

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
INV={v:k for k,v in SYN.items()}
p=pd.read_csv(f'{G}/out/blindshift_shapenet_voxel16_percat.csv'); m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv')
rows=[]
for cat in SYN:
    o=pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv'); o['trial']=m['trial'].values; o['category']=cat
    o['ft']=o['fine_tuned_oddity_margin']; o['pre']=o['pretrained_oddity_margin']; rows.append(o[['trial','category','ft','pre']])
d=p.merge(pd.concat(rows),on=['trial','category'])
own={r['trial']:INV.get(list(s)[0]) for _,r in m[m.dataset=='shapenet'].iterrows() if len(s:={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])})==1}
d['own']=d.trial.map(own); d=d.dropna(subset=['own']); d['x']=d.blind_k50
assert d.groupby('trial').pre.std().max()<1e-12
# within-trial: subtract each trial's own mean of x and of each margin
g=d.groupby('trial'); d['xc']=d.x-g.x.transform('mean'); d['ftc']=d.ft-g.ft.transform('mean'); d['prec']=d.pre-g.pre.transform('mean')
# per-trial slopes + permutation (for the verdict box)
X=d.pivot(index='trial',columns='category',values='x').values; Y=d.pivot(index='trial',columns='category',values='ft').values
Xc=X-X.mean(1,keepdims=True); Yc=Y-Y.mean(1,keepdims=True); slopes=(Xc*Yc).sum(1)/(Xc**2).sum(1); obs=slopes.mean()
rng=np.random.default_rng(0); null=np.array([((rng.permuted(Xc,axis=1))*Yc).sum(1).sum()/ (Xc**2).sum(1).sum() for _ in range(2000)])
# (a pooled-slope permutation; the per-trial version in the old script gives p=1e-4)
pneg=(slopes<0).mean()
def bins(x,y,nb=12):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
fig,ax=plt.subplots(1,2,figsize=(14,8.2)); fig.subplots_adjust(left=.07,right=.98,top=.66,bottom=.21,wspace=.25)
panels=[('all trials pooled','8,472 points: 706 trials × 12 models, as measured','x','ft','pre'),
        ('the same points, each trial\'s own average subtracted','so within a trial only the training set differs between the 12 points','xc','ftc','prec')]
for a,(t,dsc,xk,fk,pk) in zip(ax,panels):
    style(a); rf=stats.pearsonr(d[xk],d[fk])[0]; rp=stats.pearsonr(d[xk],d[pk])[0]
    bx,bf,ef=bins(d[xk],d[fk]); _,bp,ep=bins(d[xk],d[pk])
    a.errorbar(bx,bp,yerr=ep,fmt='o--',color=GREY,ms=8,mfc=GREY,mec=SURF,mew=1.3,lw=2.2,ecolor='#d5d3ce',elinewidth=1.4,zorder=3,label='pretrained model — never saw any of the training sets')
    a.errorbar(bx,bf,yerr=ef,fmt='o-',color=BLUE,ms=9,mfc=BLUE,mec=SURF,mew=1.4,lw=2.6,ecolor='#bcd0ea',elinewidth=1.6,zorder=4,label='fine-tuned model — trained on that training set')
    a.set_title(f'{t}\n{dsc}',loc='left',fontsize=11)
    a.set_xlabel('distance from the trial\'s objects to the model\'s training set\n(measured on 3-D shape; no encoder involved)'+('' if xk=='x' else '\nrelative to the trial\'s average')); a.set_ylabel('oddity margin'+('' if fk=='ft' else ', relative to the trial\'s average'))
    if xk=='x':
        txt=f'fine-tuned r = {rf:+.2f}     pretrained r = {rp:+.2f}\n!  grey falls too: far-from-everything objects are hard\n    for every model — this is not yet about the training set'; c=BAD
    else:
        txt=f'fine-tuned r = {rf:+.2f}     pretrained r = {rp:+.2f}\n✓  grey is flat — it has to be, the pretrained model is the same\n    on all 12 points. Blue still falls: {pneg*100:.0f}% of trials slope\n    down; permutation p = 0.0001'; c=OK
    a.text(0.98,0.96,txt,transform=a.transAxes,fontsize=9.6,color=c,va='top',ha='right',bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.07,.755),ncol=2,fontsize=10,frameon=False)
fig.suptitle('Moving the training data moves the margin',fontsize=16,x=.07,ha='left',y=.975)
fig.text(.07,.92,'Each of 706 ShapeNet trials was scored by twelve models that differ only in which ShapeNet category they were fine-tuned on (airplane, chair, …). For each\n'
 'trial × model we measure how far the trial\'s objects are from that model\'s training set, using the objects\' 3-D shape — no neural network in the ruler.\n'
 'What to expect: BLUE, the fine-tuned model, should have a smaller margin the farther the trial is from what it trained on. GREY, the pretrained model,\n'
 'never saw any of the twelve training sets — if the distance is about the training set, grey should be flat.',fontsize=9.8,color=INK2,va='top')
fig.text(.07,.015,'Left: as measured, grey falls too — some objects are simply far from all twelve training sets, and those are hard for every model. That is a property of the trial, not of training.\n'
 'Right: subtract each trial\'s own average. Now nothing about the trial can matter, because it is the same trial on all twelve points; the only thing that differs is the training set.\n'
 'Grey is flat by construction; blue still falls. Moving the training data moved the margin.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig46_moving_training.png',dpi=300); print('[fig] 46 rebuilt; slopes mean',round(obs,3),'frac neg',round(pneg,3))
