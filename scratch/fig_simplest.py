"""fig17: How simple can the model-free shift measure be?  (each panel on its own y-axis)

Left: the full pipeline (57-d descriptor, k-NN-50 cosine).  Middle/right: ONE scalar per
object, |z| against that category's training objects.  No neighbours, no cosine.
"""
import ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV='/vast/projects/bonnen/naturalistic-navig'
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
SURF,INK,INK2='#fcfcfb','#0b0b0b','#52514e'
COL=['#2a78d6','#1aa07a','#eb6834']
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,
 'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,
 'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343',
 'chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459',
 'sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']

def scalar_shift(bank_npz,test_npz,col):
    """|z| of one scalar vs each category's training objects, mean over trial images."""
    bz=np.load(bank_npz,allow_pickle=True); tz=np.load(test_npz,allow_pickle=True)
    B=bz['X'][:,col].astype(float); T=tz['X'][:,col].astype(float)
    syn=np.array([i.split('/')[0] for i in bz['ids']]); tix={k:i for i,k in enumerate(tz['ids'])}
    rows=[]
    for cat,s in SYN.items():
        b=B[syn==s]; mu,sd=b.mean(),b.std()+1e-12
        for _,r in mm.iterrows():
            keys=['/'.join(f[:-4].split('_')[:2]) for f in ast.literal_eval(r['images'])]
            if not all(k in tix for k in keys): continue
            rows.append(dict(trial=r['trial'],category=cat,
                             x=float(np.mean(np.abs((T[[tix[k] for k in keys]]-mu)/sd)))))
    return pd.DataFrame(rows)

names3d=list(np.load(f'{G}/bank/bank3d_shapenet_trained.npz',allow_pickle=True)['names'])
i_rstd=names3d.index('r_std')
panels=[
 ('full 57-d descriptor + k-NN-50 cosine\nthe pipeline used throughout',
  pd.read_csv(f'{G}/out/blindshift_shapenet_d57_percat.csv').rename(columns={'blind_k50':'x'})[['trial','category','x']]),
 ('|z| of r_std  — ONE number\nstd of centroid→surface distance',
  scalar_shift(f'{G}/bank/bank3d_shapenet_trained.npz',f'{G}/bank/test3d_shapenet.npz',i_rstd)),
 ('|z| of log bbox volume — ONE number\nstraight from model_normalized.json',
  scalar_shift(f'{G}/bank/bank_bbox.npz',f'{G}/bank/test_bbox.npz',6)),
]
Y=[]
for cat in SYN:
    o=pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv'); o['trial']=m['trial'].values
    o['category']=cat; o['ft']=o['fine_tuned_oddity_margin']; Y.append(o[['trial','category','ft']])
Y=pd.concat(Y)

fig,ax=plt.subplots(1,3,figsize=(20,6.6)); fig.subplots_adjust(left=.055,right=.99,top=.70,bottom=.13,wspace=.28)
for k,(ttl,P) in enumerate(panels):
    d=P.merge(Y,on=['trial','category'])
    d=d[d.groupby('trial').trial.transform('size')==12]
    xc=d.x-d.groupby('trial').x.transform('mean'); yc=d.ft-d.groupby('trial').ft.transform('mean')
    r=stats.pearsonr(xc,yc)[0]
    sl=np.array([stats.linregress(s.x,s.ft).slope for _,s in d.groupby('trial') if s.x.std()>0])
    d['rank']=d.groupby('trial').x.rank(method='first').astype(int)
    g=d.groupby('rank').agg(x=('x','mean'),y=('ft','mean'),e=('ft','sem'))
    a=ax[k]; a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
    a.errorbar(g.x,g.y,yerr=g.e,fmt='o-',color=COL[k],ms=8,mfc=COL[k],mec=SURF,mew=1.4,lw=2.2,
               ecolor='#d5d3ce',elinewidth=1.5,zorder=4)
    lo,hi=g.y.min()-g.e.max(),g.y.max()+g.e.max(); pad=.08*(hi-lo)
    a.set_ylim(lo-pad,hi+pad)                          # each panel on its OWN range
    a.set_title(f'{ttl}\nwithin-trial r = {r:+.3f},  {100*(sl<0).mean():.0f}% neg  '
                f'(n = {d.trial.nunique()} trials)',loc='left')
    a.set_xlabel('distance to training set (raw)')
    if k==0: a.set_ylabel('fine-tuned oddity margin (raw)')
    print(f'panel {k}: r={r:+.3f}  neg={100*(sl<0).mean():.1f}%  y-range [{g.y.min():.4f}, {g.y.max():.4f}]')
fig.suptitle('How simple can the model-free shift measure be?',fontsize=15,x=.055,ha='left',y=.965)
fig.text(.055,.86,'Right two panels use ONE scalar per object and |z| against that category\'s training objects — no neighbours, no cosine, no covariance.\n'
         'Each point is one within-trial rank (nearest → farthest training set), averaged over all trials. Y-axes are independent per panel.',
         fontsize=9.6,color=INK2,va='top')
p=f'{G}/out/figures/fig17_simplest.png'; fig.savefig(p,dpi=300); print('[fig]',p)
