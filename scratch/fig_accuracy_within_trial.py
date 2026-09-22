"""fig48: the missing panel of fig25 — accuracy vs geometric shift in the WITHIN-TRIAL design.

Ranks hold the same 706 trials, so the pretrained accuracy is flat by construction and
the fine-tuned accuracy's fall is attributable to the training set alone.
"""
import ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV='/vast/projects/bonnen/naturalistic-navig'
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
GEOM,GREY,HL,SURF,INK,INK2='#2a78d6','#8a8884','#eb6834','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,
 'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,
 'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11.5})
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343',
 'chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459',
 'sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
INV={v:k for k,v in SYN.items()}
p=pd.read_csv(f'{G}/out/blindshift_shapenet_voxel16_percat.csv')
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv')
rows=[]
for cat in SYN:
    o=pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv'); o['trial']=m['trial'].values
    o['category']=cat; o['fc']=o['fine_tuned_correct']; o['pc']=o['pretrained_correct']
    o['ft']=o['fine_tuned_oddity_margin']; o['pre']=o['pretrained_oddity_margin']
    rows.append(o[['trial','category','fc','pc','ft','pre']])
d=p.merge(pd.concat(rows),on=['trial','category'])
own={}
for _,r in m[m.dataset=='shapenet'].iterrows():
    s={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s)==1: own[r['trial']]=INV.get(list(s)[0])
d['own']=d.trial.map(own); d=d.dropna(subset=['own'])
d=d[d.groupby('trial').trial.transform('size')==12]
d['rank']=d.groupby('trial').blind_k50.rank(method='first').astype(int)
d['xc']=d.blind_k50-d.groupby('trial').blind_k50.transform('mean')
g=d.groupby('rank').agg(x=('blind_k50','mean'),fc=('fc','mean'),fse=('fc','sem'),
                        pc=('pc','mean'),pse=('pc','sem'))
assert g.pc.std()<1e-12, 'pretrained accuracy must be identical across ranks'
n=d.trial.nunique()

fig,a0=plt.subplots(1,1,figsize=(9.6,6.6)); fig.subplots_adjust(left=.09,right=.975,top=.70,bottom=.13)
ax=[a0]
for a in ax:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
    a.axhline(1/3,color='#c9c7c2',lw=1.2,zorder=1); a.set_ylabel('accuracy')
# A: rank design, raw x
a=ax[0]
a.errorbar(g.x,g.pc,yerr=g.pse,fmt='o--',color=GREY,ms=8,mfc=GREY,mec=SURF,mew=1.3,lw=2,
           ecolor='#d5d3ce',elinewidth=1.4,zorder=3,label='pretrained DINOv2 (no fine-tuning)')
a.errorbar(g.x,g.fc,yerr=g.fse,fmt='o-',color=GEOM,ms=9,mfc=GEOM,mec=SURF,mew=1.4,lw=2.4,
           ecolor='#bcd0ea',elinewidth=1.6,zorder=4,label='fine-tuned on that category')
a.scatter([g.x.iloc[0]],[g.fc.iloc[0]],s=180,color=HL,edgecolor=SURF,lw=1.6,zorder=5)
a.text(g.x.iloc[0]+.02,g.fc.iloc[0]+.012,'rank 1: nearest\ntraining set',fontsize=8.6,color=HL,va='bottom')
a.text(g.x.iloc[-1],1/3+.01,'chance',ha='right',fontsize=8.5,color=INK2)
sp=stats.spearmanr(g.index,g.fc)[0]
a.set_title(f'Within-trial rank design  (12 ranks × {n} trials)\n'
            f'fine-tuned {100*g.fc.iloc[0]:.1f}% → {100*g.fc.iloc[-1]:.1f}%  '
            f'({100*(g.fc.iloc[0]-g.fc.iloc[-1]):.1f} points, Spearman {sp:+.2f});  '
            f'pretrained flat by construction',loc='left')
a.set_xlabel('geometric distance to that model\'s training set (mean within rank)')
a.legend(loc='upper right',fontsize=8.8); a.set_ylim(.28,.9)
fig.suptitle('Accuracy as a function of model-free geometric distribution shift\n— within-trial design',fontsize=14,x=.09,ha='left',y=.975)
fig.text(.09,.845,'The panel fig 25 was missing. Same shift measure (16³ voxels, k-NN-50 cosine, no encoder), same 706 trials × 12 category fine-tunes.\n'
         'Every rank contains the same 706 trials, so d(A,B), trial difficulty and the pretrained model\'s accuracy cannot vary along x.\n'
         'The pretrained line is flat, and the fine-tuned line\'s fall is attributable to the training set alone.',
         fontsize=9.4,color=INK2,va='top')
pth=f'{G}/out/figures/fig48_accuracy_within_trial.png'; fig.savefig(pth,dpi=300)
print(f'rank1 {g.fc.iloc[0]:.3f} rank12 {g.fc.iloc[-1]:.3f} pretrained {g.pc.iloc[0]:.3f} sd {g.pc.std():.1e}')
print('[fig]',pth)
