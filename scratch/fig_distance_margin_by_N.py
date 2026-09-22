"""Distance (x) vs margin (y), one line per training-set size. Bank-built chair trials; every model with a bank evaluation.
For each trial × model: distance from the trial's objects to that model's actual training chairs (mean to the 10 nearest, 3-D shape),
and the model's margin on the trial. Left: as measured. Right: within trial — each trial's mean over the models of that size removed."""
import json, ast, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G
SURF,INK,INK2,GREY='#fcfcfb','#0b0b0b','#52514e','#8a8884'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); B=bz['X'].astype(float); ids=np.array(bz['ids'])
sel=np.array([i.startswith('03001627/') for i in ids]); X=B[sel]; objs=np.array([i.split('/')[1] for i in ids[sel]]); mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1
Xn=(X-mu)/sd; Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12; oix={o:i for i,o in enumerate(objs)}
split=json.load(open(f'{K}/design_clusters_chair_split.json')); kd=json.load(open(f'{K}/design_knockin.json')); dk=json.load(open(f'{K}/design.json'))
T=pd.read_csv(f'{K}/banktrials/banktrials_chair.csv'); T['objs']=T.images.apply(lambda s: sorted({n[len('chair_'):-8] for n in ast.literal_eval(s)})); Tt=[Xn[[oix[o] for o in l]] for l in T.objs]
allchairs=set(os.listdir(f'{NAV}/Dist-shift-data/shapenet_rendered/white/chair'))
models=[]   # (name, N label, training objects)
for c in range(3):
    tr=split[str(c)]['train']; models+= [(f'chair_c{c}_n25',25,tr[:25]),(f'chair_c{c}_n50',50,tr[:50]),(f'chair_c{c}_all',100,tr)]
for i in range(8): models.append((f'chair_random_100_{i}',100,kd['random'][f'random_100_{i}']))
for g in ['1','2','3','4']: models.append((f'chair_g{g}_k10',1800,[o for o in allchairs if o not in set(dk['chair']['groups'][g]['k10']['removed'])]))
models.append(('chair_random_k10',1800,[o for o in allchairs if o not in set(dk['chair']['random']['k10']['removed'])])); models.append(('chair_full',2000,sorted(allchairs)))
rows=[]; have=[]
for name,N,tr in models:
    f=f'{K}/eval_bank/{name}/ood_analysis_results.csv'
    if not os.path.exists(f): continue
    have.append(name); mo=pd.read_csv(f); S=Xn[[oix[o] for o in tr if o in oix]]
    for i,t in enumerate(Tt):
        D=1-t@S.T; rows.append((name,N,i,np.sort(D,axis=1)[:,:10].mean(1).mean(),mo.fine_tuned_oddity_margin[i],mo.pretrained_oddity_margin[i]))
L=pd.DataFrame(rows,columns=['model','N','trial','dist','margin','pre']); L.to_csv(f'{G}/out/distance_margin_by_N.csv',index=False)
print('models with bank evaluations:',len(have)); print(L.groupby('N').agg(models=('model','nunique'),dist=('dist','mean'),dist_sd=('dist','std'),margin=('margin','mean')).round(3).to_string())
L['margin_c']=L.margin-L.groupby(['N','trial']).margin.transform('mean'); L['dist_c']=L.dist-L.groupby(['N','trial']).dist.transform('mean')
def bins(x,y,nb=8):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
cmap=plt.get_cmap('viridis'); Ns=sorted(L.N.unique()); col={N:cmap(i/(len(Ns)-1+1e-9)*0.9) for i,N in enumerate(Ns)}
fig,ax=plt.subplots(1,2,figsize=(15,6.8)); fig.subplots_adjust(left=.06,right=.98,top=.70,bottom=.24,wspace=.22)
a=ax[0]; style(a)
for N in Ns:
    d=L[L.N==N]; bx,by,be=bins(d.dist,d.margin); r=stats.pearsonr(d.dist,d.margin)[0]; k=d.model.nunique()
    a.errorbar(bx,by,yerr=be,fmt='o-',color=col[N],ms=8,mec=SURF,lw=2.4,ecolor='#d5d3ce',zorder=4,label=f'{N if N<1000 else ("~1,800" if N==1800 else "2,000")} chairs  ({k} model{"s" if k>1 else ""}; r = {r:+.2f})')
d2=L[L.N==2000] if (L.N==2000).any() else L[L.N==L.N.max()]
bx,by,be=bins(d2.dist,d2.pre); a.errorbar(bx,by,yerr=be,fmt='o--',color=GREY,ms=7,mec=SURF,lw=2,ecolor='#d5d3ce',zorder=3,label='pretrained model, against distance to all 2,000 chairs')
a.set_xlabel('distance from the trial\'s objects to the model\'s training chairs  (3-D shape)   farther →'); a.set_ylabel('oddity margin'); a.set_title('as measured',loc='left'); a.legend(fontsize=8.8,frameon=False,loc='lower right'); a.set_ylim(0.0,0.19)
a=ax[1]; style(a)
for N in Ns:
    d=L[L.N==N]
    if d.model.nunique()<2: continue
    bx,by,be=bins(d.dist_c,d.margin_c); r=stats.pearsonr(d.dist_c,d.margin_c)[0]
    a.errorbar(bx,by,yerr=be,fmt='o-',color=col[N],ms=8,mec=SURF,lw=2.4,ecolor='#d5d3ce',zorder=4,label=f'{N if N<1000 else "~1,800"} chairs  (r = {r:+.2f})')
a.axhline(0,color=GREY,ls='--',lw=2,label='pretrained (flat by construction)')
a.set_xlabel('distance to the model\'s training chairs, relative to the trial\'s mean over models of that size   farther →'); a.set_ylabel('oddity margin, relative to the trial\'s mean over models of that size'); a.set_title('within trial: same trial, different training sets of the same size',loc='left'); a.legend(fontsize=8.8,frameon=False,loc='upper right')
fig.suptitle('Distance to the training set vs margin, by training-set size',fontsize=15,x=.06,ha='left',y=.975)
fig.text(.06,.92,'Bank-built chair trials (882), every chair model with an evaluation on them. Each point is a trial scored by a model: x = how far the trial\'s objects are from the chairs that\n'
 'model was trained on; y = its margin. Lines are coloured by how many chairs the model was trained on. Left: as measured — note that larger training sets are simply nearer\n'
 'to everything (the lines shift left). Right: the same trial under different training sets of the same size, so only the composition differs. Expect: falling lines, if the\n'
 'measure tracks what the model learned.',fontsize=9.6,color=INK2,va='top')
fig.text(.06,.015,'Left: two things are entangled. Larger training sets are nearer to every trial (the lines shift left), and inside any one size the curve rises before it falls — the pretrained model\n'
 'rises with distance too, because in this category the unusual objects make the easy trials. The fall at the far right is the genuine shift: a model scored on a kind of chair it never saw.\n'
 'Right: hold the trial fixed and compare training sets of the same size; the confound is gone and every size gives the same falling line, about 0.04 of margin across the range of distances.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig72_distance_margin_by_N.png',dpi=300); print('[fig] 72')
