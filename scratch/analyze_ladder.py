"""Regime ladder (D33): how much of the margin is the trial's pre-existing level, rung by rung."""
import json, ast, numpy as np, pandas as pd
from scipy import stats
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'
T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv'); chair=(T.dataset=='chair').values; cl=T.condition.str[-1].astype(int).values
L=pd.read_csv(f'{G}/out/all_categories_long.csv'); L=L[(L.kind=='cluster')&(L.model.str.startswith('chair_c'))]
dist={(m,t):d for m,t,d in zip(L.model,L.trial,L.dist)}   # distance from each chair trial to each chair cluster model's 25 (from the voxel16 table)
def load(n): return pd.read_csv(f'{K}/eval_bank_all/{n}/ood_analysis_results.csv')
print('RUNGS WITH THREE CLUSTER MODELS AT N = 25 (chair trials, 882; three models per trial)')
print(f'{"rung":34s} {"mean margin":>11s} {"%var = trial level":>18s} {"r(level, pretrained)":>20s} {"r(level, distance)":>18s} {"within-trial r":>15s} {"own-kind adv":>12s}')
for rung,suffix in [('LoRA lr 1e-6 (have)',''),('LoRA lr 1e-5','_lora1e5'),('full fine-tune lr 1e-5','_fullft')]:
    rows=[]
    for c in range(3):
        o=load(f'chair_c{c}_n25{suffix}'); base=f'chair_c{c}_n25'
        for t in np.where(chair)[0]: rows.append((c,t,o.fine_tuned_oddity_margin[t],o.pretrained_oddity_margin[t],dist[(base,t)],cl[t]))
    d=pd.DataFrame(rows,columns=['model','trial','ft','pre','dist','cluster']); g=d.groupby('trial'); lvl=g.ft.mean(); pre=g.pre.first(); md=g.dist.mean()
    between=g.ft.transform('mean').var()/d.ft.var(); wd=d.dist-g.dist.transform('mean'); wm=d.ft-g.ft.transform('mean')
    mat=np.array([[d[(d.model==c)&(d.cluster==t)].ft.mean() for t in range(3)] for c in range(3)]); I=mat-mat.mean(1,keepdims=True)-mat.mean(0,keepdims=True)+mat.mean()
    print(f'{rung:34s} {d.ft.mean():11.3f} {100*between:17.0f}% {stats.pearsonr(pre,lvl)[0]:+20.2f} {stats.pearsonr(md,lvl)[0]:+18.2f} {stats.pearsonr(wd,wm)[0]:+15.2f} {np.trace(I)/3:+12.4f}')
Tc=pd.read_csv(f'{K}/banktrials/banktrials_chair.csv')
def load2(name):   # chair_full was scored on the chair-only trial file; the ViT-S runs on the all-category file
    if name=='chair_full': o=pd.read_csv(f'{K}/eval_bank/chair_full/ood_analysis_results.csv'); return o, Tc, np.ones(len(o),bool)
    o=load(name); return o, T, chair
print('\nSINGLE MODELS ON THE FULL 2,000-CHAIR BANK')
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True)
X=bz['X'].astype(float); ids=np.array(bz['ids']); sel=np.array([i.startswith('03001627/') for i in ids]); Xc=X[sel]; oix={i.split('/')[1]:k for k,i in enumerate(ids[sel])}
mu,sd=Xc.mean(0),Xc.std(0); sd[sd<1e-9]=1; Xn=(Xc-mu)/sd; Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12
def partial(x,y,z):
    rxy,rxz,ryz=[stats.pearsonr(a,b)[0] for a,b in [(x,y),(x,z),(y,z)]]; return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))
print(f'{"model":44s} {"margin chair":>12s} {"r(margin, pretrained)":>21s} {"r(margin, dist to bank)":>23s} {"same, pretrained":>16s} {"pair held fixed: ft / pre":>26s}')
for name,label in [('chair_full','ViT-L LoRA lr 1e-6, 2,000 chairs (rung 0)'),('chair_full_smallft','ViT-S pretrained, full FT, 2,000 chairs (3b)'),('chair_full_scratch_s','ViT-S FROM SCRATCH, 2,000 chairs (rung 3)')]:
    o,TT,mask=load2(name); f=o.fine_tuned_oddity_margin.values[mask]; pr=o.pretrained_oddity_margin.values[mask]
    tobjs=[sorted({n[len('chair_'):-8] for n in ast.literal_eval(s_)}) for s_ in TT.images.values[mask]]
    sp=json.load(open(f'{K}/design_clusters_all.json'))['chair']['split']; test_objs=set(x for c in sp for x in sp[c]['test']); S=Xn[[oix[x] for x in oix if x not in test_objs]]
    dfull=np.array([np.mean([np.sort(1-Xn[oix[x]]@S.T)[:10].mean() for x in l]) for l in tobjs])
    pair=np.array([1-Xn[oix[l[0]]]@Xn[oix[l[-1]]] for l in tobjs])
    print(f'{label:44s} {f.mean():12.3f} {stats.pearsonr(f,pr)[0]:+21.2f} {stats.pearsonr(dfull,f)[0]:+23.2f} {stats.pearsonr(dfull,pr)[0]:+16.2f} {partial(dfull,f,pair):+13.2f} / {partial(dfull,pr,pair):+.2f}')
