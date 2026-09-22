"""Trials built from the TRAINING objects (lead, 2026-09-21): for each cluster model's 25 training objects, 3 oddity trials
per object with the distractor among its 5 nearest training-set neighbours (same model's set), at the training views.
Scored by every model: one model sees each trial at distance 0 (memorisation anchor), the rest at graded distances."""
import json, os, numpy as np, pandas as pd
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'; R=f'{NAV}/Dist-shift-data/shapenet_rendered'
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); B=bz['X'].astype(float); ids=np.array(bz['ids']); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12; bix={i:k for k,i in enumerate(ids)}
D=json.load(open(f'{K}/design_clusters_all.json')); rng=np.random.default_rng(11); imgdir=f'{K}/banktrials/images'; rows=[]
for cat,syn in SYN.items():
    for c in D[cat]['split']:
        tr=D[cat]['split'][c]['train'][:25]; X=BN[[bix[f'{syn}/{o}'] for o in tr]]; Dm=1-X@X.T; np.fill_diagonal(Dm,np.inf); NN=np.argsort(Dm,axis=1)[:,:5]
        for i,a in enumerate(tr):
            for r in range(3):
                b=tr[rng.choice(NN[i])]; va=rng.choice(15,2,replace=False); vb=rng.integers(15)
                names=[f'{cat}_{a}_{va[0]:03d}.png',f'{cat}_{a}_{va[1]:03d}.png',f'{cat}_{b}_{vb:03d}.png']
                for nm in names:
                    ob,vv=nm[len(cat)+1:-4].rsplit('_',1); dst=f'{imgdir}/{nm}'
                    if not os.path.lexists(dst): os.symlink(f'{R}/white/{cat}/{ob}/{vv}.png',dst)
                order=rng.permutation(3); names=[names[j] for j in order]; odd=int(np.where(order==2)[0][0])
                rows.append((cat,f'{cat}_train_c{c}',f'{cat}_c{c}_train_{a}_{r}',odd,str(names),float(Dm[i,tr.index(b)])))
T=pd.DataFrame(rows,columns=['dataset','condition','trial','oddity_index','images','pair_distance']); T.to_csv(f'{K}/banktrials/traintrials_all.csv',index=False)
print('training-object trials:',len(T),'| conditions:',T.condition.nunique(),'| images now linked:',len(os.listdir(imgdir)))
