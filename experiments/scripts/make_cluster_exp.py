"""Sub-category calibration (lead's design, 2026-09-20). Three chair clusters (design_clusters_chair.json), each
split in half: a TRAIN half (models are fine-tuned on N of these) and a TEST half (oddity trials are built from these).
Conditions: chair_c{1,2,3}_n{all,50,25}. Trials: for each test-half object A, 3 trials with B drawn from A's 10 nearest
test-half neighbours in the same cluster (hard, like MOCHI), 2 random views of A + 1 of B, in MOCHI's CSV format."""
import json, os, ast, numpy as np, pandas as pd
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'; R=f'{NAV}/Dist-shift-data/shapenet_rendered'
rng=np.random.default_rng(0); cat='chair'
D=json.load(open(f'{K}/design_clusters_chair.json'))
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); B=bz['X'].astype(float); ids=np.array(bz['ids'])
sel=np.array([i.startswith('03001627/') for i in ids]); X=B[sel]; objs=np.array([i.split('/')[1] for i in ids[sel]]); mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1
Xn=(X-mu)/sd; Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12; oix={o:i for i,o in enumerate(objs)}
split={}; rows=[]; imgdir=f'{K}/banktrials/images'; os.makedirs(imgdir,exist_ok=True)
for c,info in D['clusters'].items():
    o=np.array(info['objects']); rng.shuffle(o); half=len(o)//2; tr,te=o[:half],o[half:]; split[c]={'train':tr.tolist(),'test':te.tolist()}
    # training conditions
    for n,name in [(len(tr),'all'),(50,'n50'),(25,'n25')]:
        sub=tr[:n]; cond=f'c{c}_{name}'; root=f'{K}/data/{cat}_{cond}'
        for bg in ['white','black','random']:
            d=f'{root}/{bg}/{cat}'; os.makedirs(d,exist_ok=True)
            for ob in sub:
                dst=f'{d}/{ob}'
                if not os.path.lexists(dst): os.symlink(f'{R}/{bg}/{cat}/{ob}',dst)
        print(f'{cat}_{cond}: {len(sub)} training objects')
    # test trials from the held-out half
    Te=Xn[[oix[x] for x in te]]; Dm=1-Te@Te.T; np.fill_diagonal(Dm,np.inf); NN=np.argsort(Dm,axis=1)[:,:10]
    for i,a in enumerate(te):
        for r in range(3):
            b=te[rng.choice(NN[i])]; va=rng.choice(15,2,replace=False); vb=rng.integers(15)
            names=[f'{cat}_{a}_{va[0]:03d}.png',f'{cat}_{a}_{va[1]:03d}.png',f'{cat}_{b}_{vb:03d}.png']
            for nm in names:
                ob,vv=nm[len(cat)+1:-4].rsplit('_',1); dst=f'{imgdir}/{nm}'
                if not os.path.lexists(dst): os.symlink(f'{R}/white/{cat}/{ob}/{vv}.png',dst)
            order=rng.permutation(3); names=[names[j] for j in order]; odd=int(np.where(order==2)[0][0])
            rows.append(('banktrial',f'{cat}_cluster{c}',f'{cat}_c{c}_{a}_{r}',odd,str(names)))
T=pd.DataFrame(rows,columns=['dataset','condition','trial','oddity_index','images']); T.to_csv(f'{K}/banktrials/banktrials_chair.csv',index=False)
json.dump(split,open(f'{K}/design_clusters_chair_split.json','w'),indent=1)
print('trials per cluster:',T.condition.value_counts().to_dict(), '| images linked:',len(os.listdir(imgdir)))
