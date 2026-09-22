"""Round 3a (lead, 2026-09-20): the chair sub-category design applied to every category, at N = 25.
Per category: k-means (k = 8) on the 16^3 shape descriptors, keep the 3 most separated clusters, split each in half,
training condition <cat>_c{0,1,2}_n25 from the training half, 3 hard oddity trials per held-out object.
One combined trial file (banktrials_all.csv) so every model is scored on every category."""
import json, os, itertools, ast, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'; R=f'{NAV}/Dist-shift-data/shapenet_rendered'
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); B=bz['X'].astype(float); ids=np.array(bz['ids'])
rng=np.random.default_rng(0); design={}; trials=[]; imgdir=f'{K}/banktrials/images'; os.makedirs(imgdir,exist_ok=True)
for cat,syn in SYN.items():
    rendered=set(os.listdir(f'{R}/white/{cat}')); sel=np.array([i.startswith(syn+'/') and i.split('/')[1] in rendered for i in ids])
    X=B[sel]; objs=np.array([i.split('/')[1] for i in ids[sel]]); mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1
    Xn=(X-mu)/sd; Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12; oix={o:i for i,o in enumerate(objs)}
    if cat=='chair':   # keep the existing chair design so the earlier models stay comparable
        old=json.load(open(f'{K}/design_clusters_chair_split.json')); clusters={c:old[c]['train']+old[c]['test'] for c in old}; split=old
    else:
        Z=PCA(20,random_state=0).fit_transform(Xn); km=KMeans(8,n_init=20,random_state=0).fit(Z); lab=km.labels_; C=km.cluster_centers_
        D=np.linalg.norm(C[:,None]-C[None],axis=2); best=max(itertools.combinations(range(8),3),key=lambda t:min(D[a,b] for a,b in itertools.combinations(t,2)))
        clusters={str(i):objs[lab==a].tolist() for i,a in enumerate(best)}; split={}
        for c,o in clusters.items():
            o=np.array(o); rng.shuffle(o); h=len(o)//2; split[c]={'train':o[:h].tolist(),'test':o[h:].tolist()}
    design[cat]={'clusters':{c:{'n':len(v)} for c,v in clusters.items()},'split':split}
    for c in split:
        tr=split[c]['train'][:25]; cond=f'c{c}_n25'; root=f'{K}/data/{cat}_{cond}'
        for bg in ['white','black','random']:
            d=f'{root}/{bg}/{cat}'; os.makedirs(d,exist_ok=True)
            for ob in tr:
                dst=f'{d}/{ob}'
                if not os.path.lexists(dst): os.symlink(f'{R}/{bg}/{cat}/{ob}',dst)
        te=split[c]['test']; Te=Xn[[oix[x] for x in te]]; Dm=1-Te@Te.T; np.fill_diagonal(Dm,np.inf); NN=np.argsort(Dm,axis=1)[:,:10]
        for i,a in enumerate(te):
            for r in range(3):
                b=te[rng.choice(NN[i])]; va=rng.choice(15,2,replace=False); vb=rng.integers(15)
                names=[f'{cat}_{a}_{va[0]:03d}.png',f'{cat}_{a}_{va[1]:03d}.png',f'{cat}_{b}_{vb:03d}.png']
                for nm in names:
                    ob,vv=nm[len(cat)+1:-4].rsplit('_',1); dst=f'{imgdir}/{nm}'
                    if not os.path.lexists(dst): os.symlink(f'{R}/white/{cat}/{ob}/{vv}.png',dst)
                order=rng.permutation(3); names=[names[j] for j in order]; odd=int(np.where(order==2)[0][0])
                trials.append((cat,f'{cat}_cluster{c}',f'{cat}_c{c}_{a}_{r}',odd,str(names),float(1-Xn[oix[a]]@Xn[oix[b]])))
    print(f'{cat}: clusters {[len(v) for v in clusters.values()]}, train 25 each, test halves {[len(split[c]["test"]) for c in split]}')
T=pd.DataFrame(trials,columns=['dataset','condition','trial','oddity_index','images','pair_distance']); T.to_csv(f'{K}/banktrials/banktrials_all.csv',index=False)
json.dump(design,open(f'{K}/design_clusters_all.json','w'),indent=1)
print('trials:',len(T),'| images linked:',len(os.listdir(imgdir)))
