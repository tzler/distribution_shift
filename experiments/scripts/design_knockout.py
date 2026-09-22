"""Knockout design: for each category, cluster its MOCHI test objects into G groups; for each
group and radius, list the rendered training objects within that radius (voxel16 cosine)."""
import ast, os, json, numpy as np, pandas as pd
NAV='/vast/projects/bonnen/naturalistic-navig'; G=f'{NAV}/Dist-shift-data/geometric_shift'; K=f'{NAV}/Dist-shift-data/knockout'; R=f'{NAV}/Dist-shift-data/shapenet_rendered/white'
SYN={'chair':'03001627','airplane':'02691156','table':'04379243'}; INV={v:k for k,v in SYN.items()}
CATS=['chair','airplane','table']; NG=4; KS=[10,50]; rng=np.random.default_rng(0)
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); tz=np.load(f'{G}/bank/test_voxel16.npz',allow_pickle=True)
B,T=bz['X'].astype(float),tz['X'].astype(float); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12; TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-12
bids=np.array(bz['ids']); tids=np.array(tz['ids']); tix={k:i for i,k in enumerate(tids)}
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
design={}
for cat in CATS:
    syn=SYN[cat]; rendered=set(os.listdir(f'{R}/{cat}'))
    bsel=np.array([i.startswith(syn+'/') and i.split('/')[1] in rendered for i in bids]); Bc=BN[bsel]; bobj=np.array([i.split('/')[1] for i in bids[bsel]])
    # test objects of this category that appear in usable single-category trials
    tobj=set(); trials=[]
    for _,r in mm.iterrows():
        ims=ast.literal_eval(r['images']); objs=['/'.join(f[:-4].split('_')[:2]) for f in ims]
        if all(o.startswith(syn+'/') for o in objs) and all(o in tix for o in objs): tobj.update(objs); trials.append(r['trial'])
    tobj=sorted(tobj); Tc=TN[[tix[o] for o in tobj]]
    # random balanced groups: each trial's own support is removed in exactly one condition per k
    perm=rng.permutation(len(tobj)); grp=np.empty(len(tobj),int); grp[perm]=np.arange(len(tobj))%NG+1
    D=1.0-Tc@Bc.T                                   # test x train cosine distance
    NN=np.argsort(D,axis=1)                          # each test object's training neighbours, nearest first
    def knn_mean(Dm,mask): # mean distance to the 50 nearest SURVIVING training objects
        Dm=np.where(mask[None,:],np.inf,Dm); return np.sort(Dm,axis=1)[:,:50].mean(1)
    design[cat]={'n_train':int(bsel.sum()),'n_test_obj':len(tobj),'n_trials':len(trials),'trials':trials,'groups':{},'random':{}}
    print(f'\n{cat}: {bsel.sum()} rendered training objects with voxels, {len(tobj)} test objects in {len(trials)} trials; group sizes {np.bincount(grp)[1:]}')
    for g in range(1,NG+1):
        sel=grp==g; Dg=D[sel]
        entry={'test_objects':[tobj[i] for i in np.where(sel)[0]]}
        for k in KS:
            hit=np.zeros(len(bobj),bool); hit[np.unique(NN[sel][:,:k])]=True; removed=sorted(bobj[hit].tolist())
            own_b,own_a=knn_mean(Dg,np.zeros(len(bobj),bool)).mean(),knn_mean(Dg,hit).mean()
            Do=D[~sel]; oth_b,oth_a=knn_mean(Do,np.zeros(len(bobj),bool)).mean(),knn_mean(Do,hit).mean()
            cov_b,cov_a=(Dg<0.12).sum(1).mean(),((Dg<0.12)&~hit[None,:]).sum(1).mean()
            entry[f'k{k}']={'removed':removed,'n_removed':len(removed),'own_knn_before':float(own_b),'own_knn_after':float(own_a),'other_knn_before':float(oth_b),'other_knn_after':float(oth_a),'own_cov_before':float(cov_b),'own_cov_after':float(cov_a)}
            print(f'  group {g} ({sel.sum():3d} test objs)  k={k:2d}: remove {len(removed):4d} ({100*len(removed)/bsel.sum():4.1f}%)   own knn_mean {own_b:.3f} -> {own_a:.3f} (Δ{own_a-own_b:+.3f})   other groups {oth_b:.3f} -> {oth_a:.3f} (Δ{oth_a-oth_b:+.3f})   own cov@.12 {cov_b:5.1f} -> {cov_a:4.1f}')
        design[cat]['groups'][g]=entry
    for k in KS:
        n=int(np.mean([design[cat]['groups'][g][f'k{k}']['n_removed'] for g in range(1,NG+1)]))
        removed=sorted(rng.choice(bobj,n,replace=False).tolist()); hit=np.isin(bobj,removed)
        kb,ka=knn_mean(D,np.zeros(len(bobj),bool)).mean(),knn_mean(D,hit).mean()
        design[cat]['random'][f'k{k}']={'removed':removed,'n_removed':n,'knn_before':float(kb),'knn_after':float(ka)}
        print(f'  random control k={k:2d}: remove {n:4d} at random   all-test knn_mean {kb:.3f} -> {ka:.3f} (Δ{ka-kb:+.3f})')
json.dump(design,open(f'{K}/design.json','w'),indent=1); print('\nwrote design.json')
