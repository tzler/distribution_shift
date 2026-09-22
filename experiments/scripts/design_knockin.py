"""Knock-in design: small training subsets of chairs, random and targeted, from pretrained.
random_N_r  : N random chair objects, seed r
target_N_t  : the N nearest chair objects (voxel16 cosine, oddity-blind: nearest to any of the
              trial's objects, by rank) to target trial t
cross_N_t   : the N nearest AIRPLANE objects to the same trial (control: adding the wrong data)"""
import ast, os, json, numpy as np, pandas as pd
NAV='/vast/projects/bonnen/naturalistic-navig'; G=f'{NAV}/Dist-shift-data/geometric_shift'; K=f'{NAV}/Dist-shift-data/knockout'; R=f'{NAV}/Dist-shift-data/shapenet_rendered/white'
S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
SYN={'chair':'03001627','airplane':'02691156'}; N=100; NRAND=8; NTARGET=6; rng=np.random.default_rng(7)
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); tz=np.load(f'{G}/bank/test_voxel16.npz',allow_pickle=True)
B,T=bz['X'].astype(float),tz['X'].astype(float); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12; TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-12
bids=np.array(bz['ids']); tids=np.array(tz['ids']); tix={k:i for i,k in enumerate(tids)}
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
o=pd.read_csv(f'{S}/chair/ood_analysis_results.csv'); o['trial']=m['trial'].values
def bank(cat):
    syn=SYN[cat]; rendered=set(os.listdir(f'{R}/{cat}')); sel=np.array([i.startswith(syn+'/') and i.split('/')[1] in rendered for i in bids]); return BN[sel],np.array([i.split('/')[1] for i in bids[sel]])
Bc,cobj=bank('chair'); Ba,aobj=bank('airplane')
# chair trials with voxels
rows=[]
for _,r in mm.iterrows():
    ims=ast.literal_eval(r['images']); objs=['/'.join(f[:-4].split('_')[:2]) for f in ims]
    if all(x.startswith(SYN['chair']+'/') for x in objs) and all(x in tix for x in objs):
        Tt=TN[[tix[x] for x in objs]]; D=1.0-Tt@Bc.T
        rows.append(dict(trial=r['trial'],objs=objs,pre=float(o.loc[o.trial==r['trial'],'pretrained_oddity_margin'].iloc[0]),
                         ft=float(o.loc[o.trial==r['trial'],'fine_tuned_oddity_margin'].iloc[0]),nn_dist=float(D.min(1).mean()),cov12=float((D<0.12).sum(1).mean())))
tr=pd.DataFrame(rows).sort_values('pre'); print(f'{len(tr)} chair trials; pretrained margin median {tr.pre.median():.3f}')
# targets: lowest pretrained margin among trials that HAVE neighbours to select (nearest-neighbour distance below the chair median)
cand=tr[tr.nn_dist<tr.nn_dist.median()].head(NTARGET); print('\nTARGET TRIALS (low pretrained margin, neighbours available):'); print(cand[['trial','pre','ft','nn_dist','cov12']].to_string(index=False))
design={'N':N,'random':{},'target':{},'cross':{}}
for r_ in range(NRAND): design['random'][f'random_{N}_{r_}']=sorted(rng.choice(cobj,N,replace=False).tolist())
for _,t in cand.iterrows():
    Tt=TN[[tix[x] for x in t.objs]]
    D=1.0-Tt@Bc.T; rank=np.argsort(D.min(0)); design['target'][f'target_{N}_{t.trial}']={'trial':t.trial,'objects':sorted(cobj[rank[:N]].tolist()),'mean_dist_selected':float(np.sort(D.min(0))[:N].mean())}
    Da=1.0-Tt@Ba.T; ranka=np.argsort(Da.min(0)); design['cross'][f'cross_{N}_{t.trial}']={'trial':t.trial,'objects':sorted(aobj[ranka[:N]].tolist()),'mean_dist_selected':float(np.sort(Da.min(0))[:N].mean())}
# how much does each random subset cover each target trial? (the x for condition 2)
def xs(D,mask):  # the trial's distance to a training subset: nearest neighbour, and mean of 10 nearest (oddity-blind: mean over the trial's objects)
    Ds=D[:,mask]; return float(Ds.min(1).mean()), float(np.sort(Ds,axis=1)[:,:10].mean())
print('\nx per subset, same definition everywhere:  nn / knn10   (targeted first, then the 8 random subsets)')
for _,t in cand.iterrows():
    Tt=TN[[tix[x] for x in t.objs]]; D=1.0-Tt@Bc.T
    tg=xs(D,np.isin(cobj,design['target'][f'target_{N}_{t.trial}']['objects']))
    rv=[xs(D,np.isin(cobj,design['random'][k])) for k in design['random']]
    print(f'  {t.trial:12s} targeted {tg[0]:.3f}/{tg[1]:.3f}   random: '+' '.join(f'{a:.2f}/{b:.2f}' for a,b in rv))
json.dump(design,open(f'{K}/design_knockin.json','w'),indent=1); print(f'\nwrote design_knockin.json: {len(design["random"])} random + {len(design["target"])} targeted + {len(design["cross"])} cross = {len(design["random"])+len(design["target"])+len(design["cross"])} runs')
