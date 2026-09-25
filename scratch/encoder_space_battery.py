"""Track A: the oddity-blind estimates (knn_mean, coverage) computed INSIDE each encoder's own
feature space — pretrained DINOv2-L, and the chair / airplane / table fine-tunes — against the
same 20,885-object training bank, through the same battery as the geometric estimates.
Image-level (per render) and object-level (mean over the 15 views) variants."""
import ast, os, numpy as np, pandas as pd
from scipy import stats
from _repo import G as _RESOLVED_G

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117',
 'lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
INV={v:k for k,v in SYN.items()}; EPS={'image':0.3,'object':0.2}   # cosine radii in feature space; swept below if needed
keep=set(np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True)['ids'])   # 'synset/obj'
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
Y=[]
for c in SYN:
    o=pd.read_csv(f'{S}/{c}/ood_analysis_results.csv'); o['trial']=m['trial'].values; o['category']=c; o['ft']=o['fine_tuned_oddity_margin']; o['pre']=o['pretrained_oddity_margin']; Y.append(o[['trial','category','ft','pre']])
Y=pd.concat(Y); own={}
for _,r in mm.iterrows():
    s={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s)==1: own[r['trial']]=INV.get(list(s)[0])
def battery(P,label):
    d=P.merge(Y,on=['trial','category']); d['own']=d.trial.map(own); d=d.dropna(subset=['own']); d=d[d.groupby('trial').trial.transform('size')==12]
    w=d[d.category==d.own]; out={}
    for col in ['knn','cov']:
        xc=d[col]-d.groupby('trial')[col].transform('mean'); yc=d.ft-d.groupby('trial').ft.transform('mean')
        sl=np.array([stats.linregress(s[col],s.ft).slope for _,s in d.groupby('trial') if s[col].std()>0])
        xo=w[col]-w.groupby('own')[col].transform('mean'); yo=w.ft-w.groupby('own').ft.transform('mean')
        out[col]=dict(within_r=stats.pearsonr(xc,yc)[0],pct_neg=100*(sl<0).mean(),pooled_r=stats.pearsonr(d[col],d.ft)[0],pooled_ctrl=stats.pearsonr(d[col],d.pre)[0],
                      oncat_r=stats.pearsonr(w[col],w.ft)[0],oncat_ctrl=stats.pearsonr(w[col],w.pre)[0],oncat_catcentred_r=stats.pearsonr(xo,yo)[0],oncat_catcentred_p=stats.pearsonr(xo,yo)[1])
    return d,out
rows=[]
for model in ['pretrained','ft_chair','ft_airplane','ft_table']:
    z=np.load(f'{K}/eval/encoder_features/{model}.npz',allow_pickle=True)
    tids=z['test_ids']; T=z['test_X'].astype(np.float32); bids=z['train_ids']; B=z['train_X'].astype(np.float32)
    bobj=np.array(['/'.join([SYN[i.split('/')[0]],i.split('/')[1]]) for i in bids]); sel=np.array([o in keep for o in bobj]); B=B[sel]; bobj=bobj[sel]; bsyn=np.array([o.split('/')[0] for o in bobj])
    print(f'[{model}] bank {len(B)} images / {len(set(bobj))} objects; test {len(T)}',flush=True)
    mu,sd=B.mean(0),B.std(0); sd[sd<1e-6]=1; BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-9; TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-9
    # object-level bank: mean over views
    objs,inv=np.unique(bobj,return_inverse=True); BO=np.zeros((len(objs),B.shape[1]),np.float32); np.add.at(BO,inv,BN); BO/=np.bincount(inv)[:,None]; BO/=np.linalg.norm(BO,axis=1,keepdims=True)+1e-9; osyn=np.array([o.split('/')[0] for o in objs])
    tix={k:i for i,k in enumerate(tids)}
    for level,(BB,ssyn) in {'image':(BN,bsyn),'object':(BO,osyn)}.items():
        per={}
        for c,s in SYN.items():
            Bc=BB[ssyn==s]; knn=np.empty(len(T)); cnt=np.empty(len(T))
            for i0 in range(0,len(T),512):
                D=1.0-TN[i0:i0+512]@Bc.T; knn[i0:i0+512]=np.sort(np.partition(D,49,axis=1)[:,:50],axis=1).mean(1); cnt[i0:i0+512]=(D<EPS[level]).sum(1)
            per[c]=(knn,-np.log1p(cnt))
        recs=[]
        for c in SYN:
            for _,r in mm.iterrows():
                ims=ast.literal_eval(r['images'])
                if all(f in tix for f in ims): ix=[tix[f] for f in ims]; recs.append(dict(trial=r['trial'],category=c,knn=float(per[c][0][ix].mean()),cov=float(per[c][1][ix].mean())))
        d,out=battery(pd.DataFrame(recs),f'{model}/{level}')
        for col,o in out.items():
            rows.append(dict(space=model,level=level,estimator=col,**o))
            # own-category subset for a fine-tuned space: the within-category question in the model's OWN space
            if model.startswith('ft_'):
                cat=model[3:]; w=d[(d.category==d.own)&(d.own==cat)]
                if len(w)>10: rows[-1].update(owncat_n=len(w),owncat_r=stats.pearsonr(w[col],w.ft)[0],owncat_p=stats.pearsonr(w[col],w.ft)[1],owncat_ctrl=stats.pearsonr(w[col],w.pre)[0])
        d.to_csv(f'{K}/eval/encoder_space_{model}_{level}.csv',index=False)
        print(f'  {level:6s} '+'  '.join(f'{col}: within {o["within_r"]:+.3f} pooled {o["pooled_r"]:+.3f}/{o["pooled_ctrl"]:+.3f} oncat {o["oncat_r"]:+.3f}/{o["oncat_ctrl"]:+.3f} catcentred {o["oncat_catcentred_r"]:+.3f}' for col,o in out.items()),flush=True)
R=pd.DataFrame(rows); R.to_csv(f'{K}/eval/encoder_space_battery.csv',index=False); pd.set_option('display.width',250); print(R.round(3).to_string())
