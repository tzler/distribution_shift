"""Shift estimate as a function of training-set size, no training needed. For the chair bank: draw training subsets of
size N (random from the whole category; random from one cluster), compute each held-out trial's distance to the subset
(nearest-neighbour mean of 10, single nearest, and coverage = log count within a radius), and plot (a) the estimate vs N,
(b) the within-trial SPREAD of the estimate across different subsets of the same size — the dose available to a design."""
import json, ast, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'
BLUE,GREY,SURF,INK,INK2,OK,BAD,GRN='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834','#1baf7a'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); B=bz['X'].astype(float); ids=np.array(bz['ids'])
sel=np.array([i.startswith('03001627/') for i in ids]); X=B[sel]; objs=np.array([i.split('/')[1] for i in ids[sel]]); mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1
Xn=(X-mu)/sd; Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12; oix={o:i for i,o in enumerate(objs)}
split=json.load(open(f'{K}/design_clusters_chair_split.json')); T=pd.read_csv(f'{K}/banktrials/banktrials_chair.csv'); T['cluster']=T.condition.str[-1].astype(int)
T['objs']=T.images.apply(lambda s: sorted({n[len('chair_'):-8] for n in ast.literal_eval(s)}))
test_objs=set(o for l in T.objs for o in l); pool=np.array([o for o in objs if o not in test_objs])   # anything not in a held-out trial can be training
Tt=[Xn[[oix[o] for o in l]] for l in T.objs]
def est(S):
    D=[1-t@S.T for t in Tt]
    return (np.array([np.sort(d,axis=1)[:,:min(10,S.shape[0])].mean(1).mean() for d in D]),
            np.array([np.sort(d,axis=1)[:,0].mean() for d in D]),
            np.array([np.log1p((d<0.12).sum(1)).mean() for d in D]))
rng=np.random.default_rng(0); Ns=[10,25,50,100,200,400,800,len(pool)]; R=8
rows=[]
for N in Ns:
    draws=[]
    for r in range(R if N<len(pool) else 1):
        S=Xn[[oix[o] for o in rng.choice(pool,N,replace=False)]]; k,n1,c=est(S); draws.append((k,n1,c))
        rows.append(('random',N,r,k.mean(),n1.mean(),c.mean(),(c==0).mean()))
    if len(draws)>1:
        for j,name in enumerate(['knn10','nn1','cov']):
            M=np.stack([d[j] for d in draws]); rows.append(('random_spread_'+name,N,-1,M.std(0).mean(),M.mean(1).std(),np.nan,np.nan))
# cluster subsets at the sizes we trained
for c in range(3):
    tr=split[str(c)]['train']
    for N in [25,50,len(tr)]:
        S=Xn[[oix[o] for o in tr[:N]]]; k,n1,cv=est(S)
        for t in range(3):
            m=T.cluster.values==t; rows.append((f'cluster{c}_on_{t}',N,0,k[m].mean(),n1[m].mean(),cv[m].mean(),(cv[m]==0).mean()))
L=pd.DataFrame(rows,columns=['kind','N','draw','knn10','nn1','cov','frac_zero']); L.to_csv(f'{G}/out/shift_vs_setsize.csv',index=False)
rnd=L[L.kind=='random'].groupby('N').agg(knn10=('knn10','mean'),knn_sd=('knn10','std'),nn1=('nn1','mean'),cov=('cov','mean'),frac_zero=('frac_zero','mean'))
sp=L[L.kind.str.startswith('random_spread')].pivot(index='N',columns='kind',values='knn10')
print(rnd.round(3).to_string()); print('\nwithin-trial spread across 8 random subsets (sd of the estimate for a fixed trial):'); print(sp.round(4).to_string())
# ---------- figure
fig,ax=plt.subplots(1,3,figsize=(16,5.8)); fig.subplots_adjust(left=.06,right=.97,top=.72,bottom=.16,wspace=.42)
a=ax[0]; style(a); a.plot(rnd.index,rnd.knn10,'o-',color=BLUE,ms=8,lw=2.4,label='distance to the 10 nearest training objects (mean)')
a.plot(rnd.index,rnd.nn1,'o--',color=BLUE,ms=7,lw=1.8,alpha=.6,label='distance to the single nearest')
for c,col in zip(range(3),['#2a78d6','#1baf7a','#eb6834']):
    own=L[L.kind==f'cluster{c}_on_{c}']; oth=L[L.kind.str.startswith(f'cluster{c}_on_')&(L.kind!=f'cluster{c}_on_{c}')].groupby('N').knn10.mean()
    a.plot(own.N,own.knn10,'s',color=col,ms=8,mec=SURF,zorder=5); a.plot(oth.index,oth.values,'s',color=col,ms=8,mfc='none',mew=1.8,zorder=5)
a.plot([],[],'s',color=INK,ms=8,label='cluster model on its own cluster (filled) / other clusters (open)')
a.set_xscale('log'); a.set_xlabel('training-set size (objects)'); a.set_ylabel('shift estimate: distance from a held-out trial\nto the training set (3-D shape)'); a.set_title('the estimate itself',loc='left')
a.legend(fontsize=8.5,loc='lower left',frameon=False)
a=ax[1]; style(a); a.plot(rnd.index,rnd['cov'],'o-',color=GRN,ms=8,lw=2.4,label='coverage: log(1 + count within a fixed radius)'); a2=a.twinx(); a2.plot(rnd.index,100*rnd.frac_zero,'o--',color=BAD,ms=7,lw=1.8,label='% of trials with a count of zero'); a2.set_ylabel('% of trials with no training object within the radius',color=BAD,labelpad=12); a2.spines['top'].set_visible(False)
a.set_xscale('log'); a.set_xlabel('training-set size (objects)'); a.set_ylabel('coverage: log(1 + count)  (higher = more training nearby)'); a.set_title('the count-based estimate saturates at small sizes',loc='left')
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels(); a.legend(h1+h2,l1+l2,fontsize=8.5,loc='center right',frameon=False)
a=ax[2]; style(a); a.plot(sp.index,sp['random_spread_knn10'],'o-',color=BLUE,ms=8,lw=2.4,label='distance to the 10 nearest'); a.plot(sp.index,sp['random_spread_nn1'],'o--',color=BLUE,ms=7,lw=1.8,alpha=.6,label='distance to the single nearest'); a.plot(sp.index,sp['random_spread_cov'],'o-',color=GRN,ms=8,lw=2.4,label='coverage')
a.set_xscale('log'); a.set_xlabel('training-set size (objects)'); a.set_ylabel('sd of the estimate for a fixed trial\nacross 8 random training sets of that size'); a.set_title('the dose a design has to work with',loc='left'); a.legend(fontsize=8.5,frameon=False)
fig.suptitle('The shift estimate as a function of training-set size (chair bank, 882 held-out trials, no training involved)',fontsize=14.5,x=.06,ha='left',y=.975)
fig.text(.06,.91,'Left: how far a held-out trial is from a random training set of N chairs; the estimate falls as N grows (more chairs, some of them near). Cluster training sets sit\n'
 'below the random line on their own cluster and above it on others. Middle: the count-based estimate is zero for most trials until N is in the hundreds. Right: how much\n'
 'the estimate differs between two random training sets of the same size, for the same trial — the contrast an experiment can create by choosing training data.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig70_shift_vs_setsize.png',dpi=300); print('[fig] 70')
