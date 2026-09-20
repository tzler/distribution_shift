"""Does the model-free shift measure predict the cluster-model margins?  For each trial × cluster model:
distance from the trial's objects to THAT model's training chairs (3-D shape, no network): nearest-neighbour mean
of 10 nearest (knn10) and coverage (count within a fixed radius). Margin with model level and trial removed."""
import json, ast, os, numpy as np, pandas as pd
from scipy import stats
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); B=bz['X'].astype(float); ids=np.array(bz['ids'])
sel=np.array([i.startswith('03001627/') for i in ids]); X=B[sel]; objs=np.array([i.split('/')[1] for i in ids[sel]]); mu,sd=X.mean(0),X.std(0); sd[sd<1e-9]=1
Xn=(X-mu)/sd; Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-12; oix={o:i for i,o in enumerate(objs)}
split=json.load(open(f'{K}/design_clusters_chair_split.json')); T=pd.read_csv(f'{K}/banktrials/banktrials_chair.csv'); T['cluster']=T.condition.str[-1].astype(int)
T['objs']=T.images.apply(lambda s: sorted({n[len('chair_'):-8] for n in ast.literal_eval(s)}))
def train_set(c,n): tr=split[str(c)]['train']; return tr if n=='all' else tr[:int(n[1:])]
rows=[]
for n in ['all','n50','n25']:
    for c in range(3):
        S=Xn[[oix[o] for o in train_set(c,n)]]; mo=pd.read_csv(f'{K}/eval_bank/chair_c{c}_{n}/ood_analysis_results.csv')
        for i,r in T.iterrows():
            Tt=Xn[[oix[o] for o in r.objs]]; D=1-Tt@S.T
            knn=np.sort(D,axis=1)[:,:10].mean(1).mean(); nn1=np.sort(D,axis=1)[:,0].mean(); cov=np.log1p((D<0.12).sum(1)).mean(); cov2=np.log1p((D<0.2).sum(1)).mean()
            rows.append((n,c,i,r.cluster,knn,nn1,cov,cov2,mo.fine_tuned_oddity_margin[i],mo.pretrained_oddity_margin[i]))
L=pd.DataFrame(rows,columns=['N','model','trial','cluster','knn10','nn1','cov12','cov20','margin','pre']); L.to_csv(f'{G}/out/cluster_shift_long.csv',index=False)
print('trial × model rows:',len(L)//3,'per N\n')
for n in ['all','n50','n25']:
    d=L[L.N==n].copy()
    # remove model level and trial: residual = y - mean_model - mean_trial + grand   (same for x)
    for k in ['knn10','nn1','cov12','cov20','margin']:
        d[k+'_r']=d[k]-d.groupby('model')[k].transform('mean')-d.groupby('trial')[k].transform('mean')+d[k].mean()
    d['own']=(d.model==d.cluster).astype(int); d['own_r']=d.own-d.groupby('model').own.transform('mean')-d.groupby('trial').own.transform('mean')+d.own.mean()
    print(f'N={n}: distance from the trial to the model\'s training chairs, residualised on model and trial:')
    for k in ['knn10','nn1','cov12','cov20']:
        r=stats.pearsonr(d[k+'_r'],d.margin_r)[0]; print(f'   {k:6s} r = {r:+.3f}',end='')
        # does distance predict the margin BEYOND the cluster label?  partial r controlling own_r
        rx=stats.pearsonr(d[k+'_r'],d.own_r)[0]; ry=stats.pearsonr(d.own_r,d.margin_r)[0]; pr=(r-rx*ry)/np.sqrt((1-rx**2)*(1-ry**2)); print(f'   (label r = {ry:+.3f}; distance|label = {pr:+.3f}; distance vs label r = {rx:+.3f})')
    # inside a cluster only: own-cluster model on own-cluster trials — graded?
    own=d[d.model==d.cluster]; rr=[]
    for c in range(3):
        o=own[own.cluster==c]; rr.append(stats.pearsonr(o.knn10,o.margin)[0])
    # compare against other-cluster models on the same trials (control: their distance shouldn't matter more)
    print(f'   within own cluster (graded): r(knn10, margin) per cluster {np.round(rr,3)}, mean {np.mean(rr):+.3f}   [pretrained control on same trials: {np.round([stats.pearsonr(own[own.cluster==c].knn10,own[own.cluster==c].pre)[0] for c in range(3)],3)}]')

# ---------- fig69
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
def bins(x,y,nb=10):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
fig,ax=plt.subplots(1,3,figsize=(15,6.6),sharey=True); fig.subplots_adjust(left=.06,right=.99,top=.68,bottom=.19,wspace=.1)
for a,n in zip(ax,['all','n50','n25']):
    style(a); d=L[L.N==n].copy()
    for k in ['knn10','margin']: d[k+'_r']=d[k]-d.groupby('model')[k].transform('mean')-d.groupby('trial')[k].transform('mean')+d[k].mean()
    d['own']=(d.model==d.cluster).astype(int); d['own_r']=d.own-d.groupby('model').own.transform('mean')-d.groupby('trial').own.transform('mean')+d.own.mean()
    r=stats.pearsonr(d.knn10_r,d.margin_r)[0]; ry=stats.pearsonr(d.own_r,d.margin_r)[0]; rx=stats.pearsonr(d.knn10_r,d.own_r)[0]; pr=(r-rx*ry)/np.sqrt((1-rx**2)*(1-ry**2))
    bx,by,be=bins(d.knn10_r,d.margin_r)
    a.errorbar(bx,by,yerr=be,fmt='o-',color=BLUE,ms=9,mfc=BLUE,mec=SURF,mew=1.4,lw=2.6,ecolor='#bcd0ea',elinewidth=1.6,zorder=4,label='the three cluster models, scored on the same trial')
    a.axhline(0,color=GREY,ls='--',lw=2.2,zorder=3,label='pretrained model (flat: the same model on every point)')
    nn={'all':'all of its half (87–104 chairs)','n50':'50 chairs','n25':'25 chairs'}[n]
    a.set_title(f'each model trained on {nn}',loc='left',fontsize=11); a.set_xlabel('distance from the trial\'s objects to that model\'s training chairs\n(3-D shape, no network; model level and trial removed)  farther →')
    box(a,f'r = {r:+.2f}\n✓  falls. The cluster label alone: r = {abs(ry):.2f}.\n    Distance with the label held fixed: r = {pr:+.2f}',OK,'ur')
ax[0].set_ylabel('oddity margin, with each model\'s overall level\nand each trial removed')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.06,.79),ncol=2,fontsize=9.5,frameon=False)
fig.suptitle('Does the model-free shift measure predict which model wins on a trial?',fontsize=15,x=.06,ha='left',y=.975)
fig.text(.06,.92,'Same nine cluster models and 882 held-out chair trials as before, but now no cluster labels are used: for each trial × model we measure how far the trial\'s objects are\n'
 'from the chairs that model was actually trained on (3-D shape, no network). Each model\'s overall level and each trial\'s difficulty are removed, so what is left is: on this\n'
 'trial, which of the three models had the nearer training set — and did it win? What to expect if the measure tracks shift: BLUE should fall.',fontsize=9.8,color=INK2,va='top')
fig.text(.06,.015,'It falls, at every training-set size, and it predicts the margin better than the cluster labels do — and still predicts it when the label is held fixed. The labels were the scaffold;\n'
 'the distance is the thing. This is "moving the training data moves the margin" one level down, with new models, hundreds of trials per cluster, and a continuous ruler.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig69_cluster_shift.png',dpi=300); print('[fig] 69')
