"""Coverage through the figure templates that worked: fig40 (all points + bins), fig25A (pooled
accuracy with the pretrained control), fig9 (rank profile), fig41 (on-category only)."""
import ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
GEOM,HL,GREY,SURF,INK,INK2='#2a78d6','#eb6834','#8a8884','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans',
 'text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117',
 'lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
INV={v:k for k,v in SYN.items()}; EPS=0.12

# ---- build coverage and knn_mean per (trial, category), voxel16
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); tz=np.load(f'{G}/bank/test_voxel16.npz',allow_pickle=True)
B,T=bz['X'].astype(float),tz['X'].astype(float); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12; TN=(T-mu)/sd; TN/=np.linalg.norm(TN,axis=1,keepdims=True)+1e-12
bsyn=np.array([i.split('/')[0] for i in bz['ids']]); tix={k:i for i,k in enumerate(tz['ids'])}; D=1.0-TN@BN.T
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mm=m[m.dataset=='shapenet']
per={c:dict(coverage=-np.log1p((D[:,bsyn==s]<EPS).sum(1)), count=(D[:,bsyn==s]<EPS).sum(1),
            knn=np.sort(np.partition(D[:,bsyn==s],49,axis=1)[:,:50],axis=1).mean(1)) for c,s in SYN.items()}
rows=[]
for c in SYN:
    for _,r in mm.iterrows():
        keys=['/'.join(f[:-4].split('_')[:2]) for f in ast.literal_eval(r['images'])]
        if all(k in tix for k in keys):
            ix=[tix[k] for k in keys]; rows.append(dict(trial=r['trial'],category=c,**{k:float(v[ix].mean()) for k,v in per[c].items()}))
P=pd.DataFrame(rows)
Y=[]
for c in SYN:
    o=pd.read_csv(f'{S}/{c}/ood_analysis_results.csv'); o['trial']=m['trial'].values; o['category']=c
    o['ft']=o['fine_tuned_oddity_margin']; o['pre']=o['pretrained_oddity_margin']; o['fc']=o['fine_tuned_correct']; o['pc']=o['pretrained_correct']
    Y.append(o[['trial','category','ft','pre','fc','pc']])
d=P.merge(pd.concat(Y),on=['trial','category'])
own={}
for _,r in mm.iterrows():
    s={f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s)==1: own[r['trial']]=INV.get(list(s)[0])
d['own']=d.trial.map(own); d=d.dropna(subset=['own']); d=d[d.groupby('trial').trial.transform('size')==12].copy()
d.to_csv(f'{G}/out/coverage_shapenet_voxel16_percat.csv',index=False)
print(f'n={len(d)} trials={d.trial.nunique()}  coverage zero-count fraction: on-cat {100*(d[d.category==d.own]["count"]<0.5).mean():.0f}%  off-cat {100*(d[d.category!=d.own]["count"]<0.5).mean():.0f}%')

CB=[-0.5,0.5,2.5,5.5,10.5,20.5,50.5,100.5,200.5,1e9]; CL=['0','1–2','3–5','6–10','11–20','21–50','51–100','101–200','>200']
def cbins(cnt,y):
    q=pd.cut(cnt,CB,labels=False); g=pd.DataFrame({'q':q,'y':y}).groupby('q'); return g.y.mean(),g.y.sem(),g.size()
def cax(a):
    a.set_xticks(range(len(CL))); a.set_xticklabels(CL,fontsize=8.6); a.set_xlabel('# training objects of that category within ε of the trial\'s objects'); a.invert_xaxis()
def qbins(x,y,nb):
    q=pd.qcut(x.rank(method='first'),nb,labels=False)      # rank-based so ties don't collapse bins
    g=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return g.x.mean(),g.y.mean(),g.y.sem()

# ================= fig51: every trial x every model  (fig40 template), x = training mass nearby
fig,ax=plt.subplots(1,3,figsize=(19,5.4)); fig.subplots_adjust(left=.05,right=.99,top=.70,bottom=.19,wspace=.28)
lc=np.log1p(d['count']); sl,ic,r,p,_=stats.linregress(lc,d.ft)
a=ax[0]; style(a); jit=lc+np.random.default_rng(0).normal(0,.04,len(lc))
a.scatter(jit,d.ft,s=5,color=GEOM,alpha=.16,linewidths=0,zorder=2); xs=np.linspace(0,lc.max(),50); a.plot(xs,ic+sl*xs,color=GEOM,lw=2.2,zorder=3)
a.set_title(f'every point: one trial × one model\nn = {len(d):,}   r = {r:+.3f}, p = {p:.1e}',loc='left'); a.set_xlabel('log(1 + #training objects within ε)   (jittered)'); a.set_ylabel("that model's oddity margin (raw)")
a.text(0.05,a.get_ylim()[1]*.93,f'{100*(d["count"]<.5).mean():.0f}% of pairs sit at 0:\nno training objects nearby',fontsize=8.3,color=INK2,va='top')
for k,(yf,yp,yl) in enumerate([('ft','pre','oddity margin'),('fc','pc','accuracy')]):
    a=ax[k+1]; style(a); bf,ef,n=cbins(d['count'],d[yf]); bp,ep,_=cbins(d['count'],d[yp]); xs=np.arange(len(bf))
    a.errorbar(xs,bp,yerr=ep,fmt='o--',color=GREY,ms=7,mfc=GREY,mec=SURF,mew=1.2,lw=1.8,ecolor='#d5d3ce',elinewidth=1.3,zorder=3,label='pretrained (control)')
    a.errorbar(xs,bf,yerr=ef,fmt='o-',color=GEOM,ms=8,mfc=GEOM,mec=SURF,mew=1.3,lw=2.2,ecolor='#bcd0ea',elinewidth=1.5,zorder=4,label='fine-tuned')
    for kk,nn in enumerate(n): a.text(kk,a.get_ylim()[0]+.01*(a.get_ylim()[1]-a.get_ylim()[0]),f'{nn:,}',ha='center',fontsize=7.2,color=INK2)
    a.set_title(f'{yl} by training mass nearby\nfine-tuned r = {stats.pearsonr(d.coverage,d[yf])[0]:+.3f}   pretrained r = {stats.pearsonr(d.coverage,d[yp])[0]:+.3f}',loc='left'); cax(a); a.set_ylabel(yl); a.legend(fontsize=8.3,loc='upper right')
fig.suptitle('Every trial × every model — COVERAGE  (voxel16, ε = 0.12)',fontsize=15,x=.05,ha='left',y=.965)
fig.text(.05,.855,'Same 8,472 observations as fig 40, x replaced by how many of that category\'s training objects sit within cosine radius ε of the trial\'s objects (mean over images).\n'
         'Coverage saturates: 85 % of off-category pairs and 13 % of on-category pairs have none. Bins are on the count itself; small numbers under each point are bin sizes.',fontsize=9.2,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig51_coverage_allpoints.png',dpi=300); plt.close(fig); print('[fig] 51')

# ================= fig52: pooled accuracy + margin with the pretrained control  (fig25A template), knn vs coverage
fig,ax=plt.subplots(2,2,figsize=(14.5,10.6)); fig.subplots_adjust(left=.06,right=.985,top=.83,bottom=.07,hspace=.48,wspace=.22)
for i,(yf,yp,yl) in enumerate([('fc','pc','accuracy'),('ft','pre','oddity margin')]):
    # left: knn_mean, rank-quantile bins
    a=ax[i,0]; style(a); bx,bf,ef=qbins(d.knn,d[yf],12); _,bp,ep=qbins(d.knn,d[yp],12)
    a.errorbar(bx,bp,yerr=ep,fmt='o--',color=GREY,ms=7.5,mfc=GREY,mec=SURF,mew=1.2,lw=2,ecolor='#d5d3ce',elinewidth=1.3,zorder=3,label='pretrained DINOv2 (control)')
    a.errorbar(bx,bf,yerr=ef,fmt='o-',color=GEOM,ms=8.5,mfc=GEOM,mec=SURF,mew=1.3,lw=2.4,ecolor='#bcd0ea',elinewidth=1.5,zorder=4,label='fine-tuned on that category')
    rf=stats.pearsonr(d.knn,d[yf])[0]; rp=stats.pearsonr(d.knn,d[yp])[0]
    a.set_title(f'knn_mean  (distance to the 50 nearest training objects)\nfine-tuned r = {rf:+.3f}     pretrained r = {rp:+.3f}',loc='left'); a.set_xlabel('knn_mean shift  (12 quantile bins)'); a.set_ylabel(yl)
    # right: coverage, count bins
    a=ax[i,1]; style(a); bf,ef,n=cbins(d['count'],d[yf]); bp,ep,_=cbins(d['count'],d[yp]); xs=np.arange(len(bf))
    a.errorbar(xs,bp,yerr=ep,fmt='o--',color=GREY,ms=7.5,mfc=GREY,mec=SURF,mew=1.2,lw=2,ecolor='#d5d3ce',elinewidth=1.3,zorder=3,label='pretrained DINOv2 (control)')
    a.errorbar(xs,bf,yerr=ef,fmt='o-',color=GEOM,ms=8.5,mfc=GEOM,mec=SURF,mew=1.3,lw=2.4,ecolor='#bcd0ea',elinewidth=1.5,zorder=4,label='fine-tuned on that category')
    for k,nn in enumerate(n): a.text(k,a.get_ylim()[0]+.01*(a.get_ylim()[1]-a.get_ylim()[0]),f'n={nn:,}',ha='center',fontsize=7.4,color=INK2)
    rf=stats.pearsonr(d.coverage,d[yf])[0]; rp=stats.pearsonr(d.coverage,d[yp])[0]
    a.set_title(f'coverage  (training mass within ε = 0.12)\nfine-tuned r = {rf:+.3f}     pretrained r = {rp:+.3f}',loc='left'); cax(a); a.set_ylabel(yl)
    if i==0:
        for a in ax[0]: a.axhline(1/3,color='#c9c7c2',lw=1.1,zorder=1); a.legend(fontsize=8.5,loc='upper right')
        ax[0,0].text(ax[0,0].get_xlim()[1],1/3+.01,'chance',ha='right',fontsize=8.5,color=INK2)
fig.suptitle('The pooled row, fig 25 style: distance vs coverage, with the pretrained control',fontsize=15,x=.06,ha='left',y=.975)
fig.text(.06,.925,'All 8,472 observations (706 trials × 12 category models), raw x. Grey is the control: the same trials scored by the pretrained encoder, which never trained on any of these sets.\n'
         'LEFT — under a distance measure the grey line falls with the blue one: the pooled relationship contains object atypicality, which every model shares.\n'
         'RIGHT — under coverage the grey line is flat and the blue one still falls. x is read right-to-left as increasing shift; the leftmost bin is "no training objects nearby".',fontsize=9.3,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig52_coverage_pooled_control.png',dpi=300); plt.close(fig); print('[fig] 52')

# ================= fig53: rank profile (fig9) + on-category only (fig41), knn vs coverage
fig,ax=plt.subplots(1,4,figsize=(22,5.2)); fig.subplots_adjust(left=.045,right=.99,top=.70,bottom=.16,wspace=.3)
for j,(col,lbl) in enumerate([('knn','knn_mean'),('coverage','coverage')]):
    d['rk']=d.groupby('trial')[col].rank(method='first').astype(int); g=d.groupby('rk').agg(x=(col,'mean'),y=('ft','mean'),e=('ft','sem'),pre=('pre','mean'))
    a=ax[j]; style(a); a.errorbar(g.x,g.y,yerr=g.e,fmt='o-',color=GEOM,ms=8,mfc=GEOM,mec=SURF,mew=1.4,lw=2.2,ecolor='#bcd0ea',elinewidth=1.5,zorder=4)
    a.axhline(g.pre.iloc[0],color=GREY,ls='--',lw=1.8,zorder=2); a.text(g.x.iloc[-1],g.pre.iloc[0]+.003,'pretrained (flat by construction)',ha='right',fontsize=8.3,color=GREY)
    xc=d[col]-d.groupby('trial')[col].transform('mean'); yc=d.ft-d.groupby('trial').ft.transform('mean'); rw=stats.pearsonr(xc,yc)[0]
    sl=np.array([stats.linregress(s[col],s.ft).slope for _,s in d.groupby('trial') if s[col].std()>0])
    a.set_title(f'{"AB"[j]}  Rank profile · {lbl}\n12 ranks × 706 trials   within-trial r = {rw:+.3f},  {100*(sl<0).mean():.0f}% neg',loc='left'); a.set_xlabel(f'{lbl} shift (mean within rank)'); a.set_ylabel('mean oddity margin')
w=d[d.category==d.own]
a=ax[2]; style(a); bx,by,be=qbins(w.knn,w.ft,15); rb=stats.pearsonr(bx,by)[0]; r=stats.pearsonr(w.knn,w.ft)[0]; rc=stats.pearsonr(w.knn,w.pre)[0]
a.errorbar(bx,by,yerr=be,fmt='o-',color=GEOM,ms=7.5,mfc=GEOM,mec=SURF,mew=1.3,lw=1.8,ecolor='#bcd0ea',elinewidth=1.4,zorder=4,label='fine-tuned')
_,bp,ep=qbins(w.knn,w.pre,15); a.errorbar(bx,bp,yerr=ep,fmt='o--',color=GREY,ms=6.5,mfc=GREY,mec=SURF,mew=1.2,lw=1.6,ecolor='#d5d3ce',elinewidth=1.2,zorder=3,label='pretrained')
a.set_title(f'C  On-category only · knn_mean\nn = 706   r = {r:+.3f} (15-bin {rb:+.2f})   control {rc:+.3f}',loc='left'); a.set_xlabel("knn_mean shift to OWN category's training set"); a.set_ylabel('oddity margin'); a.legend(fontsize=8.3,loc='upper right')
a=ax[3]; style(a); bf,ef,n=cbins(w['count'],w.ft); bp,ep,_=cbins(w['count'],w.pre); xs=np.arange(len(bf)); r=stats.pearsonr(w.coverage,w.ft)[0]; rc=stats.pearsonr(w.coverage,w.pre)[0]; rb=stats.pearsonr(xs,bf.values)[0]
a.errorbar(xs,bp,yerr=ep,fmt='o--',color=GREY,ms=6.5,mfc=GREY,mec=SURF,mew=1.2,lw=1.6,ecolor='#d5d3ce',elinewidth=1.2,zorder=3,label='pretrained')
a.errorbar(xs,bf,yerr=ef,fmt='o-',color=GEOM,ms=7.5,mfc=GEOM,mec=SURF,mew=1.3,lw=1.8,ecolor='#bcd0ea',elinewidth=1.4,zorder=4,label='fine-tuned')
for kk,nn in enumerate(n): a.text(kk,a.get_ylim()[0]+.01*(a.get_ylim()[1]-a.get_ylim()[0]),f'{nn}',ha='center',fontsize=7.2,color=INK2)
a.set_title(f'D  On-category only · coverage\nn = 706   r = {r:+.3f} (binned {-rb:+.2f})   control {rc:+.3f}',loc='left'); cax(a); a.set_ylabel('oddity margin'); a.legend(fontsize=8.3,loc='upper right')
fig.suptitle('Rank profile (fig 9) and on-category row (fig 41): knn_mean vs coverage',fontsize=15,x=.045,ha='left',y=.965)
fig.text(.045,.86,'A–B: rank each trial\'s 12 training sets nearest → farthest; every point averages the same 706 trials. C–D: keep only the model trained on each trial\'s own category, 15 rank-quantile bins.\n'
         'Coverage steepens the rank profile and the on-category row alike; the on-category control stays small under both.',fontsize=9.2,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig53_coverage_rank_oncat.png',dpi=300); plt.close(fig); print('[fig] 53')
