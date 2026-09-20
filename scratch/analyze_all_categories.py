"""Final-form figure: distance from a trial's objects to the model's training set (x) against what training added to the
margin (y = fine-tuned − pretrained on the same trial, which cancels trial difficulty exactly). Every model with an
evaluation on banktrials_all.csv; points labelled by relationship to the training set."""
import json, ast, os, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'
SURF,INK,INK2,GREY='#fcfcfb','#0b0b0b','#52514e','#8a8884'; C_SAME,C_OTH,C_CAT='#2a78d6','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
SYN={'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117','lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
bz=np.load(f'{G}/bank/bank_voxel16.npz',allow_pickle=True); B=bz['X'].astype(float); ids=np.array(bz['ids']); mu,sd=B.mean(0),B.std(0); sd[sd<1e-9]=1
BN=(B-mu)/sd; BN/=np.linalg.norm(BN,axis=1,keepdims=True)+1e-12; bix={i:k for k,i in enumerate(ids)}   # ONE normalisation for all categories
D=json.load(open(f'{K}/design_clusters_all.json')); T=pd.read_csv(f'{K}/banktrials/banktrials_all.csv')
T['cat']=T.dataset; T['cluster']=T.condition.str[-1].astype(int); T['objs']=[[f'{SYN[c]}/'+n[len(c)+1:-8] for n in ast.literal_eval(s)] for c,s in zip(T.cat,T.images)]
T['objs']=T.objs.apply(lambda l: sorted(set(l))); Tt=[BN[[bix[o] for o in l]] for l in T.objs]
rows=[]; models=sorted(os.path.basename(p) for p in glob.glob(f'{K}/eval_bank_all/*') if os.path.exists(f'{p}/ood_analysis_results.csv'))
for name in models:
    cat=name.split('_')[0]
    if '_c' in name and name.endswith('_n25'): c=name.split('_c')[1][0]; tr=D[cat]['split'][c]['train'][:25]; kind='cluster'
    elif '_tgt_' in name: tr=json.load(open(f'{K}/design_targeted.json'))[name.split('_')[2]][{'near':'nearest','rand':'random','far':'farthest'}[name.split('_')[3]]]; c=None; kind='targeted'
    else: continue
    S=BN[[bix[f'{SYN[cat]}/{o}'] for o in tr]]; mo=pd.read_csv(f'{K}/eval_bank_all/{name}/ood_analysis_results.csv'); assert len(mo)==len(T)
    for i,t in enumerate(Tt):
        Dm=1-t@S.T; d=np.sort(Dm,axis=1)[:,:10].mean(1).mean()
        rel='other category' if T.cat[i]!=cat else ('same cluster' if (c is not None and T.cluster[i]==int(c)) else 'other cluster, same category')
        rows.append((name,kind,cat,c,i,T.cat[i],T.cluster[i],rel,d,T.pair_distance[i],mo.fine_tuned_oddity_margin[i],mo.pretrained_oddity_margin[i]))
L=pd.DataFrame(rows,columns=['model','kind','train_cat','train_cluster','trial','test_cat','test_cluster','rel','dist','pair','ft','pre']); L['adv']=L.ft-L.pre
L.to_csv(f'{G}/out/all_categories_long.csv',index=False)
print('models:',len(models),'| trial×model rows:',len(L))
print(L.groupby('rel').agg(n=('adv','size'),dist=('dist','mean'),adv=('adv','mean'),ft=('ft','mean'),pre=('pre','mean')).round(3).to_string())
for r in ['same cluster','other cluster, same category','other category']:
    d=L[L.rel==r]; print(f'  {r:30s} r(dist, adv) = {stats.pearsonr(d.dist,d.adv)[0]:+.3f}   r(dist, pre) = {stats.pearsonr(d.dist,d.pre)[0]:+.3f}')
print(f'  all rows: r(dist, adv) = {stats.pearsonr(L.dist,L.adv)[0]:+.3f}   r(dist, ft) = {stats.pearsonr(L.dist,L.ft)[0]:+.3f}   r(dist, pre) = {stats.pearsonr(L.dist,L.pre)[0]:+.3f}')
# ---------- figure: one axis
def bins(x,y,edges):
    q=pd.cut(x,edges); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q',observed=True); return gg.x.mean(),gg.y.mean(),gg.y.sem(),gg.size()
edges=np.quantile(L.dist,np.linspace(0,1,17))
fig,ax=plt.subplots(1,2,figsize=(15,6.6)); fig.subplots_adjust(left=.06,right=.98,top=.7,bottom=.19,wspace=.22)
a=ax[0]; style(a)
for r,col in [('same cluster',C_SAME),('other cluster, same category',C_OTH),('other category',C_CAT)]:
    d=L[L.rel==r]; bx,by,be,n=bins(d.dist,d.adv,edges); m=n>=30; a.errorbar(bx[m],by[m],yerr=be[m],fmt='o',color=col,ms=9,mec=SURF,mew=1.3,ecolor='#d5d3ce',zorder=4,label=f'{r}  ({len(d):,} points)')
bx,by,be,n=bins(L.dist,L.adv,edges); a.plot(bx,by,'-',color=INK,lw=2.2,zorder=3,label='all points together')
a.axhline(0,color=GREY,ls='--',lw=2,label='pretrained model (zero by definition)')
a.set_xlabel('distance from the trial\'s objects to the model\'s training set  (3-D shape)   farther →'); a.set_ylabel('what training added to the margin\n(fine-tuned − pretrained, same trial)'); a.set_title('one axis, every trial × model',loc='left')
a.legend(fontsize=9,frameon=False,loc='upper right'); box(a,f'r = {stats.pearsonr(L.dist,L.adv)[0]:+.2f} over {len(L):,} points',INK,'ll')
a=ax[1]; style(a)
for r,col in [('same cluster',C_SAME),('other cluster, same category',C_OTH),('other category',C_CAT)]:
    d=L[L.rel==r]; bx,by,be,n=bins(d.dist,d.pre,edges); m=n>=30; a.errorbar(bx[m],by[m],yerr=be[m],fmt='o--',color=col,ms=8,mec=SURF,mew=1.2,lw=1.5,alpha=.8,ecolor='#d5d3ce',zorder=3)
    bx,by,be,n=bins(d.dist,d.ft,edges); a.errorbar(bx[m],by[m],yerr=be[m],fmt='o-',color=col,ms=9,mec=SURF,mew=1.3,lw=2.2,ecolor='#d5d3ce',zorder=4)
a.plot([],[],'o-',color=INK,label='fine-tuned model (solid)'); a.plot([],[],'o--',color=GREY,label='pretrained model (dashed)')
a.set_xlabel('distance from the trial\'s objects to the model\'s training set   farther →'); a.set_ylabel('oddity margin'); a.set_title('the two margins the left panel is the difference of',loc='left'); a.legend(fontsize=9,frameon=False,loc='upper right')
fig.suptitle('Within a category and across categories on one axis',fontsize=15,x=.06,ha='left',y=.975)
fig.text(.06,.91,f'{len(models)} models, each fine-tuned on 25 objects of one kind of one category, scored on {len(T):,} held-out trials from three kinds of each of twelve categories. x: how far the trial\'s\n'
 'objects are from the 25 the model saw. y (left): what fine-tuning added to that trial\'s margin — the pretrained model scored the same trial, so its difficulty cancels. Colour: the trial is of the\n'
 'kind the model saw / another kind of the same category / another category. Expect, if the distance is what matters: one falling curve, and the three colours on it.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig74_one_axis.png',dpi=300); print('[fig] 74')

# ---------- fig75: control the pair distance explicitly, then ask whether distance alone unifies the three relations
print('\nmodels in:',models)
# residualise ft and pre on pair distance within each TEST category (the trial-difficulty confound lives there)
for k in ['ft','pre']:
    L[k+'_r']=L[k]
    for cat,g in L.groupby('test_cat'):
        b=np.polyfit(g.pair,g[k],1); L.loc[g.index,k+'_r']=g[k]-np.polyval(b,g.pair)+g[k].mean()
L['adv_r']=L.ft_r-L.pre_r
print('after removing pair distance within test category:')
for r in ['same cluster','other cluster, same category','other category']:
    d=L[L.rel==r]; print(f'  {r:30s} r(dist, ft_r) = {stats.pearsonr(d.dist,d.ft_r)[0]:+.3f}   r(dist, pre_r) = {stats.pearsonr(d.dist,d.pre_r)[0]:+.3f}   mean adv_r {d.adv_r.mean():+.3f}')
# matched-distance comparison: within a common distance band, mean advantage by relation
band=(L.dist>0.45)&(L.dist<0.8); mb=L[band].groupby('rel').agg(n=('adv_r','size'),dist=('dist','mean'),adv=('adv_r','mean'),sem=('adv_r','sem')).round(4); print('\nmatched distance band 0.45–0.80:'); print(mb.to_string())
fig,ax=plt.subplots(1,2,figsize=(15,6.6),gridspec_kw={'width_ratios':[1.6,1]}); fig.subplots_adjust(left=.06,right=.98,top=.7,bottom=.19,wspace=.25)
a=ax[0]; style(a)
for r,col in [('same cluster',C_SAME),('other cluster, same category',C_OTH),('other category',C_CAT)]:
    d=L[L.rel==r]; bx,by,be,n=bins(d.dist,d.ft_r,edges); m=n>=30; a.errorbar(bx[m],by[m],yerr=be[m],fmt='o-',color=col,ms=9,mec=SURF,mew=1.3,lw=2.2,ecolor='#d5d3ce',zorder=4,label=f'{r}  ({len(d):,} points)')
    bx,by,be,n=bins(d.dist,d.pre_r,edges); a.errorbar(bx[m],by[m],yerr=be[m],fmt='o--',color=col,ms=7,mec=SURF,mew=1.2,lw=1.4,alpha=.6,ecolor='#d5d3ce',zorder=3)
a.plot([],[],'-',color=INK,label='solid: fine-tuned model'); a.plot([],[],'--',color=GREY,label='dashed: pretrained model, same trials')
a.axvspan(0.45,0.8,color='#f3e9d9',alpha=.5,lw=0); a.text(0.625,a.get_ylim()[1]*0.98 if a.get_ylim()[1]>0 else 0.2,'matched band',ha='center',va='top',fontsize=9,color=INK2)
a.set_xlabel('distance from the trial\'s objects to the model\'s training set  (3-D shape)   farther →'); a.set_ylabel('oddity margin, with the trial\'s pair distance held fixed'); a.set_title('margin vs distance, trial difficulty removed',loc='left'); a.legend(fontsize=9,frameon=False,loc='upper right')
a=ax[1]; style(a)
for i,(r,col) in enumerate([('same cluster',C_SAME),('other cluster, same category',C_OTH),('other category',C_CAT)]):
    a.bar(i,mb.loc[r,'adv'],yerr=mb.loc[r,'sem'],color=col,width=.6,ecolor=INK2,capsize=4); a.text(i,mb.loc[r,'adv']+mb.loc[r,'sem']+0.002,f'n = {int(mb.loc[r,"n"]):,}\nd̄ = {mb.loc[r,"dist"]:.2f}',ha='center',fontsize=8.5,color=INK2)
a.set_xticks([0,1,2]); a.set_xticklabels(['same\nkind','other kind,\nsame category','other\ncategory']); a.set_ylabel('what training added (fine-tuned − pretrained),\npair distance held fixed'); a.set_title('at the same distance (0.45–0.80): does it matter what the trial is?',loc='left',fontsize=11)
a.axhline(0,color=GREY,ls='--',lw=1.5)
fig.suptitle('Is distance to the training set the whole story? — first cut, chair and airplane models only',fontsize=15,x=.06,ha='left',y=.975)
fig.text(.06,.91,f'{len(models)} models (25 objects of one kind each), {len(T):,} held-out trials across twelve categories. Left: margin against distance, with each trial\'s pair distance (how different its two\n'
 'objects are) regressed out within its category — the confound that made raw curves rise. Right: restrict to trials at the same distance from the training set and ask whether\n'
 'it matters whether the trial is of the kind the model saw, another kind of the same category, or another category. If distance were the whole story, the three bars would be equal.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig75_one_axis_controlled.png',dpi=300); print('[fig] 75')
