"""The margin as a function of training-set size (dose), on both test sets, for every chair model we have.
Re-run as evaluations land; missing ones are skipped and listed."""
import json, os, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=f'{NAV}/Dist-shift-data/geometric_shift'; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
BLUE,GREY,SURF,INK,INK2,OK,BAD,GRN='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834','#1baf7a'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
T=pd.read_csv(f'{K}/banktrials/banktrials_chair.csv'); cl=T.condition.str[-1].astype(int).values
m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); dk=json.load(open(f'{K}/design.json')); mtr=[m.index[m.trial==t][0] for t in dk['chair']['trials']]
split=json.load(open(f'{K}/design_clusters_chair_split.json'))
def bank(n): 
    f=f'{K}/eval_bank/{n}/ood_analysis_results.csv'; return pd.read_csv(f) if os.path.exists(f) else None
def mochi(n):
    f=f'{K}/eval/{n}/ood_analysis_results.csv'; return pd.read_csv(f).iloc[mtr] if os.path.exists(f) else None
rows=[]; missing=[]
def add(kind,name,N,note=''):
    b=bank(name); mo=mochi(name)
    if b is None: missing.append(f'bank:{name}')
    if mo is None: missing.append(f'mochi:{name}')
    rows.append(dict(kind=kind,name=name,N=N,note=note,
        bank_all=b.fine_tuned_oddity_margin.mean() if b is not None else np.nan,
        bank_own=(b.fine_tuned_oddity_margin[cl==int(name[7])].mean() if (b is not None and name.startswith('chair_c') and name[7].isdigit()) else np.nan),
        bank_other=(b.fine_tuned_oddity_margin[cl!=int(name[7])].mean() if (b is not None and name.startswith('chair_c') and name[7].isdigit()) else np.nan),
        mochi=mo.fine_tuned_oddity_margin.mean() if mo is not None else np.nan,
        bank_pre=b.pretrained_oddity_margin.mean() if b is not None else np.nan, mochi_pre=mo.pretrained_oddity_margin.mean() if mo is not None else np.nan))
for c in range(3):
    ntr=len(split[str(c)]['train'])
    add('cluster',f'chair_c{c}_n25',25); add('cluster',f'chair_c{c}_n50',50); add('cluster',f'chair_c{c}_all',97,f'{ntr} chairs')
for s in ['s43','s44']: add('seed',f'chair_c0_all_{s}',97,'seed replicate of chair_c0_all')
for i in range(8): add('random',f'chair_random_100_{i}',100)
for g in ['g1','g2','g3','g4','random']: add('knockout',f'chair_{g}_k10',1821 if g!='random' else 1825)
add('full','chair_full',2000)
R=pd.DataFrame(rows); R.to_csv(f'{G}/out/margin_vs_dose.csv',index=False)
pre_b=R.bank_pre.dropna().iloc[0]; pre_m=R.mochi_pre.dropna().iloc[0]
ref=pd.read_csv(f'{S}/chair/ood_analysis_results.csv').iloc[mtr].fine_tuned_oddity_margin.mean()
print(R[['kind','name','N','bank_all','bank_own','bank_other','mochi']].round(3).to_string()); print('pretrained: bank',round(pre_b,3),'mochi',round(pre_m,3),'| collaborator reference on MOCHI chair trials',round(ref,3)); print('missing:',missing)
g=R.groupby(['kind','N']).agg(bank=('bank_all','mean'),bank_sd=('bank_all','std'),own=('bank_own','mean'),other=('bank_other','mean'),mochi=('mochi','mean'),mochi_sd=('mochi','std'),n=('name','count')).reset_index()
# ---------- figure
fig,ax=plt.subplots(1,3,figsize=(16.5,6)); fig.subplots_adjust(left=.06,right=.98,top=.72,bottom=.22,wspace=.3)
def xs(N): return np.where(N==0,3,N)
a=ax[0]; style(a); a.set_xscale('log')
a.axhline(pre_b,color=GREY,ls='--',lw=2.2,label='pretrained model (N = 0)')
cc=g[g.kind=='cluster']; a.plot(cc.N,cc.own,'o-',color=BLUE,ms=9,lw=2.4,label='trained on 3 kinds of chair — trials of the kind it saw'); a.plot(cc.N,cc.other,'o:',color=BLUE,ms=9,mfc='none',mew=2,lw=2.4,label='same models — trials of the kinds it did not see')
rr=g[g.kind=='random']
if len(rr) and not np.isnan(rr.bank.iloc[0]): a.errorbar(rr.N,rr.bank,yerr=rr.bank_sd,fmt='D',color=GRN,ms=9,mec=SURF,label=f'random subsets (n = {int(rr.n.iloc[0])} models)')
kk=g[g.kind=='knockout']
if len(kk) and not np.isnan(kk.bank.iloc[0]): a.errorbar(kk.N.mean(),kk.bank.mean(),yerr=kk.bank.std(),fmt='s',color=BAD,ms=9,mec=SURF,label='~10 % removed (5 models)')
ff=g[g.kind=='full']; a.plot(ff.N,ff.bank,'*',color=INK,ms=15,label='all 2,000 chairs')
ss=g[g.kind=='seed']
if len(ss) and not np.isnan(ss.bank.iloc[0]): a.plot([ss.N.iloc[0]]*2,R[R.kind=='seed'].bank_all,'|',color=INK,ms=14,mew=2,label='seed replicates of one cluster model')
a.set_xlabel('training-set size (chairs)'); a.set_ylabel('mean oddity margin'); a.set_title('on the 882 bank-built held-out trials',loc='left'); a.legend(fontsize=8.5,frameon=False,loc='lower right'); a.set_ylim(0.04,0.18)
a=ax[1]; style(a); a.set_xscale('log')
a.axhline(pre_m,color=GREY,ls='--',lw=2.2,label='pretrained model')
a.plot(cc.N,cc.mochi,'o-',color=BLUE,ms=9,lw=2.4,label='trained on one kind of chair (mean of 3)')
rr=g[g.kind=='random']; a.errorbar(rr.N,rr.mochi,yerr=rr.mochi_sd,fmt='D',color=GRN,ms=9,mec=SURF,label='random subsets of 100 (8 models)')
kk=g[g.kind=='knockout']; a.errorbar(kk.N.mean(),kk.mochi.mean(),yerr=kk.mochi.std(),fmt='s',color=BAD,ms=9,mec=SURF,label='~10 % removed (5 models)')
a.plot(ff.N,ff.mochi,'*',color=INK,ms=15,label='all 2,000 chairs (our pipeline)'); a.plot(2000,ref,'*',color=GREY,ms=15,label="all 2,000 chairs (collaborator's model)")
a.set_xlabel('training-set size (chairs)'); a.set_ylabel('mean oddity margin'); a.set_title('on the 76 MOCHI chair trials',loc='left'); a.legend(fontsize=8.5,frameon=False,loc='upper left'); a.set_ylim(0.07,0.3)
a=ax[2]; style(a); a.set_xscale('log')
cs=pd.read_csv(f'{G}/out/cluster_summary.csv'); Nmap={'all':int(np.mean([len(split[str(c)]['train']) for c in range(3)])),'n50':50,'n25':25}
inter={'all':0.0088,'n50':0.0123,'n25':0.0152}; dist_r={'all':-0.212,'n50':-0.255,'n25':-0.243}
a.plot([Nmap[k] for k in ['n25','n50','all']],[inter[k] for k in ['n25','n50','all']],'o-',color=BLUE,ms=9,lw=2.4,label='own-kind advantage (model level and trial difficulty removed)')
a.axhspan(0,0.0194,color='#f3e9d9',alpha=.6,lw=0); a.text(28,0.0185,'per-trial sd between models trained on\ndifferent random 100-chair sets',fontsize=8.5,va='top',color=INK2)
if len(ss) and not np.isnan(ss.bank.iloc[0]):
    b0=bank('chair_c0_all'); sdev=np.mean([np.abs(bank(f'chair_c0_all_{s}').fine_tuned_oddity_margin-b0.fine_tuned_oddity_margin).mean() for s in ['s43','s44']]); a.axhline(sdev,color=INK,ls=':',lw=1.8,label=f'seed floor: same data, different seed ({sdev:.3f})')
a.set_xlabel('training-set size (chairs)'); a.set_ylabel('margin: own kind of chair minus other kinds'); a.set_title('the part that depends on WHICH chairs, by dose',loc='left'); a.legend(fontsize=8.5,frameon=False,loc='upper right'); a.set_ylim(0,0.04); a.set_xlim(15,150)
fig.suptitle('How the margin behaves as the training set changes size',fontsize=14.5,x=.06,ha='left',y=.975)
fig.text(.06,.91,'Every chair model we have, placed by the number of chairs it was trained on. Left: bank-built trials — a model trained on 25 chairs of one kind already lifts the margin on that kind\n'
 'well above the pretrained model and above models trained on other kinds; 2,000 chairs do no better than 100. Middle: the same on MOCHI (small, selected). Right: the size of the\n'
 'training-set effect on the margin at each dose, against the noise between models.',fontsize=9.6,color=INK2,va='top')
fig.text(.06,.015,'Two doses act on the margin. Any fine-tuning at all — 25 chairs — moves it from 0.08 to ~0.15, and 2,000 chairs add nothing; that is the category / render-domain effect. Which chairs\n'
 'they were adds a further 0.01–0.02 on trials of that kind, largest at 25 and shrinking as the set grows. Experiments about within-category shift live in the second effect: small N, many trials.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig71_margin_vs_dose.png',dpi=300); print('[fig] 71')
