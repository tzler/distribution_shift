"""Batch 1 analysis. Units: a trial scored under several training sets that differ only in what was removed.
Dose = how many of the trial's objects had their nearest-10 training objects removed (0, 1, 2).
Outputs: out/knockout_long.csv, out/knockout_summary.csv, and fig64 (recipe: one plot type, expectation stated)."""
import json, ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

NAV='/vast/projects/bonnen/naturalistic-navig'; K=f'{NAV}/Dist-shift-data/knockout'; G=_RESOLVED_G; S=f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
d=json.load(open(f'{K}/design.json')); m=pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv'); mi=m.set_index('trial')
def load(path):
    o=pd.read_csv(path); o['trial']=m['trial'].values; return o.set_index('trial')
rows=[]; summ=[]
for cat in ['chair','airplane','table']:
    c=d[cat]; ref=load(f'{S}/{cat}/ood_analysis_results.csv'); tr=c['trials']
    obj2g={o.split('/')[-1]:gk for gk,gv in c['groups'].items() for o in gv['test_objects']}
    models={f'g{gk}':load(f'{K}/eval/{cat}_g{gk}_k10/ood_analysis_results.csv') for gk in c['groups']}
    models['random']=load(f'{K}/eval/{cat}_random_k10/ood_analysis_results.csv')
    if cat=='chair': models['full']=load(f'{K}/eval/chair_full/ood_analysis_results.csv')
    for t in tr:
        objs={f[:-4].split('_')[1] for f in ast.literal_eval(mi.loc[t,'images'])}
        for name,mo in models.items():
            hit=sum(obj2g.get(o)==name[1:] for o in objs) if name.startswith('g') else 0
            rows.append((cat,t,name,'full' if name=='full' else ('random' if name=='random' else 'knockout'),hit,mo.fine_tuned_oddity_margin[t],ref.fine_tuned_oddity_margin[t],ref.pretrained_oddity_margin[t]))
L=pd.DataFrame(rows,columns=['cat','trial','model','kind','hit','margin','ref_margin','pre_margin']); L.to_csv(f'{G}/out/knockout_long.csv',index=False)
print('mean margin on the category\'s trials, by model')
print(L.groupby(['cat','model']).margin.mean().unstack().round(3).to_string())
print('\nreference (collaborator) models:', L.groupby('cat').ref_margin.mean().round(3).to_dict(), ' pretrained:', L.groupby('cat').pre_margin.mean().round(3).to_dict())
# within-pipeline comparison: knockout models vs each other, by dose, relative to the trial's mean over the 4 knockout models
ko=L[L.kind=='knockout'].copy(); ko['rel']=ko.margin-ko.groupby(['cat','trial']).margin.transform('mean')
print('\nmargin relative to the trial\'s own average over the four knockout models, by how many of the trial\'s objects lost their support:')
print(ko.groupby(['cat','hit']).rel.agg(['mean','sem','count']).round(4).to_string())
for cat in ['chair','airplane','table']:
    w=ko[ko.cat==cat].pivot_table(index='trial',columns='hit',values='margin',aggfunc='mean')
    con=w[[h for h in w.columns if h>0]].mean(axis=1)-w[0]
    print(f'{cat}: (support removed) − (not removed), per trial: {con.mean():+.4f} ± {con.sem():.4f}, {100*(con<0).mean():.0f}% negative, n={con.notna().sum()}, p={stats.ttest_1samp(con.dropna(),0).pvalue:.2g}')
    summ.append((cat,con.mean(),con.sem(),(con<0).mean(),con.notna().sum()))
# knock-in noise floor
ki=pd.concat([load(f'{K}/eval/chair_random_100_{i}/ood_analysis_results.csv').fine_tuned_oddity_margin[d['chair']['trials']] for i in range(8)],axis=1)
print(f'\n8 random 100-chair models on chair trials: means {ki.mean().round(3).tolist()}; per-trial sd {ki.std(axis=1).mean():.4f}')
pd.DataFrame(summ,columns=['cat','contrast','sem','frac_neg','n']).to_csv(f'{G}/out/knockout_summary.csv',index=False)
# ---------- fig64: one plot type × 3 categories. x = objects hit (0/1/2), y = margin relative to trial's mean over knockout models; grey = same for random-removal model (hit=0 by definition → shown as its offset)
fig,ax=plt.subplots(1,3,figsize=(15,6.4),sharey=True); fig.subplots_adjust(left=.07,right=.98,top=.66,bottom=.2,wspace=.12)
for a,cat in zip(ax,['chair','airplane','table']):
    style(a); s=ko[ko.cat==cat].groupby('hit').rel.agg(['mean','sem','count'])
    a.errorbar(s.index,s['mean'],yerr=s['sem'],fmt='o-',color=BLUE,ms=10,mfc=BLUE,mec=SURF,mew=1.4,lw=2.6,ecolor='#bcd0ea',elinewidth=1.6,zorder=4,label='model with those training objects removed')
    r=L[(L.cat==cat)&(L.kind=='random')].set_index('trial').margin - ko[ko.cat==cat].groupby('trial').margin.mean()
    a.axhline(r.mean(),color=GREY,ls='--',lw=2.2,zorder=3,label='same number of random objects removed'); a.axhspan(r.mean()-r.sem(),r.mean()+r.sem(),color=GREY,alpha=.12,lw=0)
    a.set_xticks([0,1,2]); a.set_xticklabels(['neither','one','both']); a.set_xlabel(f'how many of the trial\'s objects lost their\n10 nearest training objects   ({cat}, {len(d[cat]["trials"])} trials)')
    a.set_title(f'{cat}',loc='left',fontsize=12)
    for h in s.index: a.annotate(f'n={int(s.loc[h,"count"])}',(h,s.loc[h,'mean']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8.5,color=INK2)
    a.axhspan(-0.0194,0.0194,color='#f3e9d9',alpha=.6,lw=0,zorder=0); a.set_ylim(-0.025,0.025)
    if cat=='chair': a.text(2.05,0.0194,'run-to-run noise between\nmodels (±0.019 per trial)',fontsize=8.5,color=INK2,va='top',ha='right')
    con=[x for x in summ if x[0]==cat][0]
    box(a,f'removed − not removed: {con[1]:+.3f} ± {con[2]:.3f}\n!  no drop when a trial\'s own support goes;\n    a 10-object removal is far inside the noise band',BAD,'ll')
ax[0].set_ylabel('oddity margin, relative to the trial\'s average\nover the four knockout models')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.07,.76),ncol=2,fontsize=10,frameon=False)
fig.suptitle('Removing a trial\'s nearest training objects: does its margin fall?',fontsize=16,x=.07,ha='left',y=.975)
fig.text(.07,.92,'For each category, four models were fine-tuned on the category\'s training set minus the 10 nearest training objects of one group of test objects (about 10 % of the set\n'
 'each time), plus one with the same number of random objects removed. Each trial is scored by all of them; y is its margin relative to its own average over the four.\n'
 'What to expect if the margin depends on nearby training data: BLUE should fall from "neither" to "both". The random-removal model (GREY) should sit at zero.',fontsize=9.8,color=INK2,va='top')
fig.text(.07,.015,'Blue does not fall in any category. Removing a trial\'s ten nearest training objects changes its margin by less than the run-to-run noise between models trained on\n'
 'random subsets (about ±0.02 per trial). Either ten objects are too small a dose at this bank size, or the margin inside a category does not depend on nearby objects at all.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig64_knockout_batch1.png',dpi=300); print('[fig] 64')
