"""Within-category state figure. One point per trial: only the model trained on the trial's own category.
Left: as measured — the margin looks graded in coverage. Right: subtract each category's own average
from both axes — flat. The 'graded' relationship was twelve categories differing from each other."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

G=_RESOLVED_G
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
def line(a,x,y,e,c,ls,lab,z): a.errorbar(x,y,yerr=e,fmt='o'+ls,color=c,ms=8,mfc=c,mec=SURF,mew=1.3,lw=2.4,ecolor='#d5d3ce',elinewidth=1.4,zorder=z,label=lab)
d=pd.read_csv(f'{G}/out/coverage_shapenet_voxel16_percat.csv'); d=d[d.category==d.own].copy()
d['x']=np.log1p(d['count'])            # more training objects nearby → right
g=d.groupby('category'); 
for k in ['x','ft','pre']: d[k+'_c']=d[k]-g[k].transform('mean')
# bins: fixed edges on log count (left) and fixed-width on centred (right)
def fbins(x,y,edges):
    q=pd.cut(x,edges); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q',observed=True); return gg.x.mean(),gg.y.mean(),gg.y.sem()
EL=[-.1,.5,1.5,2.5,3.5,4.5,5.2,5.8,9]; ER=[-9,-2,-1.2,-.6,-.2,.2,.6,1.2,2,9]
fig,ax=plt.subplots(1,2,figsize=(14,7.4)); fig.subplots_adjust(left=.07,right=.98,top=.66,bottom=.2,wspace=.25)
for a,(k,fk,pk,E,t,dsc) in zip(ax,[('x','ft','pre',EL,'as measured','706 points: each trial scored by the model trained on its own category'),
                                     ('x_c','ft_c','pre_c',ER,'each category\'s own average subtracted from both axes','so airplanes are compared with airplanes, chairs with chairs')]):
    style(a); rf=stats.pearsonr(d[k],d[fk])[0]; rp=stats.pearsonr(d[k],d[pk])[0]
    bx,bf,ef=fbins(d[k],d[fk],E); _,bp,ep=fbins(d[k],d[pk],E)
    line(a,bx,bp,ep,GREY,'--','pretrained model — never saw this category\'s training set',3); line(a,bx,bf,ef,BLUE,'-','model trained on this category',4)
    a.set_title(f'{t}\n{dsc}',loc='left',fontsize=11); a.set_ylabel('oddity margin'+('' if k=='x' else ', relative to the category\'s average'))
    a.set_xlabel('more training objects of the category near the trial\'s objects  →\n'+('log(1 + count)' if k=='x' else 'log(1 + count), relative to the category\'s average'))
    if k=='x':
        box(a,f'fine-tuned r = {rf:+.2f}     pretrained r = {rp:+.2f}\n!  looks graded — but the left end is chairs, tables,\n    sofas, cabinets; the right end is airplanes, cars,\n    benches, telephones. Categories, not objects.',BAD,'ul')
        a.set_ylim(0,0.27)
    else:
        box(a,f'fine-tuned r = {rf:+.2f}     pretrained r = {rp:+.2f}\n✓  flat inside categories: nothing graded here.\n    12 category averages: r = +0.30, p = 0.34',OK,'ur')
        a.set_ylim(-0.05,0.075)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper left',bbox_to_anchor=(.07,.76),ncol=2,fontsize=10,frameon=False)
fig.suptitle('Inside a category, does more nearby training data mean a bigger margin?',fontsize=16,x=.07,ha='left',y=.975)
fig.text(.07,.92,'One point per trial: only the model trained on that trial\'s own category (a chair trial scored by the chair model, and so on). x counts how many of the category\'s\n'
 'training objects sit near the trial\'s objects. What to expect if the margin is graded within a category: BLUE should rise with the count, and should still rise after each\n'
 'category\'s own average is subtracted (right). If the relationship is only that some categories are more homogeneous than others, the right panel will be flat.',fontsize=9.8,color=INK2,va='top')
fig.text(.07,.015,'Left: the margin appears to rise with nearby training data. But the low-count trials are one set of categories and the high-count trials another. Right: compare each trial\n'
 'only with its own category and the relationship is gone — in every one of the twelve categories, the count and the margin are unrelated. With one training set per category, "far from\n'
 'the chair training set" and "an unusual chair" are the same fact; no score computed on this data can tell them apart. Separating them needs the training set to vary within a category.',fontsize=10,color=INK,va='bottom')
fig.savefig(f'{G}/out/figures/fig57_oncat_anatomy.png',dpi=300); print('[fig] 57', round(stats.pearsonr(d.x,d.ft)[0],3), round(stats.pearsonr(d.x_c,d.ft_c)[0],3))
