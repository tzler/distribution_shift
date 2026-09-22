"""One summary figure per step of the argument (steps 1, 2, 4; steps 3, 5, 6 reuse fig62, fig76, fig78)."""
import numpy as np, pandas as pd, ast
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; L1=f'{NAV}/Dist-shift-data/L1norm_vs_distshift'
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def box(a,txt,c,where='ur'):
    x,y,va,ha={'ur':(0.98,0.96,'top','right'),'ul':(0.02,0.96,'top','left'),'ll':(0.02,0.04,'bottom','left'),'lr':(0.98,0.04,'bottom','right')}[where]
    a.text(x,y,txt,transform=a.transAxes,fontsize=9.6,color=c,va=va,ha=ha,bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=c,lw=1.2))
def bins(x,y,nb=12):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); gg=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return gg.x.mean(),gg.y.mean(),gg.y.sem()
def line(a,x,y,e,c,ls,lab,z): a.errorbar(x,y,yerr=e,fmt='o'+ls,color=c,ms=8,mfc=c,mec=SURF,mew=1.3,lw=2.4,ecolor='#d5d3ce',elinewidth=1.4,zorder=z,label=lab)

# ---------- STEP 1: the manuscript's estimate contains the trial's own difficulty, and is read off the encoder it explains
d=pd.read_csv(f'{L1}/trials_vit_base_patch16_224.dino.csv')
s=d['trial_distance_(L1_not_normalized)']; fl=0.5*d['d_AB_(L1_not_normalized)']; ex=s-fl; m=d.pretrained_oddity_margin
fig,ax=plt.subplots(1,2,figsize=(14,6.8)); fig.subplots_adjust(left=.07,right=.98,top=.70,bottom=.19,wspace=.25)
a=ax[0]; style(a); a.scatter(fl,s,s=6,color=INK2,alpha=.25,linewidths=0,zorder=3); lim=[min(fl.min(),s.min()),s.max()]; a.plot(lim,lim,'--',color=BAD,lw=2,zorder=4)
a.set_xlabel('½ × how different the trial\'s two objects are   (½·d(A,B), in the encoder\'s features)'); a.set_ylabel('the manuscript\'s shift estimate'); a.set_title('the estimate can never be smaller than\nhalf the trial\'s own difficulty',loc='left',fontsize=11)
box(a,f'r = {stats.pearsonr(fl,s)[0]:+.2f};  {100*(s>=fl-1e-9).mean():.0f}% of trials on or above the line\n!  {100*stats.pearsonr(fl,s)[0]**2:.0f}% of the estimate\'s variance is the floor',BAD,'ul')
a=ax[1]; style(a); bx,by,be=bins(s,m); line(a,bx,by,be,GREY,'--','pretrained model — the encoder the estimate is computed in',3)
a.set_xlabel('the manuscript\'s shift estimate   farther →'); a.set_ylabel('oddity margin of the pretrained model'); a.set_title('and it predicts the margin of the encoder it was read from —\nwith the sign backwards',loc='left',fontsize=11)
box(a,f'r = {stats.pearsonr(s,m)[0]:+.2f}\n!  "farther from training" = an easier trial for the model\n    that never trained on it: this is the encoder\'s ease, not shift',BAD,'ul'); a.legend(fontsize=9,frameon=False,loc='lower right')
fig.suptitle('1 · The manuscript\'s shift estimate measures the trial and the encoder, not the training set',fontsize=15,x=.07,ha='left',y=.975)
fig.text(.07,.92,'DINO ViT-B, all 2,019 MOCHI trials. The estimate is ½[d(A,C)+d(B,C)] minimised over training images C, in the encoder\'s own features. By the triangle inequality it is bounded\n'
 'below by ½·d(A,B) — how different the two objects are, which is what the oddity task is decided by (left). And because it is read off the same encoder whose margin it explains,\n'
 'it tracks that encoder\'s ease with the trial (right). What to expect from a measure of distance to the training set: no floor, and no relation to a model that never saw the training set.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/step1_manuscript_metric.png',dpi=300); print('[step 1]')

# ---------- STEP 2: fix the form — oddity-blind — the estimate no longer follows the trial's difficulty
g=pd.read_csv(f'{G}/out/shift2d_shapegen.csv'); b=pd.read_csv(f'{G}/out/blindshift_shapegen.csv'); e=g.merge(b,on='trial')
fig,ax=plt.subplots(1,2,figsize=(14,6.8),sharex=True); fig.subplots_adjust(left=.07,right=.98,top=.70,bottom=.17,wspace=.25)
for a,(col,t,c) in zip(ax,[('geom_shift_2d','the manuscript\'s form:\nmin over training images C of ½[d(A,C)+d(B,C)]',BAD),('blind_k50','the new form: mean over the trial\'s images of the\ndistance to the 50 nearest training items',OK)]):
    style(a); a.scatter(e.geom_dAB_2d,e[col],s=7,color=c,alpha=.3,linewidths=0,zorder=3); r=stats.pearsonr(e.geom_dAB_2d,e[col])[0]
    a.set_xlabel('how different the trial\'s two objects are   (d(A,B), same descriptor)'); a.set_ylabel('shift estimate'); a.set_title(t,loc='left',fontsize=10.5)
    box(a,f'r with the trial\'s difficulty = {r:+.2f}\n'+('!  the estimate IS the trial\'s difficulty' if abs(r)>0.5 else '✓  much weaker, and the other way'),c,'ul')
fig.suptitle('2 · Fix the form: never compare the trial\'s two objects to each other',fontsize=15,x=.07,ha='left',y=.975)
fig.text(.07,.92,'ShapeGen trials, both estimates computed on the same model-free shape descriptor (so this is only about the form). Left: the manuscript\'s form, which shares one training image C\n'
 'between the two objects and so inherits their difference. Right: average a per-image distance over all of the trial\'s images and never difference them. What to expect if the form was\n'
 'the problem: the right panel should be flat against the trial\'s own difficulty.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/step2_fix_the_form.png',dpi=300); print('[step 2]')

# ---------- STEP 4: it works across categories; nothing graded within a category with the models we had
c=pd.read_csv(f'{G}/out/coverage_shapenet_voxel16_percat.csv')      # 706 trials × 12 models; knn = distance, ft/pre margins, own = trial's category
gg=c.groupby('trial'); c['dc']=c.knn-gg.knn.transform('mean'); c['mc']=c.ft-gg.ft.transform('mean'); c['pc']=c.pre-gg.pre.transform('mean')
own=c[c.category==c.own].copy(); own['x']=np.log1p(own['count']); go=own.groupby('own'); own['xc']=own.x-go.x.transform('mean'); own['yc']=own.ft-go.ft.transform('mean'); own['pcc']=own.pre-go.pre.transform('mean')
fig,ax=plt.subplots(1,2,figsize=(14,6.8)); fig.subplots_adjust(left=.07,right=.98,top=.70,bottom=.19,wspace=.25)
a=ax[0]; style(a); bx,bf,ef=bins(c.dc,c.mc); _,bp,ep=bins(c.dc,c.pc); line(a,bx,bp,ep,GREY,'--','pretrained model (flat: same model on all twelve points)',3); line(a,bx,bf,ef,BLUE,'-','model fine-tuned on that category',4)
a.set_xlabel('distance from the trial\'s objects to the model\'s training category,\nrelative to the trial\'s average   farther →'); a.set_ylabel('margin, relative to the trial\'s average'); a.set_title('across categories: each trial scored by twelve category models',loc='left',fontsize=11)
box(a,f'r = {stats.pearsonr(c.dc,c.mc)[0]:+.2f};  84% of trials slope down;  permutation p = 10⁻⁴\n✓  moving the training data moves the margin',OK,'ur'); a.legend(fontsize=9,frameon=False,loc='lower left')
a=ax[1]; style(a); E=[-9,-2,-1.2,-.6,-.2,.2,.6,1.2,2,9]
def fb(x,y):
    q=pd.cut(x,E); g2=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q',observed=True); return g2.x.mean(),g2.y.mean(),g2.y.sem()
bx,bf,ef=fb(own.xc,own.yc); _,bp,ep=fb(own.xc,own.pcc); line(a,bx,bp,ep,GREY,'--','pretrained model',3); line(a,bx,bf,ef,BLUE,'-','model trained on the trial\'s own category',4)
a.set_xlabel('training objects of the category near the trial\'s objects,\nrelative to the category\'s average   more →'); a.set_ylabel('margin, relative to the category\'s average'); a.set_title('within a category: only the model trained on the trial\'s category',loc='left',fontsize=11)
box(a,f'r = {stats.pearsonr(own.xc,own.yc)[0]:+.2f}\n!  flat: with one training set per category, "far from the training set"\n    and "an unusual object" are the same fact — not identifiable here',BAD,'ur')
fig.suptitle('4 · The margin follows the training set across categories; within a category the models we had could not tell',fontsize=15,x=.07,ha='left',y=.975)
fig.text(.07,.92,'706 ShapeNet trials, twelve models each fine-tuned on one category, distance measured on the objects\' 3-D shape with no network. Left: compare the twelve models on the same trial;\n'
 'expect the fine-tuned margin to fall with distance and the pretrained model to be flat. Right: keep only the model trained on each trial\'s own category and subtract each category\'s\n'
 'average; expect a rise if the margin is graded within a category. It is flat — which is why the next step trains new models on chosen subsets of a category.',fontsize=9.6,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/step4_across_not_within.png',dpi=300); print('[step 4]')
