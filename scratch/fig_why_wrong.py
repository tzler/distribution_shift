import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; L=f'{NAV}/Dist-shift-data/L1norm_vs_distshift'
GEOM,HL,GREY,TEAL,SURF,INK,INK2='#2a78d6','#eb6834','#8a8884','#1baf7a','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,
 'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
T='trial_distance_(L1_not_normalized)'; AB='d_AB_(L1_not_normalized)'
fig,ax=plt.subplots(1,4,figsize=(23,5.8),gridspec_kw=dict(width_ratios=[1.25,1,1,1.05])); fig.subplots_adjust(left=.02,right=.99,top=.71,bottom=.14,wspace=.28)

# ---- A: the triangle
a=ax[0]; a.set_aspect('equal'); a.axis('off'); a.set_xlim(-.3,6.6); a.set_ylim(-1.2,3.4)
def trial(a,x0,A,B,C,lab,col):
    A,B,C=np.array(A)+[x0,0],np.array(B)+[x0,0],np.array(C)+[x0,0]
    for p,q,c,ls in [(A,C,GREY,'-'),(B,C,GREY,'-'),(A,B,col,'--')]: a.plot([p[0],q[0]],[p[1],q[1]],color=c,lw=1.8,ls=ls,zorder=2)
    a.scatter(*A,s=170,color=GEOM,edgecolor=SURF,lw=1.5,zorder=4); a.scatter(*B,s=170,color=GEOM,edgecolor=SURF,lw=1.5,zorder=4); a.scatter(*C,s=190,marker='s',color=INK,edgecolor=SURF,lw=1.5,zorder=4)
    a.text(A[0]-.22,A[1],'A',ha='right',va='center',fontsize=11,weight='bold'); a.text(B[0]+.22,B[1],'B',ha='left',va='center',fontsize=11,weight='bold'); a.text(C[0],C[1]+.28,'C  nearest\ntraining image',ha='center',va='bottom',fontsize=8.4)
    dAB=np.linalg.norm(A-B); dAC=np.linalg.norm(A-C); dBC=np.linalg.norm(B-C)
    a.text((A[0]+B[0])/2,(A[1]+B[1])/2-.32,f'd(A,B) = {dAB:.1f}',ha='center',fontsize=8.6,color=col)
    a.text(x0+1.2,-1.0,f'{lab}\n"shift" = ½[d(A,C)+d(B,C)] = {0.5*(dAC+dBC):.2f}',ha='center',fontsize=8.8,color=INK)
trial(a,0,(0.4,0.3),(2.0,0.3),(1.2,2.3),'same object pair, hard trial',HL)
trial(a,3.4,(-.4,0.3),(2.8,0.3),(1.2,2.3),'objects very different, easy trial',HL)
a.set_title('A  The metric contains the trial\'s own difficulty\n½[d(A,C)+d(B,C)] ≥ ½·d(A,B) by the triangle inequality — always',loc='left')
a.text(3.15,3.15,'same distance to the training set, different "shift"',ha='center',fontsize=9,color=INK2,style='italic')

# ---- B: shift vs feature norm (ResNet-50), the audit
a=ax[1]; style(a); d=pd.read_csv(f'{L}/trials_resnet50.tv_in1k.csv'); r=stats.pearsonr(d[T],d.feat_l1)[0]
a.scatter(d.feat_l1,d[T],s=7,color=HL,alpha=.35,linewidths=0,zorder=3); b1,b0=np.polyfit(d.feat_l1,d[T],1); xs=np.linspace(d.feat_l1.min(),d.feat_l1.max(),50); a.plot(xs,b1*xs+b0,color=HL,lw=2.2,zorder=4)
a.set_xlabel('‖φ‖₁ — how long the test images\' feature vectors are'); a.set_ylabel('the manuscript\'s shift  (raw ℓ₁, ResNet-50)')
a.set_title(f'B  It is mostly the size of the feature vector\nr = {r:+.3f}, n = {len(d):,} MOCHI trials  (DeiT-III: +0.893)',loc='left')

# ---- C: encoder d(A,B) vs pretrained margin
a=ax[2]; style(a); d=pd.read_csv(f'{L}/trials_vit_base_patch16_224.dino.csv'); r=stats.pearsonr(d[AB],d.pretrained_oddity_margin)[0]
a.scatter(d[AB],d.pretrained_oddity_margin,s=7,color=GEOM,alpha=.35,linewidths=0,zorder=3); b1,b0=np.polyfit(d[AB],d.pretrained_oddity_margin,1); xs=np.linspace(d[AB].min(),d[AB].max(),50); a.plot(xs,b1*xs+b0,color=GEOM,lw=2.2,zorder=4)
a.set_xlabel('d(A,B) in the encoder\'s own feature space'); a.set_ylabel('that encoder\'s oddity margin')
a.set_title(f'C  … and d(A,B) in that space is the margin\nr = {r:+.3f}, DINO ViT-B, n = {len(d):,}  (DINOv2-L on ShapeNet: +0.72)',loc='left')

# ---- D: what it predicts, on the 12-fine-tune data
a=ax[3]; style(a); a.grid(axis='x',visible=False)
labels=['pooled\n(8,472)','on-category\n(706)']; orig_ft=[-0.013,0.331]; orig_pre=[0.381,0.608]; cov_ft=[-0.268,-0.118]; cov_pre=[-0.030,-0.047]
x=np.arange(2); w=.19
a.bar(x-1.5*w,orig_ft,w,color=HL,zorder=3,edgecolor=SURF,label='original → fine-tuned margin'); a.bar(x-.5*w,orig_pre,w,color=HL,alpha=.45,zorder=3,edgecolor=SURF,label='original → PRETRAINED margin (control)')
a.bar(x+.5*w,cov_ft,w,color=TEAL,zorder=3,edgecolor=SURF,label='coverage → fine-tuned margin'); a.bar(x+1.5*w,cov_pre,w,color=TEAL,alpha=.45,zorder=3,edgecolor=SURF,label='coverage → PRETRAINED margin (control)')
for i in range(2):
    for off,v in [(-1.5*w,orig_ft[i]),(-.5*w,orig_pre[i]),(.5*w,cov_ft[i]),(1.5*w,cov_pre[i])]: a.text(x[i]+off,v+(.02 if v>=0 else -.02),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8.2)
a.axhline(0,color=INK2,lw=1); a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel('Pearson r with the margin'); a.set_ylim(-.42,.78); a.legend(fontsize=7.6,loc='upper left')
a.set_title('D  What it predicts, given 12 fine-tuned models\nit tracks the model that never saw the data — wrong sign',loc='left')
fig.suptitle('Why the manuscript\'s shift metric is the wrong one',fontsize=15,x=.03,ha='left',y=.975)
fig.text(.03,.885,'shift = min over training images C of ½[d(φ(A),φ(C)) + d(φ(B),φ(C))], with φ the pretrained encoder\'s raw features and d the ℓ₁ distance. A: it can never be smaller than half the distance between the\n'
         'two objects being compared, which is what the task is about. B: in raw feature space that distance is mostly vector length. C: the encoder\'s own d(A,B) is its margin. D: so on data where we can check, the metric\n'
         'predicts the untouched encoder\'s ease better than any fine-tuned model\'s exposure — and a shift measure that is positively correlated with the margin has the sign backwards.',fontsize=9,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig60_why_wrong.png',dpi=300); print('[fig] 60')
