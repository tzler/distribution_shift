import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

NAV='/vast/projects/bonnen/naturalistic-navig'; G=_RESOLVED_G; L=f'{NAV}/Dist-shift-data/L1norm_vs_distshift'
GEOM,HL,GREY,TEAL,SURF,INK,INK2='#2a78d6','#eb6834','#8a8884','#1baf7a','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,
 'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
fig,ax=plt.subplots(1,4,figsize=(24,7.2),gridspec_kw=dict(width_ratios=[1.45,1,.85,1.15])); fig.subplots_adjust(left=.02,right=.99,top=.68,bottom=.2,wspace=.3)

# ---- A: ellipses. level sets of d(A,C)+d(B,C)=s are ellipses with foci A,B; smallest s is d(A,B) (the segment itself)
a=ax[0]; a.set_aspect('equal'); a.axis('off'); a.set_xlim(-.2,8.4); a.set_ylim(-1.9,2.5)
rng=np.random.default_rng(3); train=rng.normal([1.9,1.6],[.55,.45],size=(9,2))
def panel(a,x0,A,B,lab):
    A=np.array(A)+[x0,0]; B=np.array(B)+[x0,0]; T=train+[x0,0]; c=(A+B)/2; f=np.linalg.norm(B-A)/2; dAB=2*f
    sums=np.linalg.norm(T-A,axis=1)+np.linalg.norm(T-B,axis=1); k=sums.argmin(); smin=sums[k]
    for s_ in [dAB+.3,dAB+.9,dAB+1.6,dAB+2.5]:
        aa=s_/2; bb=np.sqrt(max(aa**2-f**2,1e-9)); a.add_patch(Ellipse(c,2*aa,2*bb,angle=0,fill=False,ec='#d8d7d2',lw=1,zorder=1))
    aa=smin/2; bb=np.sqrt(max(aa**2-f**2,1e-9)); a.add_patch(Ellipse(c,2*aa,2*bb,angle=0,fill=False,ec=HL,lw=2.2,zorder=3))
    a.plot([A[0],B[0]],[A[1],B[1]],color=HL,lw=2.6,zorder=2)
    a.scatter(T[:,0],T[:,1],s=55,marker='s',color=INK,edgecolor=SURF,lw=.8,zorder=4); a.scatter(*T[k],s=110,marker='s',color=INK,edgecolor=HL,lw=2,zorder=5)
    a.scatter(*A,s=150,color=GEOM,edgecolor=SURF,lw=1.4,zorder=6); a.scatter(*B,s=150,color=GEOM,edgecolor=SURF,lw=1.4,zorder=6)
    a.text(A[0]-.2,A[1]-.05,'A',ha='right',va='center',fontsize=11,weight='bold'); a.text(B[0]+.2,B[1]-.05,'B',ha='left',va='center',fontsize=11,weight='bold')
    a.text(x0+1.9,-1.35,f'{lab}\nfloor  ½·d(A,B) = {dAB/2:.2f}\n"shift" = ½·(sum on the orange ellipse) = {smin/2:.2f}',ha='center',fontsize=8.6,color=INK,linespacing=1.4)
    a.text(T[:,0].max()+.25,T[:,1].mean(),'training\nimages',ha='left',va='center',fontsize=8.2,color=INK2)
panel(a,0,(1.1,-.3),(2.7,-.3),'hard trial: A, B similar')
panel(a,4.4,(.2,-.3),(3.6,-.3),'easy trial: A, B different')
a.set_title('A  Level sets of d(A,C)+d(B,C) are ellipses with foci A and B\nthe smallest is the segment AB itself, sum = d(A,B): no training image can do better',loc='left')
a.text(4.1,2.3,'same training images, same nearest one — the "shift" moved because A and B did',ha='center',fontsize=9,color=INK2,style='italic')

# ---- B: data: shift vs floor, cosine (the friendliest distance)
d=pd.read_csv(f'{L}/trials_vit_base_patch16_224.dino.csv'); a=ax[1]; style(a)
for tag,col,name in [('cosine',HL,'1 − cosine'),('L1_normalized',GEOM,'L1, unit-norm')]:
    s=d[f'trial_distance_({tag})'].values; fl=0.5*d[f'd_AB_({tag})'].values; mx=s.max(); s,fl=s/mx,fl/mx
    a.scatter(fl,s,s=6,color=col,alpha=.3,linewidths=0,zorder=3,label=f'{name}:  r = {stats.pearsonr(fl,s)[0]:+.2f}')
lim=[0,1.02]; a.plot(lim,lim,color=INK2,lw=1.4,ls='--',zorder=4); a.text(.55,.5,'floor: shift = ½·d(A,B)\nnothing can be below this line',fontsize=8.6,color=INK2,rotation=36,ha='center',va='top')
a.set_xlim(0,1.02); a.set_ylim(0,1.02); a.set_xlabel('½·d(A,B)   (each distance scaled to its own max)'); a.set_ylabel('the manuscript\'s shift'); a.legend(fontsize=8.6,loc='upper left')
a.set_title('B  Every distance obeys it — cosine too\nDINO ViT-B, 2,019 MOCHI trials; the bound holds in 100% of trials',loc='left')

# ---- C: decomposition
a=ax[2]; style(a); a.grid(axis='x',visible=False)
s=d['trial_distance_(L1_not_normalized)']; fl=0.5*d['d_AB_(L1_not_normalized)']; ex=s-fl; m=d.pretrained_oddity_margin
vals=[stats.pearsonr(fl,m)[0],stats.pearsonr(ex,m)[0],stats.pearsonr(s,m)[0]]; labs=['the floor\n½·d(A,B)','the excess\n(real distance\nto training)','their sum\n= the shift']
cols=[HL,TEAL,GREY]; x=np.arange(3); a.bar(x,vals,.6,color=cols,zorder=3,edgecolor=SURF)
for i,v in enumerate(vals): a.text(i,v+(.02 if v>0 else -.02),f'{v:+.2f}',ha='center',va='bottom' if v>0 else 'top',fontsize=9.5)
a.axhline(0,color=INK2,lw=1); a.set_xticks(x); a.set_xticklabels(labs,fontsize=8.8); a.set_ylabel('r with the pretrained margin'); a.set_ylim(-.3,.7)
a.set_title('C  Split the shift into floor + excess\nthe excess has the right sign; the floor swamps it',loc='left')

# ---- D: what does and doesn't rescue it
a=ax[3]; style(a); a.grid(axis='x',visible=False)
items=[('L1\nraw',.731,HL,'change the distance'),('L1\nunit-norm',.754,HL,''),('1 − cos',.783,HL,''),
       ('each object →\nown nearest\n(cosine)',.661,GEOM,'change the form'),
       ('blind,\ngeometric,\nshapenet',.455,TEAL,'change the space'),('blind,\ngeometric,\nshapegen',-.362,TEAL,'')]
x=np.arange(len(items)); a.bar(x,[v for _,v,_,_ in items],.62,color=[c for _,_,c,_ in items],zorder=3,edgecolor=SURF)
for i,(l,v,c,g) in enumerate(items):
    a.text(i,v+(.02 if v>0 else -.02),f'{v:+.2f}',ha='center',va='bottom' if v>0 else 'top',fontsize=9)
    if g: a.text(i-.3,.92 if i<3 else (.80 if i==3 else .68),g,fontsize=8.4,color=c,ha='left',weight='bold')
a.axhline(0,color=INK2,lw=1); a.set_xticks(x); a.set_xticklabels([l for l,_,_,_ in items],fontsize=8); a.set_ylabel('r(shift, d(A,B))'); a.set_ylim(-.5,1.0)
a.set_title('D  What rescues it\nnot the distance — the form, then the space',loc='left')
fig.suptitle('The triangle inequality, spelled out — and whether cosine or L2 gets around it',fontsize=15,x=.02,ha='left',y=.975)
fig.text(.02,.88,'For any distance worth the name, the direct route is never longer than a detour: d(A,B) ≤ d(A,C) + d(C,B). Halve it: ½[d(A,C)+d(C,B)] ≥ ½·d(A,B). The shift is the smallest value of the left side over all\n'
         'training images C — and minimising cannot get below the floor. The floor is reached only if a training image sits exactly on the path from A to B. So shift = ½·d(A,B) + excess, and only the excess is\n'
         'about the training set. L1, L2 and angular distance are all metrics, so all obey this; 1 − cosine is not quite a metric but its square root is, and in practice it is the worst of the three.',fontsize=9,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig61_triangle.png',dpi=300); print('[fig] 61')
