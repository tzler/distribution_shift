import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

G=_RESOLVED_G
GEOM,HL,GREY,TEAL,SURF,INK,INK2='#2a78d6','#eb6834','#8a8884','#1baf7a','#fcfcfb','#0b0b0b','#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,
 'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10,'axes.titlesize':11})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='x',color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
rows=[('object voxels  (voxel16 coverage)',0.424,84.2,TEAL,'shape only, canonical frame'),
      ('depth maps · all 15 training views, averaged',0.436,86.5,GEOM,'shape, viewpoint marginalised'),
      ('depth maps · one random training view',0.426,85.9,GEOM,'single on-grid view'),
      ('depth maps · nearest training view to the MOCHI camera',0.422,86.6,GEOM,'single on-grid view, closest to the test camera'),
      ('depth maps · the actual MOCHI view (estimated)',0.266,79.2,HL,'off-grid view, median 21° from the nearest training camera')]
fig,ax=plt.subplots(1,2,figsize=(17,5.6),gridspec_kw=dict(width_ratios=[1.5,1])); fig.subplots_adjust(left=.30,right=.985,top=.70,bottom=.14,wspace=.42)
a=ax[0]; style(a); y=np.arange(len(rows))[::-1]
a.barh(y,[r[1] for r in rows],color=[r[3] for r in rows],height=.6,zorder=3,edgecolor=SURF)
for yy,r in zip(y,rows): a.text(r[1]+.006,yy,f'−{r[1]:.3f}   {r[2]:.0f}% neg',va='center',fontsize=9.5,color=INK); a.text(-.012,yy-.33,r[4],ha='right',va='center',fontsize=7.8,color=INK2)
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows],fontsize=9.5); a.set_xlim(0,.55); a.set_xlabel('within-trial −r  (706 trials × 12 training sets)')
a.set_title('A  What the test descriptor is computed from\nimage-level coverage, category bank of 313k training renders; ε = 0.3',loc='left')
a=ax[1]; style(a); a.grid(axis='y',color='#eceae5',lw=.8,zorder=0); a.grid(axis='x',visible=False)
tiers=['< 0.80\n(161)','0.80–0.88\n(150)','≥ 0.88\n(395)']; obj=[.524,.286,.407]; img=[.210,.220,.293]; x=np.arange(3)
a.bar(x-.18,obj,.34,color=TEAL,zorder=3,edgecolor=SURF,label='object coverage'); a.bar(x+.18,img,.34,color=HL,zorder=3,edgecolor=SURF,label='image coverage, actual view')
for i in range(3): a.text(x[i]-.18,obj[i]+.008,f'{obj[i]:.2f}',ha='center',fontsize=8.8); a.text(x[i]+.18,img[i]+.008,f'{img[i]:.2f}',ha='center',fontsize=8.8)
a.set_xticks(x); a.set_xticklabels(tiers,fontsize=9); a.set_xlabel('pose-estimation quality: trial mean silhouette IoU (n trials)'); a.set_ylabel('within-trial −r'); a.legend(fontsize=8.5,loc='upper right'); a.set_ylim(0,.62)
a.set_title('B  Is it pose-estimation error?\nsame trials, both metrics',loc='left')
fig.suptitle('Images, not objects: does coverage of the training IMAGES predict the margin better?',fontsize=14.5,x=.30,ha='left',y=.975)
fig.text(.30,.88,'Each MOCHI object is projected to a 32×32 depth + silhouette map (no renderer); the bank is every training object at the renderer\'s 15 fixed views.\n'
         'Coverage counts training images within cosine ε of the test image. Any on-grid view recovers the full object-level effect; only the actual off-grid\n'
         'MOCHI view loses it — and loses it even when the pose is well estimated.',fontsize=9.2,color=INK2,va='top')
fig.savefig(f'{G}/out/figures/fig58_viewdepth_ladder.png',dpi=300); print('[fig] 58')
