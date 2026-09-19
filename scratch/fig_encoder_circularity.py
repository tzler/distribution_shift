"""State 2 figure: distance to the training set measured in the encoder's own space, against
the fine-tuned margin (blue) and the pretrained margin (grey). One plot type, four times.
If the distance measured training exposure, grey — a model that never saw the training set — is flat."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
K='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout'; G='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
BLUE,GREY,SURF,INK,INK2,OK,BAD='#2a78d6','#8a8884','#fcfcfb','#0b0b0b','#52514e','#1baf7a','#eb6834'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':'#d8d7d2','font.size':10.5,'axes.titlesize':12})
def style(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(color='#eceae5',lw=.8,zorder=0); a.set_axisbelow(True)
def bins(x,y,nb=12):
    q=pd.qcut(x.rank(method='first'),nb,labels=False); g=pd.DataFrame({'q':q,'x':x,'y':y}).groupby('q'); return g.x.mean(),g.y.mean(),g.y.sem()
fig,ax=plt.subplots(2,2,figsize=(14,10.5)); fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.10,hspace=.55,wspace=.25)
for j,(model,name) in enumerate([('pretrained','the pretrained encoder\'s feature space'),('ft_chair','the chair model\'s own feature space')]):
    d=pd.read_csv(f'{K}/eval/encoder_space_{model}_image.csv')
    for i,(rowname,sub,desc) in enumerate([('every trial × every model',d,'8,472 points: each trial scored by all twelve category models'),('one model per trial: its own category',d[d.category==d.own],'706 points: only the model trained on the trial\'s category')]):
        a=ax[i,j]; style(a)
        rf,rp=stats.pearsonr(sub.knn,sub.ft)[0],stats.pearsonr(sub.knn,sub.pre)[0]
        bx,bf,ef=bins(sub.knn,sub.ft); _,bp,ep=bins(sub.knn,sub.pre)
        a.errorbar(bx,bp,yerr=ep,fmt='o--',color=GREY,ms=8,mfc=GREY,mec=SURF,mew=1.3,lw=2.2,ecolor='#d5d3ce',elinewidth=1.4,zorder=3,label='pretrained model — never saw the training set')
        a.errorbar(bx,bf,yerr=ef,fmt='o-',color=BLUE,ms=9,mfc=BLUE,mec=SURF,mew=1.4,lw=2.6,ecolor='#bcd0ea',elinewidth=1.6,zorder=4,label='fine-tuned model — trained on it')
        passed=abs(rp)<0.1
        verdict=('✓  grey is flat: the distance tracks the training set' if passed else '!  grey falls with blue: we have not separated the training set\n    from the encoder here')
        a.set_title(f'{rowname}\n{desc}',loc='left',fontsize=11)
        a.text(0.98,0.96,f'fine-tuned r = {rf:+.2f}     pretrained r = {rp:+.2f}\n{verdict}',transform=a.transAxes,fontsize=9.6,color=OK if passed else BAD,va='top',ha='right',
               bbox=dict(boxstyle='round,pad=.4',fc=SURF,ec=OK if passed else BAD,lw=1.2))
        a.set_xlabel(f'distance from the trial to the training set,\nmeasured in {name}'); a.set_ylabel('oddity margin')
        if i==0 and j==0: h,l=a.get_legend_handles_labels()
fig.suptitle('Is the distance about the training set, or about the encoder?',fontsize=16,x=.07,ha='left',y=.965)
fig.text(.07,.905,'The oddity-blind distance (mean over the trial\'s images of the distance to the 50 nearest training renders), measured inside the encoder\'s own features.\n'
         'Read every panel the same way: BLUE is the model that trained on this data, so its margin should fall as the distance grows. GREY is the pretrained model, which never saw\n'
         'this training set — if the distance measured training exposure, grey would be FLAT. Where grey falls with blue, the distance is measuring something about the\n'
         'encoder\'s view of the object, not about what the model was trained on. Left column: the pretrained encoder\'s space. Right: the chair model\'s own space.',fontsize=9.8,color=INK2,va='top')
fig.legend(h,l,loc='upper left',bbox_to_anchor=(.07,.845),ncol=2,fontsize=10,frameon=False)
fig.text(.07,.015,'Top row: the distance passes — the pretrained model is flat.\nBottom row (the within-category question we care about): not settled — the pretrained model falls about as steeply as the fine-tuned one. That is consistent with the distance\nreporting which objects the encoder finds hard, with the margin being the same encoder\'s report of the same thing. Not a kill; a concern that has not gone away.',fontsize=10,color=INK,va='bottom')
fig.subplots_adjust(bottom=.13,top=.78)
fig.savefig(f'{G}/out/figures/fig62_encoder_space.png',dpi=300); print('[fig] 62 rebuilt')
