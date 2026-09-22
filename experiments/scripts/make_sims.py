"""Per-condition similarity tables: the full category table filtered to the condition's objects,
so similarity-binned triplet mining fills every epoch instead of rejecting missing partners."""
import os, sys, pickle
K='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout'; R='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/shapenet_rendered'
_cache={}
def full(bg,cat):
    if (bg,cat) not in _cache: _cache[(bg,cat)]=pickle.load(open(f'{R}/similarities/{bg}/{cat}/object_similarities_precomputed.pkl','rb'))
    return _cache[(bg,cat)]
def build(cat,cond):
    keep=set(os.listdir(f'{K}/data/{cat}_{cond}/white/{cat}'))
    for bg in ['white','black','random']:
        d=f'{K}/data/{cat}_{cond}/sim/{bg}/{cat}'; os.makedirs(d,exist_ok=True)
        out=f'{d}/object_similarities_precomputed.pkl'
        if os.path.exists(out): continue
        f=full(bg,cat); sub={o:{p:s for p,s in f[o].items() if p in keep} for o in f if o in keep}
        pickle.dump(sub,open(out,'wb'))
    print(f'{cat}_{cond}: sims for {len(keep)} objects')
if __name__=='__main__':
    conds=sys.argv[1:] or sorted(os.listdir(f'{K}/data'))
    for c in conds:
        cat,cond=c.split('_',1); build(cat,cond)
