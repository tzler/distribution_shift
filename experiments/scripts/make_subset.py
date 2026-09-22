import json, os, sys
K='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout'; R='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/shapenet_rendered'
design=json.load(open(f'{K}/design.json'))
def build(cat, cond):
    if cond=='full': removed=set()
    elif cond.startswith('random_'): removed=set(design[cat]['random'][cond.split('_')[1]]['removed'])
    else: g,k=cond.split('_'); removed=set(design[cat]['groups'][g[1:]][k]['removed'])
    root=f'{K}/data/{cat}_{cond}'; n=0
    for bg in ['white','black','random']:
        d=f'{root}/{bg}/{cat}'; os.makedirs(d,exist_ok=True)
        for obj in sorted(os.listdir(f'{R}/{bg}/{cat}')):
            if obj in removed: continue
            dst=f'{d}/{obj}'
            if not os.path.lexists(dst): os.symlink(f'{R}/{bg}/{cat}/{obj}',dst)
            n+=1
    print(f'{cat}_{cond}: removed {len(removed)}, linked {n//3} objects per background -> {root}')
if __name__=='__main__': build(sys.argv[1],sys.argv[2])
