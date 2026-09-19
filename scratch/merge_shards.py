"""Concatenate the per-representation shards into one candidate matrix."""
import glob, os
import numpy as np
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sh = sorted(glob.glob(f'{G}/out/hc_shard_*.npz'))
if not sh:
    raise SystemExit('no shards found')
base = np.load(sh[0], allow_pickle=True)
trials = base['trials']
Ms, keys = [], []
for f in sh:
    z = np.load(f, allow_pickle=True)
    assert list(z['trials']) == list(trials), f'trial order differs in {f}'
    Ms.append(z['M']); keys += [str(k) for k in z['keys']]
    print(f'  {os.path.basename(f):28s} {z["M"].shape[1]:>5,} candidates')
M = np.concatenate(Ms, axis=1)
out = {k: base[k] for k in base.files if k.startswith('y_')}
np.savez_compressed(f'{G}/out/hillclimb_candidates.npz', M=M,
                    keys=np.array(keys), trials=trials, **out)
print(f'\nmerged: {M.shape[0]} trials x {M.shape[1]:,} candidates')
