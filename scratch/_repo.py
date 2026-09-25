"""Where things are, for every analysis script. Import this first:

    from _repo import G          # the repository root, wherever it was cloned

`G` is the repo root if the script sits inside a clone, else our lab share. Importing this
also puts `lib/` on the path (the shared descriptor and estimator modules) and makes sure
`out/figures/` exists, so a script can save straight into it.
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
G = _root if os.path.exists(os.path.join(_root, 'STATE.md')) \
    else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
for _p in (os.path.join(G, 'lib'), G):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.makedirs(os.path.join(G, 'out', 'figures'), exist_ok=True)
# the lab share, for the bulk inputs that are not in git (rendered images, checkpoints)
NAV = '/vast/projects/bonnen/naturalistic-navig'
K = f'{NAV}/Dist-shift-data/knockout'
