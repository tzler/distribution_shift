#!/bin/bash
# One representation per process: peak memory stays bounded to a single bank.
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python
while pgrep -f "scratch/hillclimb.py --rep d57" >/dev/null; do sleep 5; done
for r in bbox structure volatility multiview voxel8 voxel16 voxel32; do
  echo "=== $r ==="
  /usr/bin/time -f "  peak %M KB, wall %E" $PY scratch/hillclimb.py --rep $r
done
echo ALL_SHARDS_DONE
