# Variable subset write comparison

nensemble=2, nsteps=4

| Run | # variables | Wall time (s) | Zarr size (MB) |
|-----|-------------|----------------|-----------------|
| Full | 72 | 59.5 | 2990.2 |
| Subset | 4 | 19.7 | 166.1 |

Storage ratio (subset/full): 0.056
Time ratio (subset/full): 0.332

Model compute per step is unchanged either way (confirmed by reading
run.ensemble's source - see this script's docstring). If the time ratio
isn't close to 1.0, that means zarr write I/O is a substantial fraction of
total wall time for this run size (nsteps=4, nensemble=2),
not that the model computed less work - for a longer rollout, where I/O is
a smaller share of the total, expect the ratio to drift closer to 1.0.
