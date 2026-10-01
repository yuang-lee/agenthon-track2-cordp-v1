"""## Executive summary (read this first)

Transfer only a proposal's row-average level to an incumbent joint distribution.
Every within-row coordinate difference remains exactly the incumbent difference.
Callers must restrict this raw-unit projection to homogeneous target grids; it is
not meaningful across currency quotes or unrelated units. No outcomes are read.
"""
import numpy as np

from pipelines.numerical_combinations import combine


def common_path_mixture(baseline, proposal, weight, seed=0, radius=None):
    x, y = [np.asarray(a, dtype=float) for a in (baseline, proposal)]
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] < 2:
        raise ValueError('Require matching joint grids with at least two cells')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Require finite parent draws')
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Mixture weight must lie in [0,1]')
    if radius is not None and (not np.isfinite(radius) or radius<=0):
        raise ValueError('Common shift radius must be finite and positive')
    if weight == 0 or np.array_equal(x, y):
        return x.copy()
    shift=y.mean(axis=1)-x.mean(axis=1)
    if radius is not None:
        # One common offset: the smallest marginal width enforces every cell's
        # trust radius. This is a robustness policy, not an invalidity diagnosis.
        width=np.quantile(x,.9,axis=0)-np.quantile(x,.1,axis=0)
        limit=radius*float(width.min())
        shift=np.clip(shift,-limit,limit)
    projected = x + shift[:, None]
    return combine([x, projected], [1-weight, weight], 'joint_mixture', seed=seed)
