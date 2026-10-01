"""## Executive summary (read this first)

Combine cached joint distributions without outcomes. Center averages retain the
first parent's residual cloud. Mixtures choose one parent for a complete joint
row, using a shared row partition so identical parents reproduce the baseline
exactly. Log-level columns retain their positive support under center/width edits.
"""
import numpy as np


def conditional_common(source, scale, positive_columns=()):
    """Shrink a covariance-weighted common component in the official raw units.

    w = covariance^-1 1 / (1' covariance^-1 1), c = (X-mean)w.
    The scalar c has zero sample covariance with every column contrast. Adding
    the same (scale-1)c to every cell preserves all within-row differences and
    column means. Gaussian independence is not assumed for non-Gaussian draws.
    """
    x=np.asarray(source,dtype=float)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('Expected finite joint draws')
    if not np.isfinite(scale) or not 0<=scale<=1:
        raise ValueError('Common shrink scale must lie in[0,1]')
    audit={'method':'covariance-weighted common shrinkage','scale':scale,
           'coordinate_system':'raw official target values','applied':False}
    def fallback(reason):
        return x.copy(),{**audit,'fallback':reason}
    columns=list(positive_columns)
    if np.any(x[:,columns]<=0):
        raise ValueError('Original positive-support column is invalid')
    if x.shape[1]<2:
        return fallback('Single cell; no joint common/contrast decomposition')
    centered=x-x.mean(axis=0)
    sd=np.std(centered,axis=0,ddof=1)
    if np.any(sd<=0):
        return fallback('Degenerate column variance')
    standardized=centered/sd
    correlation=standardized.T@standardized/(len(x)-1)
    condition=float(np.linalg.cond(correlation));audit['correlation_condition']=condition
    if not np.isfinite(condition) or condition>1e10:
        return fallback('Ill-conditioned correlation matrix')
    inverse_sd=1/sd
    solved=np.linalg.solve(correlation,inverse_sd)
    denominator=float(inverse_sd@solved)
    if denominator<=0 or not np.isfinite(denominator):
        return fallback('Invalid GLS normalization')
    weights=(solved/sd)/denominator
    common=centered@weights
    result=x+(scale-1)*common[:,None]
    if not np.isfinite(result).all():
        return fallback('Nonfinite transformed draws')
    if np.any(result[:,columns]<=0):
        return fallback('Transformed FX support invalid; retain entire baseline')
    audit.update(applied=True,weights=weights.tolist(),common_variance=1/denominator,
                 removed_covariance=(1-scale**2)/denominator,
                 note='Means and raw row contrasts fixed; marginal medians and higher moments may change')
    return result,audit


def news_increment(baseline, news, no_news, scale, log_columns=()):
    """Add the paired LLM news effect while retaining baseline model residuals.

    Use at most12 generations per condition, so the complete method fits within
    the25-call allowance. This is a median contrast, not a causal-effect estimate.
    """
    baseline, news, no_news = [np.asarray(p, dtype=float) for p in (baseline, news, no_news)]
    if (baseline.ndim != 2 or news.ndim != 2 or no_news.shape != news.shape
            or news.shape[1] != baseline.shape[1] or not 1 <= len(news) <= 12):
        raise ValueError('Expected paired1-12 trajectory grids matching baseline cells')
    if not 0 <= scale <= 1 or not all(np.isfinite(p).all() for p in (baseline, news, no_news)):
        raise ValueError('Invalid news-effect weight or nonfinite trajectories')
    work = [p.copy() for p in (baseline, news, no_news)]
    columns = list(log_columns)
    for p in work:
        if np.any(p[:, columns] <= 0):
            raise ValueError('FX inputs must have positive support')
        p[:, columns] = np.log(p[:, columns])
    delta = scale * (np.median(work[1], axis=0) - np.median(work[2], axis=0))
    if not np.any(delta):
        return baseline.copy()
    result = work[0] + delta
    result[:, columns] = np.exp(result[:, columns])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite news-effect translation')
    return result


def combine(parents, weights, mode, log_columns=(), scale=1., seed=0):
    parents = [np.asarray(p, dtype=float) for p in parents]
    if not parents or any(p.shape != parents[0].shape for p in parents):
        raise ValueError("Parent joint grids must match")
    if any(p.ndim != 2 or not np.isfinite(p).all() for p in parents):
        raise ValueError("Parents must be finite matrices")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(parents),) or np.any(weights < 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.):
        raise ValueError("Use finite nonnegative weights summing to one")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Width scale must be positive")
    if mode in ("center_common", "center_contrast"):
        if len(log_columns) or scale != 1.:
            raise ValueError("Center projection requires raw additive units and fixed width")
        centers = np.array([np.median(p, axis=0) for p in parents])
        delta = np.sum(weights[:, None] * centers, axis=0) - centers[0]
        common = np.full_like(delta, delta.mean())
        correction = common if mode == "center_common" else delta - common
        return parents[0] + correction
    if mode == "common_mode":
        if len(parents) != 1 or len(log_columns):
            raise ValueError("Common-mode control requires one parent in raw additive units")
        # Scale only the component parallel to the all-ones vector. Every
        # within-row coordinate difference is unchanged, hence the variogram is
        # unchanged. Means stay fixed; individual sample medians need not.
        source = parents[0]
        common = (source - source.mean(axis=0)).mean(axis=1, keepdims=True)
        return source + (scale - 1.) * common
    if mode == "joint_mixture":
        if scale != 1.:
            raise ValueError("Keep mixture and width controls separate")
        n = len(parents[0])
        counts = weights * n
        if not np.allclose(counts, np.round(counts)):
            raise ValueError("Mixture weights must have exact integer draw counts")
        indices = np.random.default_rng(seed).permutation(n)
        result = np.empty_like(parents[0])
        offset = 0
        for parent, count in zip(parents, np.round(counts).astype(int)):
            selected = indices[offset:offset + count]
            result[selected] = parent[selected]
            offset += count
        return result
    if mode not in ("center_mean", "width"):
        raise ValueError("Unknown numerical combination")
    if mode == "width" and len(parents) != 1:
        raise ValueError("Width control requires one fixed parent")
    working = [p.copy() for p in parents]
    log_columns = list(log_columns)
    for p in working:
        if np.any(p[:, log_columns] <= 0):
            raise ValueError("Log-level parents must be positive")
        p[:, log_columns] = np.log(p[:, log_columns])
    centers = np.array([np.median(p, axis=0) for p in working])
    residual = working[0] - centers[0]
    result = np.sum(weights[:, None] * centers, axis=0) + scale * residual
    result[:, log_columns] = np.exp(result[:, log_columns])
    if not np.isfinite(result).all():
        raise ValueError("Combination produced nonfinite draws")
    return result
