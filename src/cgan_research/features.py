"""Daily physical-load features with explicitly versioned estimators."""
import numpy as np

FEATURE_NAMES = ('near_peak', 'near_base', 'peak_duration', 'rise_time', 'fall_time')
METRIC_VERSION = 'corrected-v1'


def daily_features(loads, mode='corrected'):
    x = np.asarray(loads, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 24 or not np.isfinite(x).all():
        raise ValueError('Expected finite physical loads (N,24)')
    if not len(x):
        return np.empty((0, 5), dtype=np.float64)
    if mode == 'corrected':
        base, peak = np.quantile(x, [0.025, 0.975], axis=1, method='linear')
    elif mode == 'upstream_compatibility':
        ordered = np.sort(x, axis=1)
        base, peak = ordered[:, 1], ordered[:, 22]
    else:
        raise ValueError('Unknown feature estimator')
    duration = (x > ((base+peak)/2)[:, None]).sum(axis=1)
    maximum = x.argmax(axis=1)  # first tied maximum
    if mode == 'corrected':
        rows = np.arange(len(x))
        rise = np.zeros(len(x), dtype=np.int64)
        fall = np.zeros(len(x), dtype=np.int64)
        flat = x.max(axis=1) == x.min(axis=1)
        for offset in range(1, 25):
            before = x[rows, (maximum-offset) % 24] <= base
            after = x[rows, (maximum+offset) % 24] <= base
            rise[(rise == 0) & before & ~flat] = offset
            fall[(fall == 0) & after & ~flat] = offset
    else:
        # Historical index distances, separately labeled; never used for selection.
        peak_distance = np.abs(x-peak[:, None])
        base_distance = np.abs(x-base[:, None])
        rise = np.abs(peak_distance.argmax(axis=1)-peak_distance.argmin(axis=1))
        fall = np.abs(base_distance.argmax(axis=1)-base_distance.argmin(axis=1))
    return np.column_stack((peak, base, duration, rise, fall)).astype(np.float64)
