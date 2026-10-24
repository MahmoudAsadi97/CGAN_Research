"""Corrected distribution metrics; all learned references are training-only."""
import numpy as np
from scipy.stats import wasserstein_distance
from .features import FEATURE_NAMES, daily_features, METRIC_VERSION
from .checkpoint import write_json


def _features(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError('Expected finite (N,5) features')
    return values


def fit_metric_reference(train_features, bins=30):
    x = _features(train_features)
    if len(x) < 2 or type(bins) is not int or bins < 1:
        raise ValueError('At least two training profiles and positive bins required')
    edges = {}
    for i, name in enumerate(FEATURE_NAMES):
        if i < 2:
            lo, hi = x[:, i].min(), x[:, i].max()
            if lo == hi:
                lo, hi = lo-0.5, hi+0.5
            edges[name] = np.linspace(lo, hi, bins+1).tolist()
        else:
            edges[name] = np.arange(-0.5, 25.5, 1).tolist()
    sd = x.std(axis=0, ddof=1)
    constant = np.ptp(x, axis=0) == 0
    return {'version':METRIC_VERSION, 'bins':bins, 'pseudocount':0.5,'log_base':'natural',
            'edges':edges,'mean':x.mean(axis=0).tolist(),'scale':np.where(constant,1.,sd).tolist(),
            'constant':constant.tolist(), 'n_train':len(x)}


def fit_reference_from_prepared(prepared, metric_config=None, output_path=None):
    bins = (metric_config or {}).get('bins', 30)
    ref = fit_metric_reference(daily_features(prepared.loads[prepared.splits['train']]), bins)
    ref.update(data_fingerprint=prepared.manifest['data_fingerprint'], split_fingerprint=prepared.manifest['split_fingerprint'])
    if output_path is not None:
        write_json(output_path, ref)
    return ref


def histogram_probabilities(values, edges, pseudocount=0.5):
    x = np.asarray(values, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(e).all() or (np.diff(e) <= 0).any() or pseudocount <= 0:
        raise ValueError('Invalid histogram input')
    # searchsorted includes both exact extrema in the interior bins.
    index = np.searchsorted(e, x, side='right')
    index[x == e[-1]] = len(e)-1
    counts = np.bincount(index, minlength=len(e)+1).astype(float) + pseudocount
    return counts/counts.sum()


def _psd_root(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all() or not np.allclose(a,a.T,rtol=1e-10,atol=1e-12):
        raise ValueError('Covariance must be finite and symmetric')
    eigenvalues, eigenvectors = np.linalg.eigh((a+a.T)/2)
    tolerance = 1e-10*max(1.,np.max(np.abs(eigenvalues)))
    if eigenvalues.min() < -tolerance:
        raise ValueError('Covariance is not positive semidefinite')
    return (eigenvectors*np.sqrt(np.maximum(eigenvalues,0)))@eigenvectors.T


def gaussian_frechet(mean_real, covariance_real, mean_generated, covariance_generated):
    mr, mg = np.asarray(mean_real), np.asarray(mean_generated)
    cr, cg = np.atleast_2d(covariance_real), np.atleast_2d(covariance_generated)
    if mr.shape != mg.shape or mr.ndim != 1 or cr.shape != (len(mr),len(mr)) or cg.shape != cr.shape or not np.isfinite(mr).all() or not np.isfinite(mg).all():
        raise ValueError('Incompatible Gaussian parameters')
    root = _psd_root(cr)
    generated_root = _psd_root(cg)
    # Equivalent to tr(sqrt(root @ cg @ root)); singular values avoid
    # amplifying near-zero covariance eigenvalue roundoff with another sqrt.
    cross_trace = np.linalg.svd(root @ generated_root, compute_uv=False).sum()
    value = float(np.sum((mr-mg)**2) + np.trace(cr) + np.trace(cg) - 2*cross_trace)
    tolerance = 1e-8*max(1.,float(np.trace(cr)+np.trace(cg)))
    if value < -tolerance:
        raise ValueError('Significantly negative Frechet distance')
    return max(0., value)


def compare_distributions(real, generated, reference):
    r, g = _features(real), _features(generated)
    result = {'features':{}, 'n_real':len(r),'n_generated':len(g),'joint_frechet':None,
              'joint_reason':None, 'version':METRIC_VERSION}
    for i, name in enumerate(FEATURE_NAMES):
        cell = dict.fromkeys(('mean_difference','abs_mean_difference','std_difference','abs_std_difference','kl','js','wasserstein','frechet'))
        cell.update(reason=None, variance_reason=None)
        if not len(r) or not len(g):
            cell['reason'] = 'empty_distribution'
        else:
            a, b = r[:,i],g[:,i]
            cell['mean_difference'] = float(b.mean()-a.mean())
            cell['abs_mean_difference'] = abs(cell['mean_difference'])
            p = histogram_probabilities(a, reference['edges'][name], reference['pseudocount'])
            q = histogram_probabilities(b, reference['edges'][name], reference['pseudocount'])
            midpoint = (p+q)/2
            cell['kl'] = float(np.sum(p*np.log(p/q)))
            cell['js'] = float((np.sum(p*np.log(p/midpoint))+np.sum(q*np.log(q/midpoint)))/2)
            cell['wasserstein'] = float(wasserstein_distance(a,b))
            if min(len(r),len(g)) < 2:
                cell['variance_reason'] = 'fewer_than_two_samples'
            else:
                cell['std_difference'] = float(b.std(ddof=1)-a.std(ddof=1))
                cell['abs_std_difference'] = abs(cell['std_difference'])
                cell['frechet'] = cell['mean_difference']**2+cell['std_difference']**2
        result['features'][name] = cell
    if min(len(r),len(g)) < 2:
        result['joint_reason'] = 'fewer_than_two_samples'
    else:
        scale, mean = np.asarray(reference['scale']), np.asarray(reference['mean'])
        rs, gs = (r-mean)/scale,(g-mean)/scale
        result['joint_frechet'] = gaussian_frechet(rs.mean(axis=0),np.cov(rs,rowvar=False),gs.mean(axis=0),np.cov(gs,rowvar=False))
    return result
