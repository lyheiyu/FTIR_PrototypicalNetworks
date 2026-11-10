"""Versioned partial-spectrum adapter; original preprocessing stays untouched."""
import numpy as np
from ftir_research_data import preprocess


def observed_mask(wave, grid):
    wave, grid = np.asarray(wave), np.asarray(grid)
    if np.any(np.diff(grid) <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError('Strictly ascending physical axes required')
    return (grid >= wave[0]) & (grid <= wave[-1])


def base_input_mean(c4_input, train_indices):
    """Only explicitly supplied, already-audited base_train rows are fitted."""
    if not len(train_indices):
        raise ValueError('Empty legal base training set')
    return np.asarray(c4_input[train_indices], dtype=np.float64).mean(axis=0).astype(np.float32)


def apply_missing_mask(inputs, mask, mean):
    inputs = np.asarray(inputs)
    mask, mean = np.asarray(mask, bool), np.asarray(mean)
    if inputs.shape[1:] != mean.shape or mean.shape != mask.shape:
        raise ValueError('Input/mean/mask dimensions differ')
    out = inputs.copy()
    out[:, ~mask] = mean[~mask]
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite adapted input')
    return out


def partial_input(data, grid, mean, cache_dir):
    """C4 numerical operators, applied on native observed data BEFORE imputation.

    Explicit flip=True matches C4 max(y)-y. This is a locked numerical convention,
    NOT a verified absorbance/transmittance conversion. D4 registry defaults are
    not changed. Baseline and maxabs normalization use native measured values.
    """
    mask = observed_mask(data.wave, grid)
    observed = preprocess(data, grid[mask], cache_dir, baseline=True, flip=True)
    x = np.broadcast_to(np.asarray(mean, np.float32), (len(data.x), len(grid))).copy()
    x[:, mask] = observed
    assert np.isfinite(x).all() and np.array_equal(x[:, ~mask], np.broadcast_to(mean[~mask], x[:, ~mask].shape))
    return x, mask


def fast_metrics(true, predicted, seen, base=(), previous=(), current=()):
    """Confusion-derived group metrics; same global one-vs-rest F1 as utils."""
    classes = sorted(seen)
    mapping = {c:i for i,c in enumerate(classes)}
    a = np.fromiter((mapping[c] for c in true), int)
    b = np.fromiter((mapping[c] for c in predicted), int)
    cm = np.bincount(a*len(classes)+b, minlength=len(classes)**2).reshape(len(classes),len(classes))
    tp, support, predicted_count = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(tp,support,out=np.zeros(len(classes),float),where=support>0)
    precision = np.divide(tp,predicted_count,out=np.zeros(len(classes),float),where=predicted_count>0)
    f1 = np.divide(2*tp,support+predicted_count,out=np.zeros(len(classes),float),where=(support+predicted_count)>0)
    groups = {'all':classes,'base':list(base),'previous_novel':list(previous),'current_novel':list(current),
              'seen_novel':[c for c in classes if c not in base]}
    metrics = {}
    for name,group in groups.items():
        indices = [mapping[c] for c in group]
        n = support[indices].sum()
        metrics[name+'_accuracy'] = float(tp[indices].sum()/n) if n else np.nan
        metrics[name+'_macro_f1'] = float(f1[indices].mean()) if indices else np.nan
    metrics['accuracy'],metrics['macro_f1'] = metrics['all_accuracy'],metrics['all_macro_f1']
    report = {c:{'precision':float(precision[i]),'recall':float(recall[i]),'f1-score':float(f1[i]),'support':int(support[i])}
              for i,c in enumerate(classes)}
    return metrics,report,cm
