"""Label-independent classification heads and auditable result export."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils import utils

METRIC_NAMES = ['kappa', 'macro_f1', 'accuracy', 'macro_precision', 'macro_recall']


def base_covariance(z, labels):
    residual = np.empty_like(z, dtype=float)
    for label in np.unique(labels):
        mask = labels == label
        residual[mask] = z[mask] - z[mask].mean(axis=0)
    return residual.T @ residual / len(residual)


class Head:
    """All heads receive exactly the same retained exemplars; no query fitting."""
    def __init__(self, kind, reg=1e-3, c=1.0, tau=5.0):
        self.kind, self.reg, self.c, self.tau = kind, reg, c, tau

    def fit(self, x, labels, prior=None):
        self.classes = np.unique(labels)
        self.memory_bytes = int(x.nbytes + labels.nbytes)
        if self.kind in ('svm', '1nn'):
            self.estimator = SVC(C=self.c, kernel='linear') if self.kind == 'svm' else KNeighborsClassifier(n_neighbors=1)
            self.estimator.fit(x, labels)
            return self
        self.means = np.stack([x[labels == c].mean(axis=0) for c in self.classes])
        self.inverses = []
        if self.kind == 'proto':
            return self
        for c, mean in zip(self.classes, self.means):
            residual = x[labels == c] - mean
            cov = residual.T @ residual / len(residual)
            if self.kind == 'shared_prior':
                cov = prior
            elif self.kind == 'shrinkage':
                alpha = (len(residual) - 1) / (len(residual) - 1 + self.tau)
                cov = alpha * cov + (1 - alpha) * prior
            elif self.kind != 'class':
                raise ValueError(self.kind)
            if cov is None:
                raise ValueError('Base-only covariance prior is required')
            self.inverses.append(np.linalg.inv(cov + self.reg * np.eye(x.shape[1])))
        return self

    def predict(self, x):
        if self.kind in ('svm', '1nn'):
            return self.estimator.predict(x)
        diff = x[:, None, :] - self.means[None, :, :]
        if self.kind == 'proto':
            distance = np.sum(diff**2, axis=2)
        else:
            distance = np.einsum('mci,cij,mcj->mc', diff, np.asarray(self.inverses), diff)
        return self.classes[np.argmin(distance, axis=1)]


SPECS = {
    'raw_svm': ('raw', 'svm'), 'raw_1nn': ('raw', '1nn'), 'raw_proto': ('raw', 'proto'),
    'pca_svm': ('pca', 'svm'), 'pca_proto': ('pca', 'proto'),
    'embedding_svm': ('embedding', 'svm'), 'embedding_1nn': ('embedding', '1nn'),
    'embedding_proto': ('embedding', 'proto'), 'embedding_shared_prior': ('embedding', 'shared_prior'),
    'embedding_class': ('embedding', 'class'), 'embedding_shrinkage': ('embedding', 'shrinkage'),
}


def score(true, predicted, classes):
    named, report, cm = utils.classification_metrics(true, predicted, classes, classes.tolist())
    return named, report, cm


def tune_heads(views, labels, pool, methods, prior, rng, k, q, episodes, c_grid, reg_grid, tau_grid):
    """Tune only on base validation episodes. All candidates see same episodes."""
    classes = np.unique(labels[pool])
    minimum = min(np.sum(labels[pool] == c) for c in classes)
    if minimum < 2:
        raise ValueError('Each validation class needs at least two spectra')
    actual_k = min(k, max(1, minimum // 2))
    actual_q = min(q, minimum - actual_k)
    tasks = []
    for _ in range(episodes):
        s, query = [], []
        for c in classes:
            chosen = rng.choice(pool[labels[pool] == c], actual_k + actual_q, replace=False)
            s.extend(chosen[:actual_k]); query.extend(chosen[actual_k:])
        tasks.append((np.asarray(s), np.asarray(query)))
    selected, audit = {}, []
    for method in methods:
        view, kind = SPECS[method]
        candidates = ([{'c': c} for c in c_grid] if kind == 'svm' else
                      [{'reg': r, 'tau': t} for r in reg_grid for t in tau_grid] if kind == 'shrinkage' else
                      [{'reg': r} for r in reg_grid] if kind in ('class', 'shared_prior') else [{}])
        if candidates == [{}]:
            selected[method] = {}
            continue
        best_score, best_params = -np.inf, None
        for params in candidates:
            values = []
            for s, query in tasks:
                head = Head(kind, **params).fit(views[view][s], labels[s], prior)
                metrics, _, _ = score(labels[query], head.predict(views[view][query]), classes)
                values.append(metrics['macro_f1'])
            average = float(np.mean(values))
            audit.append({'method': method, **params, 'val_macro_f1': average,
                          'requested_k': k, 'validation_k': actual_k, 'validation_q': actual_q})
            if average > best_score:
                best_score, best_params = average, params
        selected[method] = best_params
    return selected, audit


class Recorder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.metrics, self.per_class, self.cms = [], [], {}

    def add(self, meta, true, pred, classes, base, old, new, update_ms=0, predict_ms=0, memory=0):
        values, report, cm = score(true, pred, classes)
        groups = {'base_accuracy': base, 'old_accuracy': old, 'new_accuracy': new}
        for key, group in groups.items():
            mask = np.isin(true, group)
            values[key] = float(np.mean(true[mask] == pred[mask])) if mask.any() else np.nan
        self.metrics.append({**meta, **values, 'fit_ms': update_ms, 'predict_ms': predict_ms,
                             'exemplar_bytes': memory})
        for name in classes:
            self.per_class.append({**meta, 'class': name, **report[name]})
        key = (str(meta['dataset']), int(meta['shot']), int(meta['session']), str(meta['method']))
        if key not in self.cms:
            self.cms[key] = (classes.copy(), cm.copy())
        else:
            existing, total = self.cms[key]
            if not np.array_equal(existing, classes):
                raise ValueError('Class sets changed within a recorded session')
            total += cm

    def save(self):
        metrics = pd.DataFrame(self.metrics)
        keys = ['dataset', 'task', 'seed', 'shot', 'repeat', 'method']
        metrics = metrics.sort_values(keys + ['session'])
        previous = metrics.groupby(keys)['accuracy'].shift(1)
        # Same fixed old-class queries are evaluated before and after additions.
        metrics['old_accuracy_change_pp'] = 100 * (metrics['old_accuracy'] - previous)
        metrics.to_csv(self.folder / 'metrics.csv', index=False)
        pd.DataFrame(self.per_class).to_csv(self.folder / 'per_class_metrics.csv', index=False)
        index, matrices = [], {}
        for i, ((dataset, shot, session, method), (classes, cm)) in enumerate(self.cms.items()):
            key = f'cm_{i}'
            matrices[key] = cm
            index.append({'key': key, 'dataset': dataset, 'shot': shot, 'session': session,
                          'method': method, 'classes': classes.tolist()})
        np.savez_compressed(self.folder / 'confusions.npz', **matrices)
        (self.folder / 'confusion_index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')


def evaluate_head(views, labels, support, query, method, params, prior):
    view, kind = SPECS[method]
    start = time.perf_counter()
    head = Head(kind, **params).fit(views[view][support], labels[support], prior)
    fit_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    predictions = head.predict(views[view][query])
    pred_ms = (time.perf_counter() - start) * 1000
    return predictions, fit_ms, pred_ms, head.memory_bytes


def plot_run(output):
    output = Path(output)
    paths = sorted(output.glob('**/metrics.csv'))
    if not paths:
        raise ValueError('No completed metrics.csv found')
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    data.to_csv(output / 'all_metrics.csv', index=False)
    # Descriptive trial means/SD; repeat episodes are not independent training runs.
    summary_columns = METRIC_NAMES + ['base_accuracy', 'old_accuracy', 'new_accuracy',
                                      'old_accuracy_change_pp', 'fit_ms', 'predict_ms', 'exemplar_bytes']
    summary_columns = [name for name in summary_columns if name in data]
    trial = data.groupby(['dataset', 'task', 'seed', 'shot', 'session', 'method'])[summary_columns].mean().reset_index()
    trial.to_csv(output / 'trial_metrics.csv', index=False)
    summary = trial.groupby(['dataset', 'shot', 'session', 'method'])[summary_columns].agg(['mean', 'std', 'count'])
    summary.to_csv(output / 'summary_metrics.csv')
    incremental = data.session.max() > 0
    for dataset in data.dataset.unique():
        subset = data[data.dataset == dataset]
        for shot in (sorted(subset.shot.unique()) if incremental else [None]):
            rows = subset[subset.shot == shot] if incremental else subset
            axis = 'session' if incremental else 'shot'
            metrics = ['accuracy', 'base_accuracy', 'old_accuracy', 'new_accuracy'] if incremental else ['accuracy', 'macro_f1']
            fig, axes = plt.subplots(2 if incremental else 1, 2, figsize=(13, 9 if incremental else 4), squeeze=False)
            for ax, metric in zip(axes.flat, metrics):
                group = rows.groupby([axis, 'method'])[metric].mean().unstack('method')
                for method in group:
                    ax.plot(group.index, 100 * group[method], marker='o', label=method)
                ax.set(xlabel=axis, ylabel=metric, title=f'{dataset}' + (f' | {shot}-shot' if shot else ''))
                ax.grid(alpha=.2)
            axes[-1, -1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
            fig.tight_layout()
            name = f'{dataset}_{shot}shot_sessions' if incremental else f'{dataset}_diagnostic'
            for ext in ['png', 'pdf']:
                fig.savefig(figures / f'{name}.{ext}', dpi=300, bbox_inches='tight')
            plt.close(fig)
    # CM panels stay separate for each task: different held-out class orders
    # must never be averaged as though they were the same session definition.
    for metrics_path in paths:
        folder = metrics_path.parent
        index = json.loads((folder / 'confusion_index.json').read_text())
        shots = sorted({row['shot'] for row in index})
        primary = 5 if 5 in shots else shots[0]
        with np.load(folder / 'confusions.npz', allow_pickle=False) as cms:
            for dataset in sorted({row['dataset'] for row in index}):
                for session in sorted({row['session'] for row in index}):
                    entries = [row for row in index if row['dataset'] == dataset and row['session'] == session and row['shot'] == primary]
                    if not entries:
                        continue
                    fig, axes = plt.subplots(int(np.ceil(len(entries) / 3)), 3,
                                             figsize=(16, 5 * int(np.ceil(len(entries) / 3))), squeeze=False)
                    for ax, row in zip(axes.flat, entries):
                        utils.draw_confusion_matrix(ax, cms[row['key']], row['classes'], row['method'])
                        pd.DataFrame(cms[row['key']], index=row['classes'], columns=row['classes']).to_csv(
                            folder / f"{dataset}_{primary}shot_session{session}_{row['method']}_cm.csv")
                    for ax in list(axes.flat)[len(entries):]:
                        ax.set_visible(False)
                    fig.suptitle(f'{folder.name} | {dataset} | {primary}-shot | session {session}')
                    fig.tight_layout()
                    for ext in ['png', 'pdf']:
                        fig.savefig(folder / f'{dataset}_{primary}shot_session{session}_CM.{ext}', dpi=250, bbox_inches='tight')
                    plt.close(fig)
