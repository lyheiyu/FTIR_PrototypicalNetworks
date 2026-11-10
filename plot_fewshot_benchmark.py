"""Regenerate publication figures from saved benchmark results, without TensorFlow."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from utils import utils


NAMES = {
    'euclidean': 'Euclidean Proto',
    'shared_mahalanobis': 'Shared Mahalanobis',
    'diagonal_mahalanobis': 'Class-wise diagonal Mahalanobis',
    'class_adaptive_mahalanobis': 'Class-adaptive full Mahalanobis',
    'svm_raw': 'Linear SVM (spectra)',
    'knn_raw': '3-NN (spectra)',
}
COLORS = dict(zip(NAMES, ['#666666', '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']))
METRICS = ['kappa', 'macro_f1', 'accuracy', 'macro_precision', 'macro_recall']


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def summarize(data):
    records = []
    for (dataset, shot, method), group in data.groupby(['dataset', 'shot', 'method']):
        for metric in METRICS:
            values = group[metric].to_numpy()
            n = len(values)
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            records.append({'dataset': dataset, 'shot': shot, 'method': method, 'metric': metric,
                            'mean': values.mean(), 'seed_std': sd, 'ci95_halfwidth': half, 'n_seeds': n})
    return pd.DataFrame(records)


def plot_results(output):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    data = pd.read_csv(output / 'seed_metrics.csv')
    methods = [method for method in NAMES if method in set(data.method)]
    expected = len(config['datasets']) * len(config['seeds']) * len(config['shots']) * len(methods)
    if len(data) != expected or data.duplicated(['dataset', 'seed', 'shot', 'method']).any():
        raise ValueError('Incomplete/duplicate seed_metrics.csv; wait for the benchmark to finish')
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    summary = summarize(data)
    summary.to_csv(output / 'summary_metrics.csv', index=False)
    datasets = config['datasets']
    class_names = config['class_names']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 8), squeeze=False,
                             constrained_layout=True)
    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(['accuracy', 'macro_f1']):
            ax = axes[row, col]
            for method in methods:
                group = summary[(summary.dataset == dataset) & (summary.metric == metric)
                                & (summary.method == method)].sort_values('shot')
                # A single seed has no estimated between-seed CI.
                ax.errorbar(group.shot, 100 * group['mean'],
                            yerr=100 * group.ci95_halfwidth.fillna(0), color=COLORS[method],
                            marker='o', capsize=3, label=NAMES[method])
            ax.set(title=f'{dataset.upper()} | {metric.replace("_", " ")}',
                   xlabel='Real support spectra per class (K-shot)',
                   ylabel='Accuracy (%)' if metric == 'accuracy' else 'Macro-F1 (%)',
                   xticks=sorted(config['shots']))
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.09), ncol=2)
    save_figure(fig, figures / 'shot_performance')

    pooled_class_records = []
    for dataset in datasets:
        for shot in config['shots']:
            matrices = {method: np.zeros((6, 6), dtype=np.int64) for method in methods}
            for seed in config['seeds']:
                path = output / dataset / f'seed_{seed}' / f'confusion_matrices_{shot}shot.npz'
                with np.load(path, allow_pickle=False) as cms:
                    for method in methods:
                        matrices[method] += cms[method]
            fig, axes = plt.subplots(2, 3 if len(methods) > 4 else 2,
                                     figsize=(15 if len(methods) > 4 else 11, 9),
                                     constrained_layout=True)
            for ax, method in zip(axes.flat, methods):
                cm = matrices[method]
                artist = utils.draw_confusion_matrix(ax, cm, class_names, NAMES[method])
                pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
                    figures / f'{dataset}_{shot}shot_{method}_cm_counts.csv', index_label='true_class')
                counts = cm.sum(axis=1, keepdims=True)
                normalized = np.divide(cm.astype(float), counts, out=np.zeros_like(cm, dtype=float), where=counts > 0)
                pd.DataFrame(normalized, index=class_names, columns=class_names).to_csv(
                    figures / f'{dataset}_{shot}shot_{method}_cm_normalized.csv', index_label='true_class')
                for c, name in enumerate(class_names):
                    tp = cm[c, c]
                    precision = tp / cm[:, c].sum() if cm[:, c].sum() else 0
                    recall = tp / cm[c].sum() if cm[c].sum() else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                    pooled_class_records.append({'dataset': dataset, 'shot': shot, 'method': method,
                                                 'class': name, 'precision': precision, 'recall': recall,
                                                 'f1': f1, 'query_occurrences': int(cm[c].sum())})
            for ax in list(axes.flat)[len(methods):]:
                ax.set_visible(False)
            fig.colorbar(artist, ax=list(axes.flat), shrink=0.7, label='Fraction of true class')
            fig.suptitle(f'{dataset.upper()} | 6-way {shot}-shot | pooled query occurrences', fontsize=14)
            save_figure(fig, figures / f'{dataset}_{shot}shot_confusion_matrices')
    classes = pd.DataFrame(pooled_class_records)
    classes.to_csv(output / 'per_class_metrics.csv', index=False)
    primary_shot = 5 if 5 in config['shots'] else config['shots'][0]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4), squeeze=False,
                             constrained_layout=True)
    for ax, dataset in zip(axes.flat, datasets):
        group = classes[(classes.dataset == dataset) & (classes.shot == primary_shot)]
        width = 0.8 / len(methods)
        for i, method in enumerate(methods):
            values = group[group.method == method].set_index('class').loc[class_names, 'recall']
            ax.bar(np.arange(6) - 0.4 + width / 2 + i * width, 100 * values,
                   width=width, color=COLORS[method], label=NAMES[method])
        ax.set(xticks=np.arange(6), xticklabels=class_names, ylabel='Recall (%)',
               title=f'{dataset.upper()} | {primary_shot}-shot', ylim=(0, 105))
        ax.grid(axis='y', alpha=0.2)
    axes[0, -1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
    save_figure(fig, figures / f'per_class_recall_{primary_shot}shot')
    print(f'Figures saved: {figures}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Completed benchmark output directory')
    plot_results(parser.parse_args().output)
