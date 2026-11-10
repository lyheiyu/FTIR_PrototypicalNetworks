"""Descriptive paired analysis and preregistered stop rule; no model search."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import utils


def markdown(data):
    def cell(value):
        return f'{value:.3f}' if isinstance(value, (float, np.floating)) else str(value)
    rows = ['| ' + ' | '.join(map(str, data.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(data.columns)) + ' |']
    rows += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in data.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def analyze(output):
    output = Path(output)
    config = json.loads((output / 'experiment_manifest.json').read_text())
    files = sorted(output.glob('*/seed_*/metrics.csv'))
    if len(files) != len(config['tasks']) * len(config['seeds']):
        raise ValueError('Missing planned paired trials')
    metrics, costs, geometries = [], [], []
    for f in files:
        task, seed = f.parent.parent.name, int(f.parent.name.split('_')[1])
        values = pd.read_csv(f)
        expected = len(config['shots']) * config['episodes'] * 5
        if len(values) != expected or values.duplicated(['head', 'shot', 'episode']).any():
            raise ValueError(f'Incomplete/duplicate metrics: {f}')
        metrics.append(values)
        cost = pd.read_csv(f.parent / 'costs.csv'); cost['task'], cost['seed'] = task, seed
        if cost.optimizer_updates.nunique() != 1 or cost.training_order_hash.nunique() != 1:
            raise ValueError('Paired training budget/order mismatch')
        for row in cost.itertuples():
            if row.weight_hash_before != row.weight_hash_after:
                raise ValueError('Frozen weights changed')
        if abs(cost.set_index('architecture').loc['M1','parameters'] / cost.set_index('architecture').loc['M0','parameters'] - 1) > .1:
            raise ValueError('Capacity constraint failed')
        costs.append(cost)
        g = pd.read_csv(f.parent / 'feature_geometry.csv'); g['task'], g['seed'] = task, seed
        geometries.append(g)
    metrics, costs, geometries = pd.concat(metrics), pd.concat(costs), pd.concat(geometries)
    metrics.to_csv(output / 'metrics.csv', index=False)
    costs.to_csv(output / 'costs.csv', index=False)
    geometries.to_csv(output / 'feature_geometry.csv', index=False)
    trial = metrics.groupby(['task', 'seed', 'shot', 'head'])[['accuracy','macro_f1','head_fit_ms','head_predict_ms']].mean().reset_index()
    trial.to_csv(output / 'trial_metrics.csv', index=False)
    accuracy = trial.groupby(['shot','head']).accuracy.mean().unstack() * 100
    accuracy.to_csv(output / 'accuracy_table.csv')
    macro = trial.groupby(['shot','head']).macro_f1.mean().unstack() * 100
    macro.to_csv(output / 'macro_f1_table.csv')
    deltas = []
    for kind in ['svm', '1nn']:
        a = trial[trial['head'] == f'M0_{kind}'].set_index(['task','seed','shot'])
        b = trial[trial['head'] == f'M1_{kind}'].set_index(['task','seed','shot'])
        for index, value in ((b.accuracy - a.accuracy) * 100).items():
            deltas.append({'task': index[0], 'seed': index[1], 'shot': index[2], 'head': kind, 'gain_pp': value})
    deltas = pd.DataFrame(deltas); deltas.to_csv(output / 'paired_deltas.csv', index=False)
    primary = deltas[deltas.shot.isin(config['decision_rule']['primary_shots'])]
    paired = primary.groupby(['task','seed','head']).gain_pp.mean().reset_index()
    decisions = []
    for kind in ['svm', '1nn']:
        subset = paired[paired['head'] == kind]
        task_means = subset.groupby('task').gain_pp.mean()
        decisions.append({'head': kind, 'mean_gain_pp': float(subset.gain_pp.mean()),
            'winning_trials': int((subset.gain_pp > 0).sum()), 'total_trials': len(subset),
            'positive_task_means': int((task_means > 0).sum()),
            'passes': bool(subset.gain_pp.mean() >= 1 and (subset.gain_pp > 0).sum() >= 6 and (task_means > 0).sum() == 3)})
    decision = {'stage': config['stage'], 'primary': 'equal-weight mean of 5/10-shot accuracy',
                'head_decisions': decisions,
                'continue_architecture_investigation': bool(config['stage'] == 'planned' and all(d['passes'] for d in decisions))}
    write = lambda name, text: (output / name).write_text(text, encoding='utf-8')
    write('decision.json', json.dumps(decision, indent=2))
    cost_columns = ['parameters','training_seconds','optimizer_updates','inference_median_ms_per_spectrum',
                    'all_spectra_feature_extraction_seconds','model_file_bytes']
    cost_summary = costs.groupby('architecture')[cost_columns].mean().reset_index()
    cost_summary.to_csv(output / 'cost_summary.csv', index=False)
    geometry_summary = geometries.groupby(['architecture','role'])[
        ['mean_norm','effective_rank_entropy_covariance','within_class_distance',
         'between_class_distance','neighbor_consistency_excluding_self']].mean().reset_index()
    architecture_text = (
        '# Paired M0/M1 architecture comparison\n\n'
        'C4 exploratory, base-only CE, identical fixed 1868-point legacy-cubic input, saved splits, '
        'explicit epoch orders, optimizer update budget and legal unseen episodes. '
        '128-dimensional unnormalized ReLU embeddings; linear SVM C tuned only on base validation.\n\n'
        'M0: compact three-convolution network with global average pooling. '
        'M1: 9/25/49-point branches, two factor-2 pools, 32-channel local convolutions, '
        '8-channel projection and 32 contiguous position bins before the embedding. '
        'This tests the complete M1 design, not isolated causality of multiscale vs position.\n\n'
        '## Unseen two-class accuracy (%)\n\n' + markdown(accuracy.reset_index()) +
        '\n\n## Model and cost means\n\n' + markdown(cost_summary) +
        '\n\nTraining timing includes fixed-epoch fitting, base validation and checkpoint writes. '
        'Inference is a warmed-up graph, batch 64, median over ten repeats, excluding preprocessing and classifier. '
        'Raw SVM head timings are in metrics.csv; shared preprocessing is recorded separately. '
        'No speed claim is made from parameter counts alone.\n')
    write('architecture_comparison.md', architecture_text)
    write('feature_geometry.md', '# Descriptive feature geometry\n\n' + markdown(geometry_summary) +
        '\n\nEffective rank uses the entropy of the normalized covariance eigenvalues on centered features. '
        'Distances are Euclidean on unnormalized features. Neighbor consistency excludes self-neighbors. '
        'Novel-query geometry is retrospective analysis only, never used to fit transformations or select models.\n')
    write('incremental_analysis.md', '# Scope boundary\n\nThis narrowed study evaluates frozen representations on two unseen classes, '
        'not sequential 4+1+1 competition. No incremental performance or registration-speed conclusion follows. '
        'If the paired feature-quality gate fails, stop this architecture investigation; no additional architecture/loss is introduced.\n')
    if config['stage'] == 'smoke':
        verdict = 'IMPLEMENTATION VALIDATION ONLY. Two epochs do not decide representation superiority.'
    elif decision['continue_architecture_investigation']:
        verdict = 'M1 meets the predeclared descriptive consistency gate. No additional architectures or losses were tried.'
    else:
        verdict = 'STOP: M1 does not meet the predeclared consistency gate. Do not add architectures or losses to rescue this investigation.'
    write('conclusion.md', '# Conclusion\n\n' + verdict + '\n\n' + markdown(pd.DataFrame(decisions)) +
        '\n\nGate: each of SVM and 1-NN must gain at least 1 percentage point on the 5/10-shot average, '
        'win at least 6 of 9 paired task-seed trials, and have positive means on all three tasks. '
        'This is a descriptive preregistered stop rule, not a significance test; task/seed samples overlap. '
        'C4 novel results have been inspected before; exact deduplication does not establish specimen independence. '
        'No independent confirmatory dataset has been evaluated.\n')
    figures = output / 'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, table, label in [(axes[0],accuracy,'Accuracy (%)'),(axes[1],macro,'Macro-F1 (%)')]:
        for head in table: ax.plot(table.index, table[head], marker='o', label=head)
        ax.set(xlabel='Support spectra per unseen class', ylabel=label, xticks=sorted(config['shots']))
        ax.grid(alpha=.25)
    axes[1].legend(fontsize=8); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(figures / f'unseen_feature_quality.{ext}', dpi=300)
    plt.close(fig)
    for task in config['tasks']:
        classes = sorted(c for session in task['novel_sessions'] for c in session)
        requested = [k for k in [5,20] if k in config['shots']]
        if not requested: requested = [config['shots'][0]]
        fig, axes = plt.subplots(len(requested), 5, figsize=(19, 4*len(requested)), squeeze=False)
        for row, k in enumerate(requested):
            for col, head in enumerate(['raw_svm','M0_svm','M0_1nn','M1_svm','M1_1nn']):
                cm = np.zeros((2,2), dtype=np.int64)
                for seed in config['seeds']:
                    with np.load(output/task['name']/f'seed_{seed}'/'confusions.npz') as saved:
                        cm += saved[f'{head}_{k}shot']
                utils.draw_confusion_matrix(axes[row,col],cm,classes,f'{head} | {k}-shot')
                pd.DataFrame(cm,index=classes,columns=classes).to_csv(figures/f'{task["name"]}_{head}_{k}shot_CM.csv')
        fig.suptitle(task['name']+' | pooled query occurrences'); fig.tight_layout()
        for ext in ['png','pdf']: fig.savefig(figures/f'{task["name"]}_CM.{ext}', dpi=250)
        plt.close(fig)
    print(verdict, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path)
    analyze(parser.parse_args().output)
