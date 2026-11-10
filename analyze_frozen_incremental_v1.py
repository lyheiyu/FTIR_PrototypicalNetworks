"""Summarize checkpoint-reuse incremental evaluation without changing models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import utils


def table(data):
    def cell(v):
        return f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(data.columns) + ' |',
        '| ' + ' | '.join(['---'] * len(data.columns)) + ' |'] +
        ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in data.itertuples(index=False, name=None)])


def analyze(output):
    output = Path(output)
    config = json.loads((output / 'experiment_manifest.json').read_text())
    files = sorted(output.glob('*/seed_*/metrics.csv'))
    assert len(files) == len(config['tasks']) * len(config['seeds'])
    frames, costs, per_class = [], [], []
    audit_count = 0
    for file in files:
        frame = pd.read_csv(file)
        assert len(frame) == len(config['shots']) * config['repeats'] * 3 * 5
        assert not frame.duplicated(['shot', 'repeat', 'session', 'method']).any()
        for pair in json.loads((file.parent / 'frozen_check.json').read_text()).values():
            assert pair['weights_before'] == pair['weights_after'] and pair['file_before'] == pair['file_after']
        sessions = json.loads((file.parent / 'sessions.json').read_text())
        for k in config['shots']:
            for repeat in range(config['repeats']):
                seq = [s for s in sessions if s['shot'] == k and s['repeat'] == repeat]
                assert [s['session'] for s in seq] == [0, 1, 2]
                for s in seq:
                    assert len(s['support']) == 80 + s['session'] * k
                    assert len(s['query']) == config['queries_per_class'] * (4 + s['session'])
                    assert len(set(s['support'])) == len(s['support'])
                    assert len(set(s['query'])) == len(s['query'])
                    assert not set(s['support']).intersection(s['query'])
                    audit_count += 1
                for a, b in zip(seq, seq[1:]):
                    assert set(a['support']) <= set(b['support'])
                    assert set(a['query']) <= set(b['query'])
                    assert set(b['support']) - set(a['support']) == set(b['added'])
                    assert set(a['current_novel']) <= set(b['previous_novel'])
        frames.append(frame)
        costs.append(pd.read_csv(file.parent / 'registration_costs.csv'))
        per_class.append(pd.read_csv(file.parent / 'per_class_metrics.csv'))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output / 'metrics.csv', index=False)
    pd.concat(per_class, ignore_index=True).to_csv(output / 'per_class_metrics.csv', index=False)
    registration = pd.concat(costs, ignore_index=True)
    registration.to_csv(output / 'registration_costs.csv', index=False)
    key = ['task', 'seed', 'shot', 'session', 'method']
    columns = [c for c in data if c.endswith('_accuracy') or c.endswith('_macro_f1')]
    columns += ['local_registration_ms', 'memory_assembly_ms', 'svm_full_refit_ms', 'knn_index_build_ms',
                'fit_ms', 'predict_ms', 'total_head_update_ms', 'support_count', 'query_count']
    trial = data.groupby(key)[columns].mean().reset_index()
    trial.to_csv(output / 'trial_metrics.csv', index=False)
    summary = trial.groupby(['shot', 'session', 'method'])[columns].mean().reset_index()
    summary.to_csv(output / 'summary_metrics.csv', index=False)
    aia = trial.groupby(['task', 'seed', 'shot', 'method'])[['all_accuracy', 'all_macro_f1']].mean().reset_index()
    aia = aia.rename(columns={'all_accuracy': 'average_incremental_accuracy_including_base',
                              'all_macro_f1': 'average_incremental_macro_f1_including_base'})
    novel_sessions = trial[trial.session > 0].groupby(['task', 'seed', 'shot', 'method']).all_accuracy.mean()
    aia = aia.set_index(['task', 'seed', 'shot', 'method'])
    aia['average_incremental_accuracy_sessions_1_2'] = novel_sessions
    aia.reset_index().to_csv(output / 'average_incremental_metrics.csv', index=False)
    deltas = []
    for kind in ['svm', '1nn']:
        a = trial[trial.method == f'M0_{kind}'].set_index(['task', 'seed', 'shot', 'session'])
        b = trial[trial.method == f'M1_{kind}'].set_index(['task', 'seed', 'shot', 'session'])
        for index, delta in ((b.all_accuracy - a.all_accuracy) * 100).items():
            deltas.append({'task': index[0], 'seed': index[1], 'shot': index[2], 'session': index[3],
                           'head': kind, 'all_accuracy_gain_pp': delta})
    delta = pd.DataFrame(deltas)
    delta.to_csv(output / 'paired_deltas.csv', index=False)
    cost_summary = data[data.session > 0].groupby(['shot', 'method'])[
        ['local_registration_ms', 'memory_assembly_ms', 'svm_full_refit_ms', 'knn_index_build_ms',
         'total_head_update_ms', 'predict_ms']].mean().reset_index()
    cost_summary.to_csv(output / 'cost_summary.csv', index=False)
    registration[registration.session > 0].groupby(['shot', 'view'])[
        ['new_class_encoding_ms', 'local_bank_append_ms', 'local_registration_ms']].mean().reset_index().to_csv(
            output / 'local_registration_summary.csv', index=False)
    report = '# Frozen C4 4+1+1 evaluation\n\n'
    report += ('Reused architecture-V1 M0/M1 checkpoints, splits, preprocessing and exact novel support/query episodes. '
        'Encoder training calls: zero. Five existing heads: raw SVM, M0/M1 linear SVM and Euclidean 1-NN. '
        'Each head receives the same fixed 20 base-training exemplars/class and K support per arrived novel class. '
        'SVM C is reused from archived base-only tuning, with no query retuning. '
        'The full SVM refit uses all retained seen exemplars (80, 80+K, 80+2K rows), not all historical base-training data.\n\n'
        '**Base query limitation:** base queries are sampled from archived base validation. Those rows already '
        'participated in checkpoint/C selection. Base and all-seen metrics therefore are validation-reuse exploratory '
        'estimates, not independent held-out-test performance. Novel queries remain in their original disjoint pools.\n\n'
        'Accuracy for a group conditions on its true labels. Group Macro-F1 averages its global one-vs-rest class F1 '
        'over the entire seen-class confusion matrix, including false positives from other groups. '
        'Previously added novel excludes base classes and the current class; it is empty until session 2. '
        'Current novel is undefined in session 0. All-seen Macro-F1 is standard multiclass Macro-F1.\n\n')
    for k in config['shots']:
        values = summary[(summary.shot == k) & (summary.session == 2)][
            ['method', 'base_accuracy', 'previous_novel_accuracy', 'current_novel_accuracy',
             'seen_novel_accuracy', 'all_accuracy', 'all_macro_f1']].copy()
        values.iloc[:, 1:] *= 100
        report += f'## Final session {k}-shot (%)\n\n' + table(values) + '\n\n'
    report += ('Full group accuracy/Macro-F1 for every session: `summary_metrics.csv`. '
        'Average incremental accuracy is reported both across sessions 0/1/2 and across 1/2 only. '
        'Task/seed means are equally weighted; repeated episodes are not independent training runs.\n')
    (output / 'incremental_analysis.md').write_text(report, encoding='utf-8')
    (output / 'cost_analysis.md').write_text('# Cost conditions\n\n' + table(cost_summary) +
        '\n\nCosts are milliseconds. Local registration is the warm frozen encoding of K arriving spectra plus '
        'copying that class into an append-only bank. Scoring uses archived features; timed re-encoding is a replay '
        'of the same support under the frozen checkpoint, checked against saved features. Raw spectra require no '
        'encoder. All costs exclude shared row-local preprocessing. Bank materialization and classifier fitting are '
        'separate and required for serving; local append alone is not the complete update. '
        'SVM is fully rebuilt on all retained seen exemplars, including previous novel support. '
        '1-NN uses the existing sklearn head, whose index preparation is measured separately. '
        'total_head_update_ms includes encoding, append, materialization and refit/index build. '
        'Prediction uses 75 queries in session 1 and 90 in session 2, excludes cached query feature extraction, '
        'and should not be presented as end-to-end inference. Timing is sequential on one CPU with four TensorFlow '
        'threads; it is not a production throughput benchmark.\n', encoding='utf-8')
    primary = delta[(delta.session == 2) & delta.shot.isin([5, 10])]
    paired = primary.groupby(['task', 'seed', 'head']).all_accuracy_gain_pp.mean().reset_index()
    decision_rows = []
    for head in ['svm', '1nn']:
        rows = paired[paired['head'] == head]
        decision_rows.append({'head': head, 'mean_final_5_10_gain_pp': rows.all_accuracy_gain_pp.mean(),
            'positive_pairs': int((rows.all_accuracy_gain_pp > 0).sum()), 'pairs': len(rows)})
    (output / 'conclusion.md').write_text('# Exploratory conclusion\n\n' + table(pd.DataFrame(decision_rows)) +
        '\n\nThese are descriptive final-session 5/10-shot M1-minus-M0 all-seen accuracy differences. '
        'Interpret base and novel metrics together; strong novel-only features do not ensure balanced six-class '
        'incremental competition. C4 results were inspected previously, base validation is reused, and physical '
        'specimen independence remains unverified. No new encoder/loss/augmentation and no query-driven '
        'hyperparameter selection were introduced.\n', encoding='utf-8')
    write = {'passed': True, 'paired_trials': len(files), 'audited_session_episodes': audit_count,
             'encoder_training_calls': 0, 'base_query_is_independent_test': False}
    (output / 'protocol_audit.json').write_text(json.dumps(write, indent=2), encoding='utf-8')
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    for k in config['shots']:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, metric in zip(axes.flat, ['all_accuracy', 'all_macro_f1', 'base_accuracy',
                                        'previous_novel_accuracy', 'current_novel_accuracy', 'seen_novel_accuracy']):
            values = summary[summary.shot == k].pivot(index='session', columns='method', values=metric)
            for method in values:
                ax.plot(values.index, 100 * values[method], marker='o', label=method)
            ax.set(title=metric, xlabel='Session (4 / 5 / 6 seen classes)', ylabel='%', xticks=[0, 1, 2], ylim=[0, 102])
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{k}-shot | base validation reused; exploratory')
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(figures / f'{k}shot_sessions.{ext}', dpi=230)
        plt.close(fig)
    classes = sorted(set(config['tasks'][0]['base'] + [c for s in config['tasks'][0]['novel_sessions'] for c in s]))
    methods = ['raw_svm', 'M0_svm', 'M0_1nn', 'M1_svm', 'M1_1nn']
    for task in config['tasks']:
        for k in [v for v in [5, 20] if v in config['shots']]:
            fig, axes = plt.subplots(1, 5, figsize=(23, 4.5))
            for ax, method in zip(axes, methods):
                cm = np.zeros((6, 6), dtype=np.int64)
                for seed in config['seeds']:
                    with np.load(output / task['name'] / f'seed_{seed}' / 'confusions.npz') as saved:
                        cm += saved[f'{method}_{k}shot_session2']
                utils.draw_confusion_matrix(ax, cm, classes, method)
                pd.DataFrame(cm, index=classes, columns=classes).to_csv(figures / f'{task["name"]}_{k}shot_{method}_CM.csv')
            fig.suptitle(f'{task["name"]} | {k}-shot | final six-class session | pooled query occurrences')
            fig.tight_layout()
            for ext in ['png', 'pdf']:
                fig.savefig(figures / f'{task["name"]}_{k}shot_final_CM.{ext}', dpi=230)
            plt.close(fig)
    print('Incremental analysis complete: ' + str(output), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    analyze(p.parse_args().output)
