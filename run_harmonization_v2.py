"""V2 first-stage development-only method/grid screening. No Encoder training.

Selected base classes only; no final novel queries and no final-test selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ftir_harmonization_v2 import METHODS, SCHEMA_VERSION, convert_signal, harmonize_spectrum, reconstruction_metrics
from ftir_research_data import ROOT, REGISTRY, load_dataset, canonical_label
from PLS import airPLS
from utils import utils


def make_episodes(y, pool, k, q, count, rng):
    result = []
    for episode in range(count):
        support, query = [], []
        for c in np.unique(y):
            candidates = pool[y[pool] == c]
            if len(candidates) < k + q:
                raise ValueError(f'{c}: need {k + q} in each development pool; reduce shots/queries')
            chosen = rng.choice(candidates, k + q, replace=False)
            support.extend(chosen[:k]); query.extend(chosen[k:])
        result.append((np.asarray(support), np.asarray(query)))
    return result


def svm_metrics(x, y, episodes, c):
    rows = []
    classes = np.unique(y)
    for support, query in episodes:
        predictions = SVC(C=c, kernel='linear').fit(x[support], y[support]).predict(x[query])
        metrics, _, _ = utils.classification_metrics(y[query], predictions, classes, classes.tolist())
        rows.append((metrics, predictions))
    return rows


def run(args):
    data = load_dataset(args.dataset)
    base = sorted({canonical_label(c) for c in args.base_classes})
    if len(base) < 2 or any(c not in data.labels for c in base):
        raise ValueError('Supply at least two existing base classes; inspect research_inventory first')
    if len(base) == len(set(data.labels)):
        raise ValueError('Reserve at least one class from selection; do not screen on all eventual novel labels')
    eligible = np.flatnonzero(np.isin(data.labels, base))
    if args.groups:
        table = pd.read_csv(args.groups, dtype=str)
        if not {'dataset', 'row_id', 'group'} <= set(table.columns) or table.duplicated(['dataset', 'row_id']).any():
            raise ValueError('Groups CSV requires unique dataset,row_id,group')
        lookup = {(r.dataset, r.row_id): r.group for r in table.itertuples()}
        seen, representatives = {}, []
        for index in eligible:
            group = lookup.get((args.dataset, str(data.ids[index])))
            if not isinstance(group, str) or not group:
                raise ValueError('Verified groups must cover every selected base spectrum')
            if group in seen and seen[group] != data.labels[index]:
                raise ValueError('Conflicting labels within specimen group')
            if group not in seen:
                seen[group] = data.labels[index]; representatives.append(index)
        eligible = np.asarray(representatives)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'schema_version': SCHEMA_VERSION, 'base_classes': base,
        'excluded_classes': sorted(set(data.labels) - set(base)), 'source_sha256': data.report['source_sha256'],
        'scope': 'base-class development screening only; no locked final test',
        'signal_provenance': 'explicit operator declaration; legacy does not assert physical absorbance',
        'group_status': 'one representative per verified group' if args.groups else 'exact-spectrum dedup only; specimen identity unverified',
        'order': 'explicit signal conversion, native-axis airPLS, per-spectrum maxabs, harmonizer',
        'bin_mean': 'piecewise-linear interval average, not ideal band-limited reconstruction; endpoint bins half width',
        'fidelity': 'raw-native-point reconstruction; whole-spectrum dominant peak and area diagnostics only',
        'kernel': 'linear; new RBF comparison is not included'})
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    snapshot = out / 'source'; snapshot.mkdir()
    for name in ['run_harmonization_v2.py', 'ftir_harmonization_v2.py', 'ftir_research_data.py', 'utils.py', 'PLS.py']:
        (snapshot / name).write_bytes((ROOT / name).read_bytes())
    # Physical coordinates are shared by this registry dataset. Grid spacings
    # derive from native sampling, never from class scores/instrument resolution.
    spacing = float(np.median(np.diff(data.wave)))
    grids = {'native': data.wave}
    for factor in args.grid_factors:
        length = int(np.floor(np.ptp(data.wave) / (spacing * factor))) + 1
        if length < 4:
            raise ValueError('Requested grid is too coarse')
        grids[f'factor_{factor:g}'] = np.linspace(data.wave[0], data.wave[-1], length)
    qc, processed, kept = [], [], []
    started = time.perf_counter()
    for index in eligible:
        try:
            row, flags = convert_signal(data.x[index], args.signal,
                legacy_flip=data.report['default_flip'], upper_policy=args.transmittance_upper)
            if not args.no_baseline and np.ptp(row) > 0:
                row = row - airPLS(row)
            scale = np.max(np.abs(row)); row = row / scale if scale else row
            if not np.isfinite(row).all():
                raise ValueError('Non-finite preprocessing')
            processed.append(row); kept.append(index)
            qc.append({'source_index': index, 'source_id': data.ids[index], 'status': 'retained', 'detail': ';'.join(flags)})
        except ValueError as error:
            qc.append({'source_index': index, 'source_id': data.ids[index], 'status': 'rejected', 'detail': str(error)})
    pd.DataFrame(qc).to_csv(out / 'signal_qc.csv', index=False)
    if not kept:
        raise ValueError('No spectra passed explicit signal QC; inspect signal_qc.csv, do not silently clip values')
    x = np.asarray(processed); kept = np.asarray(kept); y = data.labels[kept]
    if set(y) != set(base):
        raise ValueError('QC removed a base class; resolve signal policy rather than changing class budget')
    native_preprocessing_seconds = time.perf_counter() - started
    pd.DataFrame({'index': np.arange(len(y)), 'source_index': kept, 'source_id': data.ids[kept], 'class': y}).to_csv(out / 'sample_index.csv', index=False)
    split_records, episode_records, schedules = [], [], {}
    for seed in args.seeds:
        tune, screen = train_test_split(np.arange(len(y)), test_size=.3, random_state=seed, stratify=y)
        for role, indices in [('tune', tune), ('screen', screen)]:
            for index in indices:
                split_records.append({'seed': seed, 'index': index, 'source_id': data.ids[kept[index]], 'role': role})
        for k in args.shots:
            inner = make_episodes(y, tune, k, args.queries, args.val_episodes, np.random.default_rng(seed + 1000 + k))
            outer = make_episodes(y, screen, k, args.queries, args.episodes, np.random.default_rng(seed + 2000 + k))
            schedules[(seed, k)] = (inner, outer)
            for role, episodes in [('tune', inner), ('screen', outer)]:
                for i, (s, q) in enumerate(episodes):
                    episode_records.append({'seed': seed, 'shot': k, 'role': role, 'episode': i,
                                            'support': s.tolist(), 'query': q.tolist()})
    pd.DataFrame(split_records).to_csv(out / 'development_splits.csv', index=False)
    (out / 'episodes.json').write_text(json.dumps(episode_records), encoding='utf-8')
    records, timing, fidelity, matrices, selected, cm_index, tuning_records = [], [], [], {}, [], [], []
    # Fixed reference rows from allowed development classes, independent of method.
    fidelity_indices = np.random.default_rng(57).choice(len(y), min(args.fidelity_spectra, len(y)), replace=False)
    for method in ['native', *args.methods]:
        for grid_id, grid in grids.items():
            if (method == 'native') != (grid_id == 'native'):
                continue
            name = f'{method}_{grid_id}'
            begin = time.perf_counter()
            aligned = x if method == 'native' else np.asarray([harmonize_spectrum(data.wave, row, grid, method) for row in x])
            timing.append({'candidate': name, 'spectra': len(y), 'points': len(grid),
                           'sampling_step_cm1': float(np.median(np.diff(grid))),
                           'harmonization_seconds': time.perf_counter() - begin, 'matrix_bytes': aligned.nbytes,
                           'native_preprocessing_seconds': native_preprocessing_seconds})
            np.save(out / f'{name}_grid.npy', grid)
            for i in fidelity_indices:
                row = x[i]; output = aligned[i]
                reconstruction = ({'rmse': 0., 'nrmse': 0., 'status': 'native_reference_no_reconstruction'} if method == 'native' else
                    reconstruction_metrics(data.wave, data.x[kept[i]], method, seed=57))
                original_area = np.trapz(row, data.wave)
                if method == 'bin_mean':
                    edges = np.r_[grid[0], (grid[1:] + grid[:-1]) / 2, grid[-1]]
                    output_area = float(np.sum(output * np.diff(edges)))
                else:
                    output_area = float(np.trapz(output, grid))
                fidelity.append({'candidate': name, 'source_id': data.ids[kept[i]], **reconstruction,
                    'dominant_peak_shift_cm1': abs(float(grid[np.argmax(output)] - data.wave[np.argmax(row)])),
                    'dominant_peak_height_difference': float(np.max(output) - np.max(row)),
                    'area_relative_error': abs(output_area - original_area) / max(abs(original_area), 1e-12),
                    'range_overshoot': max(0., float(output.max() - row.max()), float(row.min() - output.min()))})
            for (seed, k), (inner, outer) in schedules.items():
                candidates = []
                for c in args.c_grid:
                    values = svm_metrics(aligned, y, inner, c)
                    score = float(np.mean([r[0]['macro_f1'] for r in values]))
                    candidates.append((score, c))
                    tuning_records.append({'candidate': name, 'seed': seed, 'shot': k,
                                           'c': c, 'tune_macro_f1': score})
                best_f1, best_c = max(candidates, key=lambda item: item[0])
                selected.append({'candidate': name, 'seed': seed, 'shot': k, 'selected_c': best_c, 'tune_macro_f1': best_f1})
                cm = np.zeros((len(base), len(base)), dtype=np.int64)
                with (out / f'{name}_seed{seed}_{k}shot_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle); writer.writerow(['episode', 'index', 'true', 'predicted'])
                    for episode, ((metrics, predictions), (_, query)) in enumerate(zip(svm_metrics(aligned, y, outer, best_c), outer)):
                        _, _, matrix = utils.classification_metrics(y[query], predictions, np.asarray(base), base)
                        cm += matrix
                        records.append({'candidate': name, 'method': method, 'grid': grid_id, 'seed': seed,
                                        'shot': k, 'episode': episode, 'selected_c': best_c, **metrics})
                        writer.writerows(zip(np.repeat(episode, len(query)), query, y[query], predictions))
                key = f'{name}_seed{seed}_{k}shot'
                matrices[key] = cm
                cm_index.append({'key': key, 'candidate': name, 'seed': seed, 'shot': k, 'classes': base})
            pd.DataFrame(records).to_csv(out / 'development_metrics.csv', index=False)
            print(f'Finished {name}: {len(grid)} points', flush=True)
    pd.DataFrame(timing).to_csv(out / 'timing.csv', index=False)
    pd.DataFrame(fidelity).to_csv(out / 'spectral_fidelity.csv', index=False)
    pd.DataFrame(selected).to_csv(out / 'svm_selection.csv', index=False)
    pd.DataFrame(tuning_records).to_csv(out / 'svm_tuning.csv', index=False)
    np.savez_compressed(out / 'confusions.npz', **matrices)
    (out / 'confusion_index.json').write_text(json.dumps(cm_index, indent=2), encoding='utf-8')
    metrics = pd.DataFrame(records)
    trial = metrics.groupby(['candidate', 'seed', 'shot'])[['accuracy', 'macro_f1']].mean().reset_index()
    trial.to_csv(out / 'seed_metrics.csv', index=False)
    trial.groupby(['candidate', 'shot'])[['accuracy', 'macro_f1']].agg(['mean', 'std']).to_csv(out / 'summary.csv')
    # A shortlist is not a locked physical pipeline: QC/peak constraints and
    # signal semantics still require review. Never label it final-test winner.
    ranking = trial[trial.shot.isin([5, 10])].groupby('candidate').macro_f1.mean().sort_values(ascending=False)
    if ranking.empty:
        ranking = trial.groupby('candidate').macro_f1.mean().sort_values(ascending=False)
    (out / 'DEVELOPMENT_SHORTLIST.json').write_text(json.dumps({'scope': 'development only; review fidelity and QC before locking',
        'ranking': ranking.to_dict(), 'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, ['accuracy', 'macro_f1']):
        for candidate, rows in trial.groupby('candidate'):
            average = rows.groupby('shot')[metric].mean()
            ax.plot(average.index, average * 100, marker='o', label=candidate)
        ax.set(xlabel='K-shot', ylabel=metric + ' (%)', title='Base-class development screening')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'method_grid_development.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    primary = 5 if 5 in args.shots else args.shots[0]
    for candidate in metrics.candidate.unique():
        cm = sum(matrices[r['key']] for r in cm_index if r['candidate'] == candidate and r['shot'] == primary)
        fig, ax = plt.subplots(figsize=(6, 5))
        utils.draw_confusion_matrix(ax, cm, base, f'{candidate} | {primary}-shot | development')
        fig.tight_layout(); fig.savefig(out / f'{candidate}_{primary}shot_CM.png', dpi=250); plt.close(fig)
    (out / 'COMPLETE.json').write_text('{"complete": true}', encoding='utf-8')
    print(f'Finished v2 development screening: {out}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(REGISTRY), default='c4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-classes', nargs='+', required=True)
    parser.add_argument('--signal', choices=['legacy', 'absorbance', 'transmittance_percent', 'transmittance_fraction'], required=True)
    parser.add_argument('--transmittance-upper', choices=['reject', 'flag'], default='reject')
    parser.add_argument('--groups', type=Path)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--grid-factors', nargs='+', type=float, default=[2, 4])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--queries', type=int, default=15)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--val-episodes', type=int, default=5)
    parser.add_argument('--c-grid', nargs='+', type=float, default=[.1, 1, 10])
    parser.add_argument('--fidelity-spectra', type=int, default=12)
    parser.add_argument('--no-baseline', action='store_true')
    args = parser.parse_args(argv)
    if min([*args.shots, *args.c_grid, args.queries, args.episodes, args.val_episodes, args.fidelity_spectra]) <= 0 or min(args.seeds) < 0 or min(args.grid_factors) < 1:
        parser.error('Counts/C positive, seeds nonnegative, grid factors at least 1')
    for values in [args.methods, args.grid_factors, args.seeds, args.shots]:
        if len(values) != len(set(values)):
            parser.error('Duplicate options are not allowed')
    run(args)


if __name__ == '__main__':
    main()
