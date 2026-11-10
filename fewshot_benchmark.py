"""Paired C4/C8 fixed-encoder benchmark based on the MCCV2 class-wise metric.

Run --help for configuration. TensorFlow is imported only for training.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from deduplicate_plastics500 import spectrum_key
from PLS import airPLS
from utils import utils


ROOT = Path(__file__).resolve().parent
CLASSES = ['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS']
DISTANCES = ['euclidean', 'shared_mahalanobis', 'diagonal_mahalanobis', 'class_adaptive_mahalanobis']
METRICS = ['kappa', 'macro_f1', 'accuracy', 'macro_precision', 'macro_recall']


def read_raw(path):
    with path.open(encoding='latin-1', newline='') as handle:
        reader = csv.reader(handle)
        next(reader)
        wave = np.asarray(next(reader)[3:], dtype=float)
        rows = list(reader)
    labels = np.asarray([int(row[2]) for row in rows])
    if set(labels) != set(range(6)):
        raise ValueError(f'{path}: expected six labels 0..5')
    for row in rows:
        if row[1] != CLASSES[int(row[2])]:
            raise ValueError(f'{path}: unexpected label/name mapping')
    keys = [spectrum_key(row[3:]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{path}: duplicate spectra; run deduplicate_plastics500.py first')
    x = np.asarray([row[3:] for row in rows], dtype=np.float64)
    if x.shape[1] != len(wave) or not np.isfinite(x).all() or not np.isfinite(wave).all():
        raise ValueError(f'{path}: invalid spectral array')
    order = np.argsort(wave)
    if np.any(np.diff(wave[order]) <= 0):
        raise ValueError(f'{path}: duplicate or invalid wavenumbers')
    return wave[order], x[:, order], labels, np.asarray([row[0] for row in rows])


def prepare_data(path, grid, out, flip=True, baseline=True):
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = out / f'{path.stem}_preprocessed.npz'
    signature = json.dumps({'source_sha256': source_hash, 'grid': grid.tolist(),
                            'flip': flip, 'baseline': baseline, 'version': 1}, sort_keys=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if data['signature'].item() == signature:
                print(f'Using preprocessing cache: {cache.name}', flush=True)
                return data['x'], data['y'], data['ids'], source_hash
    wave, raw, y, ids = read_raw(path)
    if grid.min() < wave.min() or grid.max() > wave.max():
        raise ValueError('Requested grid extends beyond measured spectrum')
    x = np.empty((len(raw), len(grid)), dtype=np.float32)
    for i, spectrum in enumerate(raw):
        # Retain MCCV2 preprocessing; no class/global fitted preprocessing.
        processed = spectrum.max() - spectrum if flip else spectrum.copy()
        if baseline and np.ptp(processed) > 0:
            processed = processed - airPLS(processed)
        scale = np.max(np.abs(processed))
        processed = processed / scale if scale > 0 else processed
        x[i] = interp1d(wave, processed, kind='cubic', bounds_error=True)(grid)
        if (i + 1) % 250 == 0:
            print(f'{path.stem}: preprocessed {i + 1}/{len(raw)}', flush=True)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite values after preprocessing')
    np.savez_compressed(cache, x=x, y=y, ids=ids, signature=signature)
    return x, y, ids, source_hash


def split_indices(y, seed):
    """60/20/20; encoder train, tuning/selection val, untouched test pool."""
    indices = np.arange(len(y))
    train, rest = train_test_split(indices, test_size=0.4, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return train, val, test


def sample_episode(y, pool, k, q, rng):
    support, query = [], []
    for c in range(6):
        candidates = pool[y[pool] == c]
        if len(candidates) < k + q:
            raise ValueError(f'Class {CLASSES[c]} has {len(candidates)} samples; need {k + q}')
        chosen = rng.choice(candidates, k + q, replace=False)
        support.extend(chosen[:k])
        query.extend(chosen[k:])
    return np.asarray(support), np.asarray(query)


def quadratic_distances(diff, centered, reg):
    """Exact d^T (A^T A/n + reg I)^-1 d, via a small Woodbury solve.

    Uses the full covariance including off-diagonal entries. No pseudo-inverse
    or repeated regularization. All arithmetic is float64.
    """
    b = centered / np.sqrt(len(centered))
    projection = diff @ b.T
    small = np.eye(len(b)) + b @ b.T / reg
    correction = np.linalg.solve(small, projection.T).T
    values = np.sum(diff * diff, axis=1) / reg
    values -= np.sum(projection * correction, axis=1) / reg**2
    return np.maximum(values, 0)


def metric_predictions(support_z, support_y, query_z, method, reg=1e-3):
    support_z, query_z = np.asarray(support_z, dtype=float), np.asarray(query_z, dtype=float)
    if reg <= 0:
        raise ValueError('Regularization must be positive')
    means = np.stack([support_z[support_y == c].mean(axis=0) for c in range(6)])
    residuals = [support_z[support_y == c] - means[c] for c in range(6)]
    pooled = np.concatenate(residuals)
    scores = []
    for c in range(6):
        diff = query_z - means[c]
        if method == 'euclidean':
            d = np.sum(diff**2, axis=1)
        elif method == 'diagonal_mahalanobis':
            variance = np.mean(residuals[c]**2, axis=0)
            d = np.sum(diff**2 / (variance + reg), axis=1)
        elif method in ('shared_mahalanobis', 'class_adaptive_mahalanobis'):
            residual = pooled if method == 'shared_mahalanobis' else residuals[c]
            d = quadratic_distances(diff, residual, reg)
        else:
            raise ValueError(method)
        scores.append(d)
    return np.argmin(np.stack(scores, axis=1), axis=1)


def train_encoder(x, y, train, val, seed, args, folder, n_classes=6, backbone='compact'):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from tensorflow.keras import layers

    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    inputs = layers.Input(shape=(x.shape[1],))
    z = layers.Reshape((x.shape[1], 1))(inputs)
    if backbone == 'multiscale':
        branches = [layers.Conv1D(16, k, padding='same', activation='relu')(z)
                    for k in [9, 25, 49]]
        z = layers.Concatenate()(branches)
        for filters in [32, 64]:
            z = layers.MaxPooling1D(4)(z)
            z = layers.Conv1D(filters, 7, padding='same', activation='relu')(z)
        # Retain spectral position rather than globally averaging all peaks.
        z = layers.Flatten()(z)
    elif backbone == 'compact':
        for filters, kernel in [(32, 15), (64, 9), (64, 5)]:
            z = layers.Conv1D(filters, kernel, padding='same', activation='relu')(z)
            z = layers.MaxPooling1D(2)(z)
        z = layers.GlobalAveragePooling1D()(z)
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    embedding = layers.Dense(args.embedding_dim, activation='relu', name='embedding')(z)
    logits = layers.Dense(n_classes, activation='softmax')(embedding)
    model = tf.keras.Model(inputs, logits)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = folder / 'best.weights.h5'
    history = model.fit(x[train], y[train], validation_data=(x[val], y[val]),
                        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss',
                                save_best_only=True, save_weights_only=True),
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)
                        ])
    # Explicitly restore even if training ends at the epoch limit rather than
    # triggering EarlyStopping (behavior differs between Keras versions).
    model.load_weights(str(checkpoint))
    pd.DataFrame(history.history).to_csv(folder / 'training_history.csv', index=False)
    encoder = tf.keras.Model(inputs, embedding)
    encoder.save(str(folder / 'encoder.h5'), include_optimizer=False)
    # Frozen embeddings used by every distance; no evaluation-time fitting.
    embeddings = encoder.predict(x, batch_size=args.batch_size, verbose=0).astype(float)
    if not np.isfinite(embeddings).all():
        raise ValueError('Non-finite encoder embeddings')
    return embeddings, tf.__version__


def tune_regularization(z, y, val, k, q, args, seed):
    rng = np.random.default_rng(seed + 10000)
    episodes = [sample_episode(y, val, k, q, rng) for _ in range(args.val_episodes)]
    records, selected = [], {'euclidean': None}
    for method in DISTANCES[1:]:
        candidates = []
        for reg in args.reg_grid:
            scores = []
            for support, query in episodes:
                pred = metric_predictions(z[support], y[support], z[query], method, reg)
                score, _, _ = utils.classification_metrics(y[query], pred, list(range(6)), CLASSES)
                scores.append(score['macro_f1'])
            value = float(np.mean(scores))
            candidates.append((value, reg))
            records.append({'method': method, 'regularization': reg, 'val_macro_f1': value})
        selected[method] = max(candidates, key=lambda pair: pair[0])[1]
    return selected, records


def evaluate(x, z, y, test, args, seed, k, regs, folder):
    methods = DISTANCES + ([] if args.no_raw_baselines else ['svm_raw', 'knn_raw'])
    rng = np.random.default_rng(seed + 20000 + k * 100)
    metrics_rows, class_rows, samples = [], [], []
    cms = {method: np.zeros((6, 6), dtype=np.int64) for method in methods}
    true_all, pred_all = [], {method: [] for method in methods}
    with (folder / f'predictions_{k}shot.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['episode', 'query_index', 'true_label', *methods])
        for episode in range(args.episodes):
            support, query = sample_episode(y, test, k, args.queries, rng)
            predictions = {method: metric_predictions(z[support], y[support], z[query], method,
                           regs.get(method) or 1e-3) for method in DISTANCES}
            if not args.no_raw_baselines:
                predictions['svm_raw'] = SVC(C=1.0, kernel='linear').fit(x[support], y[support]).predict(x[query])
                predictions['knn_raw'] = KNeighborsClassifier(n_neighbors=3).fit(x[support], y[support]).predict(x[query])
            samples.append({'episode': episode, 'support_indices': support.tolist(), 'query_indices': query.tolist()})
            true_all.extend(y[query].tolist())
            for method, pred in predictions.items():
                metrics, report, cm = utils.classification_metrics(y[query], pred, list(range(6)), CLASSES)
                metrics_rows.append({'episode': episode, 'method': method, **metrics})
                cms[method] += cm
                pred_all[method].extend(pred.tolist())
                for name in CLASSES:
                    class_rows.append({'episode': episode, 'method': method, 'class': name, **report[name]})
            for j, index in enumerate(query):
                writer.writerow([episode, index, y[index], *[predictions[m][j] for m in methods]])
            if (episode + 1) % 50 == 0:
                print(f'{folder.name}, {k}-shot: {episode + 1}/{args.episodes} episodes', flush=True)
    pd.DataFrame(metrics_rows).to_csv(folder / f'episode_metrics_{k}shot.csv', index=False)
    pd.DataFrame(class_rows).to_csv(folder / f'episode_class_metrics_{k}shot.csv', index=False)
    (folder / f'episodes_{k}shot.json').write_text(json.dumps(samples), encoding='utf-8')
    aggregate_reports = {}
    for method in methods:
        _, report, _ = utils.classification_metrics(true_all, pred_all[method], list(range(6)), CLASSES)
        aggregate_reports[method] = report
    (folder / f'class_reports_{k}shot.json').write_text(json.dumps(aggregate_reports, indent=2), encoding='utf-8')
    np.savez_compressed(folder / f'confusion_matrices_{k}shot.npz', **cms)
    return pd.DataFrame(metrics_rows).groupby('method')[METRICS].mean().reset_index()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory (must not exist)')
    parser.add_argument('--datasets', nargs='+', choices=['c4', 'c8'], default=['c4', 'c8'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--val-episodes', type=int, default=30)
    parser.add_argument('--queries', type=int, default=15, help='Query spectra per class per episode')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--reg-grid', nargs='+', type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument('--points', type=int, default=2000)
    parser.add_argument('--no-flip', action='store_true', help='Disable original row_max - intensity transform')
    parser.add_argument('--no-baseline', action='store_true', help='Disable original airPLS')
    parser.add_argument('--no-raw-baselines', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    positive = [args.episodes, args.val_episodes, args.queries, args.epochs, args.batch_size,
                args.embedding_dim, args.points, args.learning_rate, *args.shots, *args.reg_grid]
    if min(positive) <= 0 or args.points < 4 or args.patience < 0 or min(args.seeds) < 0:
        parser.error('Counts, shot numbers and regularization must be positive; seeds/patience nonnegative')
    for values in [args.datasets, args.seeds, args.shots, args.reg_grid]:
        if len(set(values)) != len(values):
            parser.error('Duplicate dataset/seed/shot/regularization values are not allowed')
    return args


def main(argv=None):
    args = parse_args(argv)
    # Fail early before expensive preprocessing if the training dependency is absent.
    import importlib.util
    if importlib.util.find_spec('tensorflow') is None:
        raise SystemExit('TensorFlow is required. Use the project TensorFlow environment; see FEWSHOT_RUN.md.')
    paths = {d: ROOT / 'dataset' / 'deduplicated' / f'FTIR_PLastics500_{d}_deduplicated.csv'
             for d in ['c4', 'c8']}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f'{path}: run deduplicate_plastics500.py first')
    # One common axis even when only one dataset is selected for this run.
    waves = []
    for path in paths.values():
        with path.open(encoding='latin-1', newline='') as handle:
            reader = csv.reader(handle)
            next(reader)
            waves.append(np.asarray(next(reader)[3:], dtype=float))
    grid = np.linspace(min(w.max() for w in waves), max(w.min() for w in waves), args.points)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'protocol': 'within-resolution known-class few-shot adaptation; separate supervised encoder per dataset/seed',
                   'split': 'stratified 60/20/20 train/validation/test; all 6 classes in each',
                   'covariance': 'divide by K; pooled within-class residuals for shared covariance; ridge applied once',
                   'preprocessing': 'optional row-max flip, optional airPLS, per-spectrum max-abs normalization, cubic interpolation',
                   'uncertainty': '95% t intervals over seed means; overlapping episodes are not independent specimens',
                   'class_names': CLASSES, 'python': sys.version, 'platform': platform.platform()})
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    # Snapshot exact executable sources for provenance.
    source_dir = out / 'source'
    source_dir.mkdir()
    for name in ['fewshot_benchmark.py', 'plot_fewshot_benchmark.py', 'utils.py', 'PLS.py', 'deduplicate_plastics500.py']:
        (source_dir / name).write_bytes((ROOT / name).read_bytes())
    import importlib.metadata
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']:
        versions[name] = importlib.metadata.version(name)
    all_summaries = []
    hashes = {}
    for dataset in args.datasets:
        x, y, ids, source_hash = prepare_data(paths[dataset], grid, out,
                                             flip=not args.no_flip, baseline=not args.no_baseline)
        hashes[dataset] = source_hash
        for seed in args.seeds:
            folder = out / dataset / f'seed_{seed}'
            folder.mkdir(parents=True)
            train, val, test = split_indices(y, seed)
            # Preflight every requested K before training.
            for pool in [val, test]:
                sample_episode(y, pool, max(args.shots), args.queries, np.random.default_rng(0))
            roles = np.empty(len(y), dtype=object)
            for role, indices in [('train', train), ('validation', val), ('test', test)]:
                roles[indices] = role
            pd.DataFrame({'index': np.arange(len(y)), 'source_id': ids,
                          'class_id': y, 'class': [CLASSES[c] for c in y], 'split': roles}).to_csv(
                              folder / 'split_manifest.csv', index=False)
            print(f'Training {dataset}, seed {seed}: {len(train)}/{len(val)}/{len(test)}', flush=True)
            z, tf_version = train_encoder(x, y, train, val, seed, args, folder)
            versions['tensorflow'] = tf_version
            np.savez_compressed(folder / 'embeddings.npz', embeddings=z, labels=y)
            for k in args.shots:
                regs, tuning = tune_regularization(z, y, val, k, args.queries, args, seed + k * 100)
                pd.DataFrame(tuning).to_csv(folder / f'validation_tuning_{k}shot.csv', index=False)
                (folder / f'selected_regularization_{k}shot.json').write_text(json.dumps(regs, indent=2), encoding='utf-8')
                summary = evaluate(x, z, y, test, args, seed, k, regs, folder)
                summary['dataset'], summary['seed'], summary['shot'] = dataset, seed, k
                all_summaries.append(summary)
                pd.concat(all_summaries, ignore_index=True).to_csv(out / 'seed_metrics.csv', index=False)
    (out / 'provenance.json').write_text(json.dumps({'source_sha256': hashes, 'versions': versions}, indent=2), encoding='utf-8')
    if not args.no_plots:
        from plot_fewshot_benchmark import plot_results
        plot_results(out)
    print(f'Finished. Results: {out}', flush=True)


if __name__ == '__main__':
    main()
