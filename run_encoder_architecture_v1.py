"""Paired M0/M1 study; smoke validation followed by fixed three-task/seed run.

No new losses, augmentation, architecture search or novel-query model selection.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf

from encoder_architecture_models import build_encoder, capacity_check
from ftir_research_data import ROOT, REGISTRY, load_dataset, preprocess
from utils import utils

SOURCES = ['run_encoder_architecture_v1.py', 'encoder_architecture_models.py',
           'analyze_encoder_architecture_v1.py', 'ftir_research_data.py', 'utils.py', 'PLS.py']
HEADS = ['raw_svm', 'M0_svm', 'M0_1nn', 'M1_svm', 'M1_1nn']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def weight_hash(model):
    h = hashlib.sha256()
    for weight in model.get_weights():
        h.update(str(weight.shape).encode()); h.update(str(weight.dtype).encode()); h.update(weight.tobytes())
    return h.hexdigest()


def build_manifest(labels, task, seed, args):
    """Novel support/query pools are disjoint; all choices are shared by M0/M1."""
    rng = np.random.default_rng(1729 + seed)
    roles = np.full(len(labels), 'unused', dtype=object)
    train, val, novel_support, novel_query = [], [], {}, {}
    for c in task['base']:
        candidates = rng.permutation(np.flatnonzero(labels == c))
        nval = max(2, int(.2 * len(candidates)))
        val.extend(candidates[:nval]); train.extend(candidates[nval:])
    for session in task['novel_sessions']:
        for c in session:
            candidates = rng.permutation(np.flatnonzero(labels == c))
            half = len(candidates) // 2
            novel_support[c], novel_query[c] = candidates[:half], candidates[half:]
            if half < max(args.shots) or len(candidates) - half < args.queries:
                raise ValueError(f'Insufficient independent support/query pool for {c}')
    train, val = np.asarray(train), np.asarray(val)
    roles[train], roles[val] = 'base_train', 'base_validation'
    for pool in novel_support.values(): roles[pool] = 'novel_support_pool'
    for pool in novel_query.values(): roles[pool] = 'novel_query_pool'
    episodes = []
    for k in args.shots:
        inner_rng = np.random.default_rng(seed + 1000 + k)
        minimum = min(np.sum(labels[val] == c) for c in task['base'])
        kval = min(k, max(1, minimum // 2)); qval = min(args.queries, minimum - kval)
        for i in range(args.val_episodes):
            support, query = [], []
            for c in sorted(task['base']):
                selected = inner_rng.choice(val[labels[val] == c], kval + qval, replace=False)
                support.extend(selected[:kval]); query.extend(selected[kval:])
            episodes.append({'role': 'base_tuning', 'shot': k, 'episode': i,
                'actual_support_per_class': kval, 'support': support, 'query': query})
        outer_rng = np.random.default_rng(seed + 2000 + k)
        for i in range(args.episodes):
            support, query = [], []
            for c in sorted(novel_support):
                support.extend(outer_rng.choice(novel_support[c], k, replace=False))
                query.extend(outer_rng.choice(novel_query[c], args.queries, replace=False))
            episodes.append({'role': 'unseen_evaluation', 'shot': k, 'episode': i,
                             'support': support, 'query': query})
    # Convert numpy integers once for portable JSON and deterministic identity.
    for episode in episodes:
        episode['support'] = [int(i) for i in episode['support']]
        episode['query'] = [int(i) for i in episode['query']]
    return train, val, roles, episodes


def train_model(name, x, labels, task, train, val, seed, args, folder, train_orders):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    encoder = build_encoder(name)
    output = tf.keras.layers.Dense(len(task['base']), activation='softmax', name='base_classifier')(encoder.output)
    model = tf.keras.Model(encoder.input, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mapping = {c: i for i, c in enumerate(task['base'])}
    targets = np.asarray([mapping.get(c, -1) for c in labels])
    if np.any(targets[train] < 0) or np.any(targets[val] < 0):
        raise ValueError('Novel labels reached training/selection')
    checkpoint = folder / 'best.weights.h5'
    history, best, best_epoch = [], np.inf, 0
    start = time.perf_counter()
    for epoch, order in enumerate(train_orders):
        # Epoch orders are explicitly precomputed once and shared, not merely
        # generated by two framework shuffle streams with a nominal same seed.
        h = model.fit(x[order], targets[order], validation_data=(x[val], targets[val]),
            epochs=1, batch_size=args.batch_size, shuffle=False, verbose=0).history
        row = {key: float(value[0]) for key, value in h.items()}
        row.update({'epoch': epoch + 1, 'elapsed_seconds': time.perf_counter() - start})
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError('Non-finite training history')
        history.append(row)
        if row['val_loss'] < best:
            best, best_epoch = row['val_loss'], epoch + 1
            model.save_weights(str(checkpoint))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f'{folder.parent.name}/{name} epoch {epoch + 1}/{args.epochs}: '
                  f'loss={row["loss"]:.4f} val={row["val_loss"]:.4f} '
                  f'elapsed={row["elapsed_seconds"]:.1f}s', flush=True)
    training_seconds = time.perf_counter() - start
    steps = int(model.optimizer.iterations.numpy())
    expected = args.epochs * int(np.ceil(len(train) / args.batch_size))
    if steps != expected:
        raise ValueError('Training update budget mismatch')
    model.load_weights(str(checkpoint))
    before = weight_hash(encoder)
    begin = time.perf_counter()
    embedding = encoder.predict(x, batch_size=args.batch_size, verbose=0).astype(np.float64)
    extraction_seconds = time.perf_counter() - begin
    if not np.isfinite(embedding).all(): raise ValueError('Non-finite frozen features')
    # Batched inference benchmark on the same fixed inputs, after warm-up.
    benchmark = tf.convert_to_tensor(x[np.arange(min(args.batch_size, len(x)))])
    infer = tf.function(lambda inputs: encoder(inputs, training=False))
    for _ in range(3): infer(benchmark).numpy()
    times = []
    for _ in range(10):
        begin = time.perf_counter(); infer(benchmark).numpy()
        times.append((time.perf_counter() - begin) * 1000 / len(benchmark))
    after = weight_hash(encoder)
    if before != after: raise ValueError('Encoder state changed during frozen extraction/benchmark')
    encoder.save(str(folder / 'encoder.keras'))
    pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
    np.savez_compressed(folder / 'embeddings.npz', embeddings=embedding)
    stats = {'architecture': name, 'parameters': encoder.count_params(), 'classifier_parameters': model.count_params(),
        'training_seconds': training_seconds, 'optimizer_updates': steps, 'best_epoch': best_epoch,
        'best_base_val_loss': best, 'all_spectra_feature_extraction_seconds': extraction_seconds,
        'inference_median_ms_per_spectrum': float(np.median(times)),
        'inference_p95_ms_per_spectrum': float(np.percentile(times, 95)),
        'benchmark_batch_size': len(benchmark), 'weight_hash_before': before, 'weight_hash_after': after,
        'model_file_bytes': (folder / 'encoder.keras').stat().st_size,
        'training_order_hash': hashlib.sha256(train_orders.tobytes()).hexdigest()}
    write_json(folder / 'training_costs.json', stats)
    write_json(folder / 'MODEL_COMPLETE.json', {'complete': True})
    return embedding, stats


def tune_svm(features, labels, episodes, c_grid):
    candidates = []
    classes = np.unique(labels[episodes[0]['query']])
    for c in c_grid:
        scores = []
        for ep in episodes:
            s, q = ep['support'], ep['query']
            predicted = SVC(C=c, kernel='linear').fit(features[s], labels[s]).predict(features[q])
            values, _, _ = utils.classification_metrics(labels[q], predicted, classes, classes.tolist())
            scores.append(values['macro_f1'])
        candidates.append({'c': c, 'base_val_macro_f1': float(np.mean(scores))})
    best = max(candidates, key=lambda row: row['base_val_macro_f1'])
    return best['c'], candidates


def geometry(features, labels, train, val, novel_query):
    """Descriptive geometry, no feature transform or model selection from novel labels."""
    records = []
    for role, indices in [('base_train', train), ('base_validation', val), ('novel_query_analysis_only', novel_query)]:
        z = features[indices]; y = labels[indices]
        centered = z - z.mean(axis=0)
        eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered), 0)
        p = eigenvalues / eigenvalues.sum() if eigenvalues.sum() else eigenvalues
        entropy_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))) if p.sum() else 0
        # Distances are descriptive; nearest neighbor excludes each point itself.
        distances = np.maximum(np.sum(z*z, axis=1)[:, None] + np.sum(z*z, axis=1)[None, :] - 2*z@z.T, 0)
        same = y[:, None] == y[None, :]; np.fill_diagonal(same, False)
        different = y[:, None] != y[None, :]
        within = float(np.sqrt(distances[same]).mean()) if same.any() else np.nan
        between = float(np.sqrt(distances[different]).mean()) if different.any() else np.nan
        np.fill_diagonal(distances, np.inf)
        consistency = float(np.mean(y[np.argmin(distances, axis=1)] == y))
        records.append({'role': role, 'mean_norm': float(np.linalg.norm(z, axis=1).mean()),
            'effective_rank_entropy_covariance': entropy_rank, 'within_class_distance': within,
            'between_class_distance': between, 'neighbor_consistency_excluding_self': consistency})
    return records


def run(args):
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(0)
    if args.smoke:
        args.epochs, args.episodes, args.val_episodes, args.seeds, args.shots = 2, 5, 2, [0], [1, 5]
    tasks = json.loads(args.task_file.read_text())['tasks']
    if args.smoke: tasks = tasks[:1]
    for task in tasks:
        novel = [c for session in task['novel_sessions'] for c in session]
        if len(task['base']) != 4 or len(novel) != 2 or len(set(task['base'] + novel)) != 6:
            raise ValueError('This study requires four disjoint base and two unseen classes')
    if not args.smoke and (len(tasks) != 3 or args.seeds != [0, 1, 2]):
        raise ValueError('Planned run must use exactly the three specified tasks and seeds 0,1,2')
    source_hashes = {name: digest(ROOT / name) for name in SOURCES}
    capacity = capacity_check()
    if not args.smoke:
        if not args.validated_smoke or not (args.validated_smoke / 'VALIDATION_PASSED.json').exists():
            raise ValueError('Run paired smoke validation first; supply --validated-smoke DIRECTORY')
        check = json.loads((args.validated_smoke / 'experiment_manifest.json').read_text())
        if check['source_hashes'] != source_hashes:
            raise ValueError('Implementation changed since smoke validation; validate current sources again')
    paths = {key: ROOT / 'dataset' / REGISTRY[key][0] for key in ['c4', 'c8']}
    waves = []
    for path in paths.values():
        with path.open(encoding='latin-1', newline='') as handle:
            reader = csv.reader(handle); next(reader); waves.append(np.asarray(next(reader)[3:], float))
    grid = np.linspace(max(w.min() for w in waves), min(w.max() for w in waves), 1868)
    config = {'stage': 'smoke' if args.smoke else 'planned', 'dataset': args.dataset, 'tasks': tasks,
        'seeds': args.seeds, 'shots': args.shots, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'episodes': args.episodes, 'val_episodes': args.val_episodes,
        'queries': args.queries, 'c_grid': args.c_grid, 'threads': args.threads, 'parameters': capacity,
        'embedding_dim': 128, 'grid_sha256': hashlib.sha256(grid.astype('<f8').tobytes()).hexdigest(),
        'grid_range_cm1': [float(grid[0]), float(grid[-1])], 'grid_points': len(grid),
        'preprocessing': 'legacy row_max-y; native airPLS; maxabs; interp1d cubic; common ascending grid',
        'signal_units': 'legacy intensity; physical absorbance not established',
        'dataset_hashes': {key: digest(path) for key, path in paths.items()}, 'source_hashes': source_hashes,
        'specimen_status': 'exact-spectrum deduplication only; specimen grouping unverified',
        'evaluation': 'two unseen classes jointly; disjoint novel support/query pools; no incremental claim',
        'training': 'base-only CE; fixed optimizer steps; no early stopping; best base-validation-loss checkpoint',
        'scope': 'exploratory: C4 novel results were inspected previously; C8 not used as independent confirmation',
        'decision_rule': {'primary_shots': [5, 10], 'minimum_mean_gain_pp_each_head': 1.0,
            'minimum_winning_trials_each_head': 6, 'required_positive_task_means_each_head': 3,
            'otherwise': 'stop architecture investigation; no added models/losses'},
        'versions': {key: importlib.metadata.version(key) for key in ['tensorflow', 'numpy', 'scipy', 'scikit-learn']},
        'platform': platform.platform(), 'devices': [str(d) for d in tf.config.list_physical_devices()]}
    out = args.output.resolve()
    if out.exists():
        if not args.resume: raise FileExistsError('Use a new versioned directory or --resume this same study')
        if json.loads((out / 'experiment_manifest.json').read_text()) != config:
            raise ValueError('Resume configuration/source/data do not match')
    else:
        out.mkdir(parents=True); write_json(out / 'experiment_manifest.json', config)
        np.save(out / 'common_grid.npy', grid)
        snapshots = out / 'source'; snapshots.mkdir()
        for name in SOURCES: (snapshots / name).write_bytes((ROOT / name).read_bytes())
    data = load_dataset(args.dataset)
    started = time.perf_counter()
    x = preprocess(data, grid, ROOT / 'results' / 'research_cache', baseline=True, flip=True)
    write_json(out / 'preprocessing_cost.json', {'cache_or_preprocess_seconds': time.perf_counter() - started,
                                               'shared_once_by_both_models': True})
    for task_number, task in enumerate(tasks):
        for seed in args.seeds:
            folder = out / task['name'] / f'seed_{seed}'; folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'TRIAL_COMPLETE.json').exists():
                print(f'Reusing completed paired trial {folder}', flush=True); continue
            train, val, roles, episodes = build_manifest(data.labels, task, seed, args)
            pd.DataFrame({'index': np.arange(len(x)), 'source_id': data.ids, 'class': data.labels,
                          'role': roles}).to_csv(folder / 'sample_manifest.csv', index=False)
            write_json(folder / 'episodes.json', episodes)
            orders = np.stack([np.random.default_rng(seed + epoch * 1000).permutation(train) for epoch in range(args.epochs)])
            np.save(folder / 'shared_train_orders.npy', orders)
            views = {'raw': x.astype(float)}; costs, geometries = [], []
            query_pool = np.flatnonzero(roles == 'novel_query_pool')
            for name in ['M0', 'M1']:
                model_folder = folder / name; model_folder.mkdir(exist_ok=True)
                if (model_folder / 'MODEL_COMPLETE.json').exists():
                    with np.load(model_folder / 'embeddings.npz') as saved: embedding = saved['embeddings']
                    stats = json.loads((model_folder / 'training_costs.json').read_text())
                else:
                    embedding, stats = train_model(name, x, data.labels, task, train, val,
                        seed, args, model_folder, orders)
                views[name] = embedding; costs.append(stats)
                for row in geometry(embedding, data.labels, train, val, query_pool):
                    geometries.append({'architecture': name, **row})
            if costs[0]['training_order_hash'] != costs[1]['training_order_hash'] or costs[0]['optimizer_updates'] != costs[1]['optimizer_updates']:
                raise ValueError('Paired training protocol mismatch')
            pd.DataFrame(costs).to_csv(folder / 'costs.csv', index=False)
            pd.DataFrame(geometries).to_csv(folder / 'feature_geometry.csv', index=False)
            records, tuning, cms = [], [], {}
            with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle); writer.writerow(['head', 'shot', 'episode', 'index', 'true', 'predicted'])
                for k in args.shots:
                    inner = [ep for ep in episodes if ep['role'] == 'base_tuning' and ep['shot'] == k]
                    outer = [ep for ep in episodes if ep['role'] == 'unseen_evaluation' and ep['shot'] == k]
                    params = {}
                    for view in ['raw', 'M0', 'M1']:
                        params[view], candidates = tune_svm(views[view], data.labels, inner, args.c_grid)
                        tuning.extend([{'view': view, 'shot': k, **row} for row in candidates])
                    classes = np.asarray(sorted(c for s in task['novel_sessions'] for c in s))
                    for head in HEADS:
                        view, kind = head.split('_')
                        cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
                        for ep in outer:
                            s, q = ep['support'], ep['query']
                            estimator = SVC(C=params[view], kernel='linear') if kind == 'svm' else KNeighborsClassifier(n_neighbors=1)
                            start = time.perf_counter(); estimator.fit(views[view][s], data.labels[s])
                            fit_ms = (time.perf_counter() - start) * 1000
                            start = time.perf_counter(); predicted = estimator.predict(views[view][q])
                            predict_ms = (time.perf_counter() - start) * 1000
                            metrics, report, matrix = utils.classification_metrics(data.labels[q], predicted, classes, classes.tolist())
                            cm += matrix
                            records.append({'task': task['name'], 'seed': seed, 'shot': k, 'episode': ep['episode'],
                                'head': head, **metrics, 'head_fit_ms': fit_ms, 'head_predict_ms': predict_ms})
                            writer.writerows(zip(np.repeat(head, len(q)), np.repeat(k, len(q)),
                                np.repeat(ep['episode'], len(q)), q, data.labels[q], predicted))
                        cms[f'{head}_{k}shot'] = cm
                    print(f'Paired unseen evaluation: {task["name"]}/seed {seed}/{k}-shot complete', flush=True)
            pd.DataFrame(records).to_csv(folder / 'metrics.csv', index=False)
            pd.DataFrame(tuning).to_csv(folder / 'svm_base_tuning.csv', index=False)
            np.savez_compressed(folder / 'confusions.npz', classes=classes, **cms)
            write_json(folder / 'TRIAL_COMPLETE.json', {'complete': True, 'pairing_validated': True})
    from analyze_encoder_architecture_v1 import analyze
    analyze(out)
    write_json(out / ('VALIDATION_PASSED.json' if args.smoke else 'COMPLETE.json'),
               {'complete': True, 'paired_capacity_and_training_checks': True})
    print(f'Finished architecture comparison: {out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--validated-smoke', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dataset', choices=['c4'], default='c4')
    p.add_argument('--task-file', type=Path, default=ROOT / 'research_configs' / 'c4_4plus1plus1.json')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10, 20])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--val-episodes', type=int, default=10)
    p.add_argument('--queries', type=int, default=15)
    p.add_argument('--c-grid', nargs='+', type=float, default=[.1, 1, 10])
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.episodes, args.val_episodes, args.queries, args.threads, args.learning_rate, *args.shots, *args.c_grid) <= 0:
        p.error('Training/evaluation counts and hyperparameters must be positive')
    run(args)


if __name__ == '__main__': main()
