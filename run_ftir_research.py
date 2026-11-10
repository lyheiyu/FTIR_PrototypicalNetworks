"""Inventory, frozen-feature diagnosis and base-only class-incremental FTIR runs."""
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from fewshot_benchmark import CLASSES, train_encoder
from ftir_research_data import ROOT, REGISTRY, load_dataset, preprocess, inventory
from ftir_research_core import (SPECS, Recorder, base_covariance, evaluate_head,
                                tune_heads, plot_run)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def new_run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'python': platform.python_version(),
                   'scope': 'exploratory diagnostic' if args.command == 'diagnose' else 'base-only frozen encoder class-incremental',
                   'memory_policy': 'same retained exemplars for all heads; full base training set may fit encoder/PCA/base covariance',
                   'timing_scope': 'head fit and batched query predict; excludes shared preprocessing/feature extraction; not an optimized incremental deployment benchmark'})
    write_json(out / 'config.json', config)
    source = out / 'source'; source.mkdir()
    for name in ['run_ftir_research.py', 'ftir_research_core.py', 'ftir_research_data.py',
                 'fewshot_benchmark.py', 'utils.py', 'PLS.py']:
        (source / name).write_bytes((ROOT / name).read_bytes())
    versions = {}
    for name in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'tensorflow']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(out / 'versions.json', versions)
    return out


def build_views(raw, embeddings, labels, train, args):
    dimension = min(args.pca_dim, len(train) - 1, raw.shape[1])
    if dimension < 1:
        raise ValueError('Not enough base training samples for PCA')
    pca = PCA(n_components=dimension, svd_solver='randomized', random_state=0)
    pca.fit(raw[train])
    z = normalize(embeddings) if args.l2_embedding else embeddings
    views = {'raw': raw.astype(float), 'pca': pca.transform(raw).astype(float), 'embedding': z.astype(float)}
    prior = base_covariance(views['embedding'][train], labels[train])
    return views, prior, pca


def save_pca(folder, pca):
    np.savez_compressed(folder / 'pca.npz', components=pca.components_, mean=pca.mean_,
                        explained_variance_ratio=pca.explained_variance_ratio_)


def record_predictions(writer, meta, query, labels, predictions):
    for index, pred in zip(query, predictions):
        writer.writerow([meta['dataset'], meta['task'], meta['seed'], meta['shot'],
                         meta['repeat'], meta['session'], meta['method'], int(index), labels[index], pred])


PRED_COLUMNS = ['dataset', 'task', 'seed', 'shot', 'repeat', 'session', 'method', 'index', 'true', 'predicted']


def diagnose(args):
    previous = args.previous.resolve()
    cfg = json.loads((previous / 'config.json').read_text())
    provenance = json.loads((previous / 'provenance.json').read_text())
    # Preflight before creating output.
    for dataset in args.datasets:
        if dataset not in cfg['datasets']:
            raise ValueError(f'{dataset} not in previous run')
        for seed in args.seeds:
            if seed not in cfg['seeds']:
                raise ValueError(f'Seed {seed} not in previous run')
            for k in args.shots:
                if not (previous / dataset / f'seed_{seed}' / f'episodes_{k}shot.json').exists():
                    raise FileNotFoundError(f'Missing saved {dataset}/{seed}/{k}-shot episodes')
    out = new_run(args)
    for dataset in args.datasets:
        cache_path = previous / f'FTIR_PLastics500_{dataset}_deduplicated_preprocessed.npz'
        with np.load(cache_path, allow_pickle=False) as cache:
            raw, y, ids = cache['x'], cache['y'], cache['ids']
            signature = json.loads(cache['signature'].item())
        source = ROOT / 'dataset' / 'deduplicated' / f'FTIR_PLastics500_{dataset}_deduplicated.csv'
        expected_hash = provenance['source_sha256'][dataset]
        if signature['source_sha256'] != expected_hash or hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Source/cache hash mismatch; diagnostic cannot reuse these indices')
        labels = np.asarray([CLASSES[int(c)] for c in y])
        for seed in args.seeds:
            saved = previous / dataset / f'seed_{seed}'
            folder = out / dataset / f'seed_{seed}'; folder.mkdir(parents=True)
            manifest = pd.read_csv(saved / 'split_manifest.csv', dtype={'source_id': str})
            if not np.array_equal(manifest['index'], np.arange(len(y))) or not np.array_equal(manifest.source_id.astype(str), ids.astype(str)):
                raise ValueError('Manifest/cache row alignment mismatch')
            with np.load(saved / 'embeddings.npz', allow_pickle=False) as data:
                z = data['embeddings']
                if not np.array_equal(data['labels'], y):
                    raise ValueError('Embedding labels/cache mismatch')
            train = manifest.loc[manifest.split == 'train', 'index'].to_numpy()
            val = manifest.loc[manifest.split == 'validation', 'index'].to_numpy()
            test = set(manifest.loc[manifest.split == 'test', 'index'])
            views, prior, pca = build_views(raw, z, labels, train, args)
            save_pca(folder, pca)
            manifest.to_csv(folder / 'split_manifest.csv', index=False)
            recorder = Recorder(folder)
            with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle); writer.writerow(PRED_COLUMNS)
                for k in args.shots:
                    params, tuning = tune_heads(views, labels, val, args.methods, prior,
                        np.random.default_rng(seed + k * 100), k, args.queries, args.val_episodes,
                        args.c_grid, args.reg_grid, args.tau_grid)
                    if 'embedding_class' in params and not args.l2_embedding:
                        original = json.loads((saved / f'selected_regularization_{k}shot.json').read_text())
                        params['embedding_class'] = {'reg': original['class_adaptive_mahalanobis']}
                    write_json(folder / f'{k}shot_parameters.json', params)
                    pd.DataFrame(tuning).to_csv(folder / f'{k}shot_validation.csv', index=False)
                    episodes = json.loads((saved / f'episodes_{k}shot.json').read_text())
                    if args.episodes > len(episodes):
                        raise ValueError('Requested more episodes than saved; indices must be reused')
                    for ep in episodes[:args.episodes]:
                        support, query = np.asarray(ep['support_indices']), np.asarray(ep['query_indices'])
                        if set(support) & set(query) or not set(support.tolist() + query.tolist()) <= test:
                            raise ValueError('Invalid saved test episode')
                        for method in args.methods:
                            pred, fit_ms, pred_ms, memory = evaluate_head(views, labels, support, query, method, params[method], prior)
                            meta = {'dataset': dataset, 'task': 'diagnostic', 'seed': seed, 'shot': k,
                                    'repeat': ep['episode'], 'session': 0, 'method': method}
                            recorder.add(meta, labels[query], pred, np.unique(labels), np.unique(labels), [], [], fit_ms, pred_ms, memory)
                            record_predictions(writer, meta, query, labels, pred)
                    recorder.save()
                    print(f'Diagnostic finished: {dataset}/seed {seed}/{k}-shot', flush=True)
    if not args.no_plots:
        plot_run(out)
    write_json(out / 'COMPLETE.json', {'complete': True})
    print(f'Completed diagnosis: {out}', flush=True)


def representatives(labels, groups):
    """When verified grouping is supplied, use one spectrum per physical group."""
    seen, indices = {}, []
    for i, (label, group) in enumerate(zip(labels, groups)):
        if group in seen:
            if seen[group] != label:
                raise ValueError(f'Group {group} has conflicting class labels')
        else:
            seen[group] = label; indices.append(i)
    return np.asarray(indices)


def generate_tasks(source_y, target_y, source_pool, target_pool, args):
    if args.task_file:
        tasks = json.loads(args.task_file.read_text(encoding='utf-8'))['tasks']
    else:
        base_eligible = sorted(c for c in np.unique(source_y[source_pool])
            if np.sum(source_y[source_pool] == c) >= args.queries + 4
            and np.sum(target_y[target_pool] == c) >= args.queries)
        novel_eligible = sorted(c for c in np.unique(target_y[target_pool])
            if np.sum(target_y[target_pool] == c) >= max(args.shots) + args.queries)
        rng, tasks, seen = np.random.default_rng(args.task_seed), [], set()
        for _ in range(10000):
            if len(base_eligible) < args.base_count:
                break
            base = sorted(rng.choice(base_eligible, args.base_count, replace=False).tolist())
            available = [c for c in novel_eligible if c not in base]
            if len(available) < args.novel_sessions:
                continue
            novel = rng.choice(available, args.novel_sessions, replace=False).tolist()
            identity = (tuple(base), tuple(novel))
            if identity in seen:
                continue
            seen.add(identity)
            tasks.append({'name': f'task_{len(tasks)}', 'base': base, 'novel_sessions': [[c] for c in novel]})
            if len(tasks) == args.tasks:
                break
        if len(tasks) != args.tasks:
            raise ValueError('Not enough eligible classes/task orders. Inspect inventory; reduce shots/queries/base-count or provide --task-file.')
    names = set()
    for task in tasks:
        if task['name'] in names or Path(task['name']).name != task['name'] or task['name'] in ('.', '..'):
            raise ValueError('Task names must be unique plain filenames')
        names.add(task['name'])
        classes = task['base'] + [c for session in task['novel_sessions'] for c in session]
        if len(set(classes)) != len(classes) or len(task['base']) < 2 or not task['novel_sessions'] or any(not s for s in task['novel_sessions']):
            raise ValueError('Task must have disjoint base/novel classes and nonempty sessions')
        for c in task['base']:
            if np.sum(source_y[source_pool] == c) < args.queries + 4 or np.sum(target_y[target_pool] == c) < args.queries:
                raise ValueError(f'Insufficient base data for {c}')
        for c in classes[len(task['base']):]:
            if np.sum(target_y[target_pool] == c) < max(args.shots) + args.queries:
                raise ValueError(f'Insufficient novel support/query for {c}')
    return tasks


def make_protocol(labels, source_pool, target_pool, groups, task, seed, queries, same_domain):
    rng = np.random.default_rng(seed)
    query, novel_pool, train, val = {}, {}, [], []
    all_classes = task['base'] + [c for session in task['novel_sessions'] for c in session]
    for c in all_classes:
        candidates = rng.permutation(target_pool[labels[target_pool] == c])
        query[c] = candidates[:queries]
        if c not in task['base']:
            novel_pool[c] = candidates[queries:]
    forbidden_groups = set(groups[np.concatenate(list(query.values()))])
    for c in task['base']:
        candidates = source_pool[labels[source_pool] == c]
        candidates = np.asarray([i for i in candidates if groups[i] not in forbidden_groups], dtype=int)
        candidates = rng.permutation(candidates)
        if len(candidates) < 4:
            raise ValueError(f'Insufficient independent training/validation for base class {c}')
        nval = max(2, int(len(candidates) * 0.2))
        val.extend(candidates[:nval]); train.extend(candidates[nval:])
    return np.asarray(train), np.asarray(val), query, novel_pool


def incremental(args):
    import importlib.util
    if importlib.util.find_spec('tensorflow') is None:
        raise RuntimeError('Use your fewshotlearning TensorFlow environment')
    aliases = json.loads(args.aliases.read_text()) if args.aliases else None
    source = load_dataset(args.dataset, aliases)
    target = source if not args.target or args.target == args.dataset else load_dataset(args.target, aliases)
    same_domain = source is target
    if not same_domain and source.report['family'] == target.report['family'] and not args.groups:
        raise ValueError('These files are variants of one source family. Supply verified --groups mapping before cross-file evaluation; exported row numbers are not specimen identities.')
    if not same_domain and set(source.hashes) & set(target.hashes):
        raise ValueError('Exact spectra overlap between source/target; resolve provenance before cross-file evaluation')
    grid = np.linspace(min(source.wave.max(), target.wave.max()), max(source.wave.min(), target.wave.min()), args.points)
    if grid[0] <= grid[-1]:
        raise ValueError('Source and target have no common wave range')
    out = new_run(args)
    for data in [source] if same_domain else [source, target]:
        report = data.report.copy()
        pd.DataFrame(report.pop('row_audit')).to_csv(out / f'{data.key}_row_audit.csv', index=False)
        write_json(out / f'{data.key}_data.json', report)
    cache = args.cache.resolve()
    started = time.perf_counter()
    flip = None if args.flip == 'profile' else args.flip == 'yes'
    source_x = preprocess(source, grid, cache, baseline=not args.no_baseline, flip=flip)
    target_x = source_x if same_domain else preprocess(target, grid, cache, baseline=not args.no_baseline, flip=flip)
    preparation_seconds = time.perf_counter() - started
    raw = source_x if same_domain else np.concatenate([source_x, target_x])
    labels = source.labels if same_domain else np.concatenate([source.labels, target.labels])
    ids = source.ids if same_domain else np.concatenate([source.ids, target.ids])
    domains = np.repeat(source.key, len(source.x)) if same_domain else np.concatenate([np.repeat(source.key, len(source.x)), np.repeat(target.key, len(target.x))])
    hashes = source.hashes if same_domain else np.concatenate([source.hashes, target.hashes])
    groups = np.asarray([f'{d}:{h}' for d, h in zip(domains, hashes)])
    if args.groups:
        mapping = pd.read_csv(args.groups, dtype=str)
        if not {'dataset', 'row_id', 'group'}.issubset(mapping.columns) or mapping.duplicated(['dataset', 'row_id']).any():
            raise ValueError('Groups CSV requires unique dataset,row_id,group records')
        lookup = {(r.dataset, r.row_id): r.group for r in mapping.itertuples()}
        try:
            groups = np.asarray([lookup[(d, i)] for d, i in zip(domains, ids)])
        except KeyError as error:
            raise ValueError(f'Group map must cover all retained rows: {error}')
        if pd.isna(groups).any() or np.any(groups == ''):
            raise ValueError('Group identity cannot be empty')
        representatives(labels, groups)  # also reject cross-domain label conflicts
    source_pool = representatives(labels[:len(source.x)], groups[:len(source.x)])
    target_pool = source_pool if same_domain else representatives(target.labels, groups[len(source.x):]) + len(source.x)
    tasks = generate_tasks(labels, labels, source_pool, target_pool, args)
    write_json(out / 'tasks.json', {'tasks': tasks})
    np.save(out / 'wavenumber_grid.npy', grid)
    pd.DataFrame({'index': np.arange(len(labels)), 'dataset': domains, 'row_id': ids,
                  'class': labels, 'group': groups}).to_csv(out / 'sample_index.csv', index=False)
    dataset_name = source.key if same_domain else f'{source.key}_to_{target.key}'
    for task_index, task in enumerate(tasks):
        for seed in args.seeds:
            folder = out / dataset_name / task['name'] / f'seed_{seed}'; folder.mkdir(parents=True)
            train, val, query_map, novel_pool = make_protocol(labels, source_pool, target_pool, groups,
                task, seed + args.task_seed + task_index * 1000, args.queries, same_domain)
            all_query = np.concatenate(list(query_map.values()))
            if set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[all_query]) or set(groups[val]) & set(groups[all_query]):
                raise ValueError('Physical groups overlap across train/validation/query')
            roles = np.full(len(labels), 'unused', dtype=object)
            roles[train], roles[val], roles[all_query] = 'base_train', 'base_validation', 'fixed_query'
            for pool in novel_pool.values():
                roles[pool] = 'novel_support_pool'
            manifest = pd.DataFrame({'index': np.arange(len(labels)), 'class': labels, 'dataset': domains,
                                      'source_id': ids, 'group': groups, 'role': roles})
            manifest.to_csv(folder / 'split_manifest.csv', index=False)
            base_labels = np.full(len(labels), -1, dtype=int)
            for i, c in enumerate(task['base']):
                base_labels[labels == c] = i
            if np.any(base_labels[train] < 0) or np.any(base_labels[val] < 0):
                raise ValueError('Novel class reached encoder training')
            print(f'Training {dataset_name}/{task["name"]}/seed {seed}; base={task["base"]}; novel={task["novel_sessions"]}', flush=True)
            train_start = time.perf_counter()
            embeddings, version = train_encoder(raw, base_labels, train, val, seed, args, folder,
                                                 n_classes=len(task['base']), backbone=args.backbone)
            encoder_hash_before = hashlib.sha256((folder / 'encoder.h5').read_bytes()).hexdigest()
            write_json(folder / 'training_provenance.json', {'base_classes': task['base'], 'train_indices': train.tolist(),
                'validation_indices': val.tolist(), 'tensorflow': version, 'encoder_sha256': encoder_hash_before,
                'encoder_file_bytes': (folder / 'encoder.h5').stat().st_size,
                'identity_control': 'verified group representatives' if args.groups else 'within-file exact-spectrum deduplication only',
                'preparation_seconds': preparation_seconds, 'training_and_all_feature_extraction_seconds': time.perf_counter() - train_start})
            views, prior, pca = build_views(raw, embeddings, labels, train, args)
            save_pca(folder, pca)
            np.savez_compressed(folder / 'frozen_features.npz', embedding=views['embedding'], base_covariance=prior)
            recorder = Recorder(folder)
            rng = np.random.default_rng(seed + 40000 + task_index * 1000)
            base_memory = np.concatenate([rng.choice(train[labels[train] == c], min(args.base_memory, np.sum(labels[train] == c)), replace=False) for c in task['base']])
            episodes_audit = []
            with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle); writer.writerow(PRED_COLUMNS)
                for k in args.shots:
                    params, tuning = tune_heads(views, labels, val, args.methods, prior,
                        np.random.default_rng(seed + k * 100), k, args.queries, args.val_episodes,
                        args.c_grid, args.reg_grid, args.tau_grid)
                    write_json(folder / f'{k}shot_parameters.json', params)
                    pd.DataFrame(tuning).to_csv(folder / f'{k}shot_validation.csv', index=False)
                    for repeat in range(args.repeats):
                        support = base_memory.copy()
                        seen = list(task['base'])
                        # Query pool and base exemplars stay fixed across sessions.
                        for session in range(len(task['novel_sessions']) + 1):
                            new = [] if session == 0 else task['novel_sessions'][session - 1]
                            old = seen.copy() if session else []
                            for c in new:
                                support = np.concatenate([support, rng.choice(novel_pool[c], k, replace=False)])
                            seen.extend(new)
                            query = np.concatenate([query_map[c] for c in seen])
                            if set(groups[support]) & set(groups[query]):
                                raise ValueError('Support/query group overlap')
                            episodes_audit.append({'shot': k, 'repeat': repeat, 'session': session,
                                'seen': seen.copy(), 'new': new, 'support': support.tolist(), 'query': query.tolist()})
                            for method in args.methods:
                                pred, fit_ms, pred_ms, memory = evaluate_head(views, labels, support, query, method, params[method], prior)
                                meta = {'dataset': dataset_name, 'task': task['name'], 'seed': seed,
                                        'shot': k, 'repeat': repeat, 'session': session, 'method': method}
                                recorder.add(meta, labels[query], pred, np.asarray(sorted(seen)), task['base'], old, new,
                                             fit_ms, pred_ms, memory)
                                record_predictions(writer, meta, query, labels, pred)
                        if repeat == 0:
                            # Retained library permits later audit/reconstruction;
                            # never include query or unarrived future classes.
                            library = folder / f'{k}shot_example_library.npz'
                            np.savez_compressed(library, support_indices=support,
                                labels=labels[support], embeddings=views['embedding'][support],
                                base_covariance=prior,
                                classes=np.asarray(sorted(seen)))
                        recorder.save()
                    print(f'Finished {task["name"]}/seed {seed}/{k}-shot', flush=True)
            write_json(folder / 'sessions.json', episodes_audit)
            encoder_hash_after = hashlib.sha256((folder / 'encoder.h5').read_bytes()).hexdigest()
            if encoder_hash_before != encoder_hash_after:
                raise ValueError('Encoder file changed during incremental evaluation')
            write_json(folder / 'FROZEN_CHECK.json', {'sha256_before': encoder_hash_before,
                         'sha256_after': encoder_hash_after, 'unchanged': True})
    if not args.no_plots:
        plot_run(out)
    write_json(out / 'COMPLETE.json', {'complete': True})
    print(f'Completed incremental run: {out}', flush=True)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory'); inv.add_argument('--output', type=Path, required=True)
    inv.add_argument('--datasets', nargs='+', choices=list(REGISTRY), default=list(REGISTRY))
    plot = sub.add_parser('plot'); plot.add_argument('output', type=Path)
    for command in ['diagnose', 'incremental']:
        p = sub.add_parser(command)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
        p.add_argument('--shots', nargs='+', type=int, default=[1, 5, 10, 20])
        p.add_argument('--queries', type=int, default=15)
        p.add_argument('--val-episodes', type=int, default=10)
        p.add_argument('--c-grid', nargs='+', type=float, default=[0.1, 1, 10])
        p.add_argument('--reg-grid', nargs='+', type=float, default=[1e-3, 1e-2, 0.1])
        p.add_argument('--tau-grid', nargs='+', type=float, default=[5, 20])
        p.add_argument('--pca-dim', type=int, default=64)
        p.add_argument('--l2-embedding', action='store_true')
        p.add_argument('--no-plots', action='store_true')
        if command == 'diagnose':
            p.add_argument('--previous', type=Path, default=ROOT / 'results' / 'fewshot_main')
            p.add_argument('--datasets', nargs='+', choices=['c4', 'c8'], default=['c4', 'c8'])
            p.add_argument('--episodes', type=int, default=200)
            p.add_argument('--methods', nargs='+', choices=list(SPECS), default=['raw_svm', 'raw_1nn', 'pca_svm', 'embedding_svm', 'embedding_1nn', 'embedding_proto', 'embedding_class'])
        else:
            p.add_argument('--dataset', choices=list(REGISTRY), default='c4')
            p.add_argument('--target', choices=list(REGISTRY))
            p.add_argument('--groups', type=Path, help='Verified CSV: dataset,row_id,group; one representative per physical group')
            p.add_argument('--aliases', type=Path, help='Explicit uppercase label alias JSON; no automatic PE/HDPE/LDPE merging')
            p.add_argument('--cache', type=Path, default=ROOT / 'results' / 'research_cache')
            p.add_argument('--tasks', type=int, default=3)
            p.add_argument('--task-file', type=Path)
            p.add_argument('--task-seed', type=int, default=1729)
            p.add_argument('--base-count', type=int, default=4)
            p.add_argument('--novel-sessions', type=int, default=2)
            p.add_argument('--base-memory', type=int, default=20)
            p.add_argument('--repeats', type=int, default=20)
            p.add_argument('--epochs', type=int, default=60)
            p.add_argument('--patience', type=int, default=8)
            p.add_argument('--batch-size', type=int, default=64)
            p.add_argument('--embedding-dim', type=int, default=128)
            p.add_argument('--learning-rate', type=float, default=1e-3)
            p.add_argument('--backbone', choices=['compact', 'multiscale'], default='compact')
            p.add_argument('--points', type=int, default=2000)
            p.add_argument('--flip', choices=['profile', 'yes', 'no'], default='profile')
            p.add_argument('--no-baseline', action='store_true')
            p.add_argument('--methods', nargs='+', choices=list(SPECS), default=['raw_svm', 'raw_1nn', 'raw_proto', 'pca_proto', 'embedding_proto', 'embedding_shared_prior', 'embedding_class', 'embedding_shrinkage'])
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ('diagnose', 'incremental'):
        numbers = [args.queries, args.val_episodes, args.pca_dim, *args.shots, *args.c_grid, *args.reg_grid, *args.tau_grid]
        if min(numbers) <= 0 or min(args.seeds) < 0:
            raise ValueError('Counts/grid values must be positive; seeds nonnegative')
        for values in [args.seeds, args.shots, args.methods]:
            if len(set(values)) != len(values):
                raise ValueError('Duplicate seed/shot/method is not allowed')
        if args.command == 'diagnose' and args.episodes <= 0:
            raise ValueError('episodes must be positive')
        if args.command == 'incremental' and min(args.tasks, args.base_count, args.novel_sessions, args.base_memory,
                args.repeats, args.epochs, args.batch_size, args.embedding_dim, args.points, args.learning_rate) <= 0:
            raise ValueError('Incremental counts must be positive')
        if args.command == 'incremental' and (args.patience < 0 or args.points < 16):
            raise ValueError('patience must be nonnegative and points at least 16')
    if args.command == 'inventory':
        inventory(args.output.resolve(), args.datasets)
    elif args.command == 'diagnose':
        diagnose(args)
    elif args.command == 'incremental':
        incremental(args)
    elif args.command == 'plot':
        if not (args.output / 'COMPLETE.json').exists():
            raise ValueError('Run is not marked complete; do not plot partial trials as final results')
        plot_run(args.output)


if __name__ == '__main__':
    main()
