"""Reuse architecture-V1 checkpoints/splits/episodes for frozen C4 4+1+1.

No training. Reuses Head/score from ftir_research_core and existing CM drawing.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import encoder_architecture_models  # registers saved SpectralBinPooling
from ftir_research_data import ROOT, load_dataset, preprocess
from ftir_research_core import Head, score
from run_encoder_architecture_v1 import digest, weight_hash, write_json

METHODS = ['raw_svm', 'M0_svm', 'M0_1nn', 'M1_svm', 'M1_1nn']


class ClassBank:
    """Local append copies only an arriving class; materialization is separate."""
    def __init__(self):
        self.entries = {}

    def register(self, label, indices, features):
        if label in self.entries:
            raise ValueError('Previously registered class cannot be replaced')
        self.entries[label] = (np.array(indices, copy=True), np.array(features, copy=True))

    def materialize(self):
        classes = sorted(self.entries)
        x = np.concatenate([self.entries[c][1] for c in classes])
        y = np.concatenate([np.repeat(c, len(self.entries[c][0])) for c in classes])
        return x, y


def sessions_for_episode(samples, task, episode, base_memory, seed, queries):
    """Preserve archived novel supports/queries exactly; append sequentially."""
    labels = samples['class'].to_numpy()
    rng = np.random.default_rng(seed + 30000 + episode['episode'])
    base_query = np.concatenate([rng.choice(np.flatnonzero(
        (samples.role.to_numpy() == 'base_validation') & (labels == c)), queries, replace=False)
        for c in task['base']])
    seen, support, query = list(task['base']), list(base_memory), list(base_query)
    result = []
    for session in range(3):
        new = [] if session == 0 else task['novel_sessions'][session - 1]
        added = [i for i in episode['support'] if labels[i] in new]
        support.extend(added)
        query.extend(i for i in episode['query'] if labels[i] in new)
        previous_novel = seen[len(task['base']):].copy()
        seen.extend(new)
        if set(support).intersection(query):
            raise ValueError('Incremental support/query overlap')
        result.append({'session': session, 'seen': seen.copy(), 'current_novel': new,
            'previous_novel': previous_novel, 'support': [int(i) for i in support],
            'query': [int(i) for i in query], 'added': [int(i) for i in added]})
    return result


def grouped_metrics(true, predicted, seen, base, previous, current):
    values, report, cm = score(true, predicted, np.asarray(sorted(seen)))
    groups = {'base': base, 'previous_novel': previous, 'current_novel': current,
              'seen_novel': [c for c in seen if c not in base], 'all': seen}
    for name, classes in groups.items():
        mask = np.isin(true, classes)
        values[name + '_accuracy'] = float(np.mean(true[mask] == predicted[mask])) if mask.any() else np.nan
        # Global one-vs-rest class F1: cross-group false positives remain included.
        values[name + '_macro_f1'] = float(np.mean([report[c]['f1-score'] for c in classes])) if classes else np.nan
    return values, report, cm


def run(args):
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()
    source, out = args.source.resolve(), args.output.resolve()
    parent = json.loads((source / 'experiment_manifest.json').read_text())
    assert parent['stage'] == 'planned' and (source / 'COMPLETE.json').exists()
    assert parent['seeds'] == [0, 1, 2] and len(parent['tasks']) == 3
    # Verify the frozen study's archived implementation and dataset provenance.
    for name, expected in parent['source_hashes'].items():
        assert digest(source / 'source' / name) == expected
    tasks = parent['tasks'][:1] if args.smoke else parent['tasks']
    seeds = [0] if args.smoke else parent['seeds']
    shots = [1, 5] if args.smoke else parent['shots']
    repeats = 2 if args.smoke else parent['episodes']
    source_files = ['run_frozen_incremental_v1.py', 'analyze_frozen_incremental_v1.py',
                    'ftir_research_core.py', 'encoder_architecture_models.py', 'ftir_research_data.py', 'utils.py', 'PLS.py',
                    'run_encoder_architecture_v1.py']
    config = {'parent': str(source), 'parent_manifest_sha256': digest(source / 'experiment_manifest.json'),
        'tasks': tasks, 'seeds': seeds, 'shots': shots, 'repeats': repeats,
        'base_memory_per_class': 20, 'queries_per_class': parent['queries'], 'smoke': args.smoke,
        'base_query_status': 'reused base validation; previously used for checkpoint/C selection; not independent test',
        'access': 'all heads: same 20 base-training exemplars/class + K support/arrived novel class; retain all prior support',
        'svm_refit_scope': 'entire seen retained exemplar library, NOT all historical base-training rows',
        'local_cost_scope': 'new-class frozen encoding + class-bank append; excludes shared preprocessing; assembly/index/refit separate',
        'group_f1': 'mean global one-vs-rest class F1 over named group; includes cross-group false positives',
        'feature_reuse': 'scoring reuses archived embeddings; local cost replays frozen encoding on actual arriving support',
        'specimen_status': parent['specimen_status'], 'source_hashes': {p: digest(ROOT / p) for p in source_files}}
    if out.exists():
        if not args.resume:
            raise FileExistsError('Choose a new output or --resume')
        assert json.loads((out / 'experiment_manifest.json').read_text()) == config
    else:
        out.mkdir(parents=True)
        write_json(out / 'experiment_manifest.json', config)
        (out / 'source').mkdir()
        for name in source_files:
            (out / 'source' / name).write_bytes((ROOT / name).read_bytes())
    if not args.smoke:
        assert args.validated_smoke and (args.validated_smoke / 'COMPLETE.json').exists(), 'Validate --smoke first'
        smoke = json.loads((args.validated_smoke / 'experiment_manifest.json').read_text())
        assert smoke['source_hashes'] == config['source_hashes'] and smoke['parent_manifest_sha256'] == config['parent_manifest_sha256']
    data = load_dataset('c4')
    assert data.report['source_sha256'] == parent['dataset_hashes']['c4']
    grid = np.load(source / 'common_grid.npy')
    assert hashlib.sha256(grid.astype('<f8').tobytes()).hexdigest() == parent['grid_sha256']
    x = preprocess(data, grid, ROOT / 'results' / 'research_cache', baseline=True, flip=True)
    labels = data.labels
    for task_index, task in enumerate(tasks):
        for seed in seeds:
            original = source / task['name'] / f'seed_{seed}'
            folder = out / task['name'] / f'seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'COMPLETE.json').exists():
                continue
            samples = pd.read_csv(original / 'sample_manifest.csv', dtype={'source_id': str})
            assert np.array_equal(samples['class'].to_numpy(), labels)
            assert np.array_equal(samples.source_id.to_numpy(), data.ids.astype(str))
            roles = samples.role.to_numpy()
            rng = np.random.default_rng(seed + 40000 + task_index * 1000)
            base_memory = np.concatenate([rng.choice(np.flatnonzero((roles == 'base_train') & (labels == c)), 20, replace=False)
                                          for c in task['base']])
            archived_episodes = json.loads((original / 'episodes.json').read_text())
            selected = [e for e in archived_episodes if e['role'] == 'unseen_evaluation'
                        and e['shot'] in shots and e['episode'] < repeats]
            tuning = pd.read_csv(original / 'svm_base_tuning.csv')
            parameters = {}
            for k in shots:
                for view in ['raw', 'M0', 'M1']:
                    rows = tuning[(tuning.shot == k) & (tuning.view == view)]
                    parameters[(view, k)] = float(rows.loc[rows.base_val_macro_f1.idxmax(), 'c'])
            models, views, hashes, infer = {}, {'raw': x.astype(float)}, {}, {}
            for name in ['M0', 'M1']:
                model_path = original / name / 'encoder.keras'
                model = tf.keras.models.load_model(str(model_path), compile=False)
                model.trainable = False
                stats = json.loads((original / name / 'training_costs.json').read_text())
                assert weight_hash(model) == stats['weight_hash_after']
                with np.load(original / name / 'embeddings.npz') as saved:
                    views[name] = saved['embeddings']
                function = tf.function(lambda inputs, encoder=model: encoder(inputs, training=False))
                for size in shots:
                    for _ in range(3):
                        function(tf.convert_to_tensor(x[:size])).numpy()
                probe = function(tf.convert_to_tensor(x[:8])).numpy()
                np.testing.assert_allclose(probe, views[name][:8], rtol=2e-4, atol=2e-4)
                models[name], infer[name] = model, function
                hashes[name] = {'checkpoint': str(model_path), 'file_before': digest(model_path), 'weights_before': weight_hash(model)}
            records, registration, session_audit, class_rows, matrices = [], [], [], [], {}
            for ep in selected:
                k, repeat = ep['shot'], ep['episode']
                sessions = sessions_for_episode(samples, task, ep, base_memory, seed, parent['queries'])
                banks = {name: ClassBank() for name in views}
                for view, bank in banks.items():
                    for c in task['base']:
                        indices = base_memory[labels[base_memory] == c]
                        bank.register(c, indices, views[view][indices])
                for session in sessions:
                    s, q, added = session['support'], session['query'], session['added']
                    assert np.all(roles[np.asarray(s)] != 'base_validation')
                    assert all(roles[i] == 'novel_support_pool' for i in added)
                    assert all(roles[i] == ('base_validation' if labels[i] in task['base'] else 'novel_query_pool') for i in q)
                    local = {}
                    for view, bank in banks.items():
                        encode_ms = append_ms = 0.
                        if added:
                            if view != 'raw':
                                inputs = tf.convert_to_tensor(x[added])
                                started = time.perf_counter()
                                encoded = infer[view](inputs).numpy()
                                encode_ms = (time.perf_counter() - started) * 1000
                                if repeat == 0:
                                    np.testing.assert_allclose(encoded, views[view][added], rtol=2e-4, atol=2e-4)
                            prepared = views[view][added]
                            started = time.perf_counter()
                            bank.register(session['current_novel'][0], added, prepared)
                            append_ms = (time.perf_counter() - started) * 1000
                        local[view] = encode_ms + append_ms
                        registration.append({'task': task['name'], 'seed': seed, 'shot': k, 'repeat': repeat,
                            'session': session['session'], 'view': view, 'new_support_count': len(added),
                            'new_class_encoding_ms': encode_ms, 'local_bank_append_ms': append_ms,
                            'local_registration_ms': local[view], 'preprocessing_excluded': True})
                    session_audit.append({'shot': k, 'repeat': repeat, **session})
                    for method in METHODS:
                        view, kind = method.split('_')
                        started = time.perf_counter()
                        support_x, support_y = banks[view].materialize()
                        assembly_ms = (time.perf_counter() - started) * 1000
                        assert len(support_y) == len(s) and set(support_y) == set(session['seen'])
                        started = time.perf_counter()
                        head = Head(kind, c=parameters[(view, k)]).fit(support_x, support_y)
                        fit_ms = (time.perf_counter() - started) * 1000
                        started = time.perf_counter()
                        pred = head.predict(views[view][q])
                        predict_ms = (time.perf_counter() - started) * 1000
                        values, report, cm = grouped_metrics(labels[q], pred, session['seen'], task['base'],
                            session['previous_novel'], session['current_novel'])
                        meta = {'task': task['name'], 'seed': seed, 'shot': k, 'repeat': repeat,
                                'session': session['session'], 'method': method}
                        records.append({**meta, **values, 'support_count': len(s), 'query_count': len(q),
                            'local_registration_ms': local[view], 'memory_assembly_ms': assembly_ms,
                            'svm_full_refit_ms': fit_ms if kind == 'svm' else np.nan,
                            'knn_index_build_ms': fit_ms if kind == '1nn' else np.nan,
                            'fit_ms': fit_ms, 'predict_ms': predict_ms,
                            'total_head_update_ms': local[view] + assembly_ms + fit_ms})
                        for c, row in report.items():
                            if c in session['seen']:
                                class_rows.append({**meta, 'class': c, **row})
                        key = f'{method}_{k}shot_session{session["session"]}'
                        matrices[key] = matrices.get(key, np.zeros_like(cm)) + cm
                if (repeat + 1) % 50 == 0:
                    print(f'{task["name"]}/seed {seed}/{k}-shot {repeat + 1}/{repeats}', flush=True)
            for name, model in models.items():
                hashes[name]['file_after'] = digest(original / name / 'encoder.keras')
                hashes[name]['weights_after'] = weight_hash(model)
                assert hashes[name]['file_before'] == hashes[name]['file_after']
                assert hashes[name]['weights_before'] == hashes[name]['weights_after']
            pd.DataFrame(records).to_csv(folder / 'metrics.csv', index=False)
            pd.DataFrame(registration).to_csv(folder / 'registration_costs.csv', index=False)
            pd.DataFrame(class_rows).to_csv(folder / 'per_class_metrics.csv', index=False)
            write_json(folder / 'sessions.json', session_audit)
            write_json(folder / 'frozen_check.json', hashes)
            write_json(folder / 'input_provenance.json', {p: digest(original / p) for p in
                ['sample_manifest.csv', 'episodes.json', 'svm_base_tuning.csv', 'M0/embeddings.npz', 'M1/embeddings.npz']})
            np.savez_compressed(folder / 'confusions.npz', **matrices)
            write_json(folder / 'COMPLETE.json', {'complete': True, 'encoder_training_calls': 0})
            print(f'Completed paired incremental trial {task["name"]}/seed {seed}', flush=True)
            tf.keras.backend.clear_session()
    from analyze_frozen_incremental_v1 import analyze
    analyze(out)
    write_json(out / 'COMPLETE.json', {'complete': True, 'encoder_training_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results' / 'encoder_architecture_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--validated-smoke', type=Path)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
