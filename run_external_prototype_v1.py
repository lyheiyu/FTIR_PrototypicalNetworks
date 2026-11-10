"""C4 exact-episode spectrum/PCA baselines and class-local frozen prototypes.

D4 is audited first; incompatible external inputs are never constructed.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import tensorflow as tf
import encoder_architecture_models
from ftir_research_data import ROOT, load_dataset, preprocess
from ftir_research_core import Head, base_covariance
from run_encoder_architecture_v1 import digest, weight_hash, write_json
from run_frozen_incremental_v1 import grouped_metrics
from frozen_prototype_heads_v1 import FrozenPrototype

OLD = ['raw_svm', 'M1_1nn', 'M1_svm']
NEW = ['raw_1nn', 'pca64_1nn', 'M1_proto_euclidean', 'M1_proto_cosine', 'M1_proto_shared', 'M1_proto_shrinkage']
PROTOTYPES = {f'M1_proto_{k}': k for k in ['euclidean', 'cosine', 'shared', 'shrinkage']}
SOURCES = ['run_external_prototype_v1.py', 'analyze_external_prototype_v1.py', 'frozen_prototype_heads_v1.py',
           'audit_external_ftir_v1.py', 'ftir_research_core.py', 'run_frozen_incremental_v1.py',
           'ftir_research_data.py', 'encoder_architecture_models.py', 'run_encoder_architecture_v1.py', 'utils.py', 'PLS.py']


def fit_base_pca(x, train, seed):
    return PCA(n_components=64, svd_solver='randomized', random_state=seed, whiten=False).fit(x[train])


def feature_storage(x, labels, head=None):
    extra = 0
    if head is not None and head.kind == 'svm':
        extra = sum(getattr(head.estimator, key).nbytes for key in
                    ['support_vectors_', 'dual_coef_', 'intercept_', 'support_'])
    return {'stored_feature_bank_bytes': x.nbytes, 'class_metric_bytes': extra, 'global_metric_bytes': 0,
            'count_bytes': 0, 'total_numeric_state_bytes': x.nbytes + extra,
            'label_array_bytes_excluded': labels.nbytes}


def run(args):
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()
    out, parent = args.output.resolve(), args.parent.resolve()
    audit_root = args.audit.resolve()
    audit = json.loads((audit_root / 'd4_audit.json').read_text())
    assert audit['external_evaluations_performed'] == 0 and audit['transfer_status'] == 'blocked'
    previous = json.loads((parent / 'experiment_manifest.json').read_text())
    assert (parent / 'COMPLETE.json').exists() and not previous['smoke']
    architecture = Path(previous['parent'])
    assert digest(architecture / 'experiment_manifest.json') == previous['parent_manifest_sha256']
    tasks = previous['tasks'][:1] if args.smoke else previous['tasks']
    seeds, shots = ([0], [1, 5]) if args.smoke else (previous['seeds'], previous['shots'])
    repeats, benchmark_repeats = (2, 2) if args.smoke else (previous['repeats'], 20)
    config = {'parent_incremental': str(parent), 'parent_manifest_sha256': digest(parent / 'experiment_manifest.json'),
        'architecture_parent': str(architecture), 'audit_root': str(audit_root),
        'd4_audit_sha256': digest(audit_root / 'd4_audit.json'), 'external_status': 'blocked_no_evaluation',
        'tasks': tasks, 'seeds': seeds, 'shots': shots, 'repeats': repeats, 'benchmark_repeats': benchmark_repeats,
        'smoke': args.smoke, 'pca': {'components': 64, 'fit': 'archived base_train only', 'solver': 'randomized', 'whiten': False},
        'mahalanobis': {'reg': 1e-3, 'tau': 5., 'prior': 'within-class residual covariance on archived base_train only',
            'formula': 'existing Head shared_prior/shrinkage; alpha=(n-1)/(n-1+tau), class covariance divisor n'},
        'cosine': 'cosine of unnormalized support mean; zero norm -> zero normalized vector; canonical class tie order',
        'hyperparameter_selection': 'none new; locked existing defaults; existing SVM C reused',
        'data_access': 'exact saved sessions/exemplars/queries; PCA and covariance may fit full base_train; no query/future fitting',
        'base_query_status': previous['base_query_status'], 'numerical_threads': 1, 'tensorflow_threads': 4,
        'storage_scope': 'float64 feature/state bytes; metric matrices counted; transform bytes separate; Python overhead excluded',
        'source_hashes': {p: digest(ROOT / p) for p in SOURCES}}
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'experiment_manifest.json'
    if manifest.exists():
        assert args.resume and json.loads(manifest.read_text()) == config
    else:
        assert not (out / 'metrics.csv').exists()
        write_json(manifest, config)
        (out / 'source').mkdir(exist_ok=True)
        for p in SOURCES:
            (out / 'source' / p).write_bytes((ROOT / p).read_bytes())
    if not args.smoke:
        assert args.validated_smoke and (args.validated_smoke / 'COMPLETE.json').exists()
        checked = json.loads((args.validated_smoke / 'experiment_manifest.json').read_text())
        assert checked['source_hashes'] == config['source_hashes']
    data = load_dataset('c4')
    assert data.report['source_sha256'] == audit['c4_source_sha256']
    x = preprocess(data, np.load(architecture / 'common_grid.npy'), ROOT / 'results' / 'research_cache', baseline=True, flip=True).astype(float)
    labels = data.labels
    for task in tasks:
        for seed in seeds:
            old = parent / task['name'] / f'seed_{seed}'
            arch = architecture / task['name'] / f'seed_{seed}'
            folder = out / task['name'] / f'seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'COMPLETE.json').exists():
                continue
            sessions = [s for s in json.loads((old / 'sessions.json').read_text()) if s['shot'] in shots and s['repeat'] < repeats]
            sample = pd.read_csv(arch / 'sample_manifest.csv')
            assert np.array_equal(sample['class'].to_numpy(), labels)
            train = np.flatnonzero(sample.role.to_numpy() == 'base_train')
            assert set(labels[train]) == set(task['base'])
            model_file = arch / 'M1' / 'encoder.keras'
            model_hash = digest(model_file)
            model = tf.keras.models.load_model(str(model_file), compile=False)
            model.trainable = False
            before = weight_hash(model)
            assert before == json.loads((arch / 'M1' / 'training_costs.json').read_text())['weight_hash_after']
            with np.load(arch / 'M1' / 'embeddings.npz') as saved:
                embedding = saved['embeddings']
            infer = tf.function(lambda a: model(a, training=False), input_signature=[tf.TensorSpec([None, 1868], tf.float32)])
            probe_x = tf.convert_to_tensor(x[:8], dtype=tf.float32)
            for _ in range(3):
                probe = infer(probe_x).numpy()
            np.testing.assert_allclose(probe, embedding[:8], rtol=2e-4, atol=2e-4)
            started = time.perf_counter()
            pca = fit_base_pca(x, train, seed)
            pca_fit_ms = (time.perf_counter() - started) * 1000
            pca_features = pca.transform(x)
            np.savez_compressed(folder / 'frozen_pca64.npz', mean=pca.mean_, components=pca.components_,
                                explained_variance=pca.explained_variance_, fit_indices=train)
            views = {'raw': x, 'pca64': pca_features, 'M1': embedding}
            started = time.perf_counter()
            prior = base_covariance(embedding[train], labels[train])
            prior_fit_ms = (time.perf_counter() - started) * 1000
            np.save(folder / 'base_covariance.npy', prior)
            base_support = np.asarray(sessions[0]['support'])
            templates, initial_cost = {}, []
            for method, kind in PROTOTYPES.items():
                started = time.perf_counter()
                template = FrozenPrototype(kind, 128, prior=prior)
                for c in task['base']:
                    template.register(c, embedding[base_support[labels[base_support] == c]])
                templates[method] = template
                initial_cost.append({'method': method, 'base_head_initialization_ms': (time.perf_counter() - started) * 1000})
            transform_bytes = {'raw': 0, 'pca64': pca.components_.nbytes + pca.mean_.nbytes,
                               'M1': sum(v.nbytes for v in model.get_weights())}
            parent_metrics = pd.read_csv(old / 'metrics.csv')
            parent_metrics = parent_metrics[parent_metrics.shot.isin(shots) & (parent_metrics['repeat'] < repeats)
                                            & parent_metrics.method.isin(OLD)].copy()
            parent_metrics['metric_origin'] = 'reused_parent'
            expected = parent_metrics.set_index(['shot', 'repeat', 'session', 'method'])
            previous_class = pd.read_csv(old / 'per_class_metrics.csv')
            previous_class = previous_class[previous_class.shot.isin(shots) & (previous_class['repeat'] < repeats)
                                            & previous_class.method.isin(OLD)]
            records, class_rows, costs, cms = [], [], [], {}
            c_tuning = pd.read_csv(arch / 'svm_base_tuning.csv')
            cs = {}
            for k in shots:
                for view in ['raw', 'M1']:
                    candidates = c_tuning[(c_tuning.shot == k) & (c_tuning['view'] == view)]
                    cs[view, k] = float(candidates.loc[candidates.base_val_macro_f1.idxmax(), 'c'])
            for context in sessions:
                k, repeat, session = context['shot'], context['repeat'], context['session']
                if session == 0:
                    heads = {name: template.clone() for name, template in templates.items()}
                s, q, added = np.asarray(context['support']), np.asarray(context['query']), context['added']
                assert not set(s).intersection(q)
                assert set(s[np.isin(labels[s], task['base'])]) == set(base_support)
                ordered = np.concatenate([s[labels[s] == c] for c in sorted(context['seen'])])
                feature_ms = {'raw': 0., 'pca64': 0., 'M1': 0.}
                benchmark = repeat < benchmark_repeats
                if benchmark and added:
                    tensor = tf.convert_to_tensor(x[added], dtype=tf.float32)
                    started = time.perf_counter(); encoded = infer(tensor).numpy()
                    feature_ms['M1'] = (time.perf_counter() - started) * 1000
                    started = time.perf_counter(); projected = pca.transform(x[added])
                    feature_ms['pca64'] = (time.perf_counter() - started) * 1000
                    if repeat == 0:
                        np.testing.assert_allclose(encoded, embedding[added], rtol=2e-4, atol=2e-4)
                        np.testing.assert_allclose(projected, pca_features[added], rtol=1e-9, atol=1e-9)
                for method in NEW + (OLD if benchmark else []):
                    prototype = method in PROTOTYPES
                    view = 'M1' if prototype else method.split('_')[0]
                    started = time.perf_counter()
                    if prototype:
                        head = heads[method]
                        if added:
                            head.register(context['current_novel'][0], embedding[added])
                    else:
                        kind = 'svm' if method.endswith('_svm') else '1nn'
                        support_x, support_y = views[view][ordered], labels[ordered]
                        head = Head(kind, c=cs.get((view, k), 1.)).fit(support_x, support_y)
                    update_ms = (time.perf_counter() - started) * 1000
                    if prototype:
                        storage = head.storage()
                        storage['label_array_bytes_excluded'] = 0
                    else:
                        storage = feature_storage(support_x, support_y, head)
                    started = time.perf_counter(); prediction = head.predict(views[view][q])
                    predict_ms = (time.perf_counter() - started) * 1000
                    meta = {'task': task['name'], 'seed': seed, 'shot': k, 'repeat': repeat,
                            'session': session, 'method': method}
                    metrics, report, cm = grouped_metrics(labels[q], prediction, context['seen'], task['base'],
                                                        context['previous_novel'], context['current_novel'])
                    if method in OLD:
                        parent_row = expected.loc[k, repeat, session, method]
                        for key, value in metrics.items():
                            if key in parent_row and np.isfinite(value):
                                np.testing.assert_allclose(value, parent_row[key], atol=1e-12, rtol=1e-12)
                    else:
                        records.append({**meta, **metrics, 'support_count': len(s), 'query_count': len(q),
                                        'metric_origin': 'new_head'})
                        class_rows.extend({**meta, 'class': c, **report[c]} for c in context['seen'])
                        key = f'{method}_{k}shot_session{session}'
                        cms[key] = cms.get(key, np.zeros_like(cm)) + cm
                    if benchmark:
                        costs.append({**meta, **storage, 'global_transform_bytes': transform_bytes[view],
                            'total_with_transform_bytes': storage['total_numeric_state_bytes'] + transform_bytes[view],
                            'new_feature_encoding_ms': feature_ms[view], 'head_update_ms': update_ms,
                            'new_class_registration_ms': feature_ms[view] + update_ms if added else np.nan,
                            'svm_full_refit_and_assembly_ms': update_ms if method.endswith('_svm') else np.nan,
                            'prediction_ms': predict_ms, 'query_count': len(q), 'support_count': len(s)})
                if session == 2 and (repeat + 1) % 50 == 0:
                    print(f'{task["name"]}/seed {seed}/{k}-shot {repeat + 1}/{repeats}', flush=True)
            assert weight_hash(model) == before and digest(model_file) == model_hash
            pd.concat([parent_metrics, pd.DataFrame(records)], ignore_index=True).to_csv(folder / 'metrics.csv', index=False)
            pd.concat([previous_class, pd.DataFrame(class_rows)], ignore_index=True).to_csv(folder / 'per_class_metrics.csv', index=False)
            pd.DataFrame(costs).to_csv(folder / 'costs.csv', index=False)
            np.savez_compressed(folder / 'confusions.npz', **cms)
            write_json(folder / 'preparation_costs.json', {'pca_fit_ms': pca_fit_ms, 'prior_fit_ms': prior_fit_ms,
                'base_initialization': initial_cost, 'fit_indices': train.tolist(), 'transform_bytes': transform_bytes})
            write_json(folder / 'reuse_audit.json', {'session_file_sha256': digest(old / 'sessions.json'),
                'parent_metrics_sha256': digest(old / 'metrics.csv'), 'checkpoint_path': str(model_file),
                'checkpoint_before_after_sha256': model_hash, 'weight_hash_before_after': before,
                'cached_embedding_sha256': digest(arch / 'M1' / 'embeddings.npz'),
                'verified_parent_head_episodes': benchmark_repeats * len(shots) * 3 * len(OLD),
                'encoder_training_calls': 0, 'prior_and_pca_base_train_only': True})
            write_json(folder / 'COMPLETE.json', {'complete': True})
            print(f'Completed baseline/prototype trial {task["name"]}/seed {seed}', flush=True)
            tf.keras.backend.clear_session()
    from analyze_external_prototype_v1 import analyze
    analyze(out)
    write_json(out / 'COMPLETE.json', {'complete': True, 'd4_status': 'blocked', 'external_evaluations': 0,
                                     'encoder_training_calls': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parent', type=Path, default=ROOT / 'results' / 'frozen_incremental_v1')
    p.add_argument('--audit', type=Path, default=ROOT / 'results' / 'external_transfer_prototype_v1')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--validated-smoke', type=Path)
    p.add_argument('--resume', action='store_true')
    with threadpool_limits(limits=1):
        run(p.parse_args())
