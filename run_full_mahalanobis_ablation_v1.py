"""Matched full class covariance ablation; never trains or changes an encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from frozen_prototype_heads_v1 import FrozenPrototype
from ftir_research_core import Head
from run_frozen_incremental_v1 import grouped_metrics

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / 'results/external_transfer_prototype_v1'
METHODS = ['M1_full_class_ridge001', 'M1_full_class_legacy002',
           'M1_proto_shared', 'M1_proto_shrinkage']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


class FullClassMahalanobis(FrozenPrototype):
    def __init__(self, dimension, reg=0.001):
        super().__init__('euclidean', dimension, reg=reg)
        self.kind = 'class'

    def register(self, label, x):
        x = np.asarray(x, dtype=np.float64)
        if label in self.centers or x.ndim != 2 or not len(x) or x.shape[1] != self.dimension:
            raise ValueError('One new class with nonempty support required')
        mean = x.mean(axis=0)
        residual = x - mean
        covariance = residual.T @ residual / len(x)
        self.precisions[label] = np.linalg.inv(covariance + self.reg * np.eye(self.dimension))
        self.centers[label], self.counts[label] = mean, len(x)


def self_check():
    rng = np.random.RandomState(51)
    x, q = rng.randn(10, 8), rng.randn(17, 8)
    labels = np.repeat([0, 1], 5)
    for reg in [.001, .002]:
        head = FullClassMahalanobis(8, reg)
        for c in [0, 1]:
            head.register(c, x[labels == c])
        original = Head('class', reg=reg).fit(x, labels)
        np.testing.assert_array_equal(head.predict(q), original.predict(q))
        residual = x[:5] - x[:5].mean(axis=0)
        cov = residual.T @ residual / 5
        # Legacy adds 0.001 I when estimating covariance and again at inversion.
        if reg == .002:
            np.testing.assert_allclose(head.precisions[0], np.linalg.inv(cov + .001*np.eye(8) + .001*np.eye(8)))
        copy = head.clone()
        copy.register(2, x[:1])
        assert 2 not in head.centers and copy.precisions[0] is head.precisions[0]
        np.testing.assert_allclose(copy.precisions[2], np.eye(8)/reg)
        assert np.linalg.matrix_rank(cov) <= 4


def run(args):
    self_check()
    parent = json.loads((PARENT/'experiment_manifest.json').read_text())
    arch = Path(parent['architecture_parent'])
    incremental = Path(parent['parent_incremental'])
    out = args.output.resolve()
    tasks = parent['tasks'][:1] if args.smoke else parent['tasks']
    seeds = [0] if args.smoke else parent['seeds']
    repeats = 2 if args.smoke else parent['repeats']
    config = dict(smoke=args.smoke, tasks=tasks, seeds=seeds, shots=parent['shots'], repeats=repeats,
        parent=str(PARENT), parent_manifest_sha256=digest(PARENT/'experiment_manifest.json'),
        source_sha256=digest(__file__), head_source_sha256=digest(ROOT/'frozen_prototype_heads_v1.py'),
        methods=METHODS, dimension=128, covariance_divisor='n', full_class_ridges=[.001,.002],
        legacy='Formula equivalent to original two 0.001 I additions; float64 cached features, not bitwise Torch float32 reproduction',
        shrinkage='alpha=(n-1)/(n-1+5); prior from archived base_train; ridge=.001',
        data_access='Exact archived support/query sessions; class-full uses class support only; shared/shrinkage also access original base_train covariance',
        limitations='Exploratory C4; base queries are previously used validation data; specimen grouping unknown',
        costs='Head registration/prediction on cached embeddings; first 20 repeats; one numerical thread; encoder/preprocessing excluded',
        encoder_training_steps=0, numerical_threads=1, hyperparameter_tuning='none')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'experiment_manifest.json').exists():
        assert json.loads((out/'experiment_manifest.json').read_text()) == config
    else:
        write(out/'experiment_manifest.json', config)
        (out/'source.py').write_bytes(Path(__file__).read_bytes())
    if not args.smoke:
        assert args.validated_smoke is not None
        check = json.loads((args.validated_smoke/'experiment_manifest.json').read_text())
        assert (args.validated_smoke/'COMPLETE.json').exists() and check['source_sha256'] == config['source_sha256']
    trial_paths = []
    for task in tasks:
        for seed in seeds:
            name = task['name']
            folder = out/name/f'seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            trial_paths.append(folder)
            if (folder/'COMPLETE.json').exists():
                continue
            old = PARENT/name/f'seed_{seed}'
            original = arch/name/f'seed_{seed}'
            checkpoint = original/'M1/encoder.keras'
            checkpoint_before = digest(checkpoint)
            z = np.load(original/'M1/embeddings.npz')['embeddings']
            labels = pd.read_csv(original/'sample_manifest.csv')['class'].to_numpy()
            sessions_file = incremental/name/f'seed_{seed}/sessions.json'
            contexts = [s for s in json.loads(sessions_file.read_text()) if s['repeat'] < repeats]
            prior = np.load(old/'base_covariance.npy')
            expected = pd.read_csv(old/'metrics.csv').set_index(['shot','repeat','session','method'])
            base_support = np.asarray(contexts[0]['support'])
            templates = dict(zip(METHODS, [FullClassMahalanobis(128,.001), FullClassMahalanobis(128,.002),
                FrozenPrototype('shared',128,prior), FrozenPrototype('shrinkage',128,prior)]))
            for head in templates.values():
                for c in task['base']:
                    head.register(c, z[base_support[labels[base_support] == c]])
            records, per_class, costs, cms = [], [], [], {}
            matched_checks = 0
            for context in contexts:
                k,r,stage = context['shot'],context['repeat'],context['session']
                if stage == 0:
                    heads = {method:h.clone() for method,h in templates.items()}
                support,query,added = context['support'],context['query'],context['added']
                assert not set(support).intersection(query)
                for method,head in heads.items():
                    start = time.perf_counter()
                    if added:
                        head.register(context['current_novel'][0],z[added])
                    update = (time.perf_counter()-start)*1000
                    start = time.perf_counter()
                    pred = head.predict(z[query])
                    prediction_ms = (time.perf_counter()-start)*1000
                    metrics, report, cm = grouped_metrics(labels[query],pred,context['seen'],task['base'],
                        context['previous_novel'],context['current_novel'])
                    meta = dict(task=name,seed=seed,shot=k,repeat=r,session=stage,method=method)
                    records.append(dict(meta,**metrics))
                    per_class.extend(dict(meta,**{'class':c},**report[c]) for c in context['seen'])
                    key = f'{method}_{k}shot_session{stage}'
                    cms[key] = cms.get(key,np.zeros_like(cm))+cm
                    if method in METHODS[2:]:
                        row = expected.loc[k,r,stage,method]
                        for key,value in metrics.items():
                            if key in row and np.isfinite(value):
                                np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12)
                        matched_checks += 1
                    if r < 20:
                        costs.append(dict(meta,head_update_ms=update,cached_query_prediction_ms=prediction_ms,**head.storage()))
            pd.DataFrame(records).to_csv(folder/'metrics.csv',index=False)
            pd.DataFrame(per_class).to_csv(folder/'per_class_metrics.csv',index=False)
            pd.DataFrame(costs).to_csv(folder/'costs.csv',index=False)
            np.savez_compressed(folder/'confusions.npz',**cms)
            assert digest(checkpoint) == checkpoint_before
            write(folder/'COMPLETE.json',dict(rows=len(records),matched_shared_shrinkage_checks=matched_checks,
                checkpoint_sha256_before=checkpoint_before,checkpoint_sha256_after=digest(checkpoint),
                embeddings_sha256=digest(original/'M1/embeddings.npz'),sessions_sha256=digest(sessions_file)))
            print(f'Completed {name} seed {seed}: {len(records)} rows; {matched_checks} matched controls',flush=True)
    for filename in ['metrics.csv','per_class_metrics.csv','costs.csv']:
        pd.concat([pd.read_csv(p/filename) for p in trial_paths],ignore_index=True).to_csv(out/filename,index=False)
    write(out/'COMPLETE.json',dict(trials=len(trial_paths),encoder_training_steps=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/mahalanobis_full_ablation_v1')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--validated-smoke',type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
