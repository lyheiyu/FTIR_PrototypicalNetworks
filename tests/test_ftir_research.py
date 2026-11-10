"""Leakage boundaries, physical grouping and incremental classifier tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ftir_research_core import Head, Recorder, base_covariance, plot_run, tune_heads
from ftir_research_data import Spectra, canonical_label, load_dataset, preprocess, REGISTRY
from run_ftir_research import (build_views, generate_tasks, make_protocol, representatives,
                               incremental, parser)


class ResearchTests(unittest.TestCase):
    def test_aliases_do_not_merge_chemically_ambiguous_labels(self):
        self.assertEqual(canonical_label('Poly(propylene)'), 'PP')
        self.assertEqual(canonical_label('polyethylene'), 'PE')
        self.assertEqual(len({canonical_label(x) for x in ['PE', 'HDPE', 'LDPE']}), 3)
        self.assertNotEqual(canonical_label('polyester'), canonical_label('PET'))
        self.assertNotEqual(canonical_label('Nylon'), canonical_label('PA'))

    def test_adapter_dedup_and_invalid_rows(self):
        with tempfile.TemporaryDirectory(prefix='ftir_adapter_') as temp:
            root = Path(temp); (root / 'dataset').mkdir()
            p = root / 'dataset' / 'test.csv'
            p.write_text('1000,1100,1200,1300,PolymerName,PolyID\n'
                         '1,2,3,4,PP,0\n1.0,2.00,3,4,PP,0\n'
                         '2,3,4,5,PS,1\nNaN,1,2,3,PS,1\n')
            profile = ('test.csv', 'fourth', 'test', False, None)
            with patch('ftir_research_data.ROOT', root), patch.dict(REGISTRY, {'test': profile}):
                result = load_dataset('test')
            self.assertEqual(len(result.x), 2)
            self.assertEqual(result.report['statuses'], {'retained': 2, 'duplicate': 1, 'invalid': 1})

    def test_protocol_base_only_and_fixed_queries(self):
        labels = np.repeat(np.array(list('ABCDEF')), 50)
        groups = np.asarray([f'g{i}' for i in range(len(labels))])
        pool = np.arange(len(labels))
        task = {'base': list('ABCD'), 'novel_sessions': [['E'], ['F']]}
        train, val, query, novel = make_protocol(labels, pool, pool, groups, task, 2, 5, True)
        self.assertEqual(set(labels[train]), set('ABCD'))
        self.assertEqual(set(labels[val]), set('ABCD'))
        query_indices = np.concatenate(list(query.values()))
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(query_indices))
        self.assertFalse(set(val) & set(query_indices))
        for c, indices in novel.items():
            self.assertFalse(set(indices) & set(query[c]))
            self.assertEqual(set(labels[indices]), {c})

    def test_verified_groups_take_one_representative(self):
        selected = representatives(np.array(['A', 'A', 'B']), np.array(['one', 'one', 'two']))
        np.testing.assert_array_equal(selected, [0, 2])
        with self.assertRaises(ValueError):
            representatives(np.array(['A', 'B']), np.array(['one', 'one']))

    def test_cross_domain_protocol_excludes_target_query_groups(self):
        source = np.repeat(np.array(list('ABCD')), 20)
        target = np.repeat(np.array(list('ABCDEF')), 12)
        labels = np.concatenate([source, target])
        source_pool, target_pool = np.arange(len(source)), np.arange(len(source), len(labels))
        groups = np.asarray([f's{i}' for i in range(len(labels))], dtype=object)
        # A target specimen also occurs in source: whichever target row is used
        # for query must not have its paired source spectrum in train/val.
        for c in list('ABCD'):
            s = source_pool[labels[source_pool] == c][:12]
            t = target_pool[labels[target_pool] == c]
            groups[t] = groups[s]
        task = {'base': list('ABCD'), 'novel_sessions': [['E'], ['F']]}
        train, val, query, _ = make_protocol(labels, source_pool, target_pool, groups, task, 1, 3, False)
        forbidden = set(groups[np.concatenate(list(query.values()))])
        self.assertFalse(set(groups[train]) & forbidden)
        self.assertFalse(set(groups[val]) & forbidden)
        self.assertTrue(set(train).issubset(set(source_pool)))

    def test_added_class_keeps_existing_class_statistics(self):
        x = np.array([[0., 0.], [1., 1.], [3., 4.], [4., 3.], [9., 9.]])
        labels = np.array(['A', 'A', 'B', 'B', 'C'])
        prior = base_covariance(x[:4], labels[:4])
        old = Head('shrinkage').fit(x[:4], labels[:4], prior)
        added = Head('shrinkage').fit(x, labels, prior)
        np.testing.assert_allclose(old.means, added.means[:2])
        np.testing.assert_allclose(old.inverses, added.inverses[:2])

    def test_base_prior_and_class_full_quadratic(self):
        x = np.array([[0., 0.], [1., 2.], [3., 2.], [4., 4.]])
        labels = np.array(['old', 'old', 'new', 'new'])
        prior = base_covariance(x, labels)
        head = Head('class', reg=.01).fit(x, labels, prior)
        query = np.array([[.8, .3], [4., 3.]])
        scores = []
        for c in head.classes:
            z = x[labels == c]; mean = z.mean(axis=0); a = z - mean
            cov = a.T @ a / len(z) + .01 * np.eye(2)
            diff = query - mean
            scores.append(np.einsum('mi,ij,mj->m', diff, np.linalg.inv(cov), diff))
        expected = head.classes[np.argmin(np.stack(scores, axis=1), axis=1)]
        np.testing.assert_array_equal(head.predict(query), expected)
        one = Head('shrinkage', reg=.01).fit(x[[0, 2]], labels[[0, 2]], prior)
        np.testing.assert_allclose(one.inverses[0], np.linalg.inv(prior + .01 * np.eye(2)))

    def test_pca_does_not_fit_future_classes(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(30, 16)); z = rng.normal(size=(30, 8))
        y = np.repeat(['baseA', 'baseB', 'future'], 10); train = np.arange(20)
        args = SimpleNamespace(pca_dim=4, l2_embedding=False)
        views, prior, pca = build_views(x, z, y, train, args)
        changed = x.copy(); changed[20:] += 10000
        _, prior2, pca2 = build_views(changed, z, y, train, args)
        np.testing.assert_allclose(pca.mean_, pca2.mean_)
        np.testing.assert_allclose(pca.components_, pca2.components_)
        np.testing.assert_allclose(prior, prior2)

    def test_task_orders_have_held_out_disjoint_classes(self):
        y = np.repeat(np.array(list('ABCDEF')), 50); pool = np.arange(len(y))
        args = SimpleNamespace(task_file=None, tasks=3, base_count=4, novel_sessions=2,
                               shots=[1, 5], queries=3, task_seed=1729)
        tasks = generate_tasks(y, y, pool, pool, args)
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            novel = {c for session in task['novel_sessions'] for c in session}
            self.assertFalse(set(task['base']) & novel)
            self.assertEqual(len(novel), 2)

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow required for training smoke')
    def test_incremental_end_to_end_with_novel_isolation(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.array(list('ABCDEF')), 24)
        wave = np.linspace(700, 3900, 32)
        raw = rng.normal(size=(len(labels), 32)) + np.repeat(np.arange(6), 24)[:, None] / 3
        data = Spectra('synthetic', wave, raw, labels,
                       np.asarray([str(i) for i in range(len(labels))]),
                       np.asarray([f'h{i}' for i in range(len(labels))]),
                       {'family': 'synthetic', 'source_sha256': 'synthetic-test-only',
                        'default_flip': False, 'trusted_range': None, 'row_audit': []})
        with tempfile.TemporaryDirectory(prefix='ftir_incremental_test_') as temp:
            output = Path(temp) / 'run'
            args = parser().parse_args(['incremental', '--output', str(output), '--dataset', 'c4',
                '--seeds', '0', '--tasks', '1', '--shots', '1', '5', '--queries', '3',
                '--epochs', '1', '--repeats', '2', '--val-episodes', '1', '--embedding-dim', '8',
                '--points', '32', '--pca-dim', '4', '--base-memory', '3',
                '--c-grid', '1', '--reg-grid', '.01', '--tau-grid', '5',
                '--methods', 'raw_1nn', 'pca_proto', 'embedding_svm', 'embedding_proto', 'embedding_shrinkage'])
            with patch('run_ftir_research.load_dataset', return_value=data), patch('run_ftir_research.preprocess', return_value=raw.astype(np.float32)):
                incremental(args)
            task = json.loads((output / 'tasks.json').read_text())['tasks'][0]
            folder = next(output.glob('synthetic/*/seed_0'))
            manifest = pd.read_csv(folder / 'split_manifest.csv')
            self.assertEqual(set(manifest.loc[manifest.role == 'base_train', 'class']), set(task['base']))
            self.assertTrue(json.loads((folder / 'FROZEN_CHECK.json').read_text())['unchanged'])
            episodes = json.loads((folder / 'sessions.json').read_text())
            for k in [1, 5]:
                for repeat in [0, 1]:
                    sessions = [r for r in episodes if r['shot'] == k and r['repeat'] == repeat]
                    for earlier, later in zip(sessions, sessions[1:]):
                        self.assertTrue(set(earlier['support']).issubset(later['support']))
                        self.assertTrue(set(earlier['query']).issubset(later['query']))
                        self.assertFalse(set(later['support']) & set(later['query']))
            self.assertTrue((output / 'COMPLETE.json').exists())
            self.assertTrue(list(output.glob('**/*session2_CM.pdf')))
            metrics = pd.read_csv(folder / 'metrics.csv')
            self.assertEqual(len(metrics), 2 * 2 * 3 * 5)
            self.assertTrue('old_accuracy_change_pp' in metrics)


if __name__ == '__main__':
    unittest.main()
