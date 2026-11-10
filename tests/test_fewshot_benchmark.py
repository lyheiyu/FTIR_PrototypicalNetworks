"""Numerical correctness and a synthetic, non-training end-to-end check."""

import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from fewshot_benchmark import (
    CLASSES, DISTANCES, evaluate, metric_predictions, quadratic_distances,
    sample_episode, split_indices, tune_regularization,
    train_encoder,
)
from plot_fewshot_benchmark import plot_results
from utils import utils


def synthetic_run(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    y = np.repeat(np.arange(6), 50)
    means = rng.normal(size=(6, 8))
    z = means[y] + rng.normal(scale=0.8, size=(len(y), 8))
    args = SimpleNamespace(episodes=3, queries=2, val_episodes=2,
                           reg_grid=[1e-3, 1e-2], no_raw_baselines=False)
    config = {'datasets': ['c4'], 'seeds': [0, 1], 'shots': [1, 5], 'class_names': CLASSES,
              'synthetic_test_only': True}
    (output / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    summaries = []
    for seed in config['seeds']:
        folder = output / 'c4' / f'seed_{seed}'
        folder.mkdir(parents=True)
        train, val, test = split_indices(y, seed)
        for k in config['shots']:
            regs, _ = tune_regularization(z, y, val, k, args.queries, args, seed)
            summary = evaluate(z, z, y, test, args, seed, k, regs, folder)
            summary['dataset'], summary['seed'], summary['shot'] = 'c4', seed, k
            summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(output / 'seed_metrics.csv', index=False)
    plot_results(output)


class MetricTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow not installed')
    def test_tensorflow_training_and_checkpoint_smoke(self):
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(6), 20)
        x = rng.normal(size=(len(y), 64)).astype(np.float32)
        train, val, _ = split_indices(y, 0)
        args = SimpleNamespace(embedding_dim=8, learning_rate=1e-3, epochs=1,
                               batch_size=16, patience=1)
        with tempfile.TemporaryDirectory(prefix='ftir_training_test_') as temp:
            folder = Path(temp)
            z, version = train_encoder(x, y, train, val, 0, args, folder)
            self.assertEqual(z.shape, (len(y), 8))
            self.assertTrue(np.isfinite(z).all())
            self.assertTrue((folder / 'encoder.h5').exists())
            self.assertTrue((folder / 'best.weights.h5').exists())
            self.assertTrue(version)

    def test_full_quadratic_matches_dense_inverse(self):
        rng = np.random.default_rng(12)
        for n, dim in [(1, 8), (5, 128), (30, 12)]:
            a = rng.normal(size=(n, dim))
            a -= a.mean(axis=0)
            diff = rng.normal(size=(10, dim))
            for reg in [1e-4, 1e-3, 1e-2]:
                cov = a.T @ a / n + reg * np.eye(dim)
                expected = np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1)
                actual = quadratic_distances(diff, a, reg)
                np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-7)

    def test_off_diagonal_terms_are_used(self):
        centered = np.array([[1., 2.], [-1., -2.]])
        diff = np.array([[1., 2.], [2., 1.]])
        cov = centered.T @ centered / 2 + np.eye(2) * 0.1
        expected = np.einsum('mi,ij,mj->m', diff, np.linalg.inv(cov), diff)
        diagonal_only = np.einsum('md,dd,md->m', diff, np.linalg.inv(cov), diff)
        actual = quadratic_distances(diff, centered, 0.1)
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(np.allclose(actual, diagonal_only))

    def test_one_shot_reduces_to_euclidean(self):
        rng = np.random.default_rng(9)
        support = rng.normal(size=(6, 8))
        query = rng.normal(size=(35, 8))
        y = np.arange(6)
        expected = metric_predictions(support, y, query, 'euclidean')
        for method in DISTANCES[1:]:
            np.testing.assert_array_equal(expected, metric_predictions(support, y, query, method))

    def test_splits_and_support_query_are_disjoint(self):
        y = np.repeat(np.arange(6), 100)
        train, val, test = split_indices(y, 0)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(val) & set(test))
        self.assertEqual(set(np.concatenate([train, val, test])), set(range(len(y))))
        support, query = sample_episode(y, test, 5, 10, np.random.default_rng(2))
        self.assertFalse(set(support) & set(query))
        np.testing.assert_array_equal(np.bincount(y[support]), np.full(6, 5))
        np.testing.assert_array_equal(np.bincount(y[query]), np.full(6, 10))

    def test_metrics_preserve_true_prediction_direction(self):
        true = np.array([0, 0, 0, 1, 1, 2, 3, 4, 5])
        pred = np.array([0, 1, 1, 1, 1, 2, 3, 4, 5])
        metrics, report, cm = utils.classification_metrics(true, pred, list(range(6)), CLASSES)
        self.assertAlmostEqual(metrics['macro_precision'], precision_score(true, pred, average='macro'))
        self.assertAlmostEqual(metrics['macro_recall'], recall_score(true, pred, average='macro'))
        self.assertEqual(cm[0, 1], 2)
        self.assertAlmostEqual(report['PET']['recall'], 1 / 3)

    def test_synthetic_export_and_plotting(self):
        with tempfile.TemporaryDirectory(prefix='ftir_benchmark_test_') as temp:
            out = Path(temp)
            synthetic_run(out)
            self.assertTrue((out / 'figures' / 'shot_performance.pdf').exists())
            self.assertTrue((out / 'figures' / 'c4_5shot_confusion_matrices.png').exists())
            data = pd.read_csv(out / 'per_class_metrics.csv')
            self.assertEqual(len(data), 2 * 6 * 6)
            path = out / 'c4' / 'seed_0'
            episodes = json.loads((path / 'episodes_5shot.json').read_text())
            predictions = pd.read_csv(path / 'predictions_5shot.csv')
            self.assertEqual(len(predictions), 3 * 6 * 2)
            self.assertEqual(len(episodes), 3)
            with np.load(path / 'confusion_matrices_5shot.npz') as cms:
                for method in cms.files:
                    self.assertEqual(cms[method].sum(), len(predictions))


if __name__ == '__main__':
    unittest.main()
