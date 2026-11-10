import json
import unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ftir_research_core import Head, base_covariance
from frozen_prototype_heads_v1 import FrozenPrototype
from run_external_prototype_v1 import fit_base_pca


class ExternalPrototypeTests(unittest.TestCase):
    def test_existing_euclidean_shared_shrinkage_formulas_are_preserved(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(9, 5)); labels = np.repeat(['A', 'B', 'C'], [1, 3, 5])
        prior = base_covariance(rng.normal(size=(40, 5)), np.repeat(['oldA', 'oldB'], 20))
        q = rng.normal(size=(60, 5))
        with threadpool_limits(limits=1):
            for kind, original in [('euclidean', 'proto'), ('shared', 'shared_prior'), ('shrinkage', 'shrinkage')]:
                current = FrozenPrototype(kind, 5, prior)
                for c in sorted(set(labels)):
                    current.register(c, x[labels == c])
                reference = Head(original).fit(x, labels, prior)
                np.testing.assert_array_equal(current.predict(q), reference.predict(q))
                if kind == 'shrinkage':
                    np.testing.assert_allclose(current.precisions['A'], np.linalg.inv(prior + .001*np.eye(5)))

    def test_new_class_preserves_old_statistics_and_clone_isolation(self):
        prior = np.eye(3)
        base = FrozenPrototype('shrinkage', 3, prior)
        base.register('A', np.array([[1., 2., 3.], [3., 2., 1.]]))
        clone = base.clone(); old_mean, old_precision = clone.centers['A'].copy(), clone.precisions['A'].copy()
        clone.register('B', np.ones((1, 3)))
        np.testing.assert_array_equal(clone.centers['A'], old_mean)
        np.testing.assert_array_equal(clone.precisions['A'], old_precision)
        self.assertNotIn('B', base.centers)
        with self.assertRaises(ValueError):
            clone.register('A', np.ones((1, 3)))

    def test_cosine_definition_and_zero_norm(self):
        head = FrozenPrototype('cosine', 2)
        head.register('A', np.array([[10., 0.], [0., 2.]]))
        expected = np.array([5., 1.]); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(head.centers['A'], expected)
        head.register('B', np.zeros((1, 2)))
        self.assertEqual(head.predict(np.array([[10., 2.]])).tolist(), ['A'])
        self.assertEqual(head.predict(np.zeros((1, 2))).tolist(), ['A'])

    def test_dense_metric_storage_is_included(self):
        shared = FrozenPrototype('shared', 128, np.eye(128))
        shrink = FrozenPrototype('shrinkage', 128, np.eye(128))
        with threadpool_limits(limits=1):
            for head in [shared, shrink]:
                head.register('A', np.ones((2, 128)))
        self.assertEqual(shared.storage()['global_metric_bytes'], 128*128*8)
        self.assertEqual(shrink.storage()['class_metric_bytes'], 128*128*8)
        self.assertGreater(shrink.storage()['total_numeric_state_bytes'], 2*128*128*8)

    def test_pca_fitting_is_invariant_to_future_query_values(self):
        rng = np.random.default_rng(0); x = rng.normal(size=(110, 90)); train = np.arange(80)
        altered = x.copy(); altered[80:] += 10000
        with threadpool_limits(limits=1):
            a, b = fit_base_pca(x, train, 0), fit_base_pca(altered, train, 0)
        np.testing.assert_allclose(a.mean_, b.mean_)
        np.testing.assert_allclose(a.components_, b.components_)
        self.assertEqual(a.n_components_, 64)

    def test_audited_d4_coverage_blocks_external_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        audit = json.loads((root/'results/external_transfer_prototype_v1/d4_audit.json').read_text())
        self.assertEqual(audit['transfer_status'], 'blocked')
        self.assertEqual(audit['grid_points_below_d4'] + audit['grid_points_above_d4'], 107)
        self.assertEqual(audit['external_evaluations_performed'], 0)


if __name__ == '__main__':
    unittest.main()
