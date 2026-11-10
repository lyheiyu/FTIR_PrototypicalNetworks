import unittest
import json
import numpy as np
import pandas as pd
from run_frozen_incremental_v1 import ClassBank, sessions_for_episode, grouped_metrics


class FrozenIncrementalTests(unittest.TestCase):
    def test_archived_novel_episodes_are_extended_without_replacement(self):
        labels = np.repeat(list('ABCDEF'), 50)
        roles = np.tile(['base_train'] * 30 + ['base_validation'] * 20, 6)
        roles[200:] = np.tile(['novel_support_pool'] * 25 + ['novel_query_pool'] * 25, 2)
        samples = pd.DataFrame({'class': labels, 'role': roles})
        task = {'base': list('ABCD'), 'novel_sessions': [['E'], ['F']]}
        ep = {'episode': 0, 'support': [200, 201, 250, 251], 'query': [225, 226, 275, 276]}
        seq = sessions_for_episode(samples, task, ep, [0, 50, 100, 150], 0, 15)
        self.assertEqual(seq[1]['added'], [200, 201])
        self.assertEqual(seq[2]['added'], [250, 251])
        self.assertEqual(seq[2]['previous_novel'], ['E'])
        self.assertEqual(set(seq[2]['support'][4:]), set(ep['support']))
        self.assertEqual(set(seq[2]['query'][60:]), set(ep['query']))
        self.assertTrue(set(seq[0]['query']) <= set(seq[1]['query']) <= set(seq[2]['query']))
        json.dumps(seq)  # real-data manifests must remain portable JSON

    def test_group_f1_includes_false_positives_from_other_groups(self):
        true = np.array(['A', 'A', 'B', 'B'])
        pred = np.array(['A', 'A', 'A', 'B'])
        values, _, _ = grouped_metrics(true, pred, ['A', 'B'], ['A'], [], ['B'])
        self.assertEqual(values['base_accuracy'], 1.)
        self.assertAlmostEqual(values['base_macro_f1'], .8)
        self.assertTrue(np.isnan(values['previous_novel_accuracy']))

    def test_local_bank_copies_new_class_and_preserves_previous(self):
        bank = ClassBank()
        features = np.ones((2, 3))
        bank.register('A', [1, 2], features)
        features[:] = 0
        old_entry = bank.entries['A']
        bank.register('B', [3], np.zeros((1, 3)))
        self.assertIs(bank.entries['A'], old_entry)
        x, y = bank.materialize()
        np.testing.assert_array_equal(x[:2], np.ones((2, 3)))
        self.assertEqual(y.tolist(), ['A', 'A', 'B'])
        with self.assertRaises(ValueError):
            bank.register('A', [4], np.ones((1, 3)))


if __name__ == '__main__':
    unittest.main()
