"""Capacity, position, serialization, paired splits and unseen-class isolation."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

TF = importlib.util.find_spec('tensorflow') is not None
if TF:
    import tensorflow as tf
    from encoder_architecture_models import build_encoder, capacity_check, SpectralBinPooling
    from run_encoder_architecture_v1 import build_manifest, weight_hash


@unittest.skipUnless(TF, 'Requires TensorFlow environment')
class ArchitectureTests(unittest.TestCase):
    def test_capacity_and_embedding_dimensions(self):
        tf.keras.utils.set_random_seed(0)
        counts = capacity_check()
        self.assertEqual(counts['M0'], 47872)
        self.assertEqual(counts['M1'], 47400)
        self.assertLess(abs(counts['relative_difference']), .1)
        for name in ['M0','M1']:
            self.assertEqual(build_encoder(name).output_shape, (None,128))

    def test_position_bins_keep_separate_locations(self):
        layer = SpectralBinPooling(4)
        x = np.zeros((1,16,1),np.float32); x[0,:4,0] = 1
        shifted = np.roll(x,4,axis=1)
        a,b = layer(x).numpy(),layer(shifted).numpy()
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_allclose(a[0,:,0], [1,0,0,0])
        np.testing.assert_allclose(b[0,:,0], [0,1,0,0])

    def test_serialization_and_frozen_weights(self):
        encoder = build_encoder('M1')
        x = np.random.default_rng(0).normal(size=(2,1868)).astype(np.float32)
        before = weight_hash(encoder); expected = encoder(x,training=False).numpy()
        self.assertEqual(before,weight_hash(encoder))
        with tempfile.TemporaryDirectory(prefix='ftir_m1_serialization_') as temp:
            file = Path(temp)/'encoder.keras'; encoder.save(str(file))
            loaded = tf.keras.models.load_model(str(file),compile=False)
            np.testing.assert_allclose(expected,loaded(x,training=False).numpy(),rtol=1e-6,atol=1e-6)

    def test_paired_manifest_and_novel_isolation(self):
        labels = np.repeat(np.array(list('ABCDEF')),80)
        task = {'base':list('ABCD'),'novel_sessions':[['E'],['F']]}
        args = SimpleNamespace(shots=[1,5,10,20],queries=15,val_episodes=2,episodes=3)
        a = build_manifest(labels,task,0,args); b = build_manifest(labels,task,0,args)
        np.testing.assert_array_equal(a[0],b[0]); self.assertEqual(a[3],b[3])
        train,val,roles,episodes = a
        self.assertEqual(set(labels[train]),set('ABCD'))
        self.assertFalse(set(train)&set(val))
        for ep in episodes:
            self.assertFalse(set(ep['support'])&set(ep['query']))
            if ep['role']=='unseen_evaluation':
                self.assertEqual(set(labels[ep['query']]),set('EF'))
                self.assertTrue(np.all(roles[ep['support']]=='novel_support_pool'))
                self.assertTrue(np.all(roles[ep['query']]=='novel_query_pool'))
            else:
                self.assertTrue(np.all(roles[ep['support']]=='base_validation'))
                self.assertTrue(np.all(roles[ep['query']]=='base_validation'))


if __name__ == '__main__': unittest.main()
