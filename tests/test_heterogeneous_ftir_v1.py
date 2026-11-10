import tempfile
import unittest
from pathlib import Path
import numpy as np
from heterogeneous_ftir_adapter_v1 import observed_mask,base_input_mean,apply_missing_mask,partial_input,fast_metrics
from ftir_research_data import Spectra,preprocess
from run_frozen_incremental_v1 import grouped_metrics


class HeterogeneousTests(unittest.TestCase):
    def test_no_query_fit_and_exact_mask_values(self):
        x = np.arange(80,dtype=np.float32).reshape(10,8)
        train = np.array([0,2,3]); mask=np.array([False,True,True,True,True,True,True,False])
        mean=base_input_mean(x,train)
        poisoned=x.copy(); poisoned[4:]+=10000
        np.testing.assert_array_equal(mean,base_input_mean(poisoned,train))
        adapted=apply_missing_mask(x,mask,mean)
        np.testing.assert_array_equal(adapted[:,mask],x[:,mask])
        np.testing.assert_array_equal(adapted[:,~mask],np.broadcast_to(mean[~mask],adapted[:,~mask].shape))
        np.testing.assert_array_equal(x,np.arange(80,dtype=np.float32).reshape(10,8))

    def test_preprocessing_precedes_imputation_and_no_extrapolation(self):
        wave=np.linspace(600,3995,100)
        x=np.stack([20+np.sin(wave/100),30+np.cos(wave/80)])
        data=Spectra('synthetic_partial',wave,x,np.array(['A','B']),np.array(['1','2']),np.array(['h1','h2']),
                     dict(source_sha256='synthetic_test',default_flip=False,trusted_range=None))
        grid=np.linspace(399,4000,1868);mask=observed_mask(wave,grid);mean=np.linspace(.1,.5,1868).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            adapted,actual=partial_input(data,grid,mean,Path(tmp))
            legacy=preprocess(data,grid[mask],Path(tmp),baseline=True,flip=True)
        np.testing.assert_array_equal(actual,mask)
        np.testing.assert_array_equal(adapted[:,mask],legacy)
        np.testing.assert_array_equal(adapted[:,~mask],np.broadcast_to(mean[~mask],adapted[:,~mask].shape))

    def test_group_metrics_match_existing_utils(self):
        rng=np.random.default_rng(72);classes=np.array(['A','B','C','D'])
        for _ in range(5):
            true=rng.choice(classes,80);pred=rng.choice(classes,80)
            a,report,cm=fast_metrics(true,pred,classes,['A','B'],['C'],['D'])
            b,old,oldcm=grouped_metrics(true,pred,list(classes),['A','B'],['C'],['D'])
            for key,value in a.items():
                np.testing.assert_allclose(value,b[key],atol=1e-14)
            np.testing.assert_array_equal(cm,oldcm)
            for c in classes:
                for key,value in report[c].items():
                    np.testing.assert_allclose(value,old[c][key],atol=1e-14)


if __name__=='__main__':
    unittest.main()
