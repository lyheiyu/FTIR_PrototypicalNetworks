"""Read-only verification of archived inputs and completed full-covariance study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_full_mahalanobis_ablation_v1 import ROOT, PARENT, digest, write, self_check
from ftir_research_core import base_covariance


def main():
    out = ROOT/'results/mahalanobis_full_ablation_v1'
    assert (out/'COMPLETE.json').exists()
    config = json.loads((out/'experiment_manifest.json').read_text())
    parent = json.loads((PARENT/'experiment_manifest.json').read_text())
    assert config['source_sha256']==digest(ROOT/'run_full_mahalanobis_ablation_v1.py')==digest(out/'source.py')
    assert config['parent_manifest_sha256']==digest(PARENT/'experiment_manifest.json')
    for name,value in parent['source_hashes'].items():
        assert digest(ROOT/name)==value, name
    self_check()
    arch,incremental = Path(parent['architecture_parent']),Path(parent['parent_incremental'])
    ranks,checks,metric_checks = [],0,0
    data = pd.read_csv(out/'metrics.csv')
    assert len(data)==3*3*4*200*3*4 and not data.duplicated(['task','seed','shot','repeat','session','method']).any()
    for task in config['tasks']:
        for seed in config['seeds']:
            name = task['name']
            folder = out/name/f'seed_{seed}'
            completed = json.loads((folder/'COMPLETE.json').read_text())
            original = arch/name/f'seed_{seed}'
            assert digest(original/'M1/encoder.keras')==completed['checkpoint_sha256_before']==completed['checkpoint_sha256_after']
            previous_frozen = json.loads((incremental/name/f'seed_{seed}/frozen_check.json').read_text())['M1']
            assert previous_frozen['file_after']==completed['checkpoint_sha256_before']
            assert json.loads((original/'M1/training_costs.json').read_text())['parameters']==47400
            assert digest(original/'M1/embeddings.npz')==completed['embeddings_sha256']
            sessions_path = incremental/name/f'seed_{seed}/sessions.json'
            assert digest(sessions_path)==completed['sessions_sha256']
            z = np.load(original/'M1/embeddings.npz')['embeddings']
            sample = pd.read_csv(original/'sample_manifest.csv')
            labels = sample['class'].to_numpy()
            train = np.flatnonzero(sample.role.to_numpy()=='base_train')
            assert set(labels[train])==set(task['base'])
            np.testing.assert_allclose(base_covariance(z[train],labels[train]),np.load(PARENT/name/f'seed_{seed}/base_covariance.npy'),rtol=1e-12,atol=1e-12)
            contexts = json.loads(sessions_path.read_text())
            for context in contexts:
                s,q = context['support'],context['query']
                assert not set(s)&set(q)
                assert set(labels[s])==set(context['seen'])
                assert set(labels[q])==set(context['seen'])
                if context['repeat']==0 and context['session']==2:
                    for c in context['seen']:
                        features = z[np.asarray(s)[labels[s]==c]]
                        n = len(features)
                        residual = features-features.mean(axis=0)
                        covariance = residual.T@residual/n
                        rank = int(np.linalg.matrix_rank(covariance))
                        assert rank<=n-1 and rank<128
                        ranks.append(dict(task=name,seed=seed,shot=context['shot'],**{'class':c},support_count=n,empirical_rank=rank,dimension=128))
            metric_checks += completed['matched_shared_shrinkage_checks']
            checks += 1
    pd.DataFrame(ranks).to_csv(out/'covariance_rank_audit.csv',index=False)
    write(out/'protocol_audit.json',dict(status='passed',trials=checks,rows=len(data),matched_parent_context_method_checks=metric_checks,
        all_previous_sources_unchanged=True,checkpoints_unchanged=True,base_prior_recomputed_base_only=True,
        all_support_query_sessions_hash_verified=True,encoder_training_steps=0,
        rank_check='First final-session episode for every task/seed/shot/class; all singular before ridge'))
    print('Protocol audit passed',checks,'trials;',metric_checks,'matched parent controls;',len(ranks),'rank checks')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
