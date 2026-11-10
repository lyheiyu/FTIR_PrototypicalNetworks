"""Read-only post-run checks of saved sampling and checkpoint provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from frozen_prototype_heads_v1 import FrozenPrototype
from run_full_mahalanobis_ablation_v1 import FullClassMahalanobis
from heterogeneous_ftir_adapter_v1 import fast_metrics
from threadpoolctl import threadpool_limits
from run_bidirectional_c4_c8_final import digest,write
from ftir_research_data import ROOT


def audit(out):
    cfg=json.loads((out/'experiment_manifest.json').read_text());parent=Path(cfg['parent'])
    labels=pd.read_csv(out/'paired_group_representatives.csv').label.to_numpy();rows=[]
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            f=out/task['name']/f'seed_{seed}';p=parent/task['name']/f'seed_{seed}'
            for name in ['split.json','episodes.json','sample_manifest.csv']:
                assert digest(f/name)==digest(p/name)
            split=json.loads((f/'split.json').read_text());episodes=json.loads((f/'episodes.json').read_text())[:cfg['repeats']]
            train=set(split['train']);val=set(split['validation']);assert not train&val
            data=np.load(f/'paired_training_episodes.npz');steps=int(np.ceil(len(train)/64))
            assert data['support'].shape==(cfg['epochs'],steps,20)
            assert data['query'].shape==(cfg['epochs'],steps,40)
            assert data['validation_support'].shape==(20,20) and data['validation_query'].shape==(20,16)
            for s,q in zip(data['support'].reshape(-1,20),data['query'].reshape(-1,40)):
                assert set(s)|set(q)<=train and len(set(s)|set(q))==60 and not set(s)&set(q)
                np.testing.assert_array_equal(labels[s],np.repeat(task['base'],5));np.testing.assert_array_equal(labels[q],np.repeat(task['base'],10))
            for s,q in zip(data['validation_support'],data['validation_query']):
                assert set(s)|set(q)<=val and len(set(s)|set(q))==36 and not set(s)&set(q)
                np.testing.assert_array_equal(labels[s],np.repeat(task['base'],5));np.testing.assert_array_equal(labels[q],np.repeat(task['base'],4))
            for ep in episodes:
                queries=set(sum(ep['query'].values(),[]));assert not queries&(train|val)
                supports=set(sum(ep['base_support'].values(),[]))|set(sum(ep['novel_support_order'].values(),[]))
                assert not queries&supports
                for c,s in ep['base_support'].items():assert set(s)<=train and set(labels[s])=={c}
                for c,s in ep['novel_support_order'].items():
                    assert set(s)<=set(split['novel_support'][c]) and set(labels[s])=={c}
                    assert not set(s)&(train|val)
            initial=[]
            newmetrics=pd.read_csv(f/'new_metrics.csv')
            for source in ['c4','c8']:
                new=f/source/'M1';ce=p/source/'M1';stats=json.loads((new/'training_costs.json').read_text())
                old=json.loads((ce/'PAIRED_TRAINING_COMPLETE.json').read_text())
                assert stats['initial_encoder_hash']==old['initial_encoder_hash'];initial.append(stats['initial_encoder_hash'])
                assert stats['parameters']==47400 and stats['optimizer_updates']==cfg['epochs']*steps
                hist=pd.read_csv(new/'training_history.csv');assert len(hist)==cfg['epochs']
                assert stats['best_epoch']==int(hist.loc[hist.val_loss.idxmin(),'epoch'])
                np.testing.assert_allclose(stats['best_source_validation_loss'],hist.val_loss.min(),rtol=1e-12,atol=1e-12)
                assert digest(new/'encoder.keras')==json.loads((new/'COMPLETE.json').read_text())['encoder_sha256']
                freeze=json.loads((new/'FROZEN_CHECK.json').read_text());assert freeze['weights_before']==freeze['weights_after']==stats['weight_hash']
                # Independent dense solves on real 128-D EP embeddings, paired C/D episode0, 5-shot final session.
                case,target=('C','c8') if source=='c4' else ('D','c4')
                features=np.load(new/'frozen_views.npz');ep=episodes[0];classes=sorted(set(labels));qi=np.array([i for c in classes for i in ep['query'][c]])
                query=features['M1_'+target][qi];prior=features['M1_prior'];novels=[c[0] for c in task['novel_sessions']]
                for kind in ['full','shrinkage']:
                    head=FullClassMahalanobis(128,reg=.001) if kind=='full' else FrozenPrototype('shrinkage',128,prior,reg=.001,tau=5.)
                    distances=[]
                    for c in classes:
                        indices=ep['base_support'][c] if c in task['base'] else ep['novel_support_order'][c][:5]
                        resolution=source if c in task['base'] else target
                        values=features['M1_'+resolution][indices];mean=values.mean(0);r=values-mean;cov=r.T@r/len(values)
                        head.register(c,values)
                        if kind=='shrinkage':
                            alpha=(len(values)-1)/(len(values)-1+5);cov=alpha*cov+(1-alpha)*prior
                        delta=query-mean;direct=np.sum(delta*np.linalg.solve(cov+.001*np.eye(128),delta.T).T,axis=1)
                        distances.append(direct)
                    direct_prediction=np.array(classes)[np.argmin(distances,axis=0)]
                    np.testing.assert_array_equal(direct_prediction,head.predict(query))
                    values,_,_=fast_metrics(labels[qi],direct_prediction,classes,task['base'],[novels[0]],[novels[1]])
                    row=newmetrics[(newmetrics.setting==case)&(newmetrics.shot==5)&(newmetrics.session==2)&
                        (newmetrics['repeat']==0)&(newmetrics.method=='EP_'+kind)].iloc[0]
                    for key in ['all_accuracy','all_macro_f1','base_accuracy','seen_novel_accuracy','previous_novel_accuracy','current_novel_accuracy']:
                        np.testing.assert_allclose(values[key],row[key],atol=1e-12)
            assert len(set(initial))==1
            rows.append(dict(task=task['name'],seed=seed,steps_per_epoch=steps,updates=cfg['epochs']*steps,
                training_episode_file_sha256=digest(f/'paired_training_episodes.npz'),split_sha256=digest(f/'split.json'),episodes_sha256=digest(f/'episodes.json'),
                train_episode_count=cfg['epochs']*steps,validation_episode_count=20,evaluation_repeat_count=len(episodes)))
    for name,h in cfg['source_hashes'].items():assert digest(ROOT/name)==h
    for name,h in cfg['supervised_checkpoint_hashes'].items():assert digest(ROOT/name)==h
    parent_cfg=json.loads((parent/'experiment_manifest.json').read_text())
    for name,h in parent_cfg['source_hashes'].items():assert digest(ROOT/name)==h
    pd.DataFrame(rows).to_csv(out/'sampling_provenance.csv',index=False)
    write(out/'sampling_audit.json',dict(status='passed',paired_trials=len(rows),source_hashes_unchanged=True,ce_checkpoints_unchanged=True,
        no_train_validation_query_overlap=True,no_novel_class_encoder_training_or_validation=True,all_training_and_validation_local_labels_verified=True,
        support_query_disjoint_within_every_saved_episode=True,checkpoint_selection_matches_minimum_source_validation_loss=True,
        paired_source_initialization=True,parent_preprocessing_and_evaluation_sources_unchanged=True,
        real_128D_full_and_shrinkage_dense_solves_match_predictions_and_saved_cross_resolution_metrics=True,
        physical_specimen_independence_verified=False))
    print('Sampling and checkpoint audit passed:',len(rows),'paired trials')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/protonet_bidirectional_final')
    with threadpool_limits(limits=1):audit(parser.parse_args().output.resolve())
