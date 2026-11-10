"""Reuse existing full-class heads on the EXACT heterogeneous-transfer episodes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_full_mahalanobis_ablation_v1 import FullClassMahalanobis
from run_heterogeneous_ftir_transfer_v1 import ROOT,ARCH,INC,Recorder,digest,write

METHODS={'M1_full_class_ridge001':.001,'M1_full_class_legacy002':.002}


def fit(method,z,labels,s):
    head=FullClassMahalanobis(128,METHODS[method])
    for c in sorted(set(labels[s])):head.register(c,z[s[labels[s]==c]])
    return head


def run(out):
    parent=json.loads((out/'experiment_manifest.json').read_text())
    assert (out/'COMPLETE.json').exists() and not parent['smoke']
    root=out/'full_mahalanobis_control';root.mkdir(exist_ok=True)
    config=dict(parent_manifest_sha256=digest(out/'experiment_manifest.json'),methods=METHODS,
        source_sha256=digest(__file__),head_source_sha256=digest(ROOT/'run_full_mahalanobis_ablation_v1.py'),
        protocol='Exact same supports/queries/adapter/frozen features as seven-head parent; no query tuning; no encoder operations',
        covariance='Class support only; divisor n; fixed ridge .001 and original effective .002; cached inverse; float64 formula reproduction',
        costs='head only on cached embeddings; within episode fit all classes, incremental registration arriving class only; first20 repeats; encoder excluded')
    if (root/'experiment_manifest.json').exists():assert json.loads((root/'experiment_manifest.json').read_text())==config
    else:write(root/'experiment_manifest.json',config);(root/'source.py').write_bytes(Path(__file__).read_bytes())
    folders=[]
    for task in parent['tasks']:
        for seed in parent['seeds']:
            name=task['name'];folder=root/name/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);folders.append(folder)
            if (folder/'COMPLETE.json').exists():continue
            original=ARCH/name/f'seed_{seed}';trial=out/name/f'seed_{seed}'
            z4=np.load(original/'M1/embeddings.npz')['embeddings'];labels=pd.read_csv(original/'sample_manifest.csv')['class'].to_numpy()
            features=np.load(trial/'frozen_features.npz');z8,zd,zm=features['c8'],features['d4'],features['c4_masked']
            dlabels=np.load(trial/'adapted_d4_inputs.npz')['labels']
            contexts=json.loads((INC/name/f'seed_{seed}/sessions.json').read_text())
            pairs={(p['shot'],p['repeat']):p for p in json.loads((trial/'c8_paired_queries.json').read_text())}
            rec=Recorder(name,seed);checks=0
            expected=pd.read_csv(ROOT/'results/mahalanobis_full_ablation_v1'/name/f'seed_{seed}/metrics.csv').set_index(['shot','repeat','session','method'])
            for ctx in contexts:
                if ctx['session']!=2:continue
                k,r=ctx['shot'],ctx['repeat'];s,q=np.array(ctx['support']),np.array(ctx['query'])
                pair=pairs[k,r];qa,qb=np.array(pair['c4_query'],int),np.array(pair['c8_query'],int)
                for method in METHODS:
                    head=fit(method,z4,labels,s)
                    for scenario,z,qi in [('C4_original',z4,q),('C4_masked_query',zm,q),('C4_paired_C8',z4,qa),('C8_resolution',z8,qb)]:
                        if not len(qi):continue
                        protocol='six_class_coverage' if scenario in ['C4_original','C4_masked_query'] else 'six_class_paired_resolution'
                        true=labels[q] if scenario in ['C4_original','C4_masked_query'] else labels[qa]
                        metrics=rec.add(scenario,protocol,k,r,2,method,true,head.predict(z[qi]),ctx['seen'],task['base'],ctx['previous_novel'],ctx['current_novel'])
                        if scenario=='C4_original':
                            reference=expected.loc[k,r,2,method]
                            for field in ['all_accuracy','all_macro_f1','base_accuracy','seen_novel_accuracy']:
                                np.testing.assert_allclose(metrics[field],reference[field],atol=1e-12)
                            checks+=1
            partition=json.loads((out/f'd4_partitions_seed_{seed}.json').read_text())
            episodes=json.loads((out/f'd4_episodes_seed_{seed}.json').read_text())
            core=sorted(c for c in partition if len(np.flatnonzero(dlabels==c))>=25)
            for k in [1,5,10,20]:
                eligible=sorted(c for c,p in partition.items() if len(p['support_pool'])>=k)
                for panel,classes in [('core9',core)]+([('eligible',eligible)] if eligible!=core else []):
                    base=[c for c in classes if c in task['base']];novel=[c for c in classes if c not in base]
                    for r,episode in enumerate(episodes):
                        s=np.array([i for c in classes for i in episode[c]['support'][:k]])
                        q=np.array([i for c in classes for i in episode[c]['query']])
                        for method in METHODS:
                            start=time.perf_counter();head=fit(method,zd,dlabels,s);fit_ms=1000*(time.perf_counter()-start)
                            start=time.perf_counter();pred=head.predict(zd[q]);pred_ms=1000*(time.perf_counter()-start)
                            rec.add('D4_within',panel,k,r,0,method,dlabels[q],pred,classes,base,(),novel)
                            if r<20:rec.costs.append(dict(scenario='D4_within',protocol=panel,shot=k,repeat=r,session=0,method=method,
                                episode_head_fit_ms=fit_ms,cached_query_prediction_ms=pred_ms,head_state_bytes=head.storage()['total_numeric_state_bytes']))
            if {'PP','PS'}.issubset(task['base']):
                base=['PP','PS'];old=np.array(contexts[0]['support']);base_s=old[np.isin(labels[old],base)]
                templates={m:fit(m,z4,labels,base_s) for m in METHODS}
                for k in [1,5,10,20]:
                    for r,episode in enumerate(episodes):
                        heads={m:h.clone() for m,h in templates.items()};seen=base.copy()
                        for stage in range(3):
                            current=[] if stage==0 else [['PE'],['PEVA']][stage-1]
                            previous=seen[2:].copy();seen+=current
                            q=np.array([i for c in seen for i in episode[c]['query']])
                            for method,head in heads.items():
                                snapshot={c:(head.centers[c].copy(),head.precisions[c].copy()) for c in head.centers} if r==0 else {}
                                start=time.perf_counter()
                                if current:head.register(current[0],zd[episode[current[0]]['support'][:k]])
                                reg_ms=1000*(time.perf_counter()-start)
                                for c,(mean,precision) in snapshot.items():
                                    np.testing.assert_array_equal(head.centers[c],mean);np.testing.assert_array_equal(head.precisions[c],precision)
                                start=time.perf_counter();pred=head.predict(zd[q]);pred_ms=1000*(time.perf_counter()-start)
                                rec.add('C4_to_D4_incremental','PP_PS_to_PE_PEVA',k,r,stage,method,dlabels[q],pred,seen,base,previous,current)
                                if r<20:rec.costs.append(dict(scenario='C4_to_D4_incremental',protocol='PP_PS_to_PE_PEVA',shot=k,repeat=r,session=stage,method=method,
                                    new_class_statistics_or_bank_ms=reg_ms,cached_query_prediction_ms=pred_ms,head_state_bytes=head.storage()['total_numeric_state_bytes']))
            rec.save(folder)
            write(folder/'COMPLETE.json',dict(rows=len(rec.rows),original_C4_full_metric_checks=checks,
                features_sha256=digest(trial/'frozen_features.npz'),episodes_sha256=digest(out/f'd4_episodes_seed_{seed}.json'),
                previous_class_state_checks='first repeat of every shot/stage; means and precisions exactly unchanged',encoder_training_steps=0))
            print('Full-class control complete',name,seed,len(rec.rows),flush=True)
    for filename in ['metrics.csv','per_class_results.csv','costs.csv']:
        data=pd.concat([pd.read_csv(p/filename) for p in folders],ignore_index=True)
        data.to_csv(root/filename,index=False)
        if filename=='metrics.csv':data.to_csv(out/'full_class_mahalanobis_control_metrics.csv',index=False)
    write(root/'COMPLETE.json',dict(trials=len(folders),encoder_training_steps=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/heterogeneous_ftir_transfer_v1')
    with threadpool_limits(limits=1):run(parser.parse_args().output.resolve())
