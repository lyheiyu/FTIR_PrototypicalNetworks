"""Final six-setting C4/C8 experiment, fixed CE M1 and duplicate-group isolation."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from bidirectional_protocol_final import paired_group_table,split_groups,paired_episodes
from encoder_architecture_models import build_encoder
from ftir_research_data import ROOT,load_dataset,preprocess
from ftir_research_core import Head,base_covariance
from frozen_prototype_heads_v1 import FrozenPrototype
from run_full_mahalanobis_ablation_v1 import FullClassMahalanobis
from run_encoder_architecture_v1 import train_model,weight_hash
from heterogeneous_ftir_adapter_v1 import fast_metrics

CASES={'A':('c4','c4','c4','c4'),'B':('c8','c8','c8','c8'),
       'C':('c4','c4','c8','c8'),'D':('c8','c8','c4','c4'),
       'E':('c4','c8','c8','c8'),'F':('c8','c4','c4','c4')}
METHODS=['raw_1nn','raw_svm','PCA64_shrinkage','M1_1nn','M1_full','M1_shrinkage']
SOURCES=['run_bidirectional_c4_c8_final.py','bidirectional_protocol_final.py','tests/test_bidirectional_final.py',
         'encoder_architecture_models.py','run_encoder_architecture_v1.py','ftir_research_data.py','ftir_research_core.py',
         'frozen_prototype_heads_v1.py','run_full_mahalanobis_ablation_v1.py','heterogeneous_ftir_adapter_v1.py','utils.py','PLS.py']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


class FrozenHead:
    def __init__(self,method,prior):
        self.method=method;self.view='raw' if method.startswith('raw') else 'PCA64' if method.startswith('PCA64') else 'M1'
        self.bank={};self.estimator=None;self.prototype=None
        if method=='M1_full':self.prototype=FullClassMahalanobis(128,reg=.001)
        elif method.endswith('shrinkage'):self.prototype=FrozenPrototype('shrinkage',prior.shape[0],prior,reg=.001,tau=5.)

    def clone(self):
        result=copy.copy(self);result.bank=self.bank.copy()
        result.prototype=self.prototype.clone() if self.prototype else None
        return result

    def register(self,c,x):
        if self.prototype:self.prototype.register(c,x)
        else:
            if c in self.bank:raise ValueError('Existing class cannot be replaced')
            self.bank[c]=np.array(x,copy=True)

    def prepare(self):
        if not self.prototype:
            cs=sorted(self.bank);x=np.concatenate([self.bank[c] for c in cs])
            y=np.concatenate([np.repeat(c,len(self.bank[c])) for c in cs])
            self.estimator=Head('svm' if self.method.endswith('svm') else '1nn',c=1.).fit(x,y)

    def predict(self,x):return (self.prototype or self.estimator).predict(x)

    def distances(self,x):
        cs=sorted(self.prototype.centers)
        means=np.stack([self.prototype.centers[c] for c in cs])
        diff=x[:,None,:]-means[None,:,:]
        scores=np.stack([np.sum((diff[:,i]@self.prototype.precisions[c])*diff[:,i],axis=1) for i,c in enumerate(cs)],axis=1)
        np.testing.assert_array_equal(np.asarray(cs)[scores.argmin(axis=1)],self.predict(x))
        return cs,scores


def encoder_features(encoder,x):
    infer=tf.function(lambda a:encoder(a,training=False),input_signature=[tf.TensorSpec([None,1868],tf.float32)])
    return np.concatenate([infer(tf.convert_to_tensor(x[i:i+256],tf.float32)).numpy() for i in range(0,len(x),256)]).astype(float)


def evaluate(folder,task,seed,split,episodes,labels,features,priors):
    rows,costs,competitions=[],[],[];cms={}
    templates={}
    base=episodes[0]['base_support']
    for case,(enc,bs,ns,qs) in CASES.items():
        for method in METHODS:
            view='raw' if method.startswith('raw') else 'PCA64' if method.startswith('PCA64') else 'M1'
            head=FrozenHead(method,priors[enc].get(view))
            for c in sorted(task['base']):head.register(c,features[enc][bs][view][base[c]])
            templates[case,method]=head
    state_checks=0;invariance_checks=0
    for k in [1,5,10,20]:
        for episode in episodes:
            r=episode['repeat'];heads={key:h.clone() for key,h in templates.items()}
            seen=list(task['base']);previous_values={};base_values={}
            for stage in range(3):
                current=[] if stage==0 else task['novel_sessions'][stage-1]
                previous=seen[len(task['base']):].copy();seen+=current
                q=np.asarray([i for c in sorted(seen) for i in episode['query'][c]])
                added=np.asarray([] if not current else episode['novel_support_order'][current[0]][:k],int)
                support_groups=set(sum(episode['base_support'].values(),[]))
                for c in seen:
                    if c not in task['base']:support_groups.update(episode['novel_support_order'][c][:k])
                assert not support_groups&set(q)
                assert not set(q)&(set(split['train'])|set(split['validation']))
                current_metrics={}
                for case,(enc,bs,ns,qs) in CASES.items():
                    for method in METHODS:
                        head=heads[case,method]
                        # First repeat verifies exact old-state preservation after every registration.
                        snapshot={c:(v.copy(),head.prototype.precisions[c].copy()) for c,v in head.prototype.centers.items()} if head.prototype and r==0 else {}
                        begin=time.perf_counter()
                        if current:head.register(current[0],features[enc][ns][head.view][added])
                        reg_ms=1000*(time.perf_counter()-begin)
                        for c,(mean,precision) in snapshot.items():
                            np.testing.assert_array_equal(head.prototype.centers[c],mean)
                            np.testing.assert_array_equal(head.prototype.precisions[c],precision);state_checks+=1
                        begin=time.perf_counter();head.prepare();prepare_ms=1000*(time.perf_counter()-begin)
                        begin=time.perf_counter();prediction=head.predict(features[enc][qs][head.view][q]);predict_ms=1000*(time.perf_counter()-begin)
                        metrics,_,cm=fast_metrics(labels[q],prediction,seen,task['base'],previous,current)
                        truebase=np.isin(labels[q],task['base']);truenovel=~truebase
                        metrics['base_to_novel_error_rate']=float(np.mean(np.isin(prediction[truebase],seen[len(task['base']):])))
                        metrics['novel_to_base_error_rate']=float(np.mean(np.isin(prediction[truenovel],task['base']))) if truenovel.any() else np.nan
                        if stage==0:base_values[case,method]=metrics['base_accuracy']
                        if stage==1:previous_values[case,method]=metrics['current_novel_accuracy']
                        metrics['previous_novel_change_pp']=100*(metrics['previous_novel_accuracy']-previous_values[case,method]) if stage==2 else np.nan
                        metrics['base_change_from_session0_pp']=100*(metrics['base_accuracy']-base_values[case,method])
                        meta=dict(task=task['name'],seed=seed,setting=case,encoder_source=enc,base_source=bs,novel_source=ns,query_source=qs,
                                  shot=k,repeat=r,session=stage,method=method)
                        rows.append(dict(meta,**metrics,query_count=len(q)))
                        current_metrics[case,method]=metrics
                        key=(case,k,stage,method,tuple(sorted(seen)))
                        cms[key]=cms.get(key,np.zeros_like(cm))+cm
                        if r<20:
                            costs.append(dict(meta,head_local_registration_ms=reg_ms,full_assembly_refit_ms=prepare_ms,cached_query_prediction_ms=predict_ms,
                                scope='cached features; encoder/preprocessing excluded; raw SVM full retained-bank refit separate'))
                            if head.prototype and stage>0:
                                classes,scores=head.distances(features[enc][qs][head.view][q])
                                trueindices=np.array([classes.index(c) for c in labels[q]])
                                true_score=scores[np.arange(len(q)),trueindices]
                                novel_columns=[i for i,c in enumerate(classes) if c not in task['base']]
                                base_columns=[i for i,c in enumerate(classes) if c in task['base']]
                                other_novel=scores[:,novel_columns].min(axis=1)
                                other_base=scores[:,base_columns].min(axis=1)
                                competitions.append(dict(meta,base_mean_true_quadratic=float(true_score[truebase].mean()),
                                    base_mean_closest_novel_quadratic=float(other_novel[truebase].mean()),
                                    novel_mean_true_quadratic=float(true_score[truenovel].mean()),
                                    novel_mean_closest_base_quadratic=float(other_base[truenovel].mean())))
                for a,b in [('A','F'),('B','E')]:
                    for method in ['raw_1nn','raw_svm']:
                        for field in ['all_accuracy','all_macro_f1','base_accuracy']:
                            np.testing.assert_allclose(current_metrics[a,method][field],current_metrics[b,method][field],atol=1e-14)
                        invariance_checks+=1
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(costs).to_csv(folder/'costs.csv',index=False)
    pd.DataFrame(competitions).to_csv(folder/'score_competition.csv',index=False)
    matrices,index,perclass={},{},[]
    for i,(key,cm) in enumerate(cms.items()):
        case,k,stage,method,classes=key;token=f'cm_{i}'
        matrices[token]=cm;meta=dict(task=task['name'],seed=seed,setting=case,shot=k,session=stage,method=method)
        index[token]=dict(meta,classes=list(classes))
        tp,support,predicted=np.diag(cm),cm.sum(axis=1),cm.sum(axis=0)
        for j,c in enumerate(classes):
            perclass.append(dict(meta,**{'class':c},precision=float(tp[j]/predicted[j]) if predicted[j] else 0.,recall=float(tp[j]/support[j]) if support[j] else 0.,
                f1=float(2*tp[j]/(support[j]+predicted[j])) if support[j]+predicted[j] else 0.,query_occurrences=int(support[j]),
                aggregation='pooled repeat confusion; occurrences are not independent specimens'))
    pd.DataFrame(perclass).to_csv(folder/'per_class_metrics.csv',index=False)
    np.savez_compressed(folder/'confusions.npz',**matrices);write(folder/'confusion_index.json',index)
    return dict(metric_rows=len(rows),old_class_state_checks=state_checks,raw_source_invariance_checks=invariance_checks)


def run(args):
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism();tf.keras.utils.set_random_seed(0)
    historical=ROOT/'results/encoder_architecture_v1'
    old=json.loads((historical/'experiment_manifest.json').read_text())
    tasks=old['tasks'][:1] if args.smoke else old['tasks'];seeds=[0] if args.smoke else [0,1,2]
    epochs,repeats=(2,2) if args.smoke else (60,200)
    data={s:load_dataset(s) for s in ['c4','c8']};grid=np.load(historical/'common_grid.npy')
    assert len(grid)==1868
    for d in data.values():assert d.wave[0]<=grid[0] and grid[-1]<=d.wave[-1]
    inputs={s:preprocess(d,grid,ROOT/'results/research_cache',baseline=True,flip=True) for s,d in data.items()}
    mappings={s:pd.read_csv(ROOT/'dataset/deduplicated'/f'FTIR_PLastics500_{s}_row_mapping.csv',dtype=str) for s in data}
    groups,aliases,group_audit=paired_group_table(data,inputs,mappings)
    labels=groups.label.to_numpy();paired_x={s:inputs[s][groups[f'{s}_index'].to_numpy()].copy() for s in data}
    config=dict(experiment='bidirectional_c4_c8_final',smoke=args.smoke,tasks=tasks,seeds=seeds,epochs=epochs,repeats=repeats,
        batch_size=64,learning_rate=.001,shots=[1,5,10,20],primary_shots=[5,10],queries_per_class=15,base_references_per_class=20,
        cases={k:dict(zip(['encoder','base_references','novel_support','query'],v)) for k,v in CASES.items()},methods=METHODS,
        group_audit=group_audit,group_counts=groups.groupby('label').size().to_dict(),
        grouping='union of same original record IDs across resolutions, both native duplicate alias mappings, exact common-grid processed fingerprints; ONE representative/group/resolution, prefer shared retained ID',
        split='base groups: max(15,20%) query, max(5,15%) validation, remainder train; novel groups 50/50 support/query; shared roles across resolutions',
        retraining_C4_reason='All historical C4 base rows were train/validation. Fresh C4 and C8 models are necessary to keep new query groups out of checkpoint selection. Same original M1, CE, Adam .001, batch64, 60 epochs; no model search.',
        training='paired group epoch orders and initialization seeds; same update counts; best SOURCE base-validation loss; no early stop; temporary linear head discarded for encoder evaluation',
        preprocessing='unchanged native row_max-y -> airPLS -> maxabs -> cubic -> locked common physical grid; no extrapolation',
        classifier=dict(svm='linear C=1',covariance_divisor='n',ridge=.001,tau=5.,alpha='(n-1)/(n-1+5)',
            full='independent class full128 covariance + .001 I; not original double-ridge .002',prior='source-specific within-class covariance on source base_train only; frozen in all six settings'),
        PCA=dict(dimensions=64,svd_solver='randomized',whiten=False,fit='source base_train only; no target fit',
                 regularisation='same numeric ridge .001/tau5 as M1-128, but different dimensions and scales; NOT equivalent relative regularisation'),
        scope='EXPLORATORY: both datasets/results previously inspected; no genuinely untouched confirmatory holdout; record-group exclusion established, physical specimen independence unverified; C4/C8 not independent instruments',
        historical_reference='old C4 novel-support -> C8 query is separate H, not setting C, and uses old checkpoints/splits',
        D4='excluded',dataset_hashes={s:d.report['source_sha256'] for s,d in data.items()},
        mapping_hashes={s:digest(ROOT/'dataset/deduplicated'/f'FTIR_PLastics500_{s}_row_mapping.csv') for s in data},
        grid_sha256=digest(historical/'common_grid.npy'),source_hashes={p:digest(ROOT/p) for p in SOURCES},
        old_checkpoint_hashes={str(p.relative_to(ROOT)):digest(p) for p in historical.glob('*/seed_*/M1/encoder.keras')})
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment_manifest.json').exists():assert json.loads((out/'experiment_manifest.json').read_text())==config
    else:
        write(out/'experiment_manifest.json',config);(out/'source').mkdir(exist_ok=True)
        for p in SOURCES:
            target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    if not args.smoke:
        assert args.validated_smoke and (args.validated_smoke/'COMPLETE.json').exists()
        assert json.loads((args.validated_smoke/'experiment_manifest.json').read_text())['source_hashes']==config['source_hashes']
    groups.to_csv(out/'paired_group_representatives.csv',index=False);aliases.to_csv(out/'all_original_ID_groups.csv',index=False)
    np.save(out/'common_grid.npy',grid)
    for s,d in data.items():np.save(out/f'{s}_native_physical_coordinates.npy',d.wave)
    pd.DataFrame([dict(dataset=s,**{'class':c},retained_rows=n) for s,d in data.items() for c,n in d.report['unique_counts'].items()]).to_csv(out/'raw_unique_counts.csv',index=False)
    training_rows,folders,geometry=[],[],[]
    for task in tasks:
        for seed in seeds:
            folder=out/task['name']/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);folders.append(folder)
            split=split_groups(labels,task,seed);episodes=paired_episodes(labels,task,split,seed,repeats)
            write(folder/'split.json',split);write(folder/'episodes.json',episodes)
            sample=groups.copy();sample['role']=split['roles'];sample.to_csv(folder/'sample_manifest.csv',index=False)
            train,val=np.array(split['train']),np.array(split['validation'])
            rng=np.random.default_rng(56000+seed)
            train_orders=np.array([rng.permutation(train) for _ in range(epochs)])
            np.save(folder/'paired_training_orders.npy',train_orders)
            features,priors={},{}
            for source in ['c4','c8']:
                model_folder=folder/source/'M1';model_folder.mkdir(parents=True,exist_ok=True)
                paired_complete=model_folder/'PAIRED_TRAINING_COMPLETE.json'
                if not paired_complete.exists():
                    print(f'TRAIN {task["name"]} seed {seed} source {source}; {len(train)} train / {len(val)} validation groups',flush=True)
                    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
                    initial=build_encoder('M1');initial_hash=weight_hash(initial);del initial
                    namespace=SimpleNamespace(epochs=epochs,batch_size=64,learning_rate=.001)
                    _,stats=train_model('M1',paired_x[source],labels,task,train,val,seed,namespace,model_folder,train_orders)
                    write(paired_complete,dict(initial_encoder_hash=initial_hash,training_order_hash=hashlib.sha256(train_orders.tobytes()).hexdigest(),
                        split_sha256=digest(folder/'split.json'),source=source,stats=stats))
                meta=json.loads(paired_complete.read_text());stats=meta['stats']
                assert meta['split_sha256']==digest(folder/'split.json')
                training_rows.append(dict(task=task['name'],seed=seed,source=source,**stats,initial_encoder_hash=meta['initial_encoder_hash']))
                if (folder/'COMPLETE.json').exists():continue
                print(f'FREEZE / EXTRACT {task["name"]} seed {seed} source {source}',flush=True)
                encoder=tf.keras.models.load_model(str(model_folder/'encoder.keras'),compile=False);encoder.trainable=False
                before=weight_hash(encoder);file_before=digest(model_folder/'encoder.keras')
                assert before==stats['weight_hash_after'] and encoder.count_params()==47400
                z={s:encoder_features(encoder,x) for s,x in paired_x.items()}
                pca=PCA(n_components=64,svd_solver='randomized',random_state=seed,whiten=False).fit(paired_x[source][train].astype(float))
                features[source]={s:dict(raw=paired_x[s].astype(float),M1=z[s],PCA64=pca.transform(paired_x[s].astype(float))) for s in paired_x}
                priors[source]={view:base_covariance(features[source][source][view][train],labels[train]) for view in ['M1','PCA64']}
                np.savez_compressed(model_folder/'frozen_views.npz',M1_c4=z['c4'],M1_c8=z['c8'],PCA64_c4=features[source]['c4']['PCA64'],PCA64_c8=features[source]['c8']['PCA64'],
                    pca_mean=pca.mean_,pca_components=pca.components_,PCA64_prior=priors[source]['PCA64'],M1_prior=priors[source]['M1'],fit_indices=train)
                source_geometry=[]
                for view,cov in priors[source].items():
                    eigen=np.maximum(np.linalg.eigvalsh(cov),0);p=eigen/eigen.sum() if eigen.sum() else eigen
                    source_geometry.append(dict(task=task['name'],seed=seed,source=source,view=view,dimension=len(cov),ridge=.001,
                        mean_residual_variance=float(np.trace(cov)/len(cov)),ridge_over_mean_variance=float(.001/(np.trace(cov)/len(cov))) if np.trace(cov)>0 else np.nan,
                        effective_rank=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))) if p.sum() else 0.,
                        mean_training_norm=float(np.linalg.norm(features[source][source][view][train],axis=1).mean())))
                pd.DataFrame(source_geometry).to_csv(model_folder/'regularisation_geometry.csv',index=False)
                assert before==weight_hash(encoder) and file_before==digest(model_folder/'encoder.keras')
                write(model_folder/'FROZEN_CHECK.json',dict(file_before=file_before,file_after=digest(model_folder/'encoder.keras'),weights_before=before,weights_after=weight_hash(encoder),
                    parameters=encoder.count_params(),incremental_training_updates=0))
                tf.keras.backend.clear_session()
            # Equal initial weights / epoch orders / optimizer budgets across sources.
            both=[json.loads((folder/s/'M1/PAIRED_TRAINING_COMPLETE.json').read_text()) for s in ['c4','c8']]
            assert both[0]['initial_encoder_hash']==both[1]['initial_encoder_hash']
            assert both[0]['stats']['optimizer_updates']==both[1]['stats']['optimizer_updates']==epochs*int(np.ceil(len(train)/64))
            if (folder/'COMPLETE.json').exists():continue
            print(f'EVALUATE SIX SETTINGS {task["name"]} seed {seed}',flush=True)
            result=evaluate(folder,task,seed,split,episodes,labels,features,priors)
            write(folder/'COMPLETE.json',dict(**result,incremental_encoder_updates=0,split_sha256=digest(folder/'split.json'),episodes_sha256=digest(folder/'episodes.json')))
            print(f'COMPLETE {task["name"]} seed {seed}: {result}',flush=True)
    pd.DataFrame(training_rows).to_csv(out/'reverse_training_summary.csv',index=False)
    pd.concat([pd.read_csv(f/s/'M1/regularisation_geometry.csv') for f in folders for s in ['c4','c8']],ignore_index=True).to_csv(out/'regularisation_geometry.csv',index=False)
    for filename in ['metrics.csv','per_class_metrics.csv','costs.csv','score_competition.csv']:
        aggregate=pd.concat([pd.read_csv(f/filename) for f in folders],ignore_index=True)
        aggregate.to_csv(out/('bidirectional_transfer_metrics.csv' if filename=='metrics.csv' else filename),index=False)
    for path,value in config['old_checkpoint_hashes'].items():assert digest(ROOT/path)==value
    write(out/'COMPLETE.json',dict(paired_trials=len(folders),trained_encoders=2*len(folders),incremental_encoder_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/bidirectional_c4_c8_final')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--validated-smoke',type=Path)
    with threadpool_limits(limits=1):run(parser.parse_args())
