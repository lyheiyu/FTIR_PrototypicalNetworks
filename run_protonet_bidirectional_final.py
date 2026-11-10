"""Final locked 2-objective x 3-metric ablation, reusing final CE protocol."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from threadpoolctl import threadpool_limits
from encoder_architecture_models import build_encoder
from ftir_research_data import ROOT,load_dataset,preprocess
from ftir_research_core import base_covariance
from frozen_prototype_heads_v1 import FrozenPrototype
from run_full_mahalanobis_ablation_v1 import FullClassMahalanobis
from run_bidirectional_c4_c8_final import CASES,encoder_features,digest,write
from run_encoder_architecture_v1 import weight_hash
from heterogeneous_ftir_adapter_v1 import fast_metrics
from protonet_training_final import prototype_loss,episode_indices,local_labels

PARENT=ROOT/'results/bidirectional_c4_c8_final'
SOURCES=['run_protonet_bidirectional_final.py','protonet_training_final.py','tests/test_protonet_final.py',
         'encoder_architecture_models.py','run_bidirectional_c4_c8_final.py','frozen_prototype_heads_v1.py',
         'run_full_mahalanobis_ablation_v1.py','ftir_research_data.py','ftir_research_core.py','heterogeneous_ftir_adapter_v1.py']


def train(folder,x,labels,task,split,seed,source,epochs,parent_model,draws,valdraws):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        return json.loads((folder/'training_costs.json').read_text())
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    model=build_encoder('M1');initial=weight_hash(model)
    expected=json.loads((parent_model/'PAIRED_TRAINING_COMPLETE.json').read_text())['initial_encoder_hash']
    assert initial==expected and model.count_params()==47400
    opt=tf.keras.optimizers.Adam(.001);sy,qy=local_labels(5,10);vsy,vqy=local_labels(5,4)
    @tf.function(input_signature=[tf.TensorSpec([60,1868],tf.float32)])
    def step(a):
        with tf.GradientTape() as tape:
            z=model(a,training=True);loss,acc=prototype_loss(z[:20],z[20:],sy,qy)
        gs=tape.gradient(loss,model.trainable_variables)
        for g in gs:tf.debugging.assert_all_finite(g,'non-finite ProtoNet gradient')
        opt.apply_gradients(zip(gs,model.trainable_variables));return loss,acc
    val=np.asarray(split['validation']);val_positions={int(i):j for j,i in enumerate(val)}
    vs=tf.constant([[val_positions[int(i)] for i in d[0]] for d in valdraws],tf.int32)
    vq=tf.constant([[val_positions[int(i)] for i in d[1]] for d in valdraws],tf.int32)
    @tf.function(input_signature=[tf.TensorSpec([None,1868],tf.float32)])
    def validation(a):
        z=model(a,training=False);ss=tf.gather(z,vs);qq=tf.gather(z,vq)
        means=tf.reduce_mean(tf.reshape(ss,[20,4,5,128]),axis=2)
        logits=-tf.reduce_sum(tf.square(qq[:,:,None,:]-means[:,None,:,:]),-1)
        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.tile(vqy[None,:],[20,1]),logits=logits))
        acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,-1,output_type=tf.int32),vqy[None,:]),tf.float32))
        return loss,acc
    start=time.perf_counter();history=[];best=float('inf');best_epoch=0
    for epoch in range(epochs):
        vals=[]
        for s,q in draws[epoch]:
            loss,acc=step(tf.convert_to_tensor(x[np.r_[s,q]],tf.float32));vals.append([float(loss),float(acc)])
        vl,va=validation(tf.convert_to_tensor(x[val],tf.float32));vl=float(vl)
        if not np.isfinite(vl):raise ValueError('Non-finite source-validation loss')
        if vl<best:
            best=vl;best_epoch=epoch+1;model.save_weights(str(folder/'best.weights.h5'))
        history.append(dict(epoch=epoch+1,loss=np.mean(vals,axis=0)[0],accuracy=np.mean(vals,axis=0)[1],val_loss=vl,val_accuracy=float(va)))
        if (epoch+1)%10==0 or epoch==epochs-1:print(f'  {source} epoch {epoch+1}/{epochs}; train loss {history[-1]["loss"]:.5f}; val loss {vl:.5f}',flush=True)
    seconds=time.perf_counter()-start;updates=int(opt.iterations.numpy())
    assert updates==epochs*int(np.ceil(len(split['train'])/64))
    model.load_weights(str(folder/'best.weights.h5'));model.trainable=False
    model.save(str(folder/'encoder.keras'));before=weight_hash(model);file_hash=digest(folder/'encoder.keras')
    # Caller extracts both resolutions after the training model is safely saved.
    sample=tf.convert_to_tensor(x[:64],tf.float32)
    infer=tf.function(lambda a:model(a,training=False));infer(sample)
    timing=[]
    for _ in range(20):
        t=time.perf_counter();infer(sample).numpy();timing.append(1000*(time.perf_counter()-t)/len(sample))
    assert weight_hash(model)==before and digest(folder/'encoder.keras')==file_hash
    stats=dict(parameters=47400,classifier_parameters=47400,training_seconds=seconds,optimizer_updates=updates,
        best_epoch=best_epoch,best_source_validation_loss=best,initial_encoder_hash=initial,weight_hash=before,
        inference_median_ms_per_spectrum=float(np.median(timing)),inference_p95_ms_per_spectrum=float(np.percentile(timing,95)),
        inference_batch_size=64,incremental_encoder_updates=0,source=source,
        budget='60 rows/update: 4 classes x (5 support + 10 query); 40 query labels/update; CE up to64 labels/update; equal optimizer updates, not equal role/sample exposures')
    pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False);write(folder/'training_costs.json',stats)
    write(folder/'COMPLETE.json',dict(encoder_sha256=file_hash,**stats));return stats


def evaluate(folder,task,seed,split,episodes,labels,features,priors):
    rows,costs,perclass=[],[],[];cms={};checks=0
    methods=['CE_euclidean','EP_euclidean','EP_full','EP_shrinkage']
    templates={}
    for case,(enc,bs,ns,qs) in CASES.items():
        for method in methods:
            objective,kind=method.split('_');prior=priors[objective,enc]
            h=FullClassMahalanobis(128,reg=.001) if kind=='full' else FrozenPrototype(kind,128,prior,reg=.001,tau=5.)
            for c in sorted(task['base']):h.register(c,features[objective,enc,bs][episodes[0]['base_support'][c]])
            templates[case,method]=h
    forbidden=set(split['train'])|set(split['validation'])
    for k in [1,5,10,20]:
        for ep in episodes:
            heads={key:h.clone() for key,h in templates.items()};seen=list(task['base']);first={};base0={}
            for stage in range(3):
                current=[] if stage==0 else task['novel_sessions'][stage-1];previous=seen[4:].copy();seen+=current
                q=np.asarray([i for c in sorted(seen) for i in ep['query'][c]],int)
                supports=set(sum(ep['base_support'].values(),[]))
                for c in seen[4:]:supports.update(ep['novel_support_order'][c][:k])
                assert not set(q)&(supports|forbidden)
                for case,(enc,bs,ns,qs) in CASES.items():
                    for method in methods:
                        objective,kind=method.split('_');h=heads[case,method]
                        snapshot={c:(m.copy(),h.precisions[c].copy() if c in h.precisions else None) for c,m in h.centers.items()} if ep['repeat']==0 else {}
                        begin=time.perf_counter()
                        if current:h.register(current[0],features[objective,enc,ns][ep['novel_support_order'][current[0]][:k]])
                        registration_ms=1000*(time.perf_counter()-begin)
                        for c,(m,p) in snapshot.items():
                            np.testing.assert_array_equal(m,h.centers[c])
                            if p is not None:np.testing.assert_array_equal(p,h.precisions[c])
                            checks+=1
                        begin=time.perf_counter();pred=h.predict(features[objective,enc,qs][q]);infer_ms=1000*(time.perf_counter()-begin)
                        met,_,cm=fast_metrics(labels[q],pred,seen,task['base'],previous,current)
                        tb=np.isin(labels[q],task['base']);tn=~tb
                        met['base_to_novel_error_rate']=float(np.mean(np.isin(pred[tb],seen[4:])))
                        met['novel_to_base_error_rate']=float(np.mean(np.isin(pred[tn],task['base']))) if tn.any() else np.nan
                        if stage==0:base0[case,method]=met['base_accuracy']
                        if stage==1:first[case,method]=met['current_novel_accuracy']
                        met['previous_novel_change_pp']=100*(met['previous_novel_accuracy']-first[case,method]) if stage==2 else np.nan
                        met['base_change_from_session0_pp']=100*(met['base_accuracy']-base0[case,method])
                        meta=dict(task=task['name'],seed=seed,setting=case,encoder_source=enc,base_source=bs,novel_source=ns,query_source=qs,
                            shot=k,repeat=ep['repeat'],session=stage,method=method)
                        rows.append(dict(meta,**met,query_count=len(q)))
                        key=(case,k,stage,method,tuple(sorted(seen)));cms[key]=cms.get(key,np.zeros_like(cm))+cm
                        if ep['repeat']<20:costs.append(dict(meta,head_local_registration_ms=registration_ms,cached_query_prediction_ms=infer_ms,
                            scope='cached embedding only; encoding excluded; no refit of old classes'))
    pd.DataFrame(rows).to_csv(folder/'new_metrics.csv',index=False);pd.DataFrame(costs).to_csv(folder/'new_costs.csv',index=False)
    arrays,index={},{}
    for i,(key,cm) in enumerate(cms.items()):
        case,k,stage,method,cs=key;token=f'cm_{i}';arrays[token]=cm
        meta=dict(task=task['name'],seed=seed,setting=case,shot=k,session=stage,method=method);index[token]=dict(meta,classes=list(cs))
        for j,c in enumerate(cs):
            tp=cm[j,j];n=cm[j].sum();p=cm[:,j].sum()
            perclass.append(dict(meta,**{'class':c},recall=tp/n if n else 0,precision=tp/p if p else 0,f1=2*tp/(n+p) if n+p else 0,query_occurrences=int(n)))
    np.savez_compressed(folder/'new_confusions.npz',**arrays);write(folder/'new_confusion_index.json',index)
    pd.DataFrame(perclass).to_csv(folder/'new_per_class_metrics.csv',index=False)
    return dict(rows=len(rows),old_class_state_checks=checks,incremental_encoder_updates=0)


def run(args):
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    old=json.loads((PARENT/'experiment_manifest.json').read_text());assert (PARENT/'COMPLETE.json').exists()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    tasks=old['tasks'][:1] if args.smoke else old['tasks'];seeds=[0] if args.smoke else [0,1,2]
    epochs,repeats=(2,2) if args.smoke else (60,200)
    checkpoints={str(p.relative_to(ROOT)):digest(p) for p in PARENT.glob('*/seed_*/*/M1/encoder.keras')}
    config=dict(experiment='protonet_bidirectional_final',parent=str(PARENT),smoke=args.smoke,tasks=tasks,seeds=seeds,epochs=epochs,repeats=repeats,
        architecture='M1 unchanged, 47400 parameters, 128 embedding, no L2 normalization or classifier head',
        train_episode=dict(ways=4,support=5,query=10),validation_episode=dict(ways=4,support=5,query=4,episodes=20,fixed=True),
        objective='mean query CE of negative squared Euclidean distances to differentiable support means; no other loss',
        optimizer='Adam .001; updates per epoch=ceil(CE legal train count /64); 60 epochs; minimum mean fixed source-base-validation episode CE',
        caveat='Optimizer updates and initialization matched; sampling roles, row exposure and validation objectives differ. No hyperparameter search.',
        classifier=old['classifier'],preprocessing=old['preprocessing'],scope=old['scope'],cases=old['cases'],
        parent_manifest_sha256=digest(PARENT/'experiment_manifest.json'),grid_sha256=digest(PARENT/'common_grid.npy'),
        dataset_hashes=old['dataset_hashes'],group_manifest_sha256=digest(PARENT/'paired_group_representatives.csv'),
        supervised_checkpoint_hashes=checkpoints,source_hashes={p:digest(ROOT/p) for p in SOURCES},
        old_protonet_reuse='Not reusable: legacy encoder architecture differs (BN/global pooling/L2), lacks this locked grid/group partition provenance. Reuse audited Euclidean objective, train new M1 only.',
        old_mahalanobis_audit='FTIR_ProtoNetwork4 dists_maha_full einsum ndd selects diagonal; maha_distance_full is correct. Old full validation calls the diagonal-only variant; final heads use verified dense matrices.')
    if (out/'experiment_manifest.json').exists():assert json.loads((out/'experiment_manifest.json').read_text())==config
    else:
        write(out/'experiment_manifest.json',config)
        for p in SOURCES:
            target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    if not args.smoke:
        assert args.validated_smoke and (args.validated_smoke/'COMPLETE.json').exists()
        smoke=json.loads((args.validated_smoke/'experiment_manifest.json').read_text());assert smoke['source_hashes']==config['source_hashes']
    groups=pd.read_csv(PARENT/'paired_group_representatives.csv');labels=groups.label.to_numpy();grid=np.load(PARENT/'common_grid.npy')
    assert len(grid)==1868
    x={}
    for source in ['c4','c8']:
        data=load_dataset(source);assert data.report['source_sha256']==old['dataset_hashes'][source]
        x[source]=preprocess(data,grid,ROOT/'results/research_cache',baseline=True,flip=True)[groups[f'{source}_index'].to_numpy()]
    for filename in ['common_grid.npy','paired_group_representatives.csv','all_original_ID_groups.csv']:(out/filename).write_bytes((PARENT/filename).read_bytes())
    folders=[];training=[]
    for task in tasks:
        for seed in seeds:
            parent=PARENT/task['name']/f'seed_{seed}';folder=out/task['name']/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);folders.append(folder)
            for name in ['split.json','episodes.json','sample_manifest.csv']:(folder/name).write_bytes((parent/name).read_bytes())
            split=json.loads((parent/'split.json').read_text());episodes=json.loads((parent/'episodes.json').read_text())[:repeats]
            assert set(labels[split['train']])==set(task['base']) and set(labels[split['validation']])==set(task['base'])
            steps=int(np.ceil(len(split['train'])/64));rng=np.random.default_rng(82000+seed)
            draws=[[episode_indices(labels,split['train'],task['base'],rng) for _ in range(steps)] for _ in range(epochs)]
            vrng=np.random.default_rng(92000+seed);vdraws=[episode_indices(labels,split['validation'],task['base'],vrng,5,4) for _ in range(20)]
            np.savez_compressed(folder/'paired_training_episodes.npz',support=np.array([[d[0] for d in e] for e in draws]),query=np.array([[d[1] for d in e] for e in draws]),
                validation_support=np.array([d[0] for d in vdraws]),validation_query=np.array([d[1] for d in vdraws]))
            features,priors={},{}
            for source in ['c4','c8']:
                model_folder=folder/source/'M1';parent_model=parent/source/'M1'
                print(f'TRAIN EP {task["name"]} seed {seed} {source}; {epochs*steps} updates',flush=True)
                stats=train(model_folder,x[source],labels,task,split,seed,source,epochs,parent_model,draws,vdraws)
                training.append(dict(task=task['name'],seed=seed,**stats))
                if (folder/'COMPLETE.json').exists():continue
                ce=np.load(parent_model/'frozen_views.npz');np.testing.assert_array_equal(ce['fit_indices'],split['train'])
                for target in ['c4','c8']:features['CE',source,target]=ce[f'M1_{target}']
                priors['CE',source]=ce['M1_prior']
                if not (model_folder/'frozen_views.npz').exists():
                    model=tf.keras.models.load_model(str(model_folder/'encoder.keras'),compile=False);model.trainable=False
                    before=weight_hash(model);t=time.perf_counter();z={s:encoder_features(model,a) for s,a in x.items()};elapsed=time.perf_counter()-t
                    prior=base_covariance(z[source][split['train']],labels[split['train']])
                    np.savez_compressed(model_folder/'frozen_views.npz',M1_c4=z['c4'],M1_c8=z['c8'],M1_prior=prior,fit_indices=split['train'])
                    assert before==weight_hash(model)==stats['weight_hash']
                    write(model_folder/'FROZEN_CHECK.json',dict(weights_before=before,weights_after=weight_hash(model),parameters=model.count_params(),
                        extraction_both_resolutions_seconds=elapsed,incremental_encoder_updates=0,prior_fit='source base training only'))
                ep=np.load(model_folder/'frozen_views.npz')
                for target in ['c4','c8']:features['EP',source,target]=ep[f'M1_{target}']
                priors['EP',source]=ep['M1_prior'];tf.keras.backend.clear_session()
            if not (folder/'COMPLETE.json').exists():
                print(f'EVALUATE {task["name"]} seed {seed}; six settings, four new heads',flush=True)
                result=evaluate(folder,task,seed,split,episodes,labels,features,priors)
                write(folder/'COMPLETE.json',dict(**result,split_sha256=digest(parent/'split.json'),episodes_sha256=digest(parent/'episodes.json')))
            print(f'COMPLETE {task["name"]} seed {seed}',flush=True)
    pd.DataFrame(training).to_csv(out/'training_summary.csv',index=False)
    for p,h in checkpoints.items():assert digest(ROOT/p)==h
    write(out/'COMPLETE.json',dict(paired_trials=len(folders),trained_episodic_encoders=2*len(folders),ce_retrained=0,incremental_encoder_updates=0,
        original_ce_checkpoints_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--validated-smoke',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/protonet_bidirectional_final')
    with threadpool_limits(limits=1):run(parser.parse_args())
