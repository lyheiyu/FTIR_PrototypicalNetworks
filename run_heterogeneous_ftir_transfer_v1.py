"""Frozen C4 M1: controlled coverage, paired C8, and exploratory D4 transfer."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from threadpoolctl import threadpool_limits
import encoder_architecture_models
from ftir_research_data import ROOT,load_dataset,preprocess
from ftir_research_core import Head,base_covariance
from frozen_prototype_heads_v1 import FrozenPrototype
from heterogeneous_ftir_adapter_v1 import observed_mask,base_input_mean,apply_missing_mask,partial_input,fast_metrics
from run_encoder_architecture_v1 import weight_hash

ARCH=ROOT/'results/encoder_architecture_v1'
INC=ROOT/'results/frozen_incremental_v1'
PREVIOUS=ROOT/'results/external_transfer_prototype_v1'
METHODS=['spectrum_1nn','pca64_1nn','M1_1nn','M1_proto_euclidean','M1_proto_shared','M1_proto_shrinkage','spectrum_svm']
EMBEDDING=[m for m in METHODS if m.startswith('M1')]
NEW_SOURCES=['run_heterogeneous_ftir_transfer_v1.py','heterogeneous_ftir_adapter_v1.py','tests/test_heterogeneous_ftir_v1.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def views(raw,pca,z):
    return dict(spectrum=raw,pca64=pca,M1=z)


class EpisodeHead:
    def __init__(self,method,prior):
        self.method=method
        self.view='M1' if method.startswith('M1') else method.split('_')[0]
        self.prototype=FrozenPrototype(method.split('_')[-1],128,prior) if '_proto_' in method else None
        self.bank={}
        self.estimator=None

    def clone(self):
        result=copy.copy(self)
        result.bank=self.bank.copy()
        result.prototype=self.prototype.clone() if self.prototype else None
        return result

    def register(self,label,features):
        if self.prototype:
            self.prototype.register(label,features)
        else:
            if label in self.bank:
                raise ValueError('Existing class cannot be replaced')
            self.bank[label]=np.array(features,copy=True)

    def prepare(self):
        if not self.prototype:
            labels=sorted(self.bank)
            self.support=np.concatenate([self.bank[c] for c in labels])
            self.labels=np.concatenate([np.repeat(c,len(self.bank[c])) for c in labels])
            self.estimator=Head('svm' if self.method.endswith('svm') else '1nn',c=1.).fit(self.support,self.labels)

    def predict(self,x):
        return (self.prototype or self.estimator).predict(x)

    def state_bytes(self):
        if self.prototype:
            return self.prototype.storage()['total_numeric_state_bytes']
        n=sum(x.nbytes for x in self.bank.values())
        if self.method.endswith('svm') and self.estimator is not None:
            n+=sum(getattr(self.estimator.estimator,k).nbytes for k in ['support_vectors_','dual_coef_','intercept_','support_'])
        return n


class Recorder:
    def __init__(self,task,seed):
        self.task,self.seed=task,seed
        self.rows,self.costs,self.coverage=[],[],[]
        self.cms={}

    def add(self,scenario,protocol,k,r,session,method,true,pred,seen,base=(),previous=(),current=(),**extra):
        metrics,_,cm=fast_metrics(true,pred,seen,base,previous,current)
        meta=dict(task=self.task,seed=self.seed,scenario=scenario,protocol=protocol,shot=k,repeat=r,session=session,method=method)
        self.rows.append(dict(meta,**metrics,query_count=len(true),class_count=len(seen),**extra))
        key=(scenario,protocol,k,session,method,tuple(sorted(seen)))
        self.cms[key]=self.cms.get(key,np.zeros_like(cm))+cm
        return metrics

    def save(self,folder):
        pd.DataFrame(self.rows).to_csv(folder/'metrics.csv',index=False)
        pd.DataFrame(self.costs).to_csv(folder/'costs.csv',index=False)
        pd.DataFrame(self.coverage).to_csv(folder/'coverage_mask_comparison.csv',index=False)
        matrices,index,rows={},{},[]
        for i,(key,cm) in enumerate(self.cms.items()):
            scenario,protocol,k,session,method,classes=key
            token=f'cm_{i}'
            matrices[token]=cm
            meta=dict(task=self.task,seed=self.seed,scenario=scenario,protocol=protocol,shot=k,session=session,method=method)
            index[token]=dict(meta,classes=list(classes))
            tp,support,pc=np.diag(cm),cm.sum(axis=1),cm.sum(axis=0)
            for j,c in enumerate(classes):
                rows.append(dict(meta,**{'class':c},precision=float(tp[j]/pc[j]) if pc[j] else 0.,
                    recall=float(tp[j]/support[j]) if support[j] else 0.,
                    **{'f1-score':float(2*tp[j]/(support[j]+pc[j])) if support[j]+pc[j] else 0.,
                       'support':int(support[j])},aggregation='pooled repeated-query confusion; not independent specimens'))
        pd.DataFrame(rows).to_csv(folder/'per_class_results.csv',index=False)
        np.savez_compressed(folder/'confusions.npz',**matrices)
        write(folder/'confusion_index.json',index)


def d4_episodes(labels,seed,repeats):
    rng=np.random.default_rng(41000+seed)
    partition={}
    for c in sorted(set(labels)):
        indices=np.flatnonzero(labels==c)
        order=rng.permutation(indices)
        max_support=min(20,len(order)-5)
        n_query=max(5,min(len(order)//2,len(order)-max_support))
        partition[c]=dict(query_pool=order[:n_query].tolist(),support_pool=order[n_query:].tolist())
    episodes=[]
    for r in range(repeats):
        rng=np.random.default_rng(51000+1000*seed+r)
        episodes.append({c:dict(support=rng.permutation(p['support_pool'])[:20].tolist(),
            query=rng.choice(p['query_pool'],5,replace=False).tolist()) for c,p in partition.items()})
    return partition,episodes


def fit_episode(method,prior,features,support,labels):
    head=EpisodeHead(method,prior)
    for c in sorted(set(labels[support])):
        head.register(c,features[head.view][support[labels[support]==c]])
    head.prepare()
    return head


def encode(infer,x):
    return np.concatenate([infer(tf.convert_to_tensor(x[i:i+256],tf.float32)).numpy() for i in range(0,len(x),256)]).astype(float)


def run(args):
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()
    old=json.loads((PREVIOUS/'experiment_manifest.json').read_text())
    assert (PREVIOUS/'COMPLETE.json').exists()
    for name,value in old['source_hashes'].items():
        assert digest(ROOT/name)==value,name
    tasks=old['tasks'][:1] if args.smoke else old['tasks']
    seeds=[0] if args.smoke else old['seeds']
    repeats=2 if args.smoke else 200
    out=args.output.resolve()
    c4,c8,d4=[load_dataset(k) for k in ['c4','c8','d4']]
    grid=np.load(ARCH/'common_grid.npy');mask=observed_mask(d4.wave,grid)
    assert len(grid)==1868 and (~mask).sum()==107
    config=dict(version=1,smoke=args.smoke,tasks=tasks,seeds=seeds,repeats=repeats,shots=[1,5,10,20],
        methods=METHODS,source_hashes={p:digest(ROOT/p) for p in NEW_SOURCES},
        previous_manifest_sha256=digest(PREVIOUS/'experiment_manifest.json'),
        architecture_manifest_sha256=digest(ARCH/'experiment_manifest.json'),
        datasets={d.key:{k:v for k,v in d.report.items() if k!='row_audit'} for d in [c4,c8,d4]},
        grid_sha256=digest(ARCH/'common_grid.npy'),coverage_valid=int(mask.sum()),coverage_missing=int((~mask).sum()),
        d4_preprocessing='Explicit C4 numerical operator: native max-y -> native airPLS -> native maxabs -> in-range cubic -> base_train input-coordinate mean fill. No semantic signal conversion or extrapolation.',
        d4_registry_default_flip=False,d4_adapter_explicit_flip=True,
        signal_semantics='Unverified normalized signal; numerical orientation fixed before any query scoring; no D4-driven polarity selection',
        protocol=dict(d4_queries_per_class=5,d4_panels='fixed core9 all shots; expanded eligible11 at 1/5, eligible10 at 10',
            d4_incremental='2+1+1: C4 PP/PS exemplars -> D4 PE -> D4 PEVA; only checkpoints whose training base includes PP and PS',
            c8='Final-session C4 supports; same-label shared-ID paired C4/C8 queries balanced to available minimum per class; exclude C4 support and base_train IDs',
            masked_c4='Query-only post-preprocessing mask; same original C4 supports and heads; isolates missing input coordinates, not baseline crop effects',
            anchor='PP/PS head supports fixed C4 exemplars; matched C4 original/masked/C8 queries; D4 PP/PS separate queries, same two labels'),
        svm_C=1.,shrinkage=dict(tau=5.,ridge=.001,alpha='(n-1)/(n-1+5)',prior='unchanged original legal C4 base_train residual covariance'),
        pca='Unchanged archived PCA64 fitted only on each checkpoint base_train',
        costs='First 20 repeats; cached query features; actual new D4 support M1 encoding replayed for incremental arrivals; head-local update versus full SVM refit separated; preprocessing excluded',
        limitations='Exploratory; base validation previously used for checkpoint selection; C4/C8 resolution variants not independent instruments; cross-file specimen identity unresolved; D4 imputation does not recover chemical peaks',
        encoder_training_steps=0,numerical_threads=1,tensorflow_threads=4)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment_manifest.json').exists():
        assert json.loads((out/'experiment_manifest.json').read_text())==config
    else:
        write(out/'experiment_manifest.json',config)
        (out/'source').mkdir(exist_ok=True)
        for p in NEW_SOURCES:
            target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    if not args.smoke:
        assert args.validated_smoke and (args.validated_smoke/'COMPLETE.json').exists()
        check=json.loads((args.validated_smoke/'experiment_manifest.json').read_text())
        assert check['source_hashes']==config['source_hashes']
    np.save(out/'common_grid.npy',grid);np.save(out/'valid_wavenumber_mask.npy',mask)
    pd.DataFrame(dict(wavenumber=grid,valid=mask,missing=~mask)).to_csv(out/'coverage_mask.csv',index=False)
    for d in [c4,c8,d4]:
        pd.DataFrame(d.report['row_audit']).to_csv(out/f'{d.key}_row_audit.csv',index=False)
    pd.DataFrame([dict(dataset=d.key,**{'class':c},unique_count=n) for d in [c4,c8,d4] for c,n in d.report['unique_counts'].items()]).to_csv(out/'class_counts.csv',index=False)
    c4x=preprocess(c4,grid,ROOT/'results/research_cache',baseline=True,flip=True).astype(float)
    c8x=preprocess(c8,grid,ROOT/'results/research_cache',baseline=True,flip=True).astype(float)
    matched={s:i for i,s in enumerate(c8.ids)}
    for i,s in enumerate(c4.ids):
        if s in matched:
            assert c4.labels[i]==c8.labels[matched[s]],'ID-label conflict'
    write(out/'cross_dataset_audit.json',dict(c4_c8_shared_ids=len(set(c4.ids)&set(c8.ids)),
        c4_c8_shared_id_label_conflicts=0,c4_d4_shared_ids=len(set(c4.ids)&set(d4.ids)),
        c4_d4_native_hash_matches=len(set(c4.hashes)&set(d4.hashes)),
        native_hash_zero_is_nonexclusionary=True,specimen_groups='unverified; no invented ID-prefix grouping',
        d4_label_mapping={'PP':'poly(propylene)','PS':'poly(styrene)','PE':'kept separate from HDPE/LDPE',
            'PEVA':'retained label; not merged with PE','MORPHOTYPE 1/2':'unresolved identity; separate labels'}))
    pool_by_seed={}
    for seed in seeds:
        partition,episodes=d4_episodes(d4.labels,seed,repeats)
        write(out/f'd4_partitions_seed_{seed}.json',partition)
        write(out/f'd4_episodes_seed_{seed}.json',episodes)
        pool_by_seed[seed]=(partition,episodes)
    trial_folders=[]
    for task in tasks:
        for seed in seeds:
            name=task['name'];folder=out/name/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            trial_folders.append(folder)
            if (folder/'COMPLETE.json').exists():
                continue
            print(f'Preparing {name} seed {seed}',flush=True)
            original=ARCH/name/f'seed_{seed}';previous_folder=PREVIOUS/name/f'seed_{seed}'
            samples=pd.read_csv(original/'sample_manifest.csv')
            np.testing.assert_array_equal(samples['class'],c4.labels)
            np.testing.assert_array_equal(samples.source_id.astype(str),c4.ids)
            train=np.flatnonzero(samples.role.to_numpy()=='base_train')
            assert set(c4.labels[train])==set(task['base'])
            mean=base_input_mean(c4x,train)
            d4x,actual_mask=partial_input(d4,grid,mean,ROOT/'results/research_cache')
            np.testing.assert_array_equal(actual_mask,mask)
            masked=apply_missing_mask(c4x,mask,mean).astype(np.float32)
            np.savez_compressed(folder/'input_adapter.npz',grid=grid,mask=mask,mean=mean,base_train_indices=train)
            np.savez_compressed(folder/'adapted_d4_inputs.npz',x=d4x,ids=d4.ids,labels=d4.labels)
            pca=np.load(previous_folder/'frozen_pca64.npz')
            np.testing.assert_array_equal(pca['fit_indices'],train)
            np.testing.assert_allclose(pca['mean'],c4x[train].mean(axis=0),atol=1e-12)
            project=lambda x:(np.asarray(x,float)-pca['mean'])@pca['components'].T
            prior=np.load(previous_folder/'base_covariance.npy')
            model_file=original/'M1/encoder.keras';file_hash=digest(model_file)
            assert file_hash==json.loads((INC/name/f'seed_{seed}/frozen_check.json').read_text())['M1']['file_after']
            model=tf.keras.models.load_model(str(model_file),compile=False);model.trainable=False
            assert model.count_params()==47400
            before=weight_hash(model)
            infer=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec([None,1868],tf.float32)])
            z4=np.load(original/'M1/embeddings.npz')['embeddings']
            np.testing.assert_allclose(encode(infer,c4x[:8]),z4[:8],rtol=2e-4,atol=2e-4)
            np.testing.assert_allclose(base_covariance(z4[train],c4.labels[train]),prior,atol=1e-12)
            z8,zd,zm=encode(infer,c8x),encode(infer,d4x),encode(infer,masked)
            np.savez_compressed(folder/'frozen_features.npz',c8=z8,d4=zd,c4_masked=zm)
            v4=views(c4x,project(c4x),z4);v8=views(c8x,project(c8x),z8)
            vm=views(masked,project(masked),zm)
            vd=views(d4x[:,mask].astype(float),project(d4x),zd)
            v4d=views(c4x[:,mask],v4['pca64'],z4)
            v8d=views(c8x[:,mask],v8['pca64'],z8)
            vmd=views(masked[:,mask].astype(float),vm['pca64'],zm)
            contexts=json.loads((INC/name/f'seed_{seed}/sessions.json').read_text())
            contexts=[c for c in contexts if c['repeat']<repeats]
            rec=Recorder(name,seed)
            original_scores=pd.read_csv(previous_folder/'metrics.csv').set_index(['shot','repeat','session','method'])
            c8_pairs=[];same_score_checks=0
            for ctx in contexts:
                k,r,stage=ctx['shot'],ctx['repeat'],ctx['session']
                if stage!=2:
                    continue
                s,q=np.asarray(ctx['support']),np.asarray(ctx['query'])
                pair_groups=[[i for i in q if c4.labels[i]==c and c4.ids[i] in matched] for c in sorted(ctx['seen'])]
                min_q=min(map(len,pair_groups))
                paired4=np.array([i for g in pair_groups for i in g[:min_q]],int)
                paired8=np.array([matched[c4.ids[i]] for i in paired4],int)
                assert not set(c8.ids[paired8])&set(c4.ids[np.concatenate([train,s])])
                c8_pairs.append(dict(shot=k,repeat=r,c4_query=paired4.tolist(),c8_query=paired8.tolist(),queries_per_class=min_q))
                for method in METHODS:
                    head=fit_episode(method,prior,v4,s,c4.labels)
                    if method in EMBEDDING:
                        original_pred=head.predict(v4[head.view][q]);mask_pred=head.predict(vm[head.view][q])
                        a=rec.add('C4_original','six_class_coverage',k,r,2,method,c4.labels[q],original_pred,ctx['seen'],task['base'],ctx['previous_novel'],ctx['current_novel'])
                        b=rec.add('C4_masked_query','six_class_coverage',k,r,2,method,c4.labels[q],mask_pred,ctx['seen'],task['base'],ctx['previous_novel'],ctx['current_novel'])
                        parent_row=original_scores.loc[k,r,2,method]
                        for field in ['all_accuracy','all_macro_f1','base_accuracy','seen_novel_accuracy']:
                            np.testing.assert_allclose(a[field],parent_row[field],atol=1e-12)
                        same_score_checks+=1
                        rec.coverage.append(dict(task=name,seed=seed,shot=k,repeat=r,method=method,
                            original_accuracy=a['all_accuracy'],masked_accuracy=b['all_accuracy'],accuracy_gap_pp=100*(b['all_accuracy']-a['all_accuracy']),
                            original_macro_f1=a['all_macro_f1'],masked_macro_f1=b['all_macro_f1'],prediction_agreement=float(np.mean(original_pred==mask_pred)),
                            missing_coordinates=int((~mask).sum()),support_condition='unchanged original full C4 support'))
                    if min_q:
                        for scenario,v,indices,lab in [('C4_paired_C8',v4,paired4,c4.labels),('C8_resolution',v8,paired8,c8.labels)]:
                            pred=head.predict(v[head.view][indices])
                            rec.add(scenario,'six_class_paired_resolution',k,r,2,method,lab[indices],pred,ctx['seen'],task['base'],ctx['previous_novel'],ctx['current_novel'],queries_per_class=min_q)
            write(folder/'c8_paired_queries.json',c8_pairs)
            print(f'{name} seed {seed}: C4 coverage / C8 paired complete',flush=True)
            partition,episodes=pool_by_seed[seed]
            core=sorted(c for c,n in d4.report['unique_counts'].items() if n>=25)
            assert len(core)==9
            for k in [1,5,10,20]:
                eligible=sorted(c for c,p in partition.items() if len(p['support_pool'])>=k)
                panels=[('core9',core)]+([('eligible',eligible)] if eligible!=core else [])
                for panel,classes in panels:
                    base=[c for c in classes if c in task['base']]
                    novel=[c for c in classes if c not in base]
                    for r,episode in enumerate(episodes):
                        s=np.asarray([i for c in classes for i in episode[c]['support'][:k]],int)
                        q=np.asarray([i for c in classes for i in episode[c]['query']],int)
                        assert len(s)==len(classes)*k and not set(s)&set(q)
                        for method in METHODS:
                            started=time.perf_counter();head=fit_episode(method,prior,vd,s,d4.labels)
                            fit_ms=1000*(time.perf_counter()-started)
                            started=time.perf_counter();pred=head.predict(vd[head.view][q])
                            predict_ms=1000*(time.perf_counter()-started)
                            rec.add('D4_within',panel,k,r,0,method,d4.labels[q],pred,classes,base,(),novel)
                            if r<20:
                                rec.costs.append(dict(scenario='D4_within',protocol=panel,shot=k,repeat=r,session=0,method=method,
                                    episode_head_fit_ms=fit_ms,cached_query_prediction_ms=predict_ms,head_state_bytes=head.state_bytes()))
            print(f'{name} seed {seed}: D4 within complete',flush=True)
            if {'PP','PS'}.issubset(task['base']):
                base=['PP','PS'];new_order=['PE','PEVA']
                original_base=np.asarray(contexts[0]['support'])
                base_s=original_base[np.isin(c4.labels[original_base],base)]
                assert len(base_s)==40
                templates={m:fit_episode(m,prior,v4d,base_s,c4.labels) for m in METHODS}
                anchor_pools={c:np.flatnonzero((samples.role.to_numpy()=='base_validation')&(c4.labels==c)&np.isin(c4.ids,list(matched))) for c in base}
                assert min(map(len,anchor_pools.values()))>=5
                for k in [1,5,10,20]:
                    for r,episode in enumerate(episodes):
                        heads={m:h.clone() for m,h in templates.items()}
                        seen=base.copy()
                        for stage in range(3):
                            current=[] if stage==0 else [new_order[stage-1]]
                            previous=seen[2:].copy();added=[]
                            encoding_ms=0.
                            if current:
                                c=current[0];added=episode[c]['support'][:k];seen+=current
                                if r<20:
                                    tensor=tf.convert_to_tensor(d4x[added],tf.float32)
                                    started=time.perf_counter();actual=infer(tensor).numpy();encoding_ms=1000*(time.perf_counter()-started)
                                    np.testing.assert_allclose(actual,zd[added],rtol=2e-4,atol=2e-4)
                            q=np.asarray([i for c in seen for i in episode[c]['query']],int)
                            for method,head in heads.items():
                                started=time.perf_counter()
                                if added:
                                    head.register(current[0],vd[head.view][added])
                                registration_ms=1000*(time.perf_counter()-started)
                                started=time.perf_counter();head.prepare();refit_ms=1000*(time.perf_counter()-started)
                                started=time.perf_counter();pred=head.predict(vd[head.view][q]);prediction_ms=1000*(time.perf_counter()-started)
                                rec.add('C4_to_D4_incremental','PP_PS_to_PE_PEVA',k,r,stage,method,d4.labels[q],pred,seen,base,previous,current)
                                if r<20:
                                    transform_ms=0.
                                    if added and head.view=='pca64':
                                        started=time.perf_counter();project(d4x[added]);transform_ms=1000*(time.perf_counter()-started)
                                    feature_ms=encoding_ms if head.view=='M1' else transform_ms
                                    rec.costs.append(dict(scenario='C4_to_D4_incremental',protocol='PP_PS_to_PE_PEVA',shot=k,repeat=r,session=stage,method=method,
                                        feature_encoding_ms=feature_ms,new_class_statistics_or_bank_ms=registration_ms,
                                        full_head_assembly_refit_ms=refit_ms,svm_full_refit_ms=refit_ms if method.endswith('svm') else np.nan,
                                        local_class_registration_ms=(feature_ms+registration_ms) if added else 0.,
                                        total_update_ms=(feature_ms+registration_ms+refit_ms) if added else 0.,
                                        cached_query_prediction_ms=prediction_ms,head_state_bytes=head.state_bytes()))
                                if stage==0 and k==5:
                                    rng=np.random.default_rng(61000+1000*seed+r)
                                    qa=np.concatenate([rng.choice(anchor_pools[c],5,replace=False) for c in base])
                                    q8=np.array([matched[c4.ids[i]] for i in qa])
                                    assert not set(c8.ids[q8])&set(c4.ids[np.concatenate([train,base_s])])
                                    for scenario,v,qi,lab in [('anchor_C4',v4d,qa,c4.labels),('anchor_C4_masked',vmd,qa,c4.labels),('anchor_C8',v8d,q8,c8.labels)]:
                                        p=head.predict(v[head.view][qi]);rec.add(scenario,'PP_PS_base_anchor',0,r,0,method,lab[qi],p,base,base)
                                    rec.add('anchor_D4','PP_PS_base_anchor',0,r,0,method,d4.labels[q],pred,base,base)
                print(f'{name} seed {seed}: D4 incremental complete',flush=True)
            rec.save(folder)
            assert file_hash==digest(model_file) and before==weight_hash(model)
            write(folder/'COMPLETE.json',dict(rows=len(rec.rows),original_metric_checks=same_score_checks,
                checkpoint_sha256_before=file_hash,checkpoint_sha256_after=digest(model_file),weights_before=before,weights_after=weight_hash(model),
                encoder_parameters=model.count_params(),encoder_training_steps=0,
                pca_sha256=digest(previous_folder/'frozen_pca64.npz'),prior_sha256=digest(previous_folder/'base_covariance.npy'),
                d4_episodes_sha256=digest(out/f'd4_episodes_seed_{seed}.json'),incremental_eligible={'PP','PS'}.issubset(task['base'])))
            tf.keras.backend.clear_session()
            print(f'COMPLETE {name} seed {seed}: {len(rec.rows)} metrics',flush=True)
    aggregate=pd.concat([pd.read_csv(f/'metrics.csv') for f in trial_folders],ignore_index=True)
    aggregate.to_csv(out/'metrics.csv',index=False)
    for scenario,filename in [('D4_within','d4_fewshot_metrics.csv'),('C4_to_D4_incremental','c4_to_d4_incremental_metrics.csv')]:
        aggregate[aggregate.scenario==scenario].to_csv(out/filename,index=False)
    aggregate[aggregate.scenario.isin(['C8_resolution','C4_paired_C8'])].to_csv(out/'c8_transfer_metrics.csv',index=False)
    aggregate[aggregate.scenario.str.startswith('anchor_')].to_csv(out/'transfer_anchor_metrics.csv',index=False)
    for filename in ['per_class_results.csv','coverage_mask_comparison.csv','costs.csv']:
        pd.concat([pd.read_csv(f/filename) for f in trial_folders],ignore_index=True).to_csv(out/filename,index=False)
    write(out/'COMPLETE.json',dict(trials=len(trial_folders),rows=len(aggregate),encoder_training_steps=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/heterogeneous_ftir_transfer_v1')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--validated-smoke',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
