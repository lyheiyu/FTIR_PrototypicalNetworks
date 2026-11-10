# -*- coding: utf-8 -*-
"""One additional PCA128 baseline. Existing results/checkpoints are read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from ftir_research_data import ROOT,load_dataset,preprocess
from ftir_research_core import base_covariance
from frozen_prototype_heads_v1 import FrozenPrototype
from heterogeneous_ftir_adapter_v1 import fast_metrics

PARENT=ROOT/'results/bidirectional_c4_c8_final'
EP_PARENT=ROOT/'results/protonet_bidirectional_final'
METHOD='PCA128_shrinkage'
METRICS=['all_accuracy','all_macro_f1','base_accuracy','base_macro_f1','seen_novel_accuracy','seen_novel_macro_f1',
         'previous_novel_accuracy','previous_novel_macro_f1','current_novel_accuracy','current_novel_macro_f1',
         'base_to_novel_error_rate','novel_to_base_error_rate','previous_novel_change_pp','base_change_from_session0_pp']


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def evaluate(folder,task,seed,split,episodes,labels,features,priors,cases):
    templates={};rows=[];costs=[];cms={};checks=0;numeric_checks=0
    for case,(source,bs,ns,qs) in cases.items():
        h=FrozenPrototype('shrinkage',128,priors[source],reg=.001,tau=5.)
        for c in sorted(task['base']):h.register(c,features[source][bs][episodes[0]['base_support'][c]])
        templates[case]=h
    for k in [1,5,10,20]:
        for ep in episodes:
            heads={c:h.clone() for c,h in templates.items()};seen=list(task['base']);first={};base0={}
            for stage in range(3):
                current=[] if stage==0 else task['novel_sessions'][stage-1];previous=seen[4:].copy();seen+=current
                qi=np.array([i for c in sorted(seen) for i in ep['query'][c]],int)
                support=set(sum(ep['base_support'].values(),[]))
                for c in seen[4:]:support.update(ep['novel_support_order'][c][:k])
                assert not set(qi)&(support|set(split['train'])|set(split['validation']))
                for case,(source,bs,ns,qs) in cases.items():
                    h=heads[case];snapshot={c:(m.copy(),h.precisions[c].copy()) for c,m in h.centers.items()} if ep['repeat']==0 else {}
                    t=time.perf_counter()
                    if current:h.register(current[0],features[source][ns][ep['novel_support_order'][current[0]][:k]])
                    register_ms=1000*(time.perf_counter()-t)
                    for c,(m,p) in snapshot.items():
                        np.testing.assert_array_equal(m,h.centers[c]);np.testing.assert_array_equal(p,h.precisions[c]);checks+=1
                    query=features[source][qs][qi];t=time.perf_counter();prediction=h.predict(query);prediction_ms=1000*(time.perf_counter()-t)
                    # Independent full 128-D quadratic solves, one complete episode per shot/setting.
                    if ep['repeat']==0 and stage==2:
                        ds=[]
                        for c in sorted(seen):
                            si=ep['base_support'][c] if c in task['base'] else ep['novel_support_order'][c][:k]
                            z=features[source][bs if c in task['base'] else ns][si];mean=z.mean(0);r=z-mean
                            alpha=(len(z)-1)/(len(z)-1+5.)
                            cov=alpha*(r.T@r/len(z))+(1-alpha)*priors[source]+.001*np.eye(128)
                            diff=query-mean;ds.append(np.sum(diff*np.linalg.solve(cov,diff.T).T,axis=1))
                        np.testing.assert_array_equal(prediction,np.array(sorted(seen))[np.argmin(ds,axis=0)]);numeric_checks+=1
                    met,_,cm=fast_metrics(labels[qi],prediction,seen,task['base'],previous,current)
                    tb=np.isin(labels[qi],task['base']);tn=~tb
                    met['base_to_novel_error_rate']=float(np.mean(np.isin(prediction[tb],seen[4:])))
                    met['novel_to_base_error_rate']=float(np.mean(np.isin(prediction[tn],task['base']))) if tn.any() else np.nan
                    if stage==0:base0[case]=met['base_accuracy']
                    if stage==1:first[case]=met['current_novel_accuracy']
                    met['previous_novel_change_pp']=100*(met['previous_novel_accuracy']-first[case]) if stage==2 else np.nan
                    met['base_change_from_session0_pp']=100*(met['base_accuracy']-base0[case])
                    meta=dict(task=task['name'],seed=seed,setting=case,encoder_source=source,base_source=bs,novel_source=ns,query_source=qs,
                              shot=k,repeat=ep['repeat'],session=stage,method=METHOD)
                    rows.append(dict(meta,**met,query_count=len(qi)))
                    key=(case,k,stage,tuple(sorted(seen)));cms[key]=cms.get(key,np.zeros_like(cm))+cm
                    if ep['repeat']<20:costs.append(dict(meta,head_local_registration_ms=register_ms,cached_query_prediction_ms=prediction_ms))
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(costs).to_csv(folder/'costs.csv',index=False)
    arrays,index,perclass={},{},[]
    means=pd.DataFrame(rows).groupby(['setting','shot','session']).all_accuracy.mean()
    for i,((case,k,stage,cs),cm) in enumerate(cms.items()):
        token=f'cm_{i}';arrays[token]=cm
        meta=dict(task=task['name'],seed=seed,setting=case,shot=k,session=stage,method=METHOD);index[token]=dict(meta,classes=list(cs))
        np.testing.assert_allclose(np.trace(cm)/cm.sum(),means.loc[(case,k,stage)],atol=1e-12)
        for j,c in enumerate(cs):
            n=cm[j].sum();p=cm[:,j].sum();tp=cm[j,j]
            perclass.append(dict(meta,**{'class':c},recall=tp/n if n else 0,precision=tp/p if p else 0,f1=2*tp/(n+p) if n+p else 0,query_occurrences=int(n)))
    np.savez_compressed(folder/'confusions.npz',**arrays);write(folder/'confusion_index.json',index)
    pd.DataFrame(perclass).to_csv(folder/'per_class_metrics.csv',index=False)
    return dict(metric_rows=len(rows),old_class_state_checks=checks,dense_solve_prediction_checks=numeric_checks)


def summarize(out,folders):
    new=pd.concat([pd.read_csv(f/'metrics.csv') for f in folders],ignore_index=True);new.to_csv(out/'metrics.csv',index=False)
    reference=[]
    for f in folders:
        relative=f.relative_to(out)
        a=pd.read_csv(PARENT/relative/'metrics.csv');a=a[a.method=='PCA64_shrinkage'].copy()
        b=pd.read_csv(EP_PARENT/relative/'new_metrics.csv');b=b[b.method=='EP_shrinkage'].copy();b['method']='EP_M1_128_shrinkage'
        reference.extend([a,b])
    all_metrics=pd.concat([new]+reference,ignore_index=True)
    keys=['task','seed','setting','shot','repeat','session']
    assert not all_metrics.duplicated(keys+['method']).any()
    paired_rows=all_metrics.groupby(keys).size();assert (paired_rows==3).all()
    all_metrics.to_csv(out/'comparison_metrics.csv',index=False)
    units=all_metrics.groupby(['task','seed','setting','shot','session','method'],as_index=False)[METRICS].mean()
    units.to_csv(out/'task_seed_metrics.csv',index=False)
    agg=units.groupby(['setting','shot','session','method'])[METRICS].agg(['mean','std']).reset_index()
    agg.columns=['_'.join(c).strip('_') for c in agg.columns];agg.to_csv(out/'summary_all_shots_sessions.csv',index=False)
    primary=units[(units.session==2)&units.shot.isin([5,10])].groupby(['task','seed','setting','method'],as_index=False)[METRICS].mean()
    primary.to_csv(out/'primary_task_seed_metrics.csv',index=False)
    summary=primary.groupby(['setting','method'],as_index=False)[METRICS].mean();summary.to_csv(out/'primary_summary.csv',index=False)
    p=primary.pivot(index=['task','seed','setting'],columns='method',values=METRICS);diffs=[]
    for idx,row in p.iterrows():
        for baseline in ['PCA64_shrinkage','EP_M1_128_shrinkage']:
            diffs.append(dict(task=idx[0],seed=idx[1],setting=idx[2],comparison='PCA128_minus_'+baseline,
                **{m+'_delta_pp':(row[m,METHOD]-row[m,baseline])*(1 if m.endswith('_pp') else 100) for m in METRICS}))
    diffs=pd.DataFrame(diffs);diffs.to_csv(out/'paired_differences.csv',index=False)
    diffs.groupby(['setting','comparison'],as_index=False).agg(accuracy_delta_pp=('all_accuracy_delta_pp','mean'),
        macro_f1_delta_pp=('all_macro_f1_delta_pp','mean'),accuracy_positive_pairs=('all_accuracy_delta_pp',lambda x:int((x>0).sum())),
        paired_units=('seed','size')).to_csv(out/'paired_summary.csv',index=False)
    for name in ['per_class_metrics.csv','costs.csv']:
        pd.concat([pd.read_csv(f/name) for f in folders],ignore_index=True).to_csv(out/name,index=False)
    # Compact Chinese summary; figures and additional model variants are outside this request.
    lookup=summary.set_index(['setting','method']);lines=[]
    for case in 'ABCDEF':
        cells=[]
        for method in ['PCA64_shrinkage',METHOD,'EP_M1_128_shrinkage']:
            r=lookup.loc[(case,method)];cells.append(f'{100*r.all_accuracy:.2f} / {100*r.all_macro_f1:.2f}')
        lines.append('| '+case+' | '+' | '.join(cells)+' |')
    delta=diffs.groupby(['setting','comparison'])[['all_accuracy_delta_pp','all_macro_f1_delta_pp']].mean()
    contrasts=[]
    for baseline in ['PCA64_shrinkage','EP_M1_128_shrinkage']:
        contrasts.append('PCA128 相对 '+baseline+'：'+ '；'.join(f'{c} {delta.loc[(c,"PCA128_minus_"+baseline),"all_accuracy_delta_pp"]:+.2f}/{delta.loc[(c,"PCA128_minus_"+baseline),"all_macro_f1_delta_pp"]:+.2f}' for c in 'ABCDEF')+' pp（Accuracy/Macro-F1）。')
    text='''# PCA128 + Shrinkage 补充对照

仅新增 PCA128；未重训 M0、M1 或 ProtoNet，未改写历史结果。三个留出任务×三个种子，A–F、200个固定 episode、1/5/10/20-shot 和三个增量 session 均与原实验一致。

以下为最终六类 session，先在各 task/seed 内对200次重复与5/10-shot取均值，再对9个配对单元取均值。单元格为 **Accuracy / Macro-F1（%）**。所有方法均使用同一收缩头：协方差 /n，alpha=(n−1)/(n−1+5)，ridge=.001。EP-M1-128 表示既有 episodic M1 的128维嵌入加收缩头。

| 设置 | PCA64 + 收缩 | PCA128 + 收缩 | EP-M1-128 + 收缩 |
| --- | --- | --- | --- |
'''+ '\n'.join(lines)+'\n\n'+'\n\n'.join(contrasts)+'''

A/B 为 C4/C8 同域；C/D 为双向跨分辨率、基类参考保持源域；E/F 额外使用目标域的有标签基类参考。PCA128 使用与 PCA64 相同的 randomized solver、种子和不白化配置，仅维数改为128。每个 PCA 与协方差先验都只拟合对应源域原始 base-training 分组，随后冻结；没有使用 validation/query/novel support 拟合 PCA 或先验。

全部 shot/session 的 Accuracy、Macro-F1 和分组指标见 summary_all_shots_sessions.csv；逐 task/seed 配对差值见 paired_differences.csv。PCA64 与 EP 直接复用保存的逐 episode 结果。相同数值 ridge 不代表不同特征空间中的相同相对正则强度；PCA64/128 的 randomized 分解亦不是强制嵌套的同一组主轴。此为既有探索性协议中的维度对照，不能推导实物独立或外部确证结论。没有新增调参、损失、编码器或其他实验。
'''
    (out/'summary_zh.md').write_text(text,encoding='utf-8')


def run(out):
    out=out.resolve();assert out!=PARENT.resolve() and out!=EP_PARENT.resolve()
    parent=json.loads((PARENT/'experiment_manifest.json').read_text());assert (PARENT/'COMPLETE.json').exists() and (EP_PARENT/'COMPLETE.json').exists()
    cases={c:tuple(v[k] for k in ['encoder','base_references','novel_support','query']) for c,v in parent['cases'].items()}
    assert set(cases)==set('ABCDEF') and parent['seeds']==[0,1,2]
    protected=[PARENT/'experiment_manifest.json',EP_PARENT/'experiment_manifest.json',PARENT/'common_grid.npy',PARENT/'paired_group_representatives.csv']
    for task in parent['tasks']:
        for seed in parent['seeds']:
            p=PARENT/task['name']/f'seed_{seed}';e=EP_PARENT/task['name']/f'seed_{seed}'
            protected.extend([p/'split.json',p/'episodes.json',p/'metrics.csv',e/'new_metrics.csv'])
            protected.extend([p/s/'M1/frozen_views.npz' for s in ['c4','c8']])
    protected.extend(p for pattern in ['*.keras','*.h5'] for p in (ROOT/'results').rglob(pattern) if out not in p.parents)
    hashes={str(p.relative_to(ROOT)):digest(p) for p in protected}
    sources=['run_pca128_shrinkage_bidirectional_v1.py','ftir_research_data.py','ftir_research_core.py','frozen_prototype_heads_v1.py','heterogeneous_ftir_adapter_v1.py','PLS.py','utils.py']
    config=dict(experiment='pca128_shrinkage_bidirectional_v1',parent=str(PARENT),ep_parent=str(EP_PARENT),tasks=parent['tasks'],seeds=[0,1,2],
        cases=parent['cases'],shots=[1,5,10,20],repeats=200,primary_shots=[5,10],
        PCA=dict(dimensions=128,svd_solver='randomized',whiten=False,random_state='matched seed',fit='source base train only; exact saved original group-safe train indices'),
        classifier=dict(ridge=.001,tau=5.,covariance_divisor='n',alpha='(n-1)/(n-1+5)',prior='source base-train within-class covariance only; frozen'),
        preprocessing=parent['preprocessing'],scope=parent['scope'],source_hashes={p:digest(ROOT/p) for p in sources},protected_original_file_hashes=hashes)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment_manifest.json').exists():assert json.loads((out/'experiment_manifest.json').read_text())==config
    else:
        write(out/'experiment_manifest.json',config)
        for name in sources:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    groups=pd.read_csv(PARENT/'paired_group_representatives.csv');labels=groups.label.to_numpy();grid=np.load(PARENT/'common_grid.npy');assert len(grid)==1868
    x={}
    for source in ['c4','c8']:
        d=load_dataset(source);assert d.report['source_sha256']==parent['dataset_hashes'][source]
        # New cache only: the old research cache is never written.
        x[source]=preprocess(d,grid,out/'preprocessing_cache',baseline=True,flip=True)[groups[f'{source}_index'].to_numpy()].astype(float)
    for name in ['paired_group_representatives.csv','common_grid.npy']:(out/name).write_bytes((PARENT/name).read_bytes())
    folders=[];fit_rows=[];old_checks=0;numeric_checks=0
    for task in parent['tasks']:
        for seed in [0,1,2]:
            p=PARENT/task['name']/f'seed_{seed}';f=out/task['name']/f'seed_{seed}';f.mkdir(parents=True,exist_ok=True);folders.append(f)
            for name in ['split.json','episodes.json']:(f/name).write_bytes((p/name).read_bytes())
            split=json.loads((p/'split.json').read_text());episodes=json.loads((p/'episodes.json').read_text());assert len(episodes)==200
            train=np.array(split['train']);assert set(labels[train])==set(task['base'])
            features,priors={},{}
            for source in ['c4','c8']:
                target=f/source;target.mkdir(exist_ok=True);saved=target/'frozen_pca128.npz'
                if not saved.exists():
                    # Verify reconstructed input against saved PCA64 transforms without refitting PCA64.
                    ce=np.load(p/source/'M1/frozen_views.npz');np.testing.assert_array_equal(ce['fit_indices'],train)
                    np.testing.assert_allclose(x[source][train].mean(0),ce['pca_mean'],atol=1e-12)
                    for resolution in ['c4','c8']:
                        np.testing.assert_allclose((x[resolution]-ce['pca_mean'])@ce['pca_components'].T,ce['PCA64_'+resolution],atol=1e-10,rtol=1e-10)
                    t=time.perf_counter();pca=PCA(n_components=128,svd_solver='randomized',random_state=seed,whiten=False).fit(x[source][train]);seconds=time.perf_counter()-t
                    z={s:pca.transform(a) for s,a in x.items()};prior=base_covariance(z[source][train],labels[train])
                    np.savez_compressed(saved,mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_,
                        singular_values=pca.singular_values_,PCA128_c4=z['c4'],PCA128_c8=z['c8'],prior=prior,fit_indices=train)
                    write(target/'fit_cost.json',dict(task=task['name'],seed=seed,source=source,fit_seconds=seconds,training_groups=len(train),explained_variance_fraction=float(pca.explained_variance_ratio_.sum())))
                data=np.load(saved);np.testing.assert_array_equal(data['fit_indices'],train);assert data['components'].shape==(128,1868)
                features[source]={s:data['PCA128_'+s] for s in ['c4','c8']};priors[source]=data['prior']
                np.testing.assert_allclose(priors[source],base_covariance(features[source][source][train],labels[train]),atol=1e-12)
                fit_rows.append(json.loads((target/'fit_cost.json').read_text()))
            if not (f/'COMPLETE.json').exists():
                print('EVALUATE PCA128',task['name'],'seed',seed,flush=True)
                frozen_hashes={s:digest(f/s/'frozen_pca128.npz') for s in ['c4','c8']}
                result=evaluate(f,task,seed,split,episodes,labels,features,priors,cases)
                for s,h in frozen_hashes.items():assert digest(f/s/'frozen_pca128.npz')==h
                write(f/'COMPLETE.json',dict(result,pca_updates_during_incremental=0,split_sha256=digest(p/'split.json'),episodes_sha256=digest(p/'episodes.json')))
            result=json.loads((f/'COMPLETE.json').read_text());old_checks+=result['old_class_state_checks'];numeric_checks+=result['dense_solve_prediction_checks']
            print('COMPLETE',task['name'],'seed',seed,flush=True)
    pd.DataFrame(fit_rows).to_csv(out/'pca_fit_costs.csv',index=False);summarize(out,folders)
    for name,h in hashes.items():assert digest(ROOT/name)==h
    write(out/'protocol_audit.json',dict(status='passed',paired_trials=9,PCA128_fits=18,retrained_neural_encoders=0,incremental_PCA_updates=0,
        existing_protected_files_unchanged=len(hashes),old_class_state_checks=old_checks,independent_dense_128D_solve_checks=numeric_checks,
        source_training_only_PCA_and_prior=True,preprocessing_matches_saved_PCA64_features=True,original_episodes_and_splits_reused=True,
        specimen_independence_verified=False))
    write(out/'COMPLETE.json',dict(paired_trials=9,new_metric_rows=129600,comparison_metric_rows=388800,neural_retraining=0))
    print('Finished:',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/pca128_shrinkage_bidirectional_v1')
    with threadpool_limits(limits=1):run(parser.parse_args().output)
