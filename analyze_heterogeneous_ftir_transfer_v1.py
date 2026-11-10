"""Audit, plot and report frozen heterogeneous transfer without query tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from heterogeneous_ftir_adapter_v1 import base_input_mean,observed_mask,fast_metrics
from ftir_research_data import ROOT,load_dataset,preprocess

LABELS={'spectrum_1nn':'Spectrum 1-NN','pca64_1nn':'Frozen PCA64 1-NN','M1_1nn':'M1 1-NN',
        'M1_proto_euclidean':'M1 Euclidean','M1_proto_shared':'M1 Shared','M1_proto_shrinkage':'M1 Shrinkage','spectrum_svm':'Spectrum SVM'}
METRICS=['all_accuracy','all_macro_f1','base_accuracy','base_macro_f1','previous_novel_accuracy',
         'previous_novel_macro_f1','current_novel_accuracy','current_novel_macro_f1','seen_novel_accuracy','seen_novel_macro_f1']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md(frame):
    frame=frame.reset_index()
    def val(x):
        if isinstance(x,(float,np.floating)):
            return '' if np.isnan(x) else f'{x:.3f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(val,row))+' |' for row in frame.itertuples(index=False,name=None)])


def paired(data,a,b,protocol,metric='all_accuracy',primary=True):
    selected=data[(data.protocol==protocol)&data.scenario.isin([a,b])]
    if primary:
        selected=selected[selected.shot.isin([5,10])]
    trial=selected.groupby(['task','seed','shot','method','scenario'])[metric].mean().unstack('scenario')
    delta=100*(trial[a]-trial[b])
    result=delta.groupby(['task','seed','method']).mean().rename('delta_pp').reset_index()
    return result


def audit(out,config):
    arch=ROOT/'results/encoder_architecture_v1'
    previous=ROOT/'results/external_transfer_prototype_v1'
    incremental=ROOT/'results/frozen_incremental_v1'
    parent=json.loads((previous/'experiment_manifest.json').read_text())
    for name,value in config['source_hashes'].items():
        assert digest(ROOT/name)==digest(out/'source'/name)==value,name
    for name,value in parent['source_hashes'].items():
        assert digest(ROOT/name)==value,name
    assert digest(previous/'experiment_manifest.json')==config['previous_manifest_sha256']
    assert digest(arch/'experiment_manifest.json')==config['architecture_manifest_sha256']
    grid=np.load(out/'common_grid.npy');mask=np.load(out/'valid_wavenumber_mask.npy')
    d4,c4,c8=[load_dataset(k) for k in ['d4','c4','c8']]
    for data in [d4,c4,c8]:
        assert data.report['source_sha256']==config['datasets'][data.key]['source_sha256']
    np.testing.assert_array_equal(mask,observed_mask(d4.wave,grid))
    assert len(d4.x)==954 and len(set(d4.labels))==11 and (~mask).sum()==107
    x=preprocess(c4,grid,ROOT/'results/research_cache',baseline=True,flip=True).astype(float)
    x8=preprocess(c8,grid,ROOT/'results/research_cache',baseline=True,flip=True).astype(float)
    match8={s:i for i,s in enumerate(c8.ids)}
    shared4=np.array([i for i,s in enumerate(c4.ids) if s in match8])
    shared8=np.array([match8[c4.ids[i]] for i in shared4])
    a,b=x[shared4],x8[shared8]
    ac,bc=a-a.mean(axis=1,keepdims=True),b-b.mean(axis=1,keepdims=True)
    denom=np.linalg.norm(ac,axis=1)*np.linalg.norm(bc,axis=1)
    corr=np.divide(np.sum(ac*bc,axis=1),denom,out=np.full(len(a),np.nan),where=denom>0)
    alignment=pd.DataFrame(dict(source_id=c4.ids[shared4],**{'class':c4.labels[shared4]},
        common_grid_pearson=corr,common_grid_rmse=np.sqrt(np.mean((a-b)**2,axis=1))))
    alignment.to_csv(out/'c4_c8_record_alignment_audit.csv',index=False)
    alignment.groupby('class')[['common_grid_pearson','common_grid_rmse']].agg(['mean','median','min','max']).to_csv(out/'c4_c8_record_alignment_summary.csv')
    trial_checks=[]
    for task in config['tasks']:
        for seed in config['seeds']:
            name=task['name'];folder=out/name/f'seed_{seed}';original=arch/name/f'seed_{seed}'
            complete=json.loads((folder/'COMPLETE.json').read_text())
            assert complete['checkpoint_sha256_before']==complete['checkpoint_sha256_after']==digest(original/'M1/encoder.keras')
            assert complete['weights_before']==complete['weights_after']
            assert complete['encoder_parameters']==47400 and complete['encoder_training_steps']==0
            old_frozen=json.loads((incremental/name/f'seed_{seed}/frozen_check.json').read_text())['M1']
            assert old_frozen['file_after']==complete['checkpoint_sha256_before']
            assert digest(previous/name/f'seed_{seed}/frozen_pca64.npz')==complete['pca_sha256']
            assert digest(previous/name/f'seed_{seed}/base_covariance.npy')==complete['prior_sha256']
            adapter=np.load(folder/'input_adapter.npz')
            samples=pd.read_csv(original/'sample_manifest.csv')
            train=np.flatnonzero(samples.role.to_numpy()=='base_train')
            np.testing.assert_array_equal(train,adapter['base_train_indices'])
            np.testing.assert_array_equal(adapter['mean'],base_input_mean(x,train))
            adapted=np.load(folder/'adapted_d4_inputs.npz')['x']
            np.testing.assert_array_equal(adapted[:,~mask],np.broadcast_to(adapter['mean'][~mask],adapted[:,~mask].shape))
            pairs=json.loads((folder/'c8_paired_queries.json').read_text())
            sessions=json.loads((incremental/name/f'seed_{seed}/sessions.json').read_text())
            support_lookup={(c['shot'],c['repeat']):c['support'] for c in sessions if c['session']==2}
            for pair in pairs:
                qa,qb=pair['c4_query'],pair['c8_query']
                np.testing.assert_array_equal(c4.labels[qa],c8.labels[qb])
                np.testing.assert_array_equal(c4.ids[qa],c8.ids[qb])
                illegal=set(c4.ids[train])|set(c4.ids[support_lookup[pair['shot'],pair['repeat']]])
                assert not illegal&set(c8.ids[qb])
            partition=json.loads((out/f'd4_partitions_seed_{seed}.json').read_text())
            episodes=json.loads((out/f'd4_episodes_seed_{seed}.json').read_text())
            assert digest(out/f'd4_episodes_seed_{seed}.json')==complete['d4_episodes_sha256']
            for c,p in partition.items():
                assert not set(p['support_pool'])&set(p['query_pool'])
                assert set(d4.labels[p['support_pool']])==set(d4.labels[p['query_pool']])=={c}
            for episode in episodes:
                for c,e in episode.items():
                    assert set(e['support'])<=set(partition[c]['support_pool'])
                    assert set(e['query'])<=set(partition[c]['query_pool']) and len(e['query'])==5
            trial_checks.append(dict(task=name,seed=seed,**complete))
    pd.DataFrame(trial_checks).to_csv(out/'frozen_encoder_verification.csv',index=False)
    result=dict(status='passed',trials=len(trial_checks),original_C4_metric_checks=sum(c['original_metric_checks'] for c in trial_checks),
        missing_coordinates=107,observed_coordinates=int(mask.sum()),encoder_training_steps=0,
        old_sources_and_checkpoints_unchanged=True,imputation_base_train_only=True,PCA_and_prior_unchanged=True,
        c8_query_training_and_support_ID_exclusion=True,D4_support_query_pools_disjoint=True,
        specimen_independence_verified=False,physical_signal_semantics_verified=False,
        analysis_source_sha256=digest(__file__))
    (out/'protocol_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main(out):
    config=json.loads((out/'experiment_manifest.json').read_text())
    assert (out/'COMPLETE.json').exists()
    checked=audit(out,config)
    data=pd.read_csv(out/'metrics.csv')
    keys=['task','seed','scenario','protocol','shot','session','method']
    trial=data.groupby(keys)[METRICS].mean().reset_index()
    trial.to_csv(out/'trial_metrics.csv',index=False)
    summary=trial.groupby(['scenario','protocol','shot','session','method'])[METRICS].mean().reset_index()
    summary.to_csv(out/'summary_metrics.csv',index=False)
    inc_trial=trial[trial.scenario=='C4_to_D4_incremental'].groupby(['task','seed','shot','method'])[['all_accuracy','all_macro_f1']].mean()
    inc_trial.to_csv(out/'average_incremental_metrics.csv')
    strict=trial[(trial.task=='holdout_pp_ps')&(trial.scenario=='D4_within')]
    strict.groupby(['protocol','shot','method'])[METRICS].mean().to_csv(out/'strict_label_disjoint_summary.csv')
    coverage=pd.read_csv(out/'coverage_mask_comparison.csv')
    coverage_trial=coverage.groupby(['task','seed','shot','method'])[['original_accuracy','masked_accuracy','accuracy_gap_pp','prediction_agreement']].mean().reset_index()
    coverage_trial.to_csv(out/'coverage_trial_summary.csv',index=False)
    coverage_summary=coverage_trial.groupby(['shot','method'])[['original_accuracy','masked_accuracy','accuracy_gap_pp','prediction_agreement']].mean()
    coverage_summary.to_csv(out/'coverage_summary.csv')
    c8gap=paired(data,'C8_resolution','C4_paired_C8','six_class_paired_resolution',primary=False)
    c8gap.to_csv(out/'c8_transfer_gaps_pp.csv',index=False)
    method_deltas=[]
    for scenario,protocol,stage in [('D4_within','core9',0),('C4_to_D4_incremental','PP_PS_to_PE_PEVA',2)]:
        subset=trial[(trial.scenario==scenario)&(trial.protocol==protocol)&(trial.session==stage)&trial.shot.isin([5,10])]
        for metric in ['all_accuracy','all_macro_f1','seen_novel_accuracy','current_novel_accuracy','base_accuracy']:
            tab=subset.groupby(['task','seed','method'])[metric].mean().unstack('method')
            for ref in ['M1_1nn','M1_proto_euclidean','M1_proto_shared','spectrum_1nn','pca64_1nn','spectrum_svm']:
                for (task,seed),value in (100*(tab['M1_proto_shrinkage']-tab[ref])).items():
                    method_deltas.append(dict(scenario=scenario,protocol=protocol,metric=metric,task=task,seed=seed,reference=ref,shrinkage_delta_pp=value))
    deltas=pd.DataFrame(method_deltas)
    deltas.to_csv(out/'d4_paired_method_deltas_pp.csv',index=False)
    paired_summary=deltas.groupby(['scenario','metric','reference']).shrinkage_delta_pp.agg(['mean','min','max','count',lambda x:int((x>1e-9).sum())])
    paired_summary.rename(columns={'<lambda_0>':'positive_trials'},inplace=True)
    paired_summary.to_csv(out/'d4_paired_method_summary.csv')
    costs=pd.read_csv(out/'costs.csv')
    cost_summary=costs.groupby(['scenario','protocol','shot','session','method']).mean(numeric_only=True)
    cost_summary.to_csv(out/'cost_summary.csv')
    historical=pd.read_csv(ROOT/'results/external_transfer_prototype_v1/metrics.csv')
    historical['method']=historical.method.replace({'raw_1nn':'spectrum_1nn','raw_svm':'spectrum_svm'})
    historical=historical[(historical.session==2)&historical.method.isin(LABELS)]
    hist=historical.groupby(['task','seed','shot','method'])[METRICS].mean().groupby(['shot','method']).mean()
    hist.to_csv(out/'c4_historical_reference.csv')
    figures(out,config,summary,coverage_summary)
    report(out,config,checked,summary,coverage_summary,c8gap,paired_summary,cost_summary,hist)
    if (out/'full_mahalanobis_control/COMPLETE.json').exists():
        full_control_report(out,config,trial)
    print('Analysis and audit complete:',checked)
    print(summary[(summary.shot==5)&(((summary.scenario=='D4_within')&(summary.protocol=='core9'))|((summary.scenario=='C4_to_D4_incremental')&(summary.session==2)))][['scenario','method','all_accuracy','all_macro_f1','seen_novel_accuracy']].to_string(index=False))


def full_control_report(out,config,parent_trial):
    root=out/'full_mahalanobis_control'
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    assert digest(root/'source.py')==digest(ROOT/'supplement_heterogeneous_full_mahalanobis_v1.py')==manifest['source_sha256']
    assert digest(ROOT/'run_full_mahalanobis_ablation_v1.py')==manifest['head_source_sha256']
    assert digest(out/'experiment_manifest.json')==manifest['parent_manifest_sha256']
    original_checks=0
    for task in config['tasks']:
        for seed in config['seeds']:
            parent=out/task['name']/f'seed_{seed}';folder=root/task['name']/f'seed_{seed}'
            done=json.loads((folder/'COMPLETE.json').read_text())
            assert digest(parent/'frozen_features.npz')==done['features_sha256']
            assert digest(out/f'd4_episodes_seed_{seed}.json')==done['episodes_sha256']
            original_checks+=done['original_C4_full_metric_checks']
    metrics=pd.read_csv(root/'metrics.csv')
    keys=['task','seed','scenario','protocol','shot','session','method']
    trial=pd.concat([parent_trial,metrics.groupby(keys)[METRICS].mean().reset_index()],ignore_index=True)
    summary=trial.groupby(['scenario','protocol','shot','session','method'])[METRICS].mean().reset_index()
    summary.to_csv(out/'full_class_mahalanobis_comparison.csv',index=False)
    trial[(trial.task=='holdout_pp_ps')&(trial.scenario=='D4_within')].groupby(['protocol','shot','method'])[METRICS].mean().to_csv(out/'full_class_strict_label_disjoint_summary.csv')
    result=[]
    selections=[('D4_within','core9',0),('C4_to_D4_incremental','PP_PS_to_PE_PEVA',2),('C8_resolution','six_class_paired_resolution',2)]
    for scenario,protocol,stage in selections:
        sub=trial[(trial.scenario==scenario)&(trial.protocol==protocol)&(trial.session==stage)&trial.shot.isin([5,10])]
        for field in ['all_accuracy','all_macro_f1','seen_novel_accuracy','base_accuracy']:
            tab=sub.groupby(['task','seed','method'])[field].mean().unstack('method')
            for ref in manifest['methods']:
                for (task,seed),value in (100*(tab['M1_proto_shrinkage']-tab[ref])).items():
                    result.append(dict(scenario=scenario,metric=field,reference=ref,task=task,seed=seed,shrinkage_delta_pp=value))
    deltas=pd.DataFrame(result)
    deltas.to_csv(out/'full_class_mahalanobis_paired_deltas_pp.csv',index=False)
    delta_summary=deltas.groupby(['scenario','metric','reference']).shrinkage_delta_pp.agg(['mean','min','max','count',lambda x:int((x>1e-9).sum())])
    delta_summary.rename(columns={'<lambda_0>':'positive_trials'},inplace=True)
    delta_summary.to_csv(out/'full_class_mahalanobis_paired_summary.csv')
    within_delta=delta_summary.loc[('D4_within','all_accuracy','M1_full_class_legacy002'),:]
    incremental_delta=delta_summary.loc[('C4_to_D4_incremental','all_accuracy','M1_full_class_legacy002'),:]
    per_class=pd.read_csv(root/'per_class_results.csv')
    recalls=100*per_class[(per_class.scenario=='C4_to_D4_incremental')&(per_class.shot==5)&(per_class.session==2)&(per_class.method=='M1_full_class_legacy002')].groupby('class').recall.mean()
    chosen=['M1_1nn','M1_full_class_ridge001','M1_full_class_legacy002','M1_proto_shrinkage','spectrum_svm']
    names=dict(LABELS,**{'M1_full_class_ridge001':'M1 Class full .001','M1_full_class_legacy002':'M1 Class full .002'})
    lines=['','## 附：保留完整类协方差 Mahalanobis 消融','',
        '沿用上一轮完整类协方差实现，在完全相同的外部 support/query、冻结特征和 C4 支持库上补测。两组固定 ridge 为 0.001 与原代码有效值 0.002，均不使用 base 协方差 prior；收缩则额外使用 C4 base_train prior。没有按 D4 query 调参，没有重新推理或训练 encoder。',
        '协方差为类内残差外积除以 n，完整 128×128 非对角矩阵。少于 129 个支持时无正则经验协方差均奇异；这里比较的是带明确 ridge 的完整类协方差。增量仅注册新类；首个 repeat 的所有 shot/阶段逐元素验证旧均值和精度矩阵未变。','',
        f'严重的逐类限制：5-shot 最终增量阶段，原公式完整类协方差对 D4 base PP 的召回率为 {recalls["PP"]:.3f}%，对 PS 为 {recalls["PS"]:.3f}%。PP 几乎全部被新增 PE/PEVA 吸引。完整类协方差总体优于收缩，不等于已实现可靠的跨域旧类保持。','']
    for scenario,protocol,stage in selections:
        rows=summary[(summary.scenario==scenario)&(summary.protocol==protocol)&(summary.session==stage)&summary.method.isin(chosen)]
        lines += [f'### {scenario}: 总体 Accuracy（%）','',md((100*rows.pivot(index='shot',columns='method',values='all_accuracy')).rename(columns=names)),'']
    lines+=['5/10-shot 收缩减完整类协方差，单位为百分点。先对每个任务/种子重复抽样求均值，再配对汇总：','',md(delta_summary),'',
        '完整对照保存在 full_class_mahalanobis_control_metrics.csv，分类别和混淆矩阵位于 full_mahalanobis_control/，综合比较图为 figures/full_class_mahalanobis_transfer_CM.png/pdf。原七种方法结果未被覆盖。','']
    for name in ['results_overview_zh.md','transfer_comparison.md']:
        path=out/name;text=path.read_text(encoding='utf-8')
        headline=(f'**完整协方差消融后的关键结论：收缩在 D4 上不是最优距离头。** 5/10-shot 中，收缩相对原公式完整类协方差的总体 Accuracy，在 D4 固定九类任务差 {within_delta["mean"]:+.3f} 个百分点（改善 {int(within_delta["positive_trials"])}/9），在跨域增量差 {incremental_delta["mean"]:+.3f} 个百分点（改善 {int(incremental_delta["positive_trials"])}/6）。C4 上收缩更好、D4 上完整类协方差更好的反转，提示应研究协方差先验的可迁移性，而非直接推广原结论。详见文末完整消融和 full_class_mahalanobis_paired_summary.csv。\n\n')
        headline+=f'**跨域增量仍未解决。** 5-shot 最终阶段，即使采用总体更好的原公式完整类协方差，旧类 PP 召回率仍只有 {recalls["PP"]:.3f}%，PS 为 {recalls["PS"]:.3f}%。必须把逐类 base 保持作为后续硬性目标，不能仅报告新类或总体平均。\n\n'
        text=text.replace('## 结论先读\n\n','## 结论先读\n\n'+headline,1)
        text=text.replace('## 下一步算法研究判断\n\n','## 下一步算法研究判断\n\n完整类协方差控制显示，优先研究“什么时候应使用 C4 base 协方差先验”及跨域 base/novel 分数可比性更有证据基础。当前固定收缩公式没有随目标域变化，可能产生先验负迁移；这是一项待独立验证的机制假设，尚非因果结论。\n\n',1)
        path.write_text(text+'\n'.join(lines),encoding='utf-8')
    write_audit=dict(status='passed',trials=len(config['tasks'])*len(config['seeds']),
        original_C4_full_metric_checks=original_checks,feature_hashes_verified=True,episode_hashes_verified=True,
        encoder_training_steps=0,previous_class_state_checks='first repeat, all shots/stages, exact mean/precision equality')
    (root/'protocol_audit.json').write_text(json.dumps(write_audit,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(3,4,figsize=(20,15))
    methods=['M1_1nn','M1_full_class_ridge001','M1_full_class_legacy002','M1_proto_shrinkage']
    for row,(scenario,protocol,stage) in enumerate(selections):
        for col,method in enumerate(methods):
            base=root if method.startswith('M1_full_class') else out
            total=None;classes=None
            for task in config['tasks']:
                for seed in config['seeds']:
                    folder=base/task['name']/f'seed_{seed}'
                    index=json.loads((folder/'confusion_index.json').read_text())
                    with np.load(folder/'confusions.npz') as cms:
                        for key,meta in index.items():
                            if meta['scenario']==scenario and meta['protocol']==protocol and meta['session']==stage and meta['shot']==5 and meta['method']==method:
                                assert classes is None or classes==meta['classes']
                                classes=meta['classes'];total=cms[key].copy() if total is None else total+cms[key]
            norm=total/total.sum(axis=1,keepdims=True);ax=axes[row,col]
            pd.DataFrame(total,index=classes,columns=classes).to_csv(out/'figures'/f'full_control_{scenario}_{method}_5shot_CM.csv')
            n=len(classes);short=[c.replace('POLY(ETHYLENE)','PE').replace('CELLULOSE ACETATE','CELL. ACETATE').replace('ETHYLENE PROPYLENE RUBBER','EP RUBBER') for c in classes]
            ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(n));ax.set_yticks(range(n));ax.set_xticklabels(short,rotation=60,ha='right',fontsize=6);ax.set_yticklabels(short,fontsize=6)
            for i in range(n):
                for j in range(n):ax.text(j,i,f'{100*norm[i,j]:.0f}',fontsize=6,ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
            ax.set_title(names[method],fontsize=10);ax.set_ylabel(scenario+' / true',fontsize=8);ax.set_xlabel('Predicted')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/'figures'/f'full_class_mahalanobis_transfer_CM.{ext}',dpi=160)
    plt.close(fig)


def figures(out,config,summary,coverage):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    panels=[('C8_resolution','six_class_paired_resolution',2,'all_accuracy','C4 to C8: paired six-class queries'),
            ('D4_within','core9',0,'all_accuracy','Within D4: fixed nine-class panel'),
            ('C4_to_D4_incremental','PP_PS_to_PE_PEVA',2,'seen_novel_accuracy','C4 to D4: final PE / PEVA recognition')]
    for ax,(scenario,protocol,stage,metric,title) in zip(axes.flat,panels):
        for method,label in LABELS.items():
            rows=summary[(summary.scenario==scenario)&(summary.protocol==protocol)&(summary.session==stage)&(summary.method==method)].sort_values('shot')
            if len(rows):
                ax.plot(rows.shot,100*rows[metric],marker='o',label=label)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Support / class' if scenario=='D4_within' else 'Novel support / class');ax.set_ylabel('Accuracy (%)');ax.set_xticks([1,5,10,20]);ax.grid(alpha=.2)
    ax=axes[1,1]
    for method in coverage.index.get_level_values('method').unique():
        rows=coverage.xs(method,level='method')
        ax.plot(rows.index,rows.accuracy_gap_pp,marker='o',label=LABELS[method])
    ax.axhline(0,color='gray',linewidth=.8);ax.set_title('C4 query-only mask: accuracy change');ax.set_ylabel('Masked minus original (pp)');ax.set_xlabel('Novel support / class');ax.set_xticks([1,5,10,20]);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'transfer_overview.{ext}',dpi=180)
    plt.close(fig)
    anchors=summary[summary.protocol=='PP_PS_base_anchor'].pivot(index='scenario',columns='method',values='all_accuracy')
    fig,ax=plt.subplots(figsize=(11,3.5));values=100*anchors.to_numpy()
    ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(anchors.columns)));ax.set_xticklabels([LABELS[m] for m in anchors.columns],rotation=25,ha='right')
    ax.set_yticks(range(len(anchors)));ax.set_yticklabels(anchors.index)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):ax.text(c,r,f'{values[r,c]:.1f}',ha='center',va='center',color='white' if values[r,c]>55 else 'black')
    ax.set_title('Fixed PP / PS heads: accuracy (%); C4/C8 ID matching is not verified specimen matching')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'PP_PS_transfer_anchor.{ext}',dpi=180)
    plt.close(fig)
    selections=[('C8_resolution','six_class_paired_resolution',2),('D4_within','core9',0),('C4_to_D4_incremental','PP_PS_to_PE_PEVA',2)]
    for scenario,protocol,stage in selections:
        sums={}
        for task in config['tasks']:
            for seed in config['seeds']:
                base=out/task['name']/f'seed_{seed}'
                index=json.loads((base/'confusion_index.json').read_text())
                with np.load(base/'confusions.npz') as matrices:
                    for key,meta in index.items():
                        if meta['scenario']==scenario and meta['protocol']==protocol and meta['session']==stage and meta['shot']==5:
                            method=meta['method'];classes=meta['classes']
                            if method in sums:
                                assert classes==sums[method][0]
                                sums[method][1][:]+=matrices[key]
                            else:sums[method]=(classes,matrices[key].copy())
        if not sums:continue
        fig,axes=plt.subplots(2,4,figsize=(20,10))
        for ax,method in zip(axes.flat,LABELS):
            classes,cm=sums[method]
            pd.DataFrame(cm,index=classes,columns=classes).to_csv(folder/f'{scenario}_{method}_5shot_CM.csv')
            norm=cm/cm.sum(axis=1,keepdims=True)
            ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
            n=len(classes);short=[c.replace('POLY(ETHYLENE)','PE').replace('CELLULOSE ACETATE','CELL. ACETATE').replace('ETHYLENE PROPYLENE RUBBER','EP RUBBER') for c in classes]
            ax.set_xticks(range(n));ax.set_yticks(range(n));ax.set_xticklabels(short,rotation=60,ha='right',fontsize=6);ax.set_yticklabels(short,fontsize=6)
            for r in range(n):
                for c in range(n):ax.text(c,r,f'{100*norm[r,c]:.0f}',fontsize=6,ha='center',va='center',color='white' if norm[r,c]>.5 else 'black')
            ax.set_title(LABELS[method],fontsize=10);ax.set_xlabel('Predicted');ax.set_ylabel('True')
        axes.flat[-1].axis('off');fig.suptitle(f'{scenario}: 5-shot final confusion, row %',fontsize=12);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/f'{scenario}_5shot_CM.{ext}',dpi=160)
        plt.close(fig)


def report(out,config,checked,summary,coverage,c8gap,deltas,costs,historical):
    def accuracy_table(scenario,protocol,stage=0,field='all_accuracy'):
        rows=summary[(summary.scenario==scenario)&(summary.protocol==protocol)&(summary.session==stage)]
        return 100*rows.pivot(index='shot',columns='method',values=field).rename(columns=LABELS)
    c8=accuracy_table('C8_resolution','six_class_paired_resolution',2)
    within=accuracy_table('D4_within','core9')
    incremental=accuracy_table('C4_to_D4_incremental','PP_PS_to_PE_PEVA',2)
    novel=accuracy_table('C4_to_D4_incremental','PP_PS_to_PE_PEVA',2,'seen_novel_accuracy')
    anchor=summary[summary.protocol=='PP_PS_base_anchor'].pivot(index='scenario',columns='method',values='all_accuracy')*100
    base_trajectory=summary[(summary.scenario=='C4_to_D4_incremental')&(summary.shot==5)].pivot(index='session',columns='method',values='base_accuracy')*100
    primary_cov=coverage.loc[coverage.index.get_level_values('shot').isin([5,10])].groupby('method')[['accuracy_gap_pp','prediction_agreement']].mean()
    cg=c8gap.groupby('method').delta_pp.agg(['mean','min','max'])
    sd=deltas.loc[('D4_within','all_accuracy','M1_1nn'),:]
    si=deltas.loc[('C4_to_D4_incremental','seen_novel_accuracy','M1_1nn'),:]
    so=deltas.loc[('C4_to_D4_incremental','all_accuracy','M1_1nn'),:]
    def shrink5(scenario,protocol,stage):
        row=summary[(summary.scenario==scenario)&(summary.protocol==protocol)&(summary.session==stage)&(summary.shot==5)&(summary.method=='M1_proto_shrinkage')].iloc[0]
        return f'5-shot 收缩方法的 Accuracy 为 {100*row.all_accuracy:.3f}%，Macro-F1 为 {100*row.all_macro_f1:.3f}%。'
    reg=costs.loc[('C4_to_D4_incremental','PP_PS_to_PE_PEVA',5),:].reset_index()
    reg=reg[reg.session>0].groupby('method')[['feature_encoding_ms','new_class_statistics_or_bank_ms','svm_full_refit_ms','total_update_ms','cached_query_prediction_ms']].mean()
    missing=primary_cov.loc['M1_proto_shrinkage','accuracy_gap_pp']
    anchor_gap=anchor.loc['anchor_D4','M1_proto_shrinkage']-anchor.loc['anchor_C4_masked','M1_proto_shrinkage']
    counts=pd.read_csv(out/'class_counts.csv');counts=counts[counts.dataset=='d4'].set_index('class')[['unique_count']]
    adapter=['# D4 input adapter V1 audit','',
        'Status: exploratory partial-spectrum transfer executed. This new explicit mean-fill authorization supersedes the previous experiment’s no-padding blocker; previous blocked artifacts remain historical and unchanged.','',
        f'Locked grid: 1868 ascending coordinates; {config["coverage_valid"]} observed, 107 missing (104 low-end, 3 high-end; {100*107/1868:.3f}% of coordinates). Original 399.1927–4000.6045 grid and M1 weights are unchanged. D4 native 599.78354–3995.98555 range is never extrapolated.','',
        'Per trial, missing inputs are fixed per-coordinate means fitted only on that checkpoint’s archived C4 base_train input rows. The same mean and explicit boolean mask apply to all D4 support/query rows and the controlled C4 mask. Each trial saves input_adapter.npz with mean, mask, grid and fit indices; adapted D4 inputs and frozen features are saved separately.','',
        'Preprocessing is native max(y)-y → native airPLS → native maxabs normalization → cubic interpolation inside the native observed range → missing-coordinate mean insertion. No normalization follows mean insertion. This uses exactly the C4 numerical orientation, explicitly flip=True; the old D4 registry default flip=False remains unchanged. This choice was locked before D4 query scoring. It is not a verified physical transmittance/absorbance conversion. No log transform, query-selected polarity, new mask channel, altered weights or target-population statistics are introduced.','',
        'D4 contains negative numerical values and row maximum 100, without signal-unit metadata. cm^-1 interpretation follows the FTIR-scale headers but lacks explicit acquisition metadata. Physical signal compatibility, specimen grouping and cross-dataset independence are unresolved. Numeric input compatibility is established; physical measurement equivalence is not.','',
        'Masking controlled C4 happens AFTER its original full-range preprocessing, retaining its original observed coordinates and using unchanged original C4 supports/heads. D4 baseline fitting instead sees its native truncated measured interval. The C4 control isolates coordinate deletion/imputation, not baseline-boundary differences. Mean imputation is a numerical input convention, not recovery of measured chemical peaks.','',
        'Spectrum 1-NN/SVM within D4 and mixed C4→D4 use only the 1761 observed common coordinates for both support and query. PCA/M1 consume the full locked grid with the fixed imputation. C8 paired six-class spectrum references use the full common grid. The two-class PP/PS anchor spectrum references use the D4-valid window in every dataset condition. Archived PCA64 and shared covariance are fitted only on original legal C4 base_train and never updated for D4. SVM C=1 is fixed without target tuning.','',
        'Unique D4 counts:', '',md(counts),'',
        'Fixed core9 excludes PA (21) and MORPHOTYPE 2 (10), and is evaluated at 1/5/10/20 shots with five disjoint queries/class. Expanded eligible panel includes all 11 classes at 1/5 shots and 10 classes at 10 shots; at 20 shots it is identical to core9 and not duplicated. The same seed’s saved support/query partitions and episode choices are shared across all three encoder tasks. Morphotype labels are retained as unresolved categories, not asserted identified polymers.','',
        'D4 PP/PS are textually matched to C4 PP/PS only. Generic PE is not mapped to HDPE/LDPE; fouling/like/PEVA/rubber and morphotypes are separate labels. The incremental protocol starts from two C4 PP/PS banks (20 exemplars/class), then adds D4 PE and PEVA, using only the six checkpoints trained with PP/PS in base. The other three are excluded from this base protocol; all nine participate in within-D4 and C8 tests.','',
        'Exact duplicates: five removed, 954 retained, eleven labels. C4/C8 have 2201 shared retained record-ID strings and zero label conflicts; these are resolution-family pairings, not verified independent specimens. Paired C8 query IDs must be disjoint from C4 base_train and retained support IDs. C4 validation queries were previously used for checkpoint selection, so C8 base-class outcomes remain exploratory. C4/D4 have no shared retained IDs/native-axis hashes; different axes make zero native hashes non-exclusionary. No independent-specimen claim is made.','',
        f'Protocol audit passed for {checked["trials"]} checkpoints and {checked["original_C4_metric_checks"]} original C4 metric contexts. Frozen encoder has 47,400 parameters, zero training updates, equal before/after file and weight hashes.','']
    (out/'d4_input_adapter_audit.md').write_text('\n'.join(adapter),encoding='utf-8')
    text=['# 异构 FTIR 冻结迁移评估 V1','',
        f'已完成 {checked["trials"]} 个既有 M1 checkpoint 的冻结评估；每种条件 200 次配对支持/查询抽样，1/5/10/20-shot。没有训练新 encoder、没有改 loss、没有使用 D4 query 选参数。','',
        '## 结论先读','',
        '1. 既有 M1 可以通过明确、固定的部分光谱适配器处理 D4。107 个未测坐标用各 checkpoint 的合法 C4 base_train 坐标均值填补；这不等于恢复缺失峰，也不证明物理信号语义一致。',
        f'2. C4 的 5/10-shot 遮罩控制中，收缩方法 Accuracy 变化为 {missing:+.3f} 个百分点（masked−original）。该控制只改变 query 的缺失坐标，支持库和已注册状态保持原样。',
        f'3. D4 固定九类适配中，收缩相对 M1 1-NN 的 5/10-shot 总体 Accuracy 平均差为 {sd["mean"]:+.3f} 个百分点，改善 {int(sd["positive_trials"])}/{int(sd["count"])} 组；C4→D4 增量中已加入新类 Accuracy 差为 {si["mean"]:+.3f} 个百分点，改善 {int(si["positive_trials"])}/{int(si["count"])} 组，但总体 Accuracy 差为 {so["mean"]:+.3f} 个百分点。收缩的新类收益没有转化为总体增量优势，不能把 C4 上的收益直接当作 D4 上的结论。',
        f'4. 固定 PP/PS 支持库的锚点中，D4 收缩 Accuracy 相对 masked C4 差 {anchor_gap:+.3f} 个百分点。标签和分类头相同，但 D4 query 不是同一批实物；该差异包含域差异、信号语义及原生预处理区间差异，不能纯归因为缺失覆盖。','',
        f'5. 同一 PP/PS 头的收缩方法在 C8 为 {anchor.loc["anchor_C8","M1_proto_shrinkage"]:.3f}%，在 D4 为 {anchor.loc["anchor_D4","M1_proto_shrinkage"]:.3f}%；欧氏头则分别为 {anchor.loc["anchor_C8","M1_proto_euclidean"]:.3f}% 和 {anchor.loc["anchor_D4","M1_proto_euclidean"]:.3f}%。数据集难度排名依赖分类头，不能概括为 D4 一定比 C8 更难。','',
        '## A. C4 内部结果与缺失覆盖敏感性','',
        '历史六类增量结果继续作为探索性参照。下面是相同 checkpoint、相同完整 C4 支持库下，原始 query 与 masked query 的配对变化：','',md(primary_cov.rename(index=LABELS)), '',
        'prediction_agreement 是预测一致比例；变化单位为百分点。完整与遮罩都沿用原有输入尺度，不重新拟合 normalization、PCA 或协方差先验。历史完整 C4 数值见 c4_historical_reference.csv，逐 episode 遮罩结果见 coverage_mask_comparison.csv。','',
        '## B. C4→C8 跨分辨率迁移','',md(c8),'',
        '表为最终六类 Accuracy（%）。'+shrink5('C8_resolution','six_class_paired_resolution',2)+' 支持样本来自原 C4；C4/C8 查询按同记录 ID、同标签配对，并按每类可匹配查询数的最小值取平衡子集。不能用全量 C4 分数直接减这个 C8 分数。正确的配对迁移差值（C8−对应 C4，跨全部 shot 的任务/种子均值）：','',md(cg.rename(index=LABELS)), '',
        'C4/C8 属于同一数据家族的分辨率版本，不能称为独立仪器泛化。同编号配对只是记录级对应，尚无元数据证明两行来自同一实物或同一次采集，所以差值不能纯归因于分辨率。另存 c4_c8_record_alignment_audit.csv 检查同编号光谱的公共网格相似度；不按相似度挑选 query。base 查询来自曾参与 checkpoint 选择的验证集。若个别 episode 某类没有可匹配 query，两种分辨率同时跳过，匹配清单完整保存在每个 trial 的 c8_paired_queries.json。','',
        '## C. C4→D4 的 2+1+1 类增量','',
        '合法的六个 checkpoint：base 训练包含 PP、PS。头部仅保留这两类的原 C4 20-shot 支持；然后按 PE→PEVA 注册 D4 新类，每类五个固定池内抽取的独立 query。每次只更新新类均值/度量，已注册的 prototype 状态保持不变；1-NN/SVM 则保留全库并分别重建索引/重拟合 SVM。','',
        '最终四类总体 Accuracy（%）：','',md(incremental),'',
        shrink5('C4_to_D4_incremental','PP_PS_to_PE_PEVA',2),'',
        '最终已加入新类 PE/PEVA Accuracy（%）：','',md(novel),'',
        'base/current/previous/seen novel 的 Accuracy 和 Macro-F1 全部保存在 c4_to_d4_incremental_metrics.csv 和 summary_metrics.csv；base query 取 D4 的 PP/PS，不冒充 C4 内部 base 测试。','',
        '5-shot 各阶段的 base Accuracy（%）：','',md(base_trajectory.rename(columns=LABELS)), '',
        '新类高准确率不能替代 base 保持。这里的 base 下降发生在 encoder 和旧 prototype 参数完全不变时，反映跨域 query 与新增类竞争；不能称为训练覆盖了旧权重。三阶段平均总体指标另存 average_incremental_metrics.csv。','',
        '## D. D4 内部 few-shot 适配，encoder 仍来自 C4','',md(within),'',
        shrink5('D4_within','core9',0),'',
        '表为固定九类任务的 Accuracy（%）。扩大类集的 11/10 类结果另存同一 d4_fewshot_metrics.csv 的 eligible 协议，不能沿变动类数画同一条主曲线。三种 encoder base 划分中，只有 holdout_pp_ps 的 D4 标签与 encoder 训练标签全部不重合；其三个种子的独立分表为 strict_label_disjoint_summary.csv。其余六个 checkpoint 已在 C4 训练见过 PP/PS 标签，但没有用 D4 数据训练。普通 PE 与 HDPE/LDPE 的化学关系也不能被“标签不重合”掩盖。','',
        '## 同标签、同支持库的 PP/PS 锚点','',md(anchor.rename(columns=LABELS)), '',
        '这里四种条件的分类头都只用原 C4 PP/PS 支持。C4 original/masked/C8 使用相同编号的 query 对；D4 是不同样本的相同文本标签。原始与 masked C4 直接估计缺失坐标敏感性，D4 与 masked C4 的剩余差异只能作为域差异诊断。该锚点没有新增 D4 支持，不能和 D4 内部适配分数混为一谈。','',
        '## 成本与冻结验证','',
        'cost_summary.csv 分开给出新类特征编码、局部注册、全库装配/重拟合 SVM 与缓存 query 推理。新 D4 支持的 M1 编码实际重放计时，均值/精度矩阵只注册新类。查询推理时间不含 encoder 与原始预处理；D4 内部 episode_head_fit_ms 是整个 episode 头部拟合成本，不能标成单类注册。',
        '5-shot 两个新增阶段的平均成本（ms）：','',md(reg.rename(index=LABELS)),'',
        '所有 checkpoint 的文件 SHA 与权重 SHA 前后完全相同，均为 47,400 参数，新增训练更新数为零。PCA64、base prior 和旧实验脚本均通过 hash 复核。','',
        '## 下一步算法研究判断','',
        ('遮罩 C4 存在可见敏感性，覆盖感知模型值得作为后续独立实验假设；但现有结果不能证明它能修复 D4 的所有域差异。' if abs(missing)>1 else '单纯遮罩 C4 的平均影响较小，当前证据不足以把覆盖感知 encoder 直接列为首要修复；应先核实物理信号语义与域差异。'),
        ('收缩在 D4 的收益存在条件差异，应优先检验 C4 协方差先验的可迁移性、跨域 base 类校准和类内/类间结构，不能直接照搬 C4 的结论。尤其新类改善而总体下降时，下一步应把 base 保持作为硬性评估目标。' if sd['mean']<=0 or so['mean']<=0 else '收缩在本轮 D4 平均指标也有收益，但仍应结合逐类混淆和 spectrum/PCA 对照，检验收益来自距离先验还是 encoder 特征。'),
        'prototype-oriented encoder 训练只能作为未来受控比较的候选方向；本轮没有训练它，也没有证据证明某个新增 loss 会解决当前问题。若原谱/PCA 明显优于 embedding，这提示当前特征迁移不足；若两者都弱，则信号语义与采样条件更需优先核实。',
        '当前不能把缺失覆盖、未知物理信号和类别判别难度的影响完全因果分离。需要有已知信号语义、已知实物分组、未被查看的独立数据，才能开展确认性评价。','',
        '## 文件与图','',
        '- figures/transfer_overview.png/pdf：分辨率、D4 适配、增量新类与遮罩影响。',
        '- figures/PP_PS_transfer_anchor.png/pdf：固定两类头的跨数据集锚点。',
        '- figures/*_5shot_CM.png/pdf：C8 六类、D4 九类与增量四类的逐类别混淆矩阵及原计数 CSV。',
        '- per_class_results.csv：按 task/seed 分开的累计混淆矩阵召回率；重复查询计数不代表独立样本数量。',
        '- trial_metrics.csv、d4_paired_method_deltas_pp.csv：先对重复抽样求均值的配对结果，不把 200 个 episode 当作 200 个独立训练模型。',
        '- protocol_audit.json、frozen_encoder_verification.csv：输入、划分、来源和冻结核查。','',
        '所有数字属于探索性评估。D4 部分光谱结果基于固定均值填补；记录 ID 匹配不等于实物独立；原生轴不同使零 exact hash 匹配不能排除重采样/重复实物。未覆盖测量峰、未知信号单位和 morphotype 标签身份仍是未解决的数据限制。','']
    (out/'results_overview_zh.md').write_text('\n'.join(text),encoding='utf-8')
    (out/'transfer_comparison.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/heterogeneous_ftir_transfer_v1')
    with threadpool_limits(limits=1):main(parser.parse_args().output.resolve())
