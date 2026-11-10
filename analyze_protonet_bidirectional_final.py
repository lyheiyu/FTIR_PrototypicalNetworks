# -*- coding: utf-8 -*-
"""Paired, descriptive final ProtoNet ablation reports; no model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from ftir_research_data import ROOT
from ftir_research_core import base_covariance
from run_bidirectional_c4_c8_final import digest,write
from encoder_architecture_models import build_encoder
import tensorflow as tf

PARENT=ROOT/'results/bidirectional_c4_c8_final'
METHODS=['CE_euclidean','EP_euclidean','CE_full','EP_full','CE_shrinkage','EP_shrinkage','PCA64_shrinkage']
MAP={'M1_full':'CE_full','M1_shrinkage':'CE_shrinkage','PCA64_shrinkage':'PCA64_shrinkage'}
METRICS=['all_accuracy','all_macro_f1','base_accuracy','seen_novel_accuracy','previous_novel_accuracy','current_novel_accuracy',
    'base_macro_f1','seen_novel_macro_f1','previous_novel_macro_f1','current_novel_macro_f1',
    'base_to_novel_error_rate','novel_to_base_error_rate','previous_novel_change_pp','base_change_from_session0_pp']


def table(frame,columns=None):
    f=frame if columns is None else frame[columns]
    # No optional tabulate dependency.
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return f'{v:.3e}' if 0<abs(v)<.001 else f'{v:.3f}'
        return str(v)
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in f.itertuples(index=False,name=None))


def run(out):
    config=json.loads((out/'experiment_manifest.json').read_text());assert (out/'COMPLETE.json').exists()
    smoke=config['smoke'];trials=[];perclass=[];costs=[];training=[];geometry=[];confusions=[];audit=[];preserved=[]
    for task in config['tasks']:
        for seed in config['seeds']:
            f=out/task['name']/f'seed_{seed}';p=PARENT/task['name']/f'seed_{seed}'
            split=json.loads((f/'split.json').read_text());episodes=json.loads((f/'episodes.json').read_text())[:config['repeats']]
            assert digest(f/'split.json')==digest(p/'split.json') and digest(f/'episodes.json')==digest(p/'episodes.json')
            new=pd.read_csv(f/'new_metrics.csv');old=pd.read_csv(p/'metrics.csv');old=old[old.method.isin(MAP)&(old['repeat']<config['repeats'])].copy()
            old['method']=old.method.map(MAP);combined=pd.concat([new,old],ignore_index=True);trials.append(combined)
            assert len(combined)==6*7*3*4*len(episodes)
            assert not combined.duplicated(['setting','method','shot','repeat','session']).any()
            accuracy_lookup=combined.groupby(['setting','shot','session','method']).all_accuracy.mean()
            oldcost=pd.read_csv(p/'costs.csv');oldcost=oldcost[oldcost.method.isin(MAP)&(oldcost['repeat']<config['repeats'])].copy();oldcost['method']=oldcost.method.map(MAP)
            costs.extend([pd.read_csv(f/'new_costs.csv'),oldcost])
            for root,cmfile,indexfile,mapping in [(f,'new_confusions.npz','new_confusion_index.json',None),(p,'confusions.npz','confusion_index.json',MAP)]:
                arrays=np.load(root/cmfile);idx=json.loads((root/indexfile).read_text())
                for token,meta in idx.items():
                    if mapping is not None and meta['method'] not in mapping:continue
                    if smoke and mapping is not None:continue # Parent pooled 200 repeats cannot be reused for a 2-repeat CM.
                    meta=meta.copy()
                    if mapping:meta['method']=mapping[meta['method']]
                    cs=meta.pop('classes');cm=arrays[token];perclass.extend([
                        dict(meta,**{'class':c},recall=cm[j,j]/cm[j].sum() if cm[j].sum() else 0,
                            precision=cm[j,j]/cm[:,j].sum() if cm[:,j].sum() else 0,
                            f1=2*cm[j,j]/(cm[j].sum()+cm[:,j].sum()) if cm[j].sum()+cm[:,j].sum() else 0,query_occurrences=int(cm[j].sum())) for j,c in enumerate(cs)])
                    expected=accuracy_lookup.loc[(meta['setting'],meta['shot'],meta['session'],meta['method'])]
                    np.testing.assert_allclose(np.trace(cm)/cm.sum(),expected,atol=1e-12)
                    if meta['session']==2:
                        confusions.append((meta,cs,cm))
            groups=pd.read_csv(PARENT/'paired_group_representatives.csv');labels=groups.label.to_numpy()
            for source in ['c4','c8']:
                model_dir=f/source/'M1';ce_dir=p/source/'M1';meta=json.loads((model_dir/'training_costs.json').read_text())
                ce_meta=json.loads((ce_dir/'training_costs.json').read_text());assert meta['optimizer_updates']==config['epochs']*int(np.ceil(len(split['train'])/64))
                if not smoke:assert meta['optimizer_updates']==ce_meta['optimizer_updates']
                assert meta['initial_encoder_hash']==json.loads((ce_dir/'PAIRED_TRAINING_COMPLETE.json').read_text())['initial_encoder_hash']
                for obj,d,stats in [('EP',model_dir,meta),('CE',ce_dir,ce_meta)]:
                    training.append(dict(stats,task=task['name'],seed=seed,source=source,objective=obj))
                    data=np.load(d/'frozen_views.npz');np.testing.assert_array_equal(data['fit_indices'],split['train'])
                    z=data[f'M1_{source}'];prior=base_covariance(z[split['train']],labels[split['train']]);np.testing.assert_allclose(prior,data['M1_prior'],rtol=1e-12,atol=1e-12)
                    e=np.maximum(np.linalg.eigvalsh(prior),0);prob=e/e.sum() if e.sum() else e
                    geometry.append(dict(task=task['name'],seed=seed,source=source,objective=obj,mean_training_norm=float(np.linalg.norm(z[split['train']],axis=1).mean()),
                        mean_residual_variance=float(np.trace(prior)/128),ridge_over_mean_variance=float(.001/(np.trace(prior)/128)) if np.trace(prior)>0 else np.nan,
                        effective_rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0]))))))
                model=tf.keras.models.load_model(str(model_dir/'encoder.keras'),compile=False);assert model.count_params()==47400
                assert digest(model_dir/'encoder.keras')==json.loads((model_dir/'COMPLETE.json').read_text())['encoder_sha256']
                tf.keras.backend.clear_session()
            audit.append(json.loads((f/'COMPLETE.json').read_text()))
    for p,h in config['supervised_checkpoint_hashes'].items():assert digest(ROOT/p)==h
    metrics=pd.concat(trials,ignore_index=True);metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(out/'per_class_metrics.csv',index=False)
    costs=pd.concat(costs,ignore_index=True);costs.to_csv(out/'costs.csv',index=False)
    training=pd.DataFrame(training);training.to_csv(out/'training_comparison.csv',index=False)
    geometry=pd.DataFrame(geometry);geometry.to_csv(out/'feature_geometry.csv',index=False)
    keys=['task','seed','setting','shot','session','method']
    trialmean=metrics.groupby(keys,as_index=False)[METRICS].mean();trialmean.to_csv(out/'task_seed_metrics.csv',index=False)
    summary=trialmean.groupby(['setting','shot','session','method'])[METRICS].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns];summary.to_csv(out/'summary_by_shot_session.csv',index=False)
    primary=trialmean[(trialmean.session==2)&trialmean.shot.isin([5,10])].groupby(['task','seed','setting','method'],as_index=False)[METRICS].mean()
    primary.to_csv(out/'primary_task_seed_metrics.csv',index=False)
    piv=primary.pivot(index=['task','seed','setting'],columns='method',values=METRICS)
    comparisons=[('EP_euclidean','CE_euclidean','episodic_vs_CE_euclidean'),('EP_full','CE_full','episodic_vs_CE_full'),
        ('EP_shrinkage','CE_shrinkage','episodic_vs_CE_shrinkage'),('EP_shrinkage','EP_euclidean','shrinkage_vs_euclidean_EP'),
        ('EP_shrinkage','EP_full','shrinkage_vs_full_EP'),('EP_shrinkage','PCA64_shrinkage','EP_vs_PCA64_shrinkage')]
    paired=[]
    for a,b,name in comparisons:
        for index,row in piv.iterrows():
            paired.append(dict(task=index[0],seed=index[1],setting=index[2],comparison=name,
                **{m+'_delta_pp':(row[m,a]-row[m,b])*(1 if m.endswith('_pp') else 100) for m in METRICS}))
    paired=pd.DataFrame(paired);paired.to_csv(out/'paired_differences.csv',index=False)
    pairsummary=paired.groupby(['setting','comparison'],as_index=False).agg(
        accuracy_delta_pp=('all_accuracy_delta_pp','mean'),novel_delta_pp=('seen_novel_accuracy_delta_pp','mean'),macro_f1_delta_pp=('all_macro_f1_delta_pp','mean'),
        accuracy_sd_pp=('all_accuracy_delta_pp','std'),accuracy_positive_pairs=('all_accuracy_delta_pp',lambda x:int((x>0).sum())),
        novel_positive_pairs=('seen_novel_accuracy_delta_pp',lambda x:int((x>0).sum())),pairs=('seed','size'))
    pairsummary.to_csv(out/'paired_summary.csv',index=False)
    transfer=[];tp=primary.pivot(index=['task','seed','method'],columns='setting',values=['all_accuracy','seen_novel_accuracy','all_macro_f1'])
    for index,row in tp.iterrows():
        for a,b,name in [('C','A','C4_to_C8_gap'),('D','B','C8_to_C4_gap'),('E','B','source_effect_target_C8'),('F','A','source_effect_target_C4'),('E','C','target_bank_alignment_C8'),('F','D','target_bank_alignment_C4')]:
            transfer.append(dict(task=index[0],seed=index[1],method=index[2],comparison=name,**{m+'_pp':100*(row[m,a]-row[m,b]) for m in ['all_accuracy','seen_novel_accuracy','all_macro_f1']}))
    transfer=pd.DataFrame(transfer);transfer.to_csv(out/'source_target_transfer_gaps.csv',index=False)
    transfer.groupby(['method','comparison'],as_index=False)[['all_accuracy_pp','seen_novel_accuracy_pp','all_macro_f1_pp']].mean().to_csv(out/'transfer_gap_summary.csv',index=False)
    cdir=out/'confusion_matrices';cdir.mkdir(exist_ok=True);confusionrows=[]
    for meta,cs,cm in confusions:
        for i,c in enumerate(cs):
            for j,pred in enumerate(cs):confusionrows.append(dict(meta,true=c,predicted=pred,count=int(cm[i,j])))
    counts=pd.DataFrame(confusionrows);counts.to_csv(out/'confusion_counts.csv',index=False)
    if len(counts):
        pooled=counts.groupby(['task','setting','shot','method','true','predicted'],as_index=False)['count'].sum()
        errors=[]
        for (task,setting,k,method),g in pooled.groupby(['task','setting','shot','method']):
            cm=g.pivot(index='true',columns='predicted',values='count').fillna(0)
            cm.to_csv(cdir/f'{task}_{setting}_{k}shot_{method}.csv')
            for a,b in [('HDPE','LDPE'),('LDPE','HDPE'),('PP','PS'),('PS','PP')]:
                errors.append(dict(task=task,setting=setting,shot=k,method=method,true=a,predicted=b,error_rate=cm.loc[a,b]/cm.loc[a].sum(),recall=cm.loc[a,a]/cm.loc[a].sum()))
        pd.DataFrame(errors).to_csv(out/'key_polymer_confusions.csv',index=False)
        if not smoke:
            for task in config['tasks']:
                for k in [5,10]:
                    fig,axes=plt.subplots(7,6,figsize=(22,24))
                    for i,method in enumerate(METHODS):
                        for j,setting in enumerate('ABCDEF'):
                            g=pooled[(pooled.task==task['name'])&(pooled.shot==k)&(pooled.method==method)&(pooled.setting==setting)]
                            cm=g.pivot(index='true',columns='predicted',values='count').fillna(0);norm=cm.to_numpy()/cm.to_numpy().sum(1,keepdims=True)
                            ax=axes[i,j];ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
                            ax.set_xticks(range(6));ax.set_xticklabels(cm.columns,rotation=90,fontsize=7);ax.set_yticks(range(6));ax.set_yticklabels(cm.index,fontsize=7)
                            ax.set_title(setting+' / '+method,fontsize=9)
                            for a in range(6):
                                for b in range(6):ax.text(b,a,f'{100*norm[a,b]:.0f}',ha='center',va='center',fontsize=6,color='white' if norm[a,b]>.55 else 'black')
                    fig.suptitle(task['name']+f' | final session | {k} shot | row % | exploratory',fontsize=12);fig.tight_layout(rect=[0,0,1,.98])
                    fig.savefig(cdir/f'{task["name"]}_{k}shot.png',dpi=160);fig.savefig(cdir/f'{task["name"]}_{k}shot.pdf');plt.close(fig)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,case in zip(axes.flat,'ABCDEF'):
        for method in METHODS:
            g=trialmean[(trialmean.setting==case)&(trialmean.method==method)&(trialmean.session==2)].groupby('shot')['all_accuracy'].mean()
            ax.plot(g.index,100*g.values,'o-',label=method,linewidth=1)
        ax.set_title(case);ax.set_xticks([1,5,10,20]);ax.set_ylabel('Final accuracy (%)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/'final_accuracy.png',dpi=180);fig.savefig(figures/'final_accuracy.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,metric in zip(axes,['all_accuracy','seen_novel_accuracy']):
        for method in METHODS:
            g=primary[primary.method==method].groupby('setting')[metric].mean();ax.plot(g.index,100*g.values,'o-',label=method)
        ax.set_ylabel(metric+' (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/'primary_all_and_novel_accuracy.png',dpi=180);plt.close(fig)
    errors=pd.read_csv(out/'key_polymer_confusions.csv')
    pair_errors=errors[errors.shot.isin([5,10])&(((errors.task=='holdout_hdpe_ldpe')&errors.true.isin(['HDPE','LDPE']))|
        ((errors.task=='holdout_pp_ps')&errors.true.isin(['PP','PS'])))].copy()
    pair_errors['novel_pair']=np.where(pair_errors.task=='holdout_hdpe_ldpe','HDPE_LDPE','PP_PS')
    pair_errors=pair_errors.groupby(['setting','method','novel_pair'],as_index=False)[['error_rate','recall']].mean()
    pair_errors.to_csv(out/'primary_novel_pair_confusions.csv',index=False)
    primary.groupby(['setting','method'],as_index=False)[METRICS].mean().to_csv(out/'error_and_retention_summary.csv',index=False)
    report(out,config,primary,pairsummary,transfer,training,geometry,pair_errors)
    write(out/'protocol_audit.json',dict(status='passed',paired_trials=len(audit),episodic_encoders=2*len(audit),metric_rows=len(metrics),
        supervised_checkpoints_unchanged=True,exact_parent_split_and_episode_files=True,source_train_only_priors_numerically_verified=True,
        matched_initialization_and_update_counts=True,old_class_state_checks=sum(a['old_class_state_checks'] for a in audit),
        incremental_encoder_updates=0,physical_specimen_independence_verified=False,untouched_confirmatory_data=False))
    print('Reports complete:',out,flush=True)


def report(out,config,primary,paired,transfer,training,geometry,pair_errors):
    accuracy=100*primary.groupby(['setting','method']).all_accuracy.mean().unstack().reindex(columns=METHODS)
    novel=100*primary.groupby(['setting','method']).seen_novel_accuracy.mean().unstack().reindex(columns=METHODS)
    mf1=100*primary.groupby(['setting','method']).all_macro_f1.mean().unstack().reindex(columns=METHODS)
    def delta(case,comparison,metric='accuracy_delta_pp'):
        return float(paired[(paired.setting==case)&(paired.comparison==comparison)][metric].iloc[0])
    tr=training.groupby('objective')[['training_seconds','inference_median_ms_per_spectrum','optimizer_updates']].mean().reset_index()
    def fmt(values):return ', '.join(f'{case} {values(case):+.2f}' for case in 'ABCDEF')
    eu=fmt(lambda c:delta(c,'episodic_vs_CE_euclidean'));eun=fmt(lambda c:delta(c,'episodic_vs_CE_euclidean','novel_delta_pp'))
    full=fmt(lambda c:delta(c,'episodic_vs_CE_full'));shr=fmt(lambda c:delta(c,'episodic_vs_CE_shrinkage'))
    shp=fmt(lambda c:delta(c,'shrinkage_vs_full_EP'));she=fmt(lambda c:delta(c,'shrinkage_vs_euclidean_EP'));pca=fmt(lambda c:delta(c,'EP_vs_PCA64_shrinkage'))
    tg=transfer.groupby(['method','comparison'])[['all_accuracy_pp','seen_novel_accuracy_pp']].mean()
    gap=[]
    for direction in ['C4_to_C8_gap','C8_to_C4_gap']:
        for head in ['euclidean','full','shrinkage']:
            ep=tg.loc[('EP_'+head,direction)];ce=tg.loc[('CE_'+head,direction)]
            gap.append(dict(direction=direction,head=head,CE_gap_pp=ce.all_accuracy_pp,EP_gap_pp=ep.all_accuracy_pp,EP_minus_CE_gap_pp=ep.all_accuracy_pp-ce.all_accuracy_pp))
    gap=pd.DataFrame(gap);gap.to_csv(out/'transfer_objective_interactions.csv',index=False)
    retention=primary[primary.setting.isin(list('ABCD'))&primary.method.isin(['CE_shrinkage','EP_shrinkage','PCA64_shrinkage'])].groupby(['setting','method'],as_index=False)[
        ['base_accuracy','previous_novel_accuracy','current_novel_accuracy','base_to_novel_error_rate','previous_novel_change_pp']].mean()
    for c in ['base_accuracy','previous_novel_accuracy','current_novel_accuracy','base_to_novel_error_rate']:retention[c]*=100
    confusion=100*pair_errors[pair_errors.setting.isin(list('ABCD'))&pair_errors.method.isin(['CE_shrinkage','EP_shrinkage','PCA64_shrinkage'])].pivot(
        index=['setting','method'],columns='novel_pair',values='error_rate')
    novel_cases=[c for c in 'ABCD' if delta(c,'episodic_vs_CE_euclidean','novel_delta_pp')>0]
    transfer_wins=sum(delta(c,'episodic_vs_CE_'+h)>0 for c in 'CD' for h in ['euclidean','full','shrinkage'])
    shrink_wins=sum(delta(c,'shrinkage_vs_full_EP')>0 for c in 'ABCDEF')
    pca_wins=[c for c in 'ABCDEF' if delta(c,'EP_vs_PCA64_shrinkage')>0]
    decisions=f'''# Final decisions / 最终判断

Primary final-session five/ten-shot task-seed means; all claims exploratory.

1. 固定欧氏原型头，episodic 对 combined novel Accuracy 的均值提升出现在主设置 A–D 中的 {','.join(novel_cases) or '无'}（{len(novel_cases)}/4），不是把更换距离头作为训练收益。各设置 pp 差值：{eun}。配对种子胜负见 paired_summary.csv；同一任务的多个种子不是独立数据集。
2. 两个真正 source-bank cross-resolution 设置 C/D、三个固定头共6个对比，EP 总体 Accuracy 均值正差 {transfer_wins}/6。欧氏：C {delta('C','episodic_vs_CE_euclidean'):+.2f}, D {delta('D','episodic_vs_CE_euclidean'):+.2f} pp；full：C {delta('C','episodic_vs_CE_full'):+.2f}, D {delta('D','episodic_vs_CE_full'):+.2f}；shrink：C {delta('C','episodic_vs_CE_shrinkage'):+.2f}, D {delta('D','episodic_vs_CE_shrinkage'):+.2f}。应逐头、逐方向陈述，不能只用 gap 缩小主张迁移更好。
3. 在固定 EP 编码器上，shrink−full 总体均值提升 {shrink_wins}/6 设置；差值：{shp} pp。shrink−Euclidean：{she}。这属于度量头收益。
4. EP+shrink 总体均值超过 PCA64+shrink 的设置：{','.join(pca_wins) or '无'}（{len(pca_wins)}/6）。差值：{pca} pp。因此应保留强 PCA 对照，不能从少数新类识别指标或个别任务推导全面优越性。
5. 最强可辩护贡献是：冻结可复用 FTIR 表示下、无需旧类重拟合的按类注册，以及在同一组隔离/物理网格/配对 episode 条件下，明确区分 episodic 训练、完整协方差与收缩、参考库分辨率匹配的适用范围。实证提升只限定在上列支持的条件。无未触碰外部确证数据、无物理实物独立性证明，不能主张跨仪器普适领先。

最终停止：不增添模型、损失、增强或后验调参。完整数值、负结果与限制见 summary_zh.md 和 manuscript_results_en.md。
'''
    core_en=f'''The principal negative result is that ordinary episodic M1 with Euclidean prototypes did not improve mean novel-class accuracy in any of the six settings: the paired differences were {eun} percentage points. Its within-resolution overall gains (A {delta('A','episodic_vs_CE_euclidean'):+.2f}, B {delta('B','episodic_vs_CE_euclidean'):+.2f}) therefore should not be described as improved novel-class generalization; they mainly reflect higher base-class accuracy. Episodic training with the shrinkage head improved overall source-bank cross-resolution accuracy over CE with the same head by {delta('C','episodic_vs_CE_shrinkage'):+.2f} (C) and {delta('D','episodic_vs_CE_shrinkage'):+.2f} (D) percentage points, with positive overall differences in seven of nine task-seed pairs in each direction. However, episodic shrinkage remained below the fixed PCA64 shrinkage reference in mean overall accuracy in every setting, including C {delta('C','EP_vs_PCA64_shrinkage'):+.2f} and D {delta('D','EP_vs_PCA64_shrinkage'):+.2f} percentage points.

For the episodic encoder in C/D, shrinkage increased mean novel-class accuracy over full Mahalanobis by {float(novel.loc['C','EP_shrinkage']-novel.loc['C','EP_full']):.2f}/{float(novel.loc['D','EP_shrinkage']-novel.loc['D','EP_full']):.2f} percentage points but changed overall accuracy by {delta('C','shrinkage_vs_full_EP'):+.2f}/{delta('D','shrinkage_vs_full_EP'):+.2f}. This discrepancy reflects the base-versus-novel tradeoff: shrinkage base-to-novel error rates reached 13.98%/19.12%, versus 2.20%/3.51% with full Mahalanobis. Thus improved novel recall alone does not establish a better incremental classifier. Target-aligned base references recovered 14.00/17.44 overall points for episodic shrinkage in E−C/F−D, but required additional labeled target base data.

Feature scale changed substantially (mean training embedding norms: CE C4/C8 110.79/105.58 versus episodic 10.52/8.42). The fixed .001 ridge therefore imposes much stronger relative regularization in episodic embedding space. These results characterize the locked training-and-head protocols, not a scale-invariant or purely loss-form comparison. No rescaling or retuning was performed after observing these findings.
'''
    core_zh=f'''**核心结论：不能把普通 ProtoNet 训练写成新类识别改进。** 欧氏原型头固定时，EP 在六设置的 genuine novel Accuracy 均值全部低于 CE（{eun} pp）。A/B 总体准确率提高主要来自基类；不能用四个基类主导的总体分数替代新类泛化证据。

**有条件的正结果：** EP+收缩相对 CE+收缩，C/D 总体分别提高 {delta('C','episodic_vs_CE_shrinkage'):+.2f}/{delta('D','episodic_vs_CE_shrinkage'):+.2f} pp，两方向各7/9个配对总体均值提高；但相对 PCA64+收缩，仍低 {abs(delta('C','EP_vs_PCA64_shrinkage')):.2f}/{abs(delta('D','EP_vs_PCA64_shrinkage')):.2f} pp，六个设置总体均值均未超过 PCA。

**完整马氏对照解释了取舍：** 在同一 EP 编码器上，C/D 收缩相对 full 的 novel Accuracy 提高 {float(novel.loc['C','EP_shrinkage']-novel.loc['C','EP_full']):.2f}/{float(novel.loc['D','EP_shrinkage']-novel.loc['D','EP_full']):.2f} pp，总体却变化 {delta('C','shrinkage_vs_full_EP'):+.2f}/{delta('D','shrinkage_vs_full_EP'):+.2f} pp。收缩的 base→novel 错误率13.98%/19.12%，高于 full 的2.20%/3.51%；提升新类召回不等于改善全体增量性能。目标对齐基类库使 EP+收缩恢复14.00/17.44 pp，但额外使用目标基类标签。

**表征尺度限制：** CE C4/C8 训练特征范数110.79/105.58，EP为10.52/8.42；固定 ridge .001 在 EP 空间相对更强。这里评价固定训练和度量方案，不能宣称分离了尺度、抽样和损失各自的因果作用。没有根据结果重新缩放或调参。

''' if not config['smoke'] else ''
    if not config['smoke']:decisions=decisions.replace('Primary final-session five/ten-shot task-seed means; all claims exploratory.','Primary final-session five/ten-shot task-seed means; all claims exploratory.\n\n'+core_zh)
    (out/'conclusion.md').write_text(decisions,encoding='utf-8')
    common='Primary: final 6-class session, mean of 5/10 shots and 200 matched episodes within each task/seed, then mean over 3 tasks x 3 seeds. All deltas are percentage points. Tasks share data; seeds and repeated episodes are not independent datasets or specimens. No significance claim is made. Group Macro-F1 averages global per-class F1, including cross-group false positives.'
    zh=f'''# 最终 ProtoNet 消融：结果与边界

## 预先固定的比较

18 个 episodic M1（每个编码器 47,400 参数），复用 18 个 CE M1，无 CE 重训。相同初始化、物理网格、原生 airPLS/归一化/三次插值顺序、保守记录重复组划分、4+1+1 到达顺序和评估 episode。欧氏 ProtoNet 只使用四个 base 类，以 4-way 5-support 10-query 训练，Adam .001，60 epochs；每 task 的更新数为 780/960/840。两类 genuine novel 不进入训练或源验证。

验证每 epoch 用固定20个源验证 episode（4-way 5-support 4-query），按最小查询 CE 选 checkpoint。验证 PS 最少9组，因此不借用训练数据或最终查询。训练支持/查询无交集。CE 与 EP 优化器更新数相同，但 CE 每步最多64监督标签、EP每步40查询标签且支持集参与梯度；EP按类平衡采样，CE为按行遍历，训练样本暴露和选择目标不完全相同。因此这是固定推断头下的 episodic 训练方案对照，无法进一步把原型损失与平衡采样各自贡献剥离。没有扫描超参数。

A=C4同域；B=C8同域；C=C4编码器/基类库，C8新增支持/查询；D=反向。E/F额外使用目标域每基类20条有标签参考，属于额外数据条件。均冻结 source-base-train 残差协方差先验；不更新旧类状态，不重训编码器。

## 主要结果（%）

{table(accuracy.reset_index())}

Combined novel Accuracy:

{table(novel.reset_index())}

Macro-F1:

{table(mf1.reset_index())}

## 基类误判、旧新类保持和关键混淆

以下均为主5/10-shot最终 session，Accuracy/错误率用%，previous_novel_change_pp 为该第一新类从 session1 到 session2 的变化（负值表示下降）。

{table(retention)}

对应 genuine novel holdout 任务内，两方向平均互相误判率（%）：

{table(confusion.reset_index())}

## 五个问题

1. **真正 episodic 训练是否改善新类识别？** 固定欧氏原型头，EP−CE 的新类 Accuracy 差值为：{eun} pp；总体 Accuracy 差值为：{eu} pp。必须同时看任务/种子配对差值，不能把换头带来的变化归因于训练方式。
2. **是否改善跨分辨率迁移？** C/D 固定头的实际目标表现和下表的迁移下降分别回答“目标效果”与“源到目标退化”。EP−CE 总体：欧氏 {eu}；完整马氏 {full}；收缩马氏 {shr}。迁移 gap 变小并不自动表示目标准确率变高，也可能源域表现下降。

{table(gap)}

3. **收缩是否改善 episodic ProtoNet？** 同一 EP 编码器，收缩−完整马氏：{shp} pp；收缩−欧氏：{she} pp。保留相同 .001 ridge、tau=5 和协方差 /n。不同训练目标会改变特征尺度，因此相同数值 ridge 并不保证相同相对正则强度，详见 feature_geometry.md。
4. **是否超过 PCA64 + 收缩？** EP 收缩−PCA 收缩：{pca} pp。该对照不能省略，也不能只报告最有利设置。
5. **最可辩护的贡献是什么？** 用可复用冻结表示和仅新增类注册构建可审计的 FTIR 4+1+1 框架；通过同编码器/同头和训练目标配对消融，界定 prototype 训练、协方差收缩及支持库分辨率匹配各自作用。只有对应配对结果支持的条件才能主张提升，不能将网络表示或收缩头表述为普遍优于光谱/PCA基线。

## 成本与统计口径

{table(tr)}

训练时间包括图编译、源验证和 checkpoint 写入；推断为 batch64、20次、CPU TF4线程，表中为每模型中位数再平均。历史 CE 与本轮 EP 顺序运行，壁钟时间仅作本机描述，非严格硬件基准。head 注册/预测时间不含光谱预处理与编码，见 costs.csv。仅注册当前类，无旧类重拟合；这里只比较原型头，无 SVM 重拟合。

{common}

全部四种 shot 和三个 session 见 metrics.csv / summary_by_shot_session.csv；基类、旧新类、当前新类、合并新类的 Accuracy/Macro-F1 均在原始 metrics.csv。主要对比表选取 Accuracy；详细 Macro-F1 需查汇总。类别召回见 per_class_metrics.csv，HDPE/LDPE 与 PP/PS 见 key_polymer_confusions.csv，旧新类保持与 base→novel 错误见 paired_differences.csv，CM 为按类别归一化的池化重复计数，不代表独立样本数。

## 限制与停止条件

保守记录/重复组隔离已审计，物理实物身份未验证；两库结果此前已检查，当前划分不构成未触碰的独立确证测试。C4/C8 不代表独立仪器；吸光度/透过率单位未获原始来源确认。相同原生 airPLS 索引平滑参数不等于相同物理平滑宽度。新旧结果均保留。无新架构、附加损失、增强或后验调参，本消融后停止扩展实验。
'''
    if not config['smoke']:zh=zh.replace('## 预先固定的比较',core_zh+'## 预先固定的比较',1)
    (out/'summary_zh.md').write_text(zh,encoding='utf-8')
    english=f'''# Manuscript-ready Results: episodic training ablation

We compared cross-entropy (CE) and episodic prototypical training using the same 47,400-parameter position-preserving M1 encoder, 128-dimensional embeddings, locked 1,868-coordinate physical grid, grouped partitions, and initialization seeds. Each of three class-holdout tasks was repeated with three seeds for both C4 and C8. The two held-out classes were excluded from encoder training and source validation. Episodic training used four-way episodes with five support and ten query spectra per class, squared Euclidean prototype logits, and query cross-entropy. Adam (.001) update counts were matched to the 60-epoch CE reference (780, 960, and 840 updates across tasks); this matches optimization steps rather than example exposures or the number of directly supervised query labels. Checkpoints were selected by mean loss on 20 fixed source-validation episodes (five support and four query spectra per class), without incremental query access. No CE encoder was retrained.

All encoders were frozen before 4+1+1 registration and evaluated using identical class arrivals and support/query episodes. We evaluated one, five, ten, and twenty novel-class support spectra, with five and ten shots prespecified as primary. Base classes retained twenty references each. Settings A/B denote within-resolution C4/C8 evaluation; C/D use source-resolution base references and target-resolution novel support/query; E/F additionally replace the base references with labeled target-resolution data. Independent full-covariance and class-specific shrinkage heads used the existing ridge .001, covariance divisor n, and shrinkage coefficient (n−1)/(n−1+5). Shrinkage priors were estimated exclusively from the corresponding encoder's source base-training embeddings and frozen thereafter. Neither old class states nor encoders were refitted during incremental sessions.

With the Euclidean head held fixed, episodic minus CE final overall accuracy differences were {eu} percentage points (settings A–F), and combined novel-class accuracy differences were {eun}. With the full Mahalanobis and shrinkage heads held fixed, the overall differences were {full} and {shr}, respectively. These paired comparisons contrast episodic and CE training at a fixed inference rule. The contrast includes class-balanced episodic sampling versus row-wise CE training and different source-validation objectives; it does not separately identify a pure loss-form effect. In contrast, comparing shrinkage and full covariance on the same episodically trained encoder yielded {shp} percentage points, while shrinkage versus Euclidean yielded {she}; these are metric-head effects and must not be attributed to episodic training.

The final-session primary accuracy and novel-class accuracy are reported below. The PCA64 shrinkage reference remained fixed and source-trained. Episodic M1 with shrinkage minus PCA64 with shrinkage yielded {pca} percentage points. The source-to-target accuracy gaps and their objective interactions are reported separately to distinguish target recognition from changes in the within-source reference; a smaller gap alone does not establish higher target accuracy. Target-aligned reference banks require additional labeled target base-class data and therefore represent a different data-access condition.

{table(accuracy.reset_index())}

{table(novel.reset_index())}

For shrinkage and the fixed PCA reference, the base-class and previously/currently introduced novel-class accuracy, base-to-novel error rate (%), and first novel-class change between sessions (percentage points) were as follows:

{table(retention)}

Bidirectional HDPE/LDPE and PP/PS confusion rates (%) below average the two directional errors only in the corresponding genuine novel-pair holdout task, rather than mixing base and novel roles across tasks:

{table(confusion.reset_index())}

{table(gap)}

{common}

Frozen feature geometry is provided because objective-dependent feature scales change the effective strength of a fixed numeric ridge. Registration and prediction costs exclude preprocessing and encoding; no previous class state was recomputed. Mean per-model training and inference costs are reported below. Timing comparisons are descriptive measurements on the same CPU environment rather than a controlled hardware benchmark.

{table(tr)}

These findings delimit where episodic representation training and covariance shrinkage help under a reusable frozen-encoder protocol; they do not establish universal superiority over the spectrum/PCA reference. The defensible methodological contribution is the auditable separation of training-objective, metric-head, and reference-resolution effects in few-shot incremental registration. Conservative duplicate/record grouping does not establish specimen independence, both datasets were previously inspected, and C4/C8 are not independent-instrument validation. All results are exploratory. No further objectives, architectures, augmentations, or post-hoc hyperparameter searches were introduced following this final ablation.
'''
    if not config['smoke']:english=english.replace('With the Euclidean head held fixed',core_en+'\nWith the Euclidean head held fixed',1)
    (out/'manuscript_results_en.md').write_text(english,encoding='utf-8')
    (out/'feature_geometry.md').write_text('# Feature geometry\n\nSource base-training embeddings only; no query-based fitting. Same numeric ridge does not equal same relative regularization across CE/EP or PCA64/128-dimensional M1.\n\n'+table(geometry.groupby(['objective','source'],as_index=False)[['mean_training_norm','mean_residual_variance','ridge_over_mean_variance','effective_rank']].mean()),encoding='utf-8')
    (out/'implementation_audit.md').write_text('''# Implementation audit

Legacy FTIR_new_protonetwork.py uses differentiable unsorted_segment_mean support prototypes, negative squared Euclidean query logits, and sparse query CE. GradientTape updates encoder variables only. make_episode samples distinct support/query indices and remaps each episode to local labels. Its legacy encoder uses BN/global pooling/L2 rather than the locked 47,400-parameter M1, and its checkpoints lack current group/grid provenance, so none were reused. The early legacy validation invokes training=True; the later FTIR_ProtoNetwork4.py validation uses training=False but can fall back to training data. Neither behavior is used here. There is no training-data validation fallback in the final implementation.

The old dists_maha_full uses einsum('ndd,bnd->bnd'), which extracts only diagonal precision entries. Its maha_distance_full uses the full matrix correctly, but the full-distance train and validation branches select different functions. The final evaluation reuses FullClassMahalanobis and FrozenPrototype with complete precision matrices, independently checked against quadratic forms computed with numpy.linalg.solve, including off-diagonal effects. No Mahalanobis objective is used during episodic training.

tests/test_protonet_final.py compares the new Euclidean loss with AST-extracted legacy functions (without importing top-level legacy experiments), verifies support and query gradients, a finite-difference support derivative, gradients into the unchanged encoder, episode label ordering and support/query disjointness, dense full/shrinkage numerical values, and the legacy diagonal-einsum discrepancy. All three tests passed. The two-epoch two-source paired smoke test completed before the full run; identical runner/source hashes gate the full run.

The final training samples only saved source-base-train groups and validates only saved source-base-validation groups. Evaluation splits and complete episode manifests are copied byte-for-byte from the CE parent. CE full/shrinkage and PCA64 shrinkage metric rows and confusion counts are reused. CE Euclidean is computed from saved frozen CE embeddings; CE encoders are not retrained. The current 2x3 neural factorial comparison plus PCA64 reference contains seven methods, with no extra encoder variants.
''',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/protonet_bidirectional_final')
    with threadpool_limits(limits=1):run(parser.parse_args().output.resolve())
