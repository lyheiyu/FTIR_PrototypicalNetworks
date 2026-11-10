"""Audit and manuscript reporting for the pre-specified final six-setting round."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from ftir_research_data import ROOT,load_dataset,preprocess
from ftir_research_core import base_covariance
from run_bidirectional_c4_c8_final import digest,write,CASES,METHODS

NAMES={'raw_1nn':'Spectrum 1-NN','raw_svm':'Spectrum SVM','PCA64_shrinkage':'PCA64 Shrinkage',
       'M1_1nn':'M1 1-NN','M1_full':'M1 Full','M1_shrinkage':'M1 Shrinkage'}
FIELDS=['all_accuracy','all_macro_f1','base_accuracy','seen_novel_accuracy','previous_novel_accuracy',
        'current_novel_accuracy','base_to_novel_error_rate','novel_to_base_error_rate','previous_novel_change_pp','base_change_from_session0_pp']


def md(frame):
    frame=frame.reset_index()
    def val(x):return '' if isinstance(x,(float,np.floating)) and np.isnan(x) else f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(val,row))+' |' for row in frame.itertuples(index=False,name=None)])


def audit(out,config):
    for path,value in config['source_hashes'].items():assert digest(ROOT/path)==digest(out/'source'/path)==value,path
    for path,value in config['old_checkpoint_hashes'].items():assert digest(ROOT/path)==value,path
    historical=ROOT/'results/encoder_architecture_v1'
    assert digest(out/'common_grid.npy')==config['grid_sha256']==digest(historical/'common_grid.npy')
    grid=np.load(out/'common_grid.npy');groups=pd.read_csv(out/'paired_group_representatives.csv',dtype={'c4_id':str,'c8_id':str})
    aliases=pd.read_csv(out/'all_original_ID_groups.csv',dtype=str)
    assert len(set(aliases.source_id))==len(aliases) and groups.group.is_unique
    data={s:load_dataset(s) for s in ['c4','c8']}
    original_summary=json.loads((ROOT/'dataset/deduplicated/summary.json').read_text())
    originals={d['source']:d for d in original_summary['datasets']}
    x={s:preprocess(data[s],grid,ROOT/'results/research_cache',baseline=True,flip=True)[groups[f'{s}_index'].to_numpy()].astype(float) for s in data}
    axis=[]
    for s,d in data.items():
        assert d.report['source_sha256']==config['dataset_hashes'][s]
        assert d.wave.min()<=grid.min() and grid.max()<=d.wave.max()
        np.testing.assert_array_equal(np.load(out/f'{s}_native_physical_coordinates.npy'),d.wave)
        np.testing.assert_array_equal(groups[f'{s}_id'],d.ids[groups[f'{s}_index'].to_numpy()])
        np.testing.assert_array_equal(groups.label,d.labels[groups[f'{s}_index'].to_numpy()])
        mapping_path=ROOT/'dataset/deduplicated'/f'FTIR_PLastics500_{s}_row_mapping.csv'
        assert digest(mapping_path)==config['mapping_hashes'][s]
        mapping=pd.read_csv(mapping_path,dtype=str);lookup=dict(zip(aliases.source_id,aliases.group))
        assert set(mapping.source_id)==set(aliases.source_id) and set(mapping.retained_source_id)==set(d.ids)
        original_name=f'FTIR_PLastics500_{s}.csv'
        assert digest(ROOT/'dataset'/original_name)==originals[original_name]['source_sha256']
        with (ROOT/'dataset'/original_name).open(encoding='latin-1',newline='') as handle:
            reader=csv.reader(handle);next(reader);original_wave=np.sort(np.asarray(next(reader)[3:],float))
        np.testing.assert_array_equal(original_wave,d.wave)
        for a,b in zip(mapping.source_id,mapping.retained_source_id):assert lookup[a]==lookup[b]
        axis.append(dict(dataset=s,native_points=len(d.wave),native_min=d.wave.min(),native_max=d.wave.max(),
            median_spacing=np.median(np.diff(d.wave)),minimum_spacing=np.min(np.diff(d.wave)),maximum_spacing=np.max(np.diff(d.wave)),
            locked_grid_points=len(grid),locked_min=grid.min(),locked_max=grid.max(),extrapolated_coordinates=0))
    pd.DataFrame(axis).to_csv(out/'physical_grid_audit.csv',index=False)
    ac=x['c4']-x['c4'].mean(axis=1,keepdims=True);bc=x['c8']-x['c8'].mean(axis=1,keepdims=True)
    denom=np.linalg.norm(ac,axis=1)*np.linalg.norm(bc,axis=1)
    corr=np.divide(np.sum(ac*bc,axis=1),denom,out=np.full(len(ac),np.nan),where=denom>0)
    similarity=groups[['group','label','c4_id','c8_id']].copy();similarity['common_grid_pearson']=corr
    similarity['common_grid_rmse']=np.sqrt(np.mean((x['c4']-x['c8'])**2,axis=1))
    similarity.to_csv(out/'record_similarity_audit.csv',index=False)
    records=[];state_checks=0;raw_checks=0
    labels=groups.label.to_numpy()
    role_counts=[]
    for task in config['tasks']:
        for seed in config['seeds']:
            folder=out/task['name']/f'seed_{seed}';split=json.loads((folder/'split.json').read_text())
            episodes=json.loads((folder/'episodes.json').read_text());done=json.loads((folder/'COMPLETE.json').read_text())
            assert digest(folder/'split.json')==done['split_sha256'] and digest(folder/'episodes.json')==done['episodes_sha256']
            train,val=np.array(split['train']),np.array(split['validation'])
            query_pool=set(sum(split['base_query'].values(),[])+sum(split['novel_query'].values(),[]))
            support_pool=set(sum(split['novel_support'].values(),[]))
            assert not set(train)&set(val) and not query_pool&(set(train)|set(val)|support_pool)
            assert set(labels[train])==set(labels[val])==set(task['base'])
            assert len(split['roles'])==len(groups)
            for c in sorted(set(labels)):
                for role in sorted(set(split['roles'])):
                    role_counts.append(dict(task=task['name'],seed=seed,**{'class':c},role=role,record_groups=int(np.sum((labels==c)&(np.array(split['roles'])==role)))))
            for ep in episodes:
                base=set(sum(ep['base_support'].values(),[]));novel=set(sum(ep['novel_support_order'].values(),[]));q=set(sum(ep['query'].values(),[]))
                assert base<=set(train) and novel<=support_pool and q<=query_pool and not q&(base|novel)
                assert all(len(qc)==15 for qc in ep['query'].values())
            orders=np.load(folder/'paired_training_orders.npy')
            assert len(orders)==config['epochs']
            for order in orders:assert set(order)==set(train) and len(order)==len(train)
            initial=[];steps=[]
            for source in ['c4','c8']:
                model_folder=folder/source/'M1'
                meta=json.loads((model_folder/'PAIRED_TRAINING_COMPLETE.json').read_text())
                frozen=json.loads((model_folder/'FROZEN_CHECK.json').read_text())
                stats=meta['stats'];initial.append(meta['initial_encoder_hash']);steps.append(stats['optimizer_updates'])
                assert meta['split_sha256']==digest(folder/'split.json')
                assert frozen['file_before']==frozen['file_after']==digest(model_folder/'encoder.keras')
                assert frozen['weights_before']==frozen['weights_after']==stats['weight_hash_after']
                assert frozen['parameters']==47400 and frozen['incremental_training_updates']==0
                history=pd.read_csv(model_folder/'training_history.csv')
                assert len(history)==config['epochs'] and int(history.loc[history.val_loss.idxmin(),'epoch'])==stats['best_epoch']
                with np.load(model_folder/'frozen_views.npz') as features:
                    np.testing.assert_array_equal(features['fit_indices'],train)
                    np.testing.assert_allclose(features['pca_mean'],x[source][train].mean(axis=0),atol=1e-12)
                    components=features['pca_components']
                    np.testing.assert_allclose(components@components.T,np.eye(64),atol=1e-10)
                    for s in ['c4','c8']:
                        np.testing.assert_allclose((x[s]-features['pca_mean'])@components.T,features[f'PCA64_{s}'],atol=1e-10)
                    for view in ['M1','PCA64']:
                        np.testing.assert_allclose(base_covariance(features[f'{view}_{source}'][train],labels[train]),features[f'{view}_prior'],atol=1e-11)
                records.append(dict(task=task['name'],seed=seed,source=source,**frozen,source_base_train_only=True,source_validation_only=True))
            assert initial[0]==initial[1] and steps[0]==steps[1]==config['epochs']*int(np.ceil(len(train)/64))
            state_checks+=done['old_class_state_checks'];raw_checks+=done['raw_source_invariance_checks']
    pd.DataFrame(records).to_csv(out/'frozen_encoder_verification.csv',index=False)
    pd.DataFrame(role_counts).to_csv(out/'partition_counts.csv',index=False)
    result=dict(status='passed',encoders=len(records),paired_groups=len(groups),old_class_state_checks=state_checks,raw_source_invariance_checks=raw_checks,
        evaluation_query_groups_excluded_from_training_and_selection=True,source_PCA_and_covariance_fit_verified=True,
        equal_pair_initialization_orders_update_counts=True,historical_checkpoints_unchanged=True,incremental_encoder_updates=0,
        specimen_independence_verified=False,genuinely_untouched_confirmatory_holdout=False,analysis_source_sha256=digest(__file__))
    write(out/'protocol_audit.json',result)
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return groups,result


def comparisons(trial):
    final=trial[trial.session==2]
    definitions=[('forward_transfer_C_minus_A','C','A'),('reverse_transfer_D_minus_B','D','B'),
                 ('target_bank_forward_E_minus_C','E','C'),('target_bank_reverse_F_minus_D','F','D'),
                 ('source_effect_target_C8_E_minus_B','E','B'),('source_effect_target_C4_F_minus_A','F','A'),
                 ('direct_forward_minus_reverse_C_minus_D','C','D')]
    rows=[]
    for metric in FIELDS:
        tab=final.pivot(index=['task','seed','shot','method'],columns='setting',values=metric)
        factor=1 if metric.endswith('_pp') else 100
        for name,a,b in definitions:
            delta=factor*(tab[a]-tab[b])
            for (task,seed,k,method),value in delta.items():rows.append(dict(comparison=name,metric=metric,task=task,seed=seed,shot=k,method=method,delta_pp=value))
        delta=factor*((tab.C-tab.A)-(tab.D-tab.B))
        for (task,seed,k,method),value in delta.items():rows.append(dict(comparison='degradation_asymmetry_(C-A)-(D-B)',metric=metric,task=task,seed=seed,shot=k,method=method,delta_pp=value))
    paired=pd.DataFrame(rows)
    primary=paired[paired.shot.isin([5,10])].groupby(['comparison','metric','task','seed','method']).delta_pp.mean().reset_index()
    summary=primary.groupby(['comparison','metric','method']).delta_pp.agg(['mean','min','max','count',lambda x:int((x>1e-9).sum())])
    summary.rename(columns={'<lambda_0>':'positive_pairs'},inplace=True)
    return paired,primary,summary


def main(out):
    assert (out/'COMPLETE.json').exists()
    config=json.loads((out/'experiment_manifest.json').read_text());groups,checked=audit(out,config)
    data=pd.read_csv(out/'bidirectional_transfer_metrics.csv')
    key=['task','seed','shot','session','setting','method']
    trial=data.groupby(key)[FIELDS].mean().reset_index();trial.to_csv(out/'trial_metrics.csv',index=False)
    summary=trial.groupby(['shot','session','setting','method'])[FIELDS].mean().reset_index();summary.to_csv(out/'summary_metrics.csv',index=False)
    paired,primary,paired_summary=comparisons(trial)
    paired.to_csv(out/'paired_comparisons.csv',index=False);primary.to_csv(out/'primary_paired_comparisons.csv',index=False);paired_summary.to_csv(out/'primary_paired_summary.csv')
    final=trial[(trial.session==2)&trial.shot.isin([5,10])]
    means=final.groupby(['task','seed','setting','method'])[FIELDS].mean().reset_index()
    primary_table=means.groupby(['setting','method'])[FIELDS].mean();primary_table.to_csv(out/'primary_5_10shot_summary.csv')
    method_rows=[]
    for field in ['all_accuracy','all_macro_f1','base_accuracy','seen_novel_accuracy']:
        tab=means.pivot(index=['task','seed','setting'],columns='method',values=field)
        for name,a,b in [('shrink_minus_full','M1_shrinkage','M1_full'),('PCA_shrink_minus_M1_shrink','PCA64_shrinkage','M1_shrinkage')]:
            for (task,seed,case),value in (100*(tab[a]-tab[b])).items():method_rows.append(dict(comparison=name,metric=field,task=task,seed=seed,setting=case,delta_pp=value))
    method_pairs=pd.DataFrame(method_rows);method_pairs.to_csv(out/'paired_method_comparisons.csv',index=False)
    method_summary=method_pairs.groupby(['comparison','metric','setting']).delta_pp.agg(['mean','min','max','count',lambda x:int((x>1e-9).sum())])
    method_summary.rename(columns={'<lambda_0>':'positive_pairs'},inplace=True);method_summary.to_csv(out/'primary_method_summary.csv')
    history=ROOT/'results/heterogeneous_ftir_transfer_v1/c8_transfer_metrics.csv'
    if history.exists():
        old=pd.read_csv(history);old=old[old.scenario=='C8_resolution'].copy()
        old['reference_status']='H: legacy C4 encoder and C4 novel support -> C8 query; different checkpoints/splits; NOT setting C'
        old.to_csv(out/'historical_C4_support_C8_query_reference.csv',index=False)
    plots(out,config,summary,paired_summary)
    export_confusions(out,config)
    report(out,config,checked,groups,primary_table,paired_summary,method_summary)
    print('Audit passed:',checked)
    print((100*primary_table.all_accuracy.unstack('method')).round(3).to_string())


def plots(out,config,summary,paired):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ax,case in zip(axes.flat,CASES):
        for method in METHODS:
            rows=summary[(summary.setting==case)&(summary.method==method)&(summary.session==2)].sort_values('shot')
            ax.plot(rows.shot,100*rows.all_accuracy,marker='o',label=NAMES[method])
        enc,bs,ns,qs=CASES[case]
        ax.set_title(f'{case}: encoder {enc.upper()}, base {bs.upper()}, novel/query {qs.upper()}',fontsize=10)
        ax.set_xticks([1,5,10,20]);ax.set_xlabel('Novel support / class');ax.set_ylabel('Final accuracy (%)');ax.set_ylim(0,100);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=6,fontsize=8)
    fig.tight_layout(rect=[0,0,1,.96])
    for ext in ['png','pdf']:fig.savefig(folder/f'six_setting_accuracy.{ext}',dpi=180)
    plt.close(fig)
    definitions=[('forward_transfer_C_minus_A','C4 to C8: C - A'),('reverse_transfer_D_minus_B','C8 to C4: D - B'),
                 ('target_bank_forward_E_minus_C','Target C8 base bank: E - C'),('target_bank_reverse_F_minus_D','Target C4 base bank: F - D'),
                 ('source_effect_target_C8_E_minus_B','Encoder/prior source, target C8: E - B'),('source_effect_target_C4_F_minus_A','Encoder/prior source, target C4: F - A')]
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    all_effects=np.concatenate([paired.loc[(name,'all_accuracy'),:].reindex(METHODS)['mean'].to_numpy() for name,_ in definitions])
    low,high=min(0,float(all_effects.min())),max(0,float(all_effects.max()))
    padding=max(1.,.1*(high-low))
    for ax,(name,title) in zip(axes.flat,definitions):
        values=paired.loc[(name,'all_accuracy'),:].reindex(METHODS)['mean']
        ax.bar(np.arange(len(METHODS)),values,color=['#777777','#aaaaaa','#e69f00','#009e73','#0072b2','#cc79a7'])
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(len(METHODS)));ax.set_xticklabels([NAMES[m] for m in METHODS],rotation=45,ha='right',fontsize=8)
        ax.set_title(title,fontsize=10);ax.set_ylabel('Paired accuracy change (pp)');ax.set_ylim(low-padding,high+padding);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'paired_transfer_and_source_effects.{ext}',dpi=180)
    plt.close(fig)


def export_confusions(out,config):
    folder=out/'confusion_matrices';folder.mkdir(exist_ok=True)
    all_cms={};task_cms={}
    for task in config['tasks']:
        for seed in config['seeds']:
            trial=out/task['name']/f'seed_{seed}';index=json.loads((trial/'confusion_index.json').read_text())
            with np.load(trial/'confusions.npz') as matrices:
                for token,meta in index.items():
                    if meta['session']!=2:continue
                    key=(meta['setting'],meta['shot'],meta['method']);classes=meta['classes'];cm=matrices[token]
                    if key in all_cms:assert all_cms[key][0]==classes
                    else:all_cms[key]=(classes,np.zeros_like(cm))
                    all_cms[key][1][:]+=cm
                    taskkey=(task['name'],)+key
                    if taskkey not in task_cms:task_cms[taskkey]=(classes,np.zeros_like(cm))
                    task_cms[taskkey][1][:]+=cm
    for (case,k,method),(classes,cm) in all_cms.items():
        pd.DataFrame(cm,index=classes,columns=classes).to_csv(folder/f'all_tasks_{case}_{k}shot_{method}_counts.csv')
    for (task,case,k,method),(classes,cm) in task_cms.items():
        if k not in [5,10]:continue
        target=folder/task;target.mkdir(exist_ok=True)
        pd.DataFrame(cm,index=classes,columns=classes).to_csv(target/f'{case}_{k}shot_{method}_counts.csv')
        pd.DataFrame(cm/cm.sum(axis=1,keepdims=True),index=classes,columns=classes).to_csv(target/f'{case}_{k}shot_{method}_row_normalized.csv')
    pair_rows=[]
    for task in config['tasks']:
        pair=[task['novel_sessions'][0][0],task['novel_sessions'][1][0]]
        for case in CASES:
            for method in METHODS:
                for k in [5,10]:
                    classes,cm=task_cms[task['name'],case,k,method]
                    for true,wrong in [pair,pair[::-1]]:
                        i,j=classes.index(true),classes.index(wrong)
                        pair_rows.append(dict(task=task['name'],setting=case,shot=k,method=method,true_class=true,other_novel_class=wrong,
                            recall=float(cm[i,i]/cm[i].sum()),confused_with_other_novel=float(cm[i,j]/cm[i].sum()),query_occurrences=int(cm[i].sum())))
    pd.DataFrame(pair_rows).to_csv(out/'key_polymer_confusions.csv',index=False)
    for task in config['tasks']:
        for k in [5,10]:
            fig,axes=plt.subplots(2,3,figsize=(13,8))
            for row,case in enumerate(['C','D']):
                for col,method in enumerate(['PCA64_shrinkage','M1_full','M1_shrinkage']):
                    classes,cm=task_cms[task['name'],case,k,method];norm=cm/cm.sum(axis=1,keepdims=True);ax=axes[row,col]
                    ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
                    ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xticklabels(classes,rotation=45);ax.set_yticklabels(classes)
                    for i in range(6):
                        for j in range(6):ax.text(j,i,f'{100*norm[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if norm[i,j]>.5 else 'black')
                    ax.set_title(f'{case}: {NAMES[method]}');ax.set_xlabel('Predicted');ax.set_ylabel('True')
            fig.suptitle(f'{task["name"]}: {k}-shot, pooled repeated-query row %',fontsize=11);fig.tight_layout()
            for ext in ['png','pdf']:fig.savefig(folder/task['name']/f'bidirectional_{k}shot.{ext}',dpi=180)
            plt.close(fig)


def report(out,config,checked,groups,primary,paired,method_summary):
    table=100*primary.all_accuracy.unstack('method').reindex(columns=METHODS)
    f1=100*primary.all_macro_f1.unstack('method').reindex(columns=METHODS)
    base=100*primary.base_accuracy.unstack('method').reindex(columns=METHODS)
    novel=100*primary.seen_novel_accuracy.unstack('method').reindex(columns=METHODS)
    retention=primary.previous_novel_change_pp.unstack('method').reindex(columns=METHODS)
    groupcounts=groups.groupby('label').agg(record_groups=('group','size'),max_original_ID_component=('original_ID_count','max'))
    axes=pd.read_csv(out/'physical_grid_audit.csv')
    audit_text=['# Protocol and leakage audit: final C4/C8 round','',md(axes.set_index('dataset')),'',
        'Both measured native axes cover the exact locked 1868-point historical physical grid. Cubic interpolation is bounded; no new axis, extrapolation, padding or D4 input is used. The native coordinate arrays and locked grid are saved exactly.','',
        'The legacy pipeline applies airPLS on native sample-index grids BEFORE interpolation. Identical numerical settings at different native point densities do not establish identical baseline smoothing in physical wavenumber units. The observed contrast therefore includes native sampling and legacy preprocessing effects, not a pure causal optical-resolution intervention. Signal absorbance/transmittance semantics and acquisition hardware remain unverified.','',
        '## Conservative record grouping','',md(groupcounts),'',
        'Original source IDs are blocked across C4/C8; native duplicate aliases from BOTH 3000-row mapping files are joined transitively, then exact common-grid processed equality is also blocked. This yields 2047 shared groups, including only 61 PS groups. Every selected representative has the same retained original ID in both resolutions. One representative/group/resolution is used for training and evaluation, preventing sample-count and duplicate-group-weight differences between sources. ID-based grouping is a conservative blocking proxy, not verified physical specimen provenance.','',
        'Conservative transitive grouping may merge genuinely different objects, and one representative omits within-component spectral variation. The tested population is this fixed paired representative set, not every native unique spectrum. Original source-file hashes and physical axes are checked against the retained deduplication provenance.','',
        '## Why both encoders were trained afresh','',
        'Historical C4 base rows were all used for training or checkpoint validation, leaving no clean base-query pool under the new requirement. Therefore both C4 and C8 M1 were retrained from scratch on common group splits, with unchanged architecture, Adam .001, CE, batch64 and 60 epochs. Historical checkpoints are unchanged. Each pair uses identical initialization seed, epoch group orders and update count; only source spectral inputs differ. Selection uses the lowest SOURCE base-validation loss; no early stopping or target/evaluation label selection. Temporary classification heads are not included in frozen encoders.','',
        'Base groups: query=max(15,floor(.20n)); validation=max(5,floor(.15n)); remaining groups train. Base references are fixed 20/group/class sampled from train. Novel groups are split 50/50 support/query, allowing 1/5/10/20-shot and fifteen disjoint query groups/class even for PS. Roles are identical across resolutions, so target query counterparts also cannot enter source training or validation. Within every episode, query groups and support groups are disjoint. Queries and maximal support order are shared across all settings, classifiers and nested shots.','',
        '## Models and fixed heads','',
        'M1: original position-preserving multiscale CNN, 47,400 encoder parameters and 128-dimensional ReLU embedding. Independent full class covariance uses residual.T@residual/n + .001 I, cached inverse, quadratic distance, no determinant or class prior. Shrinkage uses alpha=(n−1)/(n−1+5), same .001 ridge and the encoder SOURCE base_train within-class prior; E/F do not refit it on target references. All empirical support covariances are singular before ridge because n≤20. The original script’s effective double ridge .002 is a historical formula, not this fixed matched-ridge main control.','',
        'PCA64 is source-base-train-only, randomized SVD, fixed seed, no whitening; its own 64×64 source base prior is frozen. It uses the same nominal ridge .001 and tau5 as M1-128. Different dimensions and variances mean relative regularisation is NOT matched; source-only scale diagnostics are saved in regularisation_geometry.csv. Raw SVM is linear C=1; all baselines retain exactly twenty base exemplars plus K per introduced novel class, not all source training rows.','',
        'Prototype registration changes arriving-class statistics only. Existing class means/precision arrays are checked unchanged for the first repeat of every shot/session/setting. The evaluation receives frozen feature arrays and no optimizer/encoder; checkpoint file hashes are verified after evaluation. Raw A=F and B=E metric invariance provides sanity controls for matched target-support encoder-source comparisons.','',
        '## What is and is not independent','',
        'The protocol excludes known exact duplicates, linked record IDs and their transitive aliases across roles and resolutions. No supplied metadata establishes specimen-level identity, so no result is certified specimen-independent. C4/C8 are resolution-family datasets, not independent instruments; matching record IDs is not proof of identical physical acquisition. Common-grid correlation diagnostics are descriptive only and never select queries.','',
        'Both source datasets and earlier results were already inspected. The new query groups are excluded from this round’s training/model selection, but are NOT a genuinely untouched confirmatory holdout. All results remain exploratory; nine task/seed pairs also share datasets, and 200 repeats are not independent specimens or 200 independent trained models.','',
        'Historical H (old C4 encoder, C4 base AND novel support, C8 query) is retained separately. It is not setting C, which registers novel C8 support, and uses different checkpoints/group partitions. Do not attribute H-versus-C differences solely to novel-support source.','',
        f'Audit passed: {checked["encoders"]} encoders, {checked["raw_source_invariance_checks"]} raw invariance checks, {checked["old_class_state_checks"]} old-class state checks; zero incremental encoder updates.','']
    (out/'protocol_and_leakage_audit.md').write_text('\n'.join(audit_text),encoding='utf-8')
    settings=pd.DataFrame([dict(setting=k,encoder=v[0],base_reference=v[1],novel_support=v[2],query=v[3]) for k,v in CASES.items()]).set_index('setting')
    lines=['# C4/C8 双向跨分辨率 few-shot 增量：最终结果','',
        f'本轮只使用 C4/C8。{len(config["tasks"])} 任务 × {len(config["seeds"])} 种子；每对 C4、C8 模型同初始化、同训练组顺序、同 {config["epochs"]} epoch 预算；新增类阶段 encoder 始终冻结。主要指标先在每个任务/种子内平均 {config["repeats"]} 次重复与 5/10-shot，再汇总配对值。','',
        '**研究边界：这是记录组隔离的探索性评价，不是已验证实物独立、也不是全新未查看的确认性外部测试。** 两侧都重新训练是为了让 query 组不参与训练或 checkpoint 选择；没有改架构、loss、增强或搜索超参数。','',
        '## 六种设置','',md(settings),'',
        '所有设置使用相同语义记录组、20 个 base 支持/类、相同 K 个 novel 支持/类、同一到达顺序和每类 15 个平衡 query。E/F 只改 base 支持库来源，源域协方差 prior 仍保持不变。E/F 需要额外取得目标来源的 20 条有标签 base 参考光谱/类；C/D 不使用目标 base 参考。因此参考库对齐收益不能解释为无需额外目标参考数据的算法收益。','',
        '## 主要结果：5/10-shot 平均最终 Accuracy（%）','',md(table.rename(columns=NAMES)),'',
        'Macro-F1（%）：','',md(f1.rename(columns=NAMES)),'',
        'Base Accuracy（%）：','',md(base.rename(columns=NAMES)),'',
        '两个已加入 novel 类的 Accuracy（%）：','',md(novel.rename(columns=NAMES)),'',
        '## 双向迁移与来源因素','']
    for method in ['M1_full','M1_shrinkage','PCA64_shrinkage','M1_1nn']:
        forward=paired.loc[('forward_transfer_C_minus_A','all_accuracy',method),'mean']
        reverse=paired.loc[('reverse_transfer_D_minus_B','all_accuracy',method),'mean']
        bank_f=paired.loc[('target_bank_forward_E_minus_C','all_accuracy',method),'mean']
        bank_r=paired.loc[('target_bank_reverse_F_minus_D','all_accuracy',method),'mean']
        source8=paired.loc[('source_effect_target_C8_E_minus_B','all_accuracy',method),'mean']
        source4=paired.loc[('source_effect_target_C4_F_minus_A','all_accuracy',method),'mean']
        lines.append(f'- {NAMES[method]}：C4→C8 为 {table.loc["C",method]:.3f}%，相对同源 A 变化 {forward:+.3f} pp；C8→C4 为 {table.loc["D",method]:.3f}%，相对同源 B 变化 {reverse:+.3f} pp。改用 target base 库的收益 E−C={bank_f:+.3f} pp、F−D={bank_r:+.3f} pp；目标支持/查询完全匹配时，来源变化 E−B={source8:+.3f} pp、F−A={source4:+.3f} pp。')
    for method in ['M1_full','M1_shrinkage']:
        direction=table.loc['D',method]-table.loc['C',method]
        lines.append(f'{NAMES[method]} 的绝对方向比较：本轮平均 {"C8→C4 更高" if direction>0 else "C4→C8 更高" if direction<0 else "两方向相同"}，D−C={direction:+.3f} pp；逐任务一致性仍需看配对明细。')
    lines+=['','C−D 是双向绝对准确率差，(C−A)−(D−B) 是相对于各自同源基线的退化不对称，两者不能混同。对应的每个任务/种子/shot 配对值在 paired_comparisons.csv。',
        'E−B 和 F−A 在 M1 1-NN 与独立完整类协方差中不使用全局 base prior，反映固定分类公式下的 encoder 来源变化；M1/PCA 收缩方法还包含来源 prior 的变化，不能统称纯 encoder 因果效应。E−C、F−D 则固定 encoder 和适用的 source prior，改变的是同组 base 参考光谱来源。原谱 A=F、B=E 指标必须精确一致，已作为自动检查。','',
        '## 完整协方差与收缩是否一致获益','',
        md(method_summary.loc[('shrink_minus_full','all_accuracy'),:].rename_axis('setting')),'',
        '正值表示收缩更好，positive_pairs 是九组配对值中的正差数量；这些不是九个独立数据集，也没有按 query 结果调整 ridge 或 alpha。1/20-shot 次要结果保存在 summary_metrics.csv 和曲线图中。','',
        '## PCA64＋收缩是否有竞争力','',
        md(method_summary.loc[('PCA_shrink_minus_M1_shrink','all_accuracy'),:].rename_axis('setting')),'',
        '正值表示 PCA64＋收缩更高。PCA64 与 M1-128 的容量、尺度和相对 ridge 强度不同，比较是固定既定方案的性能控制，不证明纯粹由非线性网络或某个维度产生差异。PCA 与两种 covariance prior 都只拟合各自源域合法 base_train。','',
        '## 旧 novel 保持与 base/novel 竞争','',
        '第一新增类的同一批 query，在第二类加入后 Accuracy 的变化（pp；负值表示下降）：','',md(retention.rename(columns=NAMES)),'',
        '旧类 prototype 状态没有改变，但新增类会参与距离竞争，所以旧类准确率仍可能下降。这不是梯度更新覆盖旧权重。base_to_novel_error_rate、novel_to_base_error_rate、base_change_from_session0_pp 以及抽样 quadratic score 诊断分别保存在主指标和 score_competition.csv。','',
        'HDPE/LDPE、PP/PS 的召回率与相互混淆率保存在 key_polymer_confusions.csv，可结合相应 holdout 任务的双向混淆矩阵解读。矩阵累计的是重复 query 出现次数，不是独立实物数量。主 Macro-F1 先在 episode 内计算再平均；per_class_metrics.csv 是累计混淆矩阵导出的逐类指标，聚合方式不同。','',
        '## 样本隔离与训练','',md(groupcounts),'',
        'PS 经双方 duplicate alias 的传递合并后仅有 61 个保守记录组；当它是 base 时，37 train、9 validation、15 query。采用每组每分辨率一条代表后，两侧训练组数量相同。不能把本轮绝对分数与旧版按独立去重行划分的结果直接相减归因。',
        'reverse_training_summary.csv 同时保存两侧所有 18 个模型的 epoch、最优 source-validation epoch、训练时间与更新次数。旧 C4 checkpoints 和所有历史文件均保持原样。','',
        '## 支持的结论与限制','',
        '可以检验：在共同波数网格、已知重复/记录组隔离、固定训练预算下，跨 C4/C8 来源的分类表现与方向不对称；在不更新 encoder 的情况下新增两类的效果；改用目标来源 base 参考库的影响。',
        '不能声称：已完成独立仪器或实物独立的确认性泛化；同记录编号必然代表相同物理采集；历史已查看数据上的结果是全新未触碰测试；固定 nominal ridge 意味着 PCA64 与 M1-128 的相对正则完全相同。',
        '原流程在原生采样点上先做 airPLS 再插值；相同数值参数并不保证在不同原生点密度下具有相同的物理平滑尺度。方向差异包含这一固定预处理条件，不能纯归因于仪器光学分辨率，也未确认物理信号为吸光度或透射率。',
        '本轮结束当前论文的实验开发。应按实际正负结果写作，不再据 query 结果继续更换结构、loss 或调参。','',
        '## 文件','',
        '- protocol_and_leakage_audit.md、protocol_audit.json：坐标、记录别名、query/选模隔离、冻结验证。',
        '- bidirectional_transfer_metrics.csv、trial_metrics.csv：全部阶段与 1/5/10/20-shot。',
        '- paired_comparisons.csv、primary_paired_summary.csv：方向、来源与支持库配对差值。',
        '- per_class_metrics.csv、confusion_matrices/：逐类 Precision/Recall/F1 和矩阵。',
        '- figures/：六设置曲线、来源效应配对图。',
        '- historical_C4_support_C8_query_reference.csv：独立保留历史 H，不当作 C。',
        '- manuscript_results_en.md：可用于论文的简短英文 Results 草稿。','']
    shrink_means=method_summary.loc[('shrink_minus_full','all_accuracy'),:]['mean']
    pca_cross=method_summary.loc[('PCA_shrink_minus_M1_shrink','all_accuracy'),:].loc[['C','D','E','F']]['mean']
    positive=', '.join(shrink_means.index[shrink_means>0]);negative=', '.join(shrink_means.index[shrink_means<0])
    bank_magnitude=np.mean([abs(paired.loc[(name,'all_accuracy','M1_1nn'),'mean']) for name in ['target_bank_forward_E_minus_C','target_bank_reverse_F_minus_D']])
    source_magnitude=np.mean([abs(paired.loc[(name,'all_accuracy','M1_1nn'),'mean']) for name in ['source_effect_target_C8_E_minus_B','source_effect_target_C4_F_minus_A']])
    findings=['## 主要发现','',
        f'收缩相对完整类协方差的平均 Accuracy 更高的设置为 {positive or "无"}，更低的设置为 {negative or "无"}；因此应按来源/参考库条件陈述优势，不能只挑一个方向。',
        f'四个跨来源设置 C/D/E/F 中，PCA64＋收缩减 M1＋收缩的平均差值范围为 {pca_cross.min():+.3f} 到 {pca_cross.max():+.3f} pp。'+('本轮平均结果未建立神经网络表示相对该 PCA 控制的性能优势。' if pca_cross.min()>0 else '表示优劣需要按设置分别判断。'),
        f'不含全局 covariance prior 的 M1 1-NN 对照中，两方向参考库对齐效应的平均绝对量为 {bank_magnitude:.3f} pp，匹配目标支持/查询时 encoder 来源效应的平均绝对量为 {source_magnitude:.3f} pp。'+('本协议下参考库来源的条件效应量级更大。' if bank_magnitude>source_magnitude else '本协议下 encoder 来源的条件效应不可忽略。')+' 这是所覆盖六设置的条件比较，不是完整 factorial 的唯一因果分解。','']
    if config['smoke']:lines[2:2]=['**仅为实现验证 smoke，不可作为论文结果。**','']
    key_errors=pd.read_csv(out/'key_polymer_confusions.csv')
    focus=['## HDPE/LDPE 与 PP/PS 的逐类诊断','',
           '以下是对应 novel holdout 任务在 5/10-shot 的平均 Recall（%）；不能用互相误判率低代替高 Recall，因为 query 也可能被吸引到 base 类。相互误判率另存 key_polymer_confusions.csv。','']
    for name in ['holdout_hdpe_ldpe','holdout_pp_ps']:
        subset=key_errors[(key_errors.task==name)&key_errors.setting.isin(['C','D','E','F'])&key_errors.method.isin(['PCA64_shrinkage','M1_full','M1_shrinkage'])]
        if len(subset):
            recalls=100*subset.groupby(['setting','true_class','method']).recall.mean().unstack('method')
            focus += [name,'',md(recalls.rename(columns=NAMES)),'']
    text='\n'.join(lines).replace('## 六种设置','\n'.join(findings)+'\n## 六种设置',1).replace('## 样本隔离与训练','\n'.join(focus)+'\n## 样本隔离与训练',1)
    (out/'final_results_zh.md').write_text(text,encoding='utf-8')
    en=['# Results draft','',
        f'We evaluated six prespecified within- and cross-resolution settings using {len(groups)} conservative C4/C8 record components, {len(config["tasks"])} base/novel tasks and {len(config["seeds"])} seeds. Both 47,400-parameter M1 encoders were trained afresh with matched initialization, group orders and {config["epochs"]}-epoch cross-entropy budgets; evaluation-query components were excluded from training and checkpoint selection in both resolutions. All incremental sessions used frozen encoders, 20 base references per class and 1/5/10/20 novel supports, with 5/10-shot conditions designated primary.','']
    for method in ['M1_full','M1_shrinkage']:
        en.append(f'Averaged within task/seed over the primary shots, {NAMES[method]} achieved {table.loc["C",method]:.2f}% final accuracy for C4-to-C8 and {table.loc["D",method]:.2f}% for C8-to-C4 transfer. Relative to the corresponding within-source conditions, changes were {paired.loc[("forward_transfer_C_minus_A","all_accuracy",method),"mean"]:+.2f} and {paired.loc[("reverse_transfer_D_minus_B","all_accuracy",method),"mean"]:+.2f} percentage points. Replacing source-domain base references with target-domain references changed accuracy by {paired.loc[("target_bank_forward_E_minus_C","all_accuracy",method),"mean"]:+.2f} and {paired.loc[("target_bank_reverse_F_minus_D","all_accuracy",method),"mean"]:+.2f} points, respectively.')
    shrink=method_summary.loc[('shrink_minus_full','all_accuracy'),:]
    pca=method_summary.loc[('PCA_shrink_minus_M1_shrink','all_accuracy'),:].loc[['C','D','E','F']]
    en+=['',f'The mean shrinkage-minus-full-covariance contrast ranged from {shrink["mean"].min():+.2f} to {shrink["mean"].max():+.2f} percentage points across the six settings, with positive mean contrasts in {int((shrink["mean"]>0).sum())}/6 settings. Across the four cross-source settings, PCA64-shrinkage minus M1-shrinkage ranged from {pca["mean"].min():+.2f} to {pca["mean"].max():+.2f} points. These controls qualify any claim that the neural representation or a single covariance strategy is uniformly preferable.']
    if pca_cross.min()>0:en+=['The neural representation therefore did not show a mean accuracy advantage over the fixed PCA64-shrinkage control in any of the four cross-source settings.']
    en+=['',
        'Classifier and source-prior effects are reported separately from reference-bank effects; matched-target 1-NN comparisons isolate the encoder-source contrast without covariance priors. PCA64-shrinkage is included as a fixed non-neural control, with its dimensionality and relative regularization differences disclosed. Novel classes were registered without encoder updates, but unchanged old-class statistics did not guarantee unchanged recognition when new classes entered the decision set. The paired tables and class-specific confusion matrices retain negative transfer and retention outcomes rather than selecting a favorable direction.','',
        'These findings are exploratory. Record-component blocking removes known exact duplicates and linked IDs but does not establish physical specimen independence; C4/C8 are not independent instruments, and no genuinely untouched confirmatory dataset was available. Repeated episodes were not treated as independent specimens. Legacy C4-support-to-C8-query results use different checkpoints and splits and were retained separately from target-support registration.','']
    if config['smoke']:en[2:2]=['IMPLEMENTATION SMOKE ONLY. Not manuscript evidence.','']
    (out/'manuscript_results_detailed_en.md').write_text('\n'.join(en),encoding='utf-8')
    brief=['# Results','',
        f'Six prespecified settings were evaluated using {len(groups)} conservative paired record components, {len(config["tasks"])} class tasks and {len(config["seeds"])} seeds. C4- and C8-trained M1 encoders used matched {config["epochs"]}-epoch cross-entropy budgets, with evaluation-query components excluded from training and checkpoint selection. Primary results average 5/10-shot conditions within each task/seed.', '',
        f'With source-domain base references, full-covariance M1 achieved {table.loc["C","M1_full"]:.2f}% accuracy for C4-to-C8 and {table.loc["D","M1_full"]:.2f}% for C8-to-C4 transfer; shrinkage achieved {table.loc["C","M1_shrinkage"]:.2f}% and {table.loc["D","M1_shrinkage"]:.2f}%. Thus, direction rankings depended on the classifier. Target-domain base references increased shrinkage accuracy to {table.loc["E","M1_shrinkage"]:.2f}% and {table.loc["F","M1_shrinkage"]:.2f}%, respectively, but required 20 labelled target base spectra per class. In the prior-free 1-NN control, mean absolute reference-bank and matched-target encoder-source effects were {bank_magnitude:.2f} and {source_magnitude:.2f} percentage points, supporting a larger conditional reference-bank effect.', '',
        f'Shrinkage outperformed full covariance with within-domain or target-aligned base banks but not mixed-source banks. PCA64-shrinkage exceeded M1-shrinkage by {pca_cross.min():.2f}–{pca_cross.max():.2f} points across the four cross-source settings; the neural representation therefore showed no mean accuracy advantage over this fixed control. Novel classes were registered without encoder updates, although old-class recognition could decline through decision competition.', '',
        'These results remain exploratory: record blocking does not verify specimen independence, datasets were previously inspected, and C4/C8 are not independent instruments. Repeated episodes were not independent specimens. Native-grid baseline processing and unequal relative regularisation of PCA64 versus M1-128 also limit causal interpretation.','']
    if config['smoke']:brief[2:2]=['IMPLEMENTATION SMOKE ONLY. Not manuscript evidence.','']
    (out/'manuscript_results_en.md').write_text('\n'.join(brief),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/bidirectional_c4_c8_final')
    with threadpool_limits(limits=1):main(parser.parse_args().output.resolve())
