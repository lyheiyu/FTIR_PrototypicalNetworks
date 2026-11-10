"""Summarize paired class-covariance ablation without changing previous results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/mahalanobis_full_ablation_v1'
PARENT = ROOT/'results/external_transfer_prototype_v1'
NAMES = {'M1_full_class_ridge001':'Class full (ridge .001)',
         'M1_full_class_legacy002':'Class full (legacy .002)',
         'M1_proto_shared':'Shared full', 'M1_proto_shrinkage':'Class shrinkage',
         'M1_proto_euclidean':'Euclidean', 'M1_1nn':'1-NN', 'M1_svm':'SVM'}


def markdown(frame):
    frame = frame.reset_index()
    columns = [str(c) for c in frame.columns]
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    assert (OUT/'COMPLETE.json').exists()
    data = pd.read_csv(OUT/'metrics.csv')
    parent = pd.read_csv(PARENT/'metrics.csv')
    refs = parent[parent.method.isin(['M1_proto_euclidean','M1_1nn','M1_svm'])]
    data = pd.concat([data,refs],ignore_index=True)
    fields = ['all_accuracy','all_macro_f1','base_accuracy','base_macro_f1',
              'previous_novel_accuracy','previous_novel_macro_f1','current_novel_accuracy',
              'current_novel_macro_f1','seen_novel_accuracy','seen_novel_macro_f1']
    trial = data.groupby(['task','seed','shot','session','method'])[fields].mean().reset_index()
    trial.to_csv(OUT/'trial_metrics.csv',index=False)
    summary = trial.groupby(['shot','session','method'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    final = summary[summary.session==2]
    table = 100*final.pivot(index='shot',columns='method',values='all_accuracy')
    print(table.round(3).to_string())
    primary = trial[(trial.session==2)&trial.shot.isin([5,10])].groupby(['task','seed','method'])[fields].mean()
    comparisons = []
    pairs = [('M1_proto_shrinkage','M1_full_class_ridge001'),
             ('M1_proto_shrinkage','M1_full_class_legacy002'),
             ('M1_full_class_ridge001','M1_proto_shared'),
             ('M1_full_class_legacy002','M1_proto_shared'),
             ('M1_full_class_ridge001','M1_proto_euclidean'),
             ('M1_full_class_legacy002','M1_1nn')]
    for a,b in pairs:
        delta = 100*(primary.xs(a,level='method')-primary.xs(b,level='method'))
        for (task,seed),row in delta.iterrows():
            comparisons.append(dict(task=task,seed=seed,a=a,b=b,**row.to_dict()))
    paired = pd.DataFrame(comparisons)
    paired.to_csv(OUT/'paired_primary_5_10shot_pp.csv',index=False)
    paired_summary = paired.groupby(['a','b'])[fields].mean()
    paired_summary['positive_all_accuracy_trials'] = paired.groupby(['a','b']).all_accuracy.apply(lambda x:int((x>1e-9).sum()))
    paired_summary.to_csv(OUT/'paired_summary.csv')
    print(paired_summary[['all_accuracy','all_macro_f1','seen_novel_accuracy','positive_all_accuracy_trials']].round(4).to_string())
    cost = pd.read_csv(OUT/'costs.csv')
    costs = cost[cost.session>0].groupby(['shot','method']).mean(numeric_only=True)
    costs.to_csv(OUT/'cost_summary.csv')
    pcs = pd.read_csv(OUT/'per_class_metrics.csv')
    recall = pcs[pcs.session==2].groupby(['shot','method','class']).recall.mean()
    recall.to_csv(OUT/'final_per_class_recall.csv')
    figdir = OUT/'figures'
    figdir.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    for method,label in NAMES.items():
        rows = final[final.method==method].sort_values('shot')
        for ax,metric in zip(axes,['all_accuracy','seen_novel_accuracy']):
            ax.plot(rows.shot,100*rows[metric],marker='o',label=label)
            ax.set_xticks([1,5,10,20]); ax.set_xlabel('Novel support / class'); ax.set_ylabel('Accuracy (%)'); ax.grid(alpha=.2)
    axes[0].set_title('Final session: all six classes'); axes[1].set_title('Final session: both novel classes')
    axes[1].legend(fontsize=8,loc='best')
    fig.tight_layout()
    for suffix in ['png','pdf']:
        fig.savefig(figdir/f'full_mahalanobis_comparison.{suffix}',dpi=180)
    plt.close(fig)
    config = json.loads((OUT/'experiment_manifest.json').read_text())
    for task in config['tasks']:
        name = task['name']
        classes = sorted(task['base']+sum(task['novel_sessions'],[]))
        fig,axes = plt.subplots(2,4,figsize=(18,9))
        for i,k in enumerate([5,10]):
            for j,method in enumerate(config['methods']):
                key = f'{method}_{k}shot_session2'
                cm = sum(np.load(OUT/name/f'seed_{seed}/confusions.npz')[key] for seed in config['seeds'])
                assert cm.shape==(6,6)
                pd.DataFrame(cm,index=classes,columns=classes).to_csv(figdir/f'{name}_{method}_{k}shot_CM.csv')
                normalized = cm/cm.sum(axis=1,keepdims=True)
                ax = axes[i,j]
                ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
                for r in range(6):
                    for c in range(6):
                        ax.text(c,r,f'{100*normalized[r,c]:.0f}',ha='center',va='center',fontsize=8,color='white' if normalized[r,c]>.5 else 'black')
                ax.set_xticks(range(6)); ax.set_yticks(range(6));ax.set_xticklabels(classes,rotation=45);ax.set_yticklabels(classes)
                ax.set_title(f'{NAMES[method]}, {k}-shot',fontsize=10)
                ax.set_xlabel('Predicted');ax.set_ylabel('True')
        fig.tight_layout()
        for suffix in ['png','pdf']:
            fig.savefig(figdir/f'{name}_final_CM.{suffix}',dpi=160)
        plt.close(fig)
    final_f1 = 100*final.pivot(index='shot',columns='method',values='all_macro_f1')
    shrink_delta = paired_summary.loc[('M1_proto_shrinkage','M1_full_class_legacy002'),:]
    full_nn_delta = paired_summary.loc[('M1_full_class_legacy002','M1_1nn'),:]
    report = ['# 完整类协方差 Mahalanobis 补充对照 V1', '',
        '已完成 C4 三任务 × 三种子 × 四种 shot × 200 个配对 episode，包含全部三个增量阶段。M1 encoder 保持原来的 47,400 参数并完全冻结，本次训练步骤为零。', '',
        f'本轮结论：在预定的 5/10-shot 主要比较中，收缩相对原公式完整类协方差的总体 Accuracy 提高 {shrink_delta.all_accuracy:.3f} 个百分点，新类 Accuracy 提高 {shrink_delta.seen_novel_accuracy:.3f} 个百分点，九组配对总体 Accuracy 改善 {int(shrink_delta.positive_all_accuracy_trials)}/9。完整类协方差随 shot 增加明显改善，但其 5/10-shot 平均总体 Accuracy 相对 1-NN 仍差 {abs(full_nn_delta.all_accuracy):.3f} 个百分点。因此支持进一步研究收缩方法，尚不能声称原始完整类协方差在少样本条件下全面优于强基线。', '',
        '## 先区分两个问题', '',
        '原来的 shared 和 shrinkage 都使用完整 128×128 矩阵。本次补充的是每类独立估计、没有向 base prior 收缩的完整协方差对照。', '',
        '你的原始方法使用每类均值和每类协方差。它是否优于欧氏原型/共享协方差，与后加的收缩估计是否进一步改善它，是两个独立问题。之前缺少这一项，不能充分归因于收缩改进；也不能因此否定原始方法。', '',
        '## 最终六类 Accuracy（%）', '',
        markdown(table.rename(columns=NAMES).round(3)), '',
        '## 最终六类 Macro-F1（%）', '',
        markdown(final_f1.rename(columns=NAMES).round(3)), '',
        '## 主要配对比较：5/10-shot', '',
        '先对每个任务/种子内的 episode 与 5/10-shot 求均值，再对九组配对值汇总。下面的差值单位为百分点。正值表示前者更好。', '',
        markdown(paired_summary[['all_accuracy','all_macro_f1','seen_novel_accuracy','positive_all_accuracy_trials']].round(4)), '',
        '## 如何解释', '']
    for a,b in pairs:
        row = paired_summary.loc[(a,b),:]
        report.append(f'- {NAMES[a]} 对比 {NAMES[b]}：总体 Accuracy {row.all_accuracy:+.3f}，Macro-F1 {row.all_macro_f1:+.3f}，已加入新类 Accuracy {row.seen_novel_accuracy:+.3f}；总体 Accuracy 改善 {int(row.positive_all_accuracy_trials)}/9 组。')
    report += ['', '不能把某个 shot 的均值优势解释为所有任务、种子、shot 都获胜；逐组结果保存在 paired_primary_5_10shot_pp.csv。', '',
        '## 公式和可比性', '',
        '原代码估计协方差时加 0.001I，计算距离时又加 0.001I，因此原公式实际是 C+0.002I。本次同时保留 C+0.001I 的同正则对照。复现的是原始距离公式，使用当前冻结 M1 的 float64 缓存特征；不是旧 float32 Torch 训练流程的逐位复现。省略平方根不改变最近类判决。', '',
        '128 维下协方差秩最多为 n−1，所有 1/5/10/20-shot 类协方差均不可直接求逆。完整类协方差这里有明确 ridge，未用伪逆冒充无正则逆。', '',
        '所有方法使用相同 support/query。完整类协方差只使用该类 support；共享/收缩还使用已存档 base_train 的类内协方差 prior。因此结果体现 prior 与收缩策略的联合价值，不能声称额外 base 数据访问已被完全控制。', '',
        '## 注册、推理与存储', '',
        markdown(costs.loc[5][['head_update_ms','cached_query_prediction_ms','total_numeric_state_bytes']].round(4)), '',
        '5-shot 的成本表平均两个新增阶段。注册只计算新类；查询推理使用缓存特征。数字不包含 encoder 推理、预处理、Python 开销，也不是端到端延迟。不同阶段类别数不同，表内存是两个阶段均值。最终六类完整类协方差头为 792,624 bytes；收缩头为 923,696 bytes，多存一份 base prior。两者仍需六个完整逆协方差矩阵。', '',
        '本次完整类协方差头与收缩头都在注册时缓存逆矩阵；原 Torch 脚本是在查询/类别循环中反复求逆。因此这里测的是高效实现下各距离公式的成本，不能当作原始脚本的运行耗时。', '',
        '## 图与实验边界', '',
        '- figures/full_mahalanobis_comparison.png/pdf：各 shot 总体与新类 Accuracy。',
        '- figures/*_final_CM.png/pdf：三个任务各自的 5/10-shot 混淆矩阵，六类分别展示；同时保存计数 CSV。',
        '- final_per_class_recall.csv：按 shot、方法、类别的最终召回率。',
        '- summary.csv 与 trial_metrics.csv：各阶段全部分组指标。', '',
        '这是已反复查看过的 C4 探索性结果。base query 复用先前验证集，实物重复测量分组未知。不能据此声称独立数据泛化、确认性显著性或所有 FTIR 数据上的优越性。原 checkpoint、历史结果和预处理均未修改。', '']
    (OUT/'conclusion.md').write_text('\n'.join(report),encoding='utf-8')


if __name__=='__main__':
    main()
