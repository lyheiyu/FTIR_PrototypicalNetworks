"""Paired frozen-feature diagnostics; blocked D4 results remain explicitly empty."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import utils

OLD = ['raw_svm', 'M1_1nn', 'M1_svm']
BASELINES = ['raw_1nn', 'pca64_1nn', 'M1_1nn', 'M1_svm', 'raw_svm']
PROTOS = ['M1_proto_euclidean', 'M1_proto_cosine', 'M1_proto_shared', 'M1_proto_shrinkage']


def table(frame):
    def fmt(x):
        return f'{x:.3f}' if isinstance(x, (float, np.floating)) else str(x)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
        ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def analyze(output):
    output = Path(output)
    config = json.loads((output / 'experiment_manifest.json').read_text())
    files = sorted(output.glob('*/seed_*/metrics.csv'))
    assert len(files) == len(config['tasks']) * len(config['seeds'])
    frames, costs, classes = [], [], []
    for f in files:
        frame = pd.read_csv(f)
        assert len(frame) == len(config['shots']) * config['repeats'] * 3 * 9
        assert not frame.duplicated(['shot', 'repeat', 'session', 'method']).any()
        frames.append(frame)
        cost = pd.read_csv(f.parent / 'costs.csv')
        assert len(cost) == len(config['shots']) * config['benchmark_repeats'] * 3 * 9
        costs.append(cost)
        cm = pd.read_csv(f.parent / 'per_class_metrics.csv')
        task = next(t for t in config['tasks'] if t['name'] == f.parent.parent.name)
        first, second = task['novel_sessions'][0][0], task['novel_sessions'][1][0]
        cm['class_role'] = np.where(cm['class'].isin(task['base']), 'base',
                                   np.where(cm['class'] == first, 'first_novel', 'second_novel'))
        classes.append(cm)
    data, cost, per_class = pd.concat(frames, ignore_index=True), pd.concat(costs, ignore_index=True), pd.concat(classes, ignore_index=True)
    data.to_csv(output / 'metrics.csv', index=False)
    data[data.method.isin(BASELINES)].to_csv(output / 'c4_baseline_comparison.csv', index=False)
    data[data.method.isin(PROTOS + ['M1_1nn', 'M1_svm'])].to_csv(output / 'prototype_comparison.csv', index=False)
    per_class.to_csv(output / 'per_class_metrics.csv', index=False)
    cost.to_csv(output / 'cost_comparison.csv', index=False)
    metrics = [c for c in data if c.endswith('_accuracy') or c.endswith('_macro_f1')]
    trial = data.groupby(['task', 'seed', 'shot', 'session', 'method'])[metrics].mean().reset_index()
    trial.to_csv(output / 'trial_metrics.csv', index=False)
    summary = trial.groupby(['shot', 'session', 'method'])[metrics].mean().reset_index()
    summary.to_csv(output / 'summary_metrics.csv', index=False)
    csummary = cost[cost.session > 0].groupby(['task', 'seed', 'shot', 'method']).mean(numeric_only=True).reset_index()
    csummary.groupby(['shot', 'method']).mean(numeric_only=True).reset_index().to_csv(output / 'cost_summary.csv', index=False)
    storage = cost[cost.session == 2].groupby(['shot', 'method'])[
        ['stored_feature_bank_bytes', 'class_metric_bytes', 'global_metric_bytes', 'total_numeric_state_bytes',
         'global_transform_bytes', 'total_with_transform_bytes']].mean().reset_index()
    storage.to_csv(output / 'storage_summary.csv', index=False)
    delta_rows = []
    for reference in ['raw_1nn', 'pca64_1nn', 'M1_1nn']:
        a = trial[trial.method == reference].set_index(['task', 'seed', 'shot', 'session'])
        comparisons = ['M1_1nn'] if reference != 'M1_1nn' else PROTOS
        for method in comparisons:
            b = trial[trial.method == method].set_index(['task', 'seed', 'shot', 'session'])
            difference = (b[metrics] - a[metrics]) * 100
            difference = difference.mask(difference.abs() < 1e-10, 0.)
            for index, row in difference.iterrows():
                delta_rows.append({'task': index[0], 'seed': index[1], 'shot': index[2], 'session': index[3],
                                   'method': method, 'reference': reference, **row.to_dict()})
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(output / 'paired_differences_pp.csv', index=False)
    keys = ['task', 'seed', 'shot', 'method']
    first, last = trial[trial.session == 1].set_index(keys), trial[trial.session == 2].set_index(keys)
    retention = pd.DataFrame({'first_novel_accuracy_change_pp': 100*(last.previous_novel_accuracy - first.current_novel_accuracy)})
    retention.to_csv(output / 'retention_deltas.csv')
    final = summary[summary.session == 2]
    primary = delta[(delta.session == 2) & delta.shot.isin([5, 10])]
    paired = primary.groupby(['task', 'seed', 'method', 'reference'])[metrics].mean().reset_index()
    decision = paired.groupby(['method', 'reference']).agg(
        all_accuracy_gain_pp=('all_accuracy', 'mean'), novel_accuracy_gain_pp=('seen_novel_accuracy', 'mean'),
        macro_f1_gain_pp=('all_macro_f1', 'mean'), positive_pairs=('all_accuracy', lambda v: int((v > 1e-9).sum())),
        pairs=('all_accuracy', 'size')).reset_index()
    decision.to_csv(output / 'primary_paired_summary.csv', index=False)
    weak = per_class[(per_class.session == 2) & per_class.shot.isin([5, 10]) & (per_class.class_role != 'base')]
    weak = weak.groupby(['task', 'seed', 'class', 'method']).recall.mean().groupby(['class', 'method']).mean().unstack()
    weak.to_csv(output / 'novel_class_recall_summary.csv')
    report = '# C4 基线、冻结原型与 D4 迁移诊断 V1\n\n'
    report += ('所有编码器保持冻结。本轮补齐光谱/PCA64 的欧氏 1-NN，并用原样的类内支持集构建四种原型分类器；'
        '这些是冻结编码器上的统计分类头，不是新训练的 Prototypical Networks。\n\n'
        '**D4 迁移已按输入兼容性规则停止。** D4 仅覆盖 599.78354–3995.98555，锁定输入为 399.1927–4000.6045，'
        '104 个低端及 3 个高端坐标未测量；没有外推、补零、改网格或重新训练。物理信号和实物身份也未确认。'
        '`d4_transfer_metrics.csv` 只有明确 blocked 状态，空分数不是零准确率。完整依据见 `d4_dataset_audit.md`。\n\n'
        '## 评估条件\n\n'
        '三任务、三种子、1/5/10/20-shot，每个 shot 200 个原配对 episode。基类每类固定 20 条，'
        '新类按原顺序逐个加入 K 条，旧样本保留。全部 support/query 和基类 exemplar 直接读取 '
        '`frozen_incremental_v1/sessions.json`。PCA64 和类内协方差只拟合原 base_train，'
        '在新类到达后不重拟合。SVM 参数沿用父实验，原型 reg=0.001、tau=5，距离和参数均不使用新类 query 选择。\n\n'
        '基类 query 仍是参与 checkpoint/C 选择的 base_validation，C4 新类也已被查看，实物分组未验证。'
        '所以总体与基类指标为探索性验证集复用结果。M1 编码器曾用全部基类训练标签，PCA/协方差有基类统计拟合阶段，'
        '光谱参考没有编码器训练；分类头支持预算一致不等于所有历史学习机会一致。\n\n')
    for k in config['shots']:
        rows = final[final.shot == k][['method', 'base_accuracy', 'previous_novel_accuracy', 'current_novel_accuracy',
                                      'seen_novel_accuracy', 'all_accuracy', 'all_macro_f1']].copy()
        rows.iloc[:, 1:] *= 100
        report += f'## 最终六类 {k}-shot（%）\n\n' + table(rows) + '\n\n'
    report += '## 配对差异：最终 5/10-shot 均值（百分点）\n\n' + table(decision) + '\n\n'
    report += ('均先平均 episode，再等权平均 task/seed；positive_pairs 是该配对平均总准确率增益为正的组数，'
               '不是独立统计显著性检验。旧类与新类 Accuracy、Macro-F1 全部在 `summary_metrics.csv`。\n\n')
    report += '## 存储与注册成本\n\n'
    target_k = 5 if 5 in config['shots'] else config['shots'][0]
    memory = storage[storage.shot == target_k].drop(columns=['shot']).copy()
    report += f'最终 {target_k}-shot，报告 float64 特征/度量数组字节数，变换参数单列：\n\n' + table(memory) + '\n\n'
    ctable = csummary.groupby(['shot', 'method']).mean(numeric_only=True).reset_index()
    ctable = ctable[ctable.shot == target_k][['method', 'new_feature_encoding_ms', 'head_update_ms',
        'new_class_registration_ms', 'svm_full_refit_and_assembly_ms', 'prediction_ms']]
    report += f'{target_k}-shot 每次新类到达的平均毫秒数，所有方法用前 20 个相同 episode 同机重新计时：\n\n' + table(ctable) + '\n\n'
    report += ('原型只计算并保存到达类的均值/度量，旧类状态不改；共享 Mahalanobis 的逆矩阵预先计算，'
        '收缩 Mahalanobis 对到达类单独计算完整精度矩阵。P0/P1 只留均值向量；P2 另有一份共享 128×128 矩阵，'
        'P3 另有每类 128×128 精度矩阵和全局先验，不能只数原型向量来宣称省内存。\n\n'
        '特征库、类度量、全局度量、PCA/CNN 变换参数分开记录；统计不含 Python 对象、标签元数据及分配器开销，'
        '不是进程 RAM 实测。所有 M1 头的共同编码器参数成本相同。PCA 映射也占存储，已列入变换参数。\n\n'
        '注册时间包括新 support 的冻结编码/投影与头更新，排除共享预处理；SVM 整理全库后完整重拟合，'
        '1-NN 采用已有 sklearn 库整理与索引准备。原型不重拟合全局分类器。预测在缓存 query 特征上计时，'
        '不是包含 query 编码的端到端延迟。NumPy/sklearn 数值线程限制为 1，TensorFlow 为 4；'
        '这里重新计时的参考结果与父运行逐 episode 数值核对，不能把父运行的不同线程耗时直接拼入比较。'
        'PCA、先验和原型基类初始化另存 `preparation_costs.json`，未隐入新类成本。\n\n')
    report += '## 哪些新类最难\n\n'
    recall = (weak[['raw_1nn', 'pca64_1nn', 'M1_1nn'] + PROTOS] * 100).reset_index()
    report += table(recall.sort_values('M1_1nn')) + '\n\n'
    report += ('以上仅汇总这些类作为新类时的 5/10-shot recall，避免基类身份掩盖困难。'
        '旧新不平衡可从 base_accuracy 与 seen_novel_accuracy 的差观察；'
        '第一新类在 Session 1→2 的变化见 `retention_deltas.csv`，比较同一类同一份 query。'
        'current_novel 在两个 session 代表不同类别，不能直接用它们的差值当遗忘。\n\n'
        '## 研究问题与下一步\n\n'
        '1. M1 是否超过光谱 1-NN、PCA64 1-NN：看配对表中 M1_1nn 对应两行的总分及新类差值；'
        '两者都必须保留为强基线，不能只与弱 M0 比。\n'
        '2. 原型能否接近 M1 1-NN：看四个原型相对 M1_1nn 的配对差值，并同时检查新类表现，'
        '不能用基类高分遮住新类低分。\n'
        '3. 存储和注册是否改善：均值原型的特征库明显更小；共享/收缩度量的矩阵可能抵消节省，'
        '具体字节和实测计时在上表，不能预设所有原型都更省。\n'
        '4. M1 能否迁移 D4：本轮无法回答，输入不兼容导致实验未执行，不能将 blocked 记为模型迁移失败或成功。\n'
        '5. 是否必须换新 prototype-learning objective：本轮不能证明“必要”。如果现有固定度量已接近邻近方法，'
        '优先验证这种无训练头；若原型明显落后，应先区分均值几何、类别内多模态、范数与度量问题，'
        '再设计仅用基类开发数据的后续对照，不根据当前新类 query 继续调参。\n'
        '6. 下一阶段首先补充覆盖锁定网格、信号语义可核实且实物身份可审计的外部数据。'
        '在独立验证前，不宣称跨数据库确认性泛化，也不依据这一轮直接训练新 ProtoNet/SupCon。\n')
    (output / 'results_overview_zh.md').write_text(report, encoding='utf-8')
    audit = {'paired_trials': len(files), 'metric_rows': len(data), 'matched_timing_rows': len(cost),
        'encoder_training_calls': 0, 'd4_evaluations': 0, 'd4_status': 'blocked',
        'checked_parent_head_scores': len(files) * len(config['shots']) * config['benchmark_repeats'] * 3 * len(OLD),
        'base_test_independent': False, 'specimen_identity_verified': False}
    (output / 'protocol_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    figures = output / 'figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ['all_accuracy', 'all_macro_f1']):
        for method in BASELINES + PROTOS:
            d = final[final.method == method]
            ax.plot(d.shot, 100*d[metric], marker='o', label=method)
        ax.set(xlabel='Shot', ylabel=metric + ' (%)', xticks=config['shots']); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7, bbox_to_anchor=(1, 1)); fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(figures / f'final_quality.{ext}', dpi=230)
    plt.close(fig)
    for task in config['tasks']:
        names = sorted(task['base'] + [c for s in task['novel_sessions'] for c in s])
        fig, axes = plt.subplots(3, 3, figsize=(16, 15))
        for ax, method in zip(axes.flat, BASELINES + PROTOS):
            cm = np.zeros((6, 6), dtype=np.int64)
            for seed in config['seeds']:
                root = Path(config['parent_incremental']) if method in OLD else output
                with np.load(root / task['name'] / f'seed_{seed}' / 'confusions.npz') as z:
                    # Smoke references select only first 2 episodes, not all parent counts.
                    if config['smoke'] and method in OLD:
                        continue
                    cm += z[f'{method}_{target_k}shot_session2']
            if config['smoke'] and method in OLD:
                ax.set_visible(False); continue
            utils.draw_confusion_matrix(ax, cm, names, method)
            pd.DataFrame(cm, index=names, columns=names).to_csv(figures / f'{task["name"]}_{method}_final_CM.csv')
        fig.suptitle(task['name'] + f' | {target_k}-shot | pooled query occurrences; exploratory'); fig.tight_layout()
        for ext in ['png', 'pdf']: fig.savefig(figures / f'{task["name"]}_final_CM.{ext}', dpi=210)
        plt.close(fig)
    print('Analysis complete: ' + str(output), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('output', type=Path)
    analyze(p.parse_args().output)
