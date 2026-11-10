# Final episodic ProtoNet ablation

Run in the project root with the existing `fewshotlearning` environment. The runner does not import or execute legacy top-level training scripts. The final CE experiment in `results/bidirectional_c4_c8_final/` must be complete. No old checkpoints/results are changed.

```powershell
$ftirPython = 'C:\Users\31734\anaconda3\envs\fewshotlearning\python.exe'
& $ftirPython -c "import runpy; d=runpy.run_path('tests/test_protonet_final.py'); [(print(k,flush=True),v()) for k,v in d.items() if k.startswith('test_')]; print('3 tests passed')"
& $ftirPython run_protonet_bidirectional_final.py --smoke --output results/protonet_bidirectional_smoke
& $ftirPython analyze_protonet_bidirectional_final.py --output results/protonet_bidirectional_smoke
& $ftirPython run_protonet_bidirectional_final.py --validated-smoke results/protonet_bidirectional_smoke
& $ftirPython audit_protonet_final.py
& $ftirPython analyze_protonet_bidirectional_final.py
```

The smoke uses two epochs and two of the original evaluation episodes for one paired C4/C8 task/seed. It validates execution, not scientific performance. The full run uses all 18 new episodic encoders, 60 epochs and 200 original evaluation episodes. Source hashes must match the smoke before full training. Completed model/trial markers permit resuming with the same command/configuration. Do not modify implementation during a run or reuse a directory for a different configuration.

Each update uses four legal base classes, five support and ten query groups per class. The objective is query CE of negative squared Euclidean prototype distances. There is no classification head, embedding normalization change, additional loss, augmentation or hyperparameter search. Adam .001, 780/960/840 optimizer updates by task, equal to the corresponding CE 60-epoch reference. Support/query draws are identical between C4/C8 within each task/seed; source-validation episodes are also paired and fixed. The same encoder initialization is verified against saved CE provenance. There are 20 fixed four-way five-support/four-query validation episodes; five+four fits the smallest PS validation pool of nine groups. Minimum mean source-validation query CE selects the checkpoint. This matches update counts and initial weights, not the CE sampling schedule, effective label count or validation objective.

The runner computes CE Euclidean from existing frozen CE embeddings and evaluates three heads on each new episodic encoder. The analyzer reuses saved CE full/shrinkage and PCA64 shrinkage metrics, yielding a 2-training-objective x 3-head factorial comparison plus PCA64. All six settings, all four shots, and all three sessions are retained. Primary reporting averages five/ten shots and repeats within each task/seed before averaging nine paired units. No independent-specimen or confirmatory inference is made.

Outputs are in `results/protonet_bidirectional_final/`:

- `summary_zh.md`, `manuscript_results_en.md`: principal interpretation and manuscript text.
- `metrics.csv`, `summary_by_shot_session.csv`, `task_seed_metrics.csv`: overall and subgroup Accuracy/Macro-F1, base/new decision errors and retention.
- `paired_differences.csv`, `paired_summary.csv`: objective effects at fixed head and head effects at fixed objective.
- `source_target_transfer_gaps.csv`, `transfer_objective_interactions.csv`: source-to-target degradation and changes with training objective.
- `per_class_metrics.csv`, `key_polymer_confusions.csv`, `confusion_counts.csv`, `confusion_matrices/`: all-class recall and confusion outputs. Confusion percentages pool repeat counts and are not independent observations.
- `training_comparison.csv`, `costs.csv`, `feature_geometry.csv`, `feature_geometry.md`: update/parameter/timing and feature-scale context. Head times exclude preprocessing/encoding; no old-class refit occurs.
- `experiment_manifest.json`, `protocol_audit.json`, `implementation_audit.md`, per-model histories/checkpoints and per-trial episode manifests: provenance and audit evidence.
- `sampling_audit.json`, `sampling_provenance.csv`: independently rechecked saved episode indices/local labels, source-only validation checkpoint choice, and CE/checkpoint/source hashes.

This is the final addition. Disappointing outcomes do not authorize further architecture/loss/augmentation variants or post-hoc tuning.
