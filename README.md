# FTIR Prototypical Networks

**English** | [简体中文](README.zh-CN.md)

Research code for few-shot FTIR polymer classification, frozen feature representations, and incremental registration of new classes. The repository compares spectral preprocessing, encoder architectures, Euclidean and Mahalanobis prototype heads, and transfer between the C4/C8 datasets.

> Start with the final experiment reports below. Historical scripts and development experiments are retained for context; they use different protocols and should not be combined as one benchmark.

## Read the results

| Question | Report |
| --- | --- |
| How do episodic ProtoNet and supervised CE training compare? | [Final ProtoNet summary](results/protonet_bidirectional_final/summary_zh.md) · [English manuscript results](results/protonet_bidirectional_final/manuscript_results_en.md) |
| What happens in bidirectional C4/C8 transfer? | [Final transfer results](results/bidirectional_c4_c8_final/final_results_zh.md) |
| Does covariance shrinkage improve full Mahalanobis classification? | [Full covariance ablation](results/mahalanobis_full_ablation_v1/conclusion.md) |
| Do Linear, Cubic and PCHIP interpolation differ? | [Interpolation and grid comparison](results/harmonization_v2_dev_c4/RESULTS_CONCLUSIONS.md) |
| How do encoder architectures compare? | [Architecture comparison](results/encoder_architecture_v1/architecture_comparison.md) |
| Can new classes be registered with a frozen encoder? | [Prototype study](results/external_transfer_prototype_v1/results_overview_zh.md) · [Incremental study](results/frozen_incremental_v1/results_overview_zh.md) |
| How does the method behave on heterogeneous FTIR inputs? | [Heterogeneous transfer](results/heterogeneous_ftir_transfer_v1/results_overview_zh.md) |

The interpolation development study found very small classification differences at a fixed grid. Cubic had lower held-out-point reconstruction error, while Linear had lower interpolation cost. In the final ProtoNet study, shrinkage and full covariance exhibit a base/novel tradeoff: better novel recall does not necessarily mean better overall incremental accuracy. See the linked reports for scope and numerical results.

## Method overview

```text
FTIR spectrum → preprocessing / common physical grid → feature representation
                                                        ↓
                                  support samples → class prototype + metric
                                                        ↓
                                    query → minimum-distance class
```

Representations include preprocessed spectra, PCA, and learned encoders. In frozen incremental evaluation, a new class is registered from its support samples without encoder updates or refitting existing prototype classes.

For the shrinkage head, class c stores its support mean μc and a full precision matrix:

```text
Sc     = Σi (zi − μc)(zi − μc)ᵀ / nc
αc     = (nc − 1) / (nc − 1 + τ)
Σ̃c     = αc Sc + (1 − αc) Σbase + λI
d²c(z) = (z − μc)ᵀ Σ̃c⁻¹ (z − μc)
```

The final comparison uses τ = 5 and λ = 0.001. The prior Σbase is pooled within-class residual covariance fitted only on source base-training features. Full Mahalanobis uses Sc + λI; shared Mahalanobis uses the same Σbase + λI for every class. The final episodic encoder is trained with Euclidean prototype distances; these Mahalanobis heads are applied after training.

## File guide

### Shared implementation

| File / directory | Purpose |
| --- | --- |
| [ftir_research_data.py](ftir_research_data.py) | Dataset registry, labels, spectral loading and preprocessing utilities. |
| [ftir_research_core.py](ftir_research_core.py) | Covariance prior, reference classifiers, head tuning and evaluation. |
| [frozen_prototype_heads_v1.py](frozen_prototype_heads_v1.py) | Class-local Euclidean, cosine, shared and shrinkage prototype heads. |
| [run_full_mahalanobis_ablation_v1.py](run_full_mahalanobis_ablation_v1.py) | Full-class covariance head and matched ablation runner. |
| [encoder_architecture_models.py](encoder_architecture_models.py) | M0/M1 encoder definitions and spectral bin pooling. |
| [protonet_training_final.py](protonet_training_final.py) | Episodic encoder training for the final ProtoNet comparison. |
| [bidirectional_protocol_final.py](bidirectional_protocol_final.py) | Shared protocol for paired C4/C8 experiments. |
| [ftir_harmonization_v2.py](ftir_harmonization_v2.py) | Explicit spectral parsing, axis handling, signal conversion and resampling. |
| [heterogeneous_ftir_adapter_v1.py](heterogeneous_ftir_adapter_v1.py) | Adapter for heterogeneous spectral coverage. |
| [deduplicate_plastics500.py](deduplicate_plastics500.py) | Preparation of deduplicated Plastics500 inputs. |
| [PLS.py](PLS.py), [utils.py](utils.py) | Baseline correction and shared metrics/plotting utilities. |
| [research_configs/](research_configs/) | Research task configuration. |
| [tests/](tests/) | Checks for data handling, metric heads and experimental protocols. |

### Experiment entry points

| Runner | Purpose / instructions |
| --- | --- |
| `fewshot_benchmark.py` | Earlier within-dataset paired few-shot benchmark. [Guide](FEWSHOT_RUN.md) |
| `run_ftir_research.py` | Research diagnostics and earlier incremental evaluation. [Guide](FTIR_RESEARCH_RUN.md) |
| `run_harmonization_v2.py` | Interpolation and target-grid development study. [Guide](HARMONIZATION_V2_RUN.md) |
| `run_encoder_architecture_v1.py` | Matched encoder architecture comparison. [Guide](ENCODER_ARCHITECTURE_V1_RUN.md) |
| `run_frozen_incremental_v1.py` | Frozen-feature incremental classification and update costs. [Guide](FROZEN_INCREMENTAL_V1_RUN.md) |
| `run_external_prototype_v1.py` | Prototype registration and reference baselines. [Guide](EXTERNAL_PROTOTYPE_V1_RUN.md) |
| `run_full_mahalanobis_ablation_v1.py` | Full covariance versus shared/shrinkage heads. [Guide](FULL_MAHALANOBIS_ABLATION_V1_RUN.md) |
| `run_heterogeneous_ftir_transfer_v1.py` | Heterogeneous-data transfer. [Guide](HETEROGENEOUS_FTIR_TRANSFER_V1_RUN.md) |
| `run_bidirectional_c4_c8_final.py` | Final paired C4/C8 supervised-encoder experiment. [Guide](BIDIRECTIONAL_C4_C8_FINAL_RUN.md) |
| `run_protonet_bidirectional_final.py` | Final episodic ProtoNet comparison against CE references. [Guide](PROTONET_BIDIRECTIONAL_FINAL_RUN.md) |
| `run_pca128_shrinkage_bidirectional_v1.py` | Supplementary PCA128 + shrinkage comparison. |
| `supplement_heterogeneous_full_mahalanobis_v1.py` | Full covariance control for heterogeneous transfer. |

`analyze_*.py` scripts summarize saved experiment artifacts. `audit_*.py` scripts check their corresponding protocols/provenance. `plot_fewshot_benchmark.py` produces benchmark plots; `build_ftir_manuscript_figures_v1.py` assembles manuscript figures. They may require local artifacts omitted from GitHub.

### Manuscript and historical files

[paper draft/](paper%20draft/) contains manuscript drafts, research plans and the figure collection. These are working documents, not a published paper. [GITHUB_SYNC.md](GITHUB_SYNC.md) describes which artifacts are synchronized.

The root also retains earlier `FTIR_*.py` scripts for ProtoNet, Mahalanobis distance, MAML, Siamese networks and augmentation, including `FTIR_AugmentationBasedOnEMSA.py` and `FTIR_AugmentationBasedOnPCA_EMSA.py`. Other standalone examples and app/demo files are historical utilities. Some legacy scripts execute work at import time or contain local paths; use the documented runners above for the current experiments.

## Environment and first checks

Use a Python environment compatible with TensorFlow 2.13. The research runners primarily require TensorFlow/Keras, NumPy, pandas, SciPy, scikit-learn, Matplotlib and threadpoolctl. `requirements.txt` is a historical mixed-project dependency list, not a minimal lockfile for the final experiments; it also includes unrelated application/CAD/PyTorch dependencies.

From the repository root, check your existing environment:

```powershell
python -c "import tensorflow as tf; import numpy, pandas, scipy, sklearn, matplotlib, threadpoolctl; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -m unittest discover -s tests -p test_harmonization_v2.py -v
```

These commands are environment/protocol checks, not full training. For data-backed runs, prepare the local datasets first and follow the experiment-specific guide. For example, the final bidirectional workflow starts with:

```powershell
python run_bidirectional_c4_c8_final.py --smoke --output results/bidirectional_c4_c8_review_smoke
```

Use a new output directory. Read the guide before starting the full run; later experiments depend on earlier saved checkpoints and manifests.

## Data and reproducibility

- Raw datasets are not distributed here. The registry in `ftir_research_data.py` documents expected inputs; C4/C8 deduplicated CSVs reside under local `dataset/deduplicated/`.
- GitHub includes source, run instructions, manuscript drafts, selected reports, summary tables and figures. Weights, feature arrays, detailed episode artifacts and smoke/check outputs stay local.
- A clone is therefore sufficient for reading the methods and published repository reports, but is **not a complete experimental archive**. Some analyzers require those local artifacts.
- Earlier within-dataset tests and later held-out-class incremental protocols answer different questions. K-shot describes the support budget, not all labels used to train the encoder.
- Final results are exploratory. Record/duplicate groups are not verified independent physical specimens, and C4/C8 comparisons do not establish generalization across independent instruments. See each report's protocol limitations.

## Updating GitHub

Review the file list before committing:

```powershell
git status --short
git add --all
git diff --cached --stat
git commit -m "Update research code and reports"
git push origin main
```

Large local artifacts are excluded through `.gitignore`. Selected result/manuscript CSVs use ordinary Git storage; see `.gitattributes`.
