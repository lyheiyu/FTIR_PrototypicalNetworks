# Figure captions and manuscript placement

All figures use existing data/results only. No encoder, PCA or classifier was fitted during figure preparation.

## Fig01_dataset_audit: Dataset structure and duplicate handling

Original records, within-file exact deduplication and conservative cross-resolution record grouping. All original CSVs were preserved. Group counts are not verified specimen counts. One representative per group and resolution was used in the final protocol. Exact duplicate removal uses numeric equality, without rounding or a similarity threshold.

Suggested placement: Materials and methods 2.1, after the dataset description.

## Fig02_polymer_spectra: Preprocessed spectra on the common physical grid

Per-polymer median spectra with 10th–90th percentile bands across the paired record representatives, shown for descriptive purposes only. Display summaries do not fit preprocessing or classifiers and use all analysis records. Spectral direction is conventional high-to-low wavenumber. Stored absorbance/transmittance semantics were not verified; the ordinate denotes the numeric preprocessed signal.

Suggested placement: Materials and methods 2.2.

## Fig03_frozen_incremental_workflow: Frozen representation and 4+1+1 protocol

Representation fitting and source-only validation precede class registration. Base reference groups are sampled from legal base-training groups, not from a mutually disjoint extra support split. Novel support/query pools and all evaluation queries remain group-disjoint from representation training/selection. The first novel class is HDPE, PP or PET for the three tasks; the second is LDPE, PS or PVC, respectively.

Suggested placement: Methods 2.3–2.6.

## Fig04_conventional_baselines: Conventional and class-statistical baselines

Final-session 5/10-shot means for spectrum 1-NN/SVM, frozen PCA128 and EP-M1 with shrinkage. Error bars are descriptive SD across nine task–seed means, not standard errors or confidence intervals. The seeds/tasks share data; repeated episodes are not independent specimens. The ordinate starts at 55%. SVM is refitted on the permitted retained reference bank.

Suggested placement: Results 4.1, following Table 2.

## Fig05_frozen_representations: Representations under the fixed shrinkage head

PCA64/PCA128, CE-M1-128 and EP-M1-128 with the same numerical shrinkage rule. Final-session primary 5/10-shot results; error bars are descriptive SD over nine dependent task–seed means. Axes start at 60%. Equal dimensions or a common numeric ridge do not equalise feature scales or relative regularisation.

Suggested placement: Results 4.2 / Table 3; PCA64 control also supports Results 4.6.

## Fig06_episodic_training_ablation: Episodic versus CE training at a fixed distance head

Paired differences in final-session overall and combined novel-class accuracy. Each unit averages 200 paired repeats and the primary 5/10 shots. Points show the mean, error bars descriptive SD across nine dependent task–seed units. The training contrast also changes sampling roles and validation objectives; it is not an isolated pure-loss causal experiment.

Suggested placement: Results 4.2, after the negative finding about Euclidean novel recognition.

## Fig07_EP_distance_head_ablation: Distance-head ablation on the same frozen EP encoder

Only the inference head changes: squared Euclidean, independent class full covariance with ridge .001, or class-specific shrinkage with tau5 and the fixed legal source prior. Overall accuracy and novel accuracy can move in opposite directions, notably in C/D. Error bars show descriptive task–seed SD.

Suggested placement: Results 4.3 / Table 4.

## Fig08_shot_sensitivity: Recognition versus novel support size

All prespecified novel support sizes, 1/5/10/20, using nested support prefixes and the same query draws. Curves are means over task–seed units; their connecting lines aid reading and are not fitted performance models. No support size was selected after inspection.

Suggested placement: Results 4.3–4.4; use as supplementary if main text space is limited.

## Fig09_incremental_retention: Base and novel recognition across incremental sessions

EP-M1 with shrinkage, primary 5/10-shot averages. The dashed first-novel curve follows the same first introduced class from session1 to session2, even though its stored state is unchanged. The combined novel curve averages one class in session1 and two classes in session2. No novel accuracy is defined in session0. Curves describe decision competition rather than encoder-update forgetting.

Suggested placement: Results 4.4 / Table 5.

## Fig10_reference_alignment: Effect of target-resolution base references

Paired source-bank to target-bank transitions C→E and D→F. Thin lines show nine task–seed pairs and bold lines their means. Representation parameters and the source covariance prior are fixed. Target-aligned banks require 80 labelled target base references, a different data-access condition from C/D.

Suggested placement: Results 4.5 / Table 6.

## Fig11_CM_holdout_hdpe_ldpe: HDPE / LDPE held out: cross-resolution confusion

Fixed 10-shot final session, three-seed pooled confusion, shown without selecting a best seed. Rows are true classes and columns predictions; each row sums to 100%. C/D retain source base references. Compare EP full/shrinkage and the dimension-matched PCA128 shrinkage reference. Asterisks mark genuine held-out classes. All six settings including target-aligned E/F are provided in the complete confusion supplement.

Suggested placement: Results 4.3–4.4; one page may move to supplementary material.

## Fig12_CM_holdout_pp_ps: PP / PS held out: cross-resolution confusion

Fixed 10-shot final session, three-seed pooled confusion, shown without selecting a best seed. Rows are true classes and columns predictions; each row sums to 100%. C/D retain source base references. Compare EP full/shrinkage and the dimension-matched PCA128 shrinkage reference. Asterisks mark genuine held-out classes. All six settings including target-aligned E/F are provided in the complete confusion supplement.

Suggested placement: Results 4.3–4.4; one page may move to supplementary material.

## FigS01_M1_architecture: Exact M1 encoder architecture

The locked multiscale M1 definition in encoder_architecture_models.py. The three convolutional branches preserve physical position; ordered spectral-bin pooling precedes the 128-dimensional embedding. CE uses an additional temporary four-class head (516 parameters) only during initial supervised training.

Suggested placement: Methods 2.4 and implementation appendix.

## FigS02_AF_data_access: Six settings and data-access conditions

The representation and covariance prior are source-trained and frozen in every setting. E/F use additional labelled target-resolution base references; gains over C/D are not free or label-free adaptation. Training uses all legal source base-training groups, whereas classifier reference banks use 20 spectra per base class and K per novel class.

Suggested placement: Experimental design 3.1; visual companion to Table 1.

## FigS03_group_safe_partitions: Role counts under the three held-out class tasks

Counts shown for seed0; class-wise role counts are identical across seeds although group membership changes. Base reference spectra are selected from base_train, so they are not an additional mutually disjoint split. Novel support/query pools remain separate; only K support groups and 15 query groups per seen class enter a given episode. C4/C8 counterparts inherit the same group role.

Suggested placement: Methods 2.1/2.3 and data-audit supplement.

## FigS04_feature_geometry: Feature scale and effective regularisation

Source base-training embeddings only. Mean feature norm, effective rank of the pooled residual covariance and numeric ridge divided by mean residual variance. The last panel uses a logarithmic ordinate. A fixed ridge value does not impose equal relative regularisation on CE/EP representations; effective rank alone does not establish better novel-class recognition.

Suggested placement: Discussion 5.2–5.3.

## FigS05_recorded_compute_costs: Recorded representation and class-update costs

Descriptive timings from the archived runs, not a controlled speed benchmark. Initial neural training includes validation/checkpoint work, whereas PCA time covers fitting only. Incremental head times exclude encoding/preprocessing: PCA registers only the arriving class; SVM refits the entire retained labelled bank. Different runs and operation scopes prohibit interpreting the ratios as intrinsic algorithmic speedups.

Suggested placement: Results 4.6; supplementary cost context only.

## FigS06_database_file_schema: Database file schema and evaluation manifests

The actual CSV layout includes a separate physical-wavenumber metadata row. Duplicate mapping relates original rows to retained records; paired group indices address the loaded deduplicated arrays. Split/episode manifests express legal roles and matched draws. A CSV row index must not be treated as a cross-resolution specimen identity.

Suggested placement: Data-availability and reproducibility appendix; companion to Fig01.

## CE_shrinkage_10shot: HDPE / LDPE held out: CE-M1 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_euclidean_10shot: HDPE / LDPE held out: EP-M1 + Euclidean, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_full_10shot: HDPE / LDPE held out: EP-M1 + full Mahalanobis, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_shrinkage_10shot: HDPE / LDPE held out: EP-M1 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA128_shrinkage_10shot: HDPE / LDPE held out: PCA128 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA64_shrinkage_10shot: HDPE / LDPE held out: PCA64 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## CE_shrinkage_5shot: HDPE / LDPE held out: CE-M1 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_euclidean_5shot: HDPE / LDPE held out: EP-M1 + Euclidean, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_full_5shot: HDPE / LDPE held out: EP-M1 + full Mahalanobis, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_shrinkage_5shot: HDPE / LDPE held out: EP-M1 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA128_shrinkage_5shot: HDPE / LDPE held out: PCA128 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA64_shrinkage_5shot: HDPE / LDPE held out: PCA64 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_hdpe_ldpe; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## CE_shrinkage_10shot: PET / PVC held out: CE-M1 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_euclidean_10shot: PET / PVC held out: EP-M1 + Euclidean, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_full_10shot: PET / PVC held out: EP-M1 + full Mahalanobis, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_shrinkage_10shot: PET / PVC held out: EP-M1 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA128_shrinkage_10shot: PET / PVC held out: PCA128 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA64_shrinkage_10shot: PET / PVC held out: PCA64 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## CE_shrinkage_5shot: PET / PVC held out: CE-M1 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_euclidean_5shot: PET / PVC held out: EP-M1 + Euclidean, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_full_5shot: PET / PVC held out: EP-M1 + full Mahalanobis, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_shrinkage_5shot: PET / PVC held out: EP-M1 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA128_shrinkage_5shot: PET / PVC held out: PCA128 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA64_shrinkage_5shot: PET / PVC held out: PCA64 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pet_pvc; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## CE_shrinkage_10shot: PP / PS held out: CE-M1 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_euclidean_10shot: PP / PS held out: EP-M1 + Euclidean, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_full_10shot: PP / PS held out: EP-M1 + full Mahalanobis, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_shrinkage_10shot: PP / PS held out: EP-M1 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA128_shrinkage_10shot: PP / PS held out: PCA128 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA64_shrinkage_10shot: PP / PS held out: PCA64 + shrinkage, 10-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## CE_shrinkage_5shot: PP / PS held out: CE-M1 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_euclidean_5shot: PP / PS held out: EP-M1 + Euclidean, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_full_5shot: PP / PS held out: EP-M1 + full Mahalanobis, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## EP_shrinkage_5shot: PP / PS held out: EP-M1 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA128_shrinkage_5shot: PP / PS held out: PCA128 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

## PCA64_shrinkage_5shot: PP / PS held out: PCA64 + shrinkage, 5-shot

Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within holdout_pp_ps; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.

Suggested placement: Detailed confusion supplement for Results 4.3–4.5.

