# D4 input adapter V1 audit

Status: exploratory partial-spectrum transfer executed. This new explicit mean-fill authorization supersedes the previous experiment’s no-padding blocker; previous blocked artifacts remain historical and unchanged.

Locked grid: 1868 ascending coordinates; 1761 observed, 107 missing (104 low-end, 3 high-end; 5.728% of coordinates). Original 399.1927–4000.6045 grid and M1 weights are unchanged. D4 native 599.78354–3995.98555 range is never extrapolated.

Per trial, missing inputs are fixed per-coordinate means fitted only on that checkpoint’s archived C4 base_train input rows. The same mean and explicit boolean mask apply to all D4 support/query rows and the controlled C4 mask. Each trial saves input_adapter.npz with mean, mask, grid and fit indices; adapted D4 inputs and frozen features are saved separately.

Preprocessing is native max(y)-y → native airPLS → native maxabs normalization → cubic interpolation inside the native observed range → missing-coordinate mean insertion. No normalization follows mean insertion. This uses exactly the C4 numerical orientation, explicitly flip=True; the old D4 registry default flip=False remains unchanged. This choice was locked before D4 query scoring. It is not a verified physical transmittance/absorbance conversion. No log transform, query-selected polarity, new mask channel, altered weights or target-population statistics are introduced.

D4 contains negative numerical values and row maximum 100, without signal-unit metadata. cm^-1 interpretation follows the FTIR-scale headers but lacks explicit acquisition metadata. Physical signal compatibility, specimen grouping and cross-dataset independence are unresolved. Numeric input compatibility is established; physical measurement equivalence is not.

Masking controlled C4 happens AFTER its original full-range preprocessing, retaining its original observed coordinates and using unchanged original C4 supports/heads. D4 baseline fitting instead sees its native truncated measured interval. The C4 control isolates coordinate deletion/imputation, not baseline-boundary differences. Mean imputation is a numerical input convention, not recovery of measured chemical peaks.

Spectrum 1-NN/SVM within D4 and mixed C4→D4 use only the 1761 observed common coordinates for both support and query. PCA/M1 consume the full locked grid with the fixed imputation. C8 paired six-class spectrum references use the full common grid. The two-class PP/PS anchor spectrum references use the D4-valid window in every dataset condition. Archived PCA64 and shared covariance are fitted only on original legal C4 base_train and never updated for D4. SVM C=1 is fixed without target tuning.

Unique D4 counts:

| class | unique_count |
| --- | --- |
| ETHYLENE PROPYLENE RUBBER | 49 |
| POLY(ETHYLENE) + FOULING | 148 |
| POLY(ETHYLENE) LIKE | 25 |
| PP | 243 |
| PA | 21 |
| MORPHOTYPE 2 | 10 |
| CELLULOSE ACETATE | 70 |
| PEVA | 43 |
| PS | 63 |
| PE | 228 |
| MORPHOTYPE 1 | 54 |

Fixed core9 excludes PA (21) and MORPHOTYPE 2 (10), and is evaluated at 1/5/10/20 shots with five disjoint queries/class. Expanded eligible panel includes all 11 classes at 1/5 shots and 10 classes at 10 shots; at 20 shots it is identical to core9 and not duplicated. The same seed’s saved support/query partitions and episode choices are shared across all three encoder tasks. Morphotype labels are retained as unresolved categories, not asserted identified polymers.

D4 PP/PS are textually matched to C4 PP/PS only. Generic PE is not mapped to HDPE/LDPE; fouling/like/PEVA/rubber and morphotypes are separate labels. The incremental protocol starts from two C4 PP/PS banks (20 exemplars/class), then adds D4 PE and PEVA, using only the six checkpoints trained with PP/PS in base. The other three are excluded from this base protocol; all nine participate in within-D4 and C8 tests.

Exact duplicates: five removed, 954 retained, eleven labels. C4/C8 have 2201 shared retained record-ID strings and zero label conflicts; these are resolution-family pairings, not verified independent specimens. Paired C8 query IDs must be disjoint from C4 base_train and retained support IDs. C4 validation queries were previously used for checkpoint selection, so C8 base-class outcomes remain exploratory. C4/D4 have no shared retained IDs/native-axis hashes; different axes make zero native hashes non-exclusionary. No independent-specimen claim is made.

Protocol audit passed for 9 checkpoints and 28800 original C4 metric contexts. Frozen encoder has 47,400 parameters, zero training updates, equal before/after file and weight hashes.
