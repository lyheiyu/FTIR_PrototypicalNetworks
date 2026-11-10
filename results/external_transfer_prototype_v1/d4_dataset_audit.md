# D4 dataset audit — frozen-M1 transfer blocked

Audited the CSV itself before experiments. Source SHA256: `a86a7056b06576066932dfe1ead6cda90db5a7d434c9cb0986ee41d3f29b1314`.

## Structure and labels

The file has 959 spectrum rows, 1765 columns, and one header row. Each sample is a row, not a column. `Nom ` is a sample/record name; `Interpretation ` is the textual interpretation. The next 1762 numeric header columns are the spectral axis. `polymerID` is an integer category code (1–11), matching the textual categories; it is not an independent specimen ID. All rows have the same width. Metadata columns are excluded from spectra.

There are 11 classes and 954 distinct numerical spectra after removal of five exact duplicate records. The five repeated spectra also repeat their record identifiers. No cross-label exact duplicate conflict was found by the existing adapter.

| Canonical label | Raw rows | Unique rows |
| --- | --- | --- |
| ETHYLENE PROPYLENE RUBBER | 49 | 49 |
| POLY(ETHYLENE) + FOULING | 148 | 148 |
| POLY(ETHYLENE) LIKE | 25 | 25 |
| PP | 243 | 243 |
| PA | 21 | 21 |
| MORPHOTYPE 2 | 11 | 10 |
| CELLULOSE ACETATE | 70 | 70 |
| PEVA | 43 | 43 |
| PS | 66 | 63 |
| PE | 228 | 228 |
| MORPHOTYPE 1 | 55 | 54 |

## Axis and physical signal

Native axis: descending 3995.98555 to 599.78354, 1762 coordinates, median absolute interval 1.92856; small interval variations reflect decimal export rounding. Numeric scale is consistent with conventional FTIR wavenumbers in cm^-1, but the file does not explicitly declare units. There is no supplied instrument, acquisition, or physical-signal metadata; the local `dataset/data.md` only contains `#dataset`.

There are no nonfinite spectral cells or missing-width rows. Numerical values range from -60.11553724 to 100.00000000; 112405 cells are negative and every spectrum has maximum 100. These facts support describing a normalized numerical signal. They do not establish absorbance or physical transmittance. In particular, negative values preclude blindly applying a physical transmittance logarithm. Neither the filename nor maximum 100 establishes percent transmittance. The existing adapter's default no-flip flag is a software convention, not physical provenance.

## Specimen and cross-dataset identity audit

Exact duplicate rows and record IDs were audited; the retained adapter yields unique record IDs. Names contain recurring prefixes and suffix patterns that may encode collection/sample/measurement information, but no metadata verifies which substrings identify physical specimens. No independent-specimen claim is justified.

C4 and D4 share 0 exact retained row-ID strings and zero exact native spectrum+axis hashes. Their native vectors have different lengths (3736 vs 1762) and axes; zero native hashes does not rule out resampling, renormalization, repeated measurements or common specimens. No verified cross-file specimen association was supplied. Therefore cross-dataset independence and transformed duplicates remain unresolved. No speculative grouping or signal conversion was imposed to manufacture an independent test.

## Label correspondence with C4

`Poly(propylene)` corresponds by label semantics to C4 PP; `Poly(styrene)` to C4 PS. This is a textual polymer-name match, not a proof of identical material grade or acquisition conditions. D4 `Poly(ethylene)` is unspecific PE and cannot be assigned to either HDPE or LDPE. Keep `Poly(ethylene) + fouling`, `Poly(ethylene) like`, PEVA, ethylene propylene rubber, cellulose acetate, PA and the two morphotype labels separate. D4 has no verified PET/PVC label counterpart here.

## Locked input compatibility and decision

The saved M1 input grid contains 1868 coordinates spanning 399.1927–4000.6045. D4 covers only 599.78354–3995.98555. **104 coordinates lie below D4 and 3 above it: 107 coordinates are unmeasured.** Cubic interpolation inside D4 cannot construct those inputs without extrapolation, padding, or changing the locked input protocol.

**Stop both within-D4 frozen-M1 adaptation and C4-to-D4 transfer.** No external accuracy is computed. The encoder, grid, crop, legacy baseline/normalization order and checkpoints remain unchanged. Do not pad, extrapolate, silently reverse the signal convention, or retrain an encoder to bypass this blocker. Signal semantics are additionally unresolved, so even resolving coverage alone would not establish physical compatibility.

Count feasibility, if a future compatible independent dataset is provided: the smallest retained class has 10 spectra, so an all-11-class 10/20-shot task cannot also reserve a disjoint query per class. This is a dataset audit observation, not an executed adaptation experiment.

C4 baseline and prototype diagnostics can continue because their original input protocol is valid. `d4_transfer_metrics.csv` records explicit blocked statuses with empty accuracy/F1 fields; those rows are not measured performance.
