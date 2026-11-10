"""Audit D4 before any external-transfer or C4 diagnostic experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ftir_research_data import ROOT, load_dataset


def audit(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'dataset' / 'D4_4_publication11.csv'
    with path.open(encoding='latin-1', newline='') as handle:
        rows = list(csv.reader(handle))
    header, records = rows[0], rows[1:]
    axis = np.asarray(header[2:-1], float)
    values = pd.DataFrame([r[2:-1] for r in records]).apply(pd.to_numeric, errors='coerce').to_numpy()
    d4, c4 = load_dataset('d4'), load_dataset('c4')
    grid_path = ROOT / 'results' / 'encoder_architecture_v1' / 'common_grid.npy'
    grid = np.load(grid_path)
    raw_ids = pd.Series([r[0] for r in records])
    counts = pd.DataFrame({'raw_count': pd.Series(d4.report['original_counts']),
                           'exact_unique_count': pd.Series(d4.report['unique_counts'])}).fillna(0).astype(int)
    counts.to_csv(output / 'd4_class_counts.csv')
    pd.DataFrame(d4.report['row_audit']).to_csv(output / 'd4_row_audit.csv', index=False)
    shared_ids = sorted(set(d4.ids).intersection(c4.ids))
    result = {'d4_source_sha256': d4.report['source_sha256'], 'c4_source_sha256': c4.report['source_sha256'],
        'raw_spectra': len(records), 'unique_spectra': len(d4.x), 'classes': len(counts),
        'spectral_points': len(axis), 'native_min': float(axis.min()), 'native_max': float(axis.max()),
        'median_axis_interval': float(np.median(np.abs(np.diff(axis)))),
        'nonfinite_spectral_cells': int((~np.isfinite(values)).sum()),
        'duplicate_spectra_removed': len(records) - len(d4.x),
        'duplicate_identifier_occurrences': int(raw_ids.duplicated().sum()),
        'negative_spectral_cells': int((values < 0).sum()),
        'signal_min': float(np.nanmin(values)), 'signal_max': float(np.nanmax(values)),
        'grid_min': float(grid.min()), 'grid_max': float(grid.max()), 'grid_points': len(grid),
        'grid_sha256': hashlib.sha256(grid.astype('<f8').tobytes()).hexdigest(),
        'grid_points_below_d4': int((grid < axis.min()).sum()),
        'grid_points_above_d4': int((grid > axis.max()).sum()),
        'signal_semantics': 'unverified normalized numerical signal; every row maximum is 100; no physical unit metadata',
        'axis_units': 'numeric headers consistent with FTIR cm^-1, but file has no explicit unit declaration',
        'specimen_identity': 'unverified; repeated IDs and name patterns are not a validated specimen grouping',
        'cross_dataset_shared_row_ids': shared_ids,
        'cross_dataset_native_axis_equal': bool(np.array_equal(c4.wave, d4.wave)),
        'cross_dataset_native_hash_intersection': len(set(c4.hashes).intersection(d4.hashes)),
        'cross_dataset_duplicate_test_interpretation': 'Unequal native axes/lengths make zero exact native-hash matches non-exclusionary; physical overlap unresolved.',
        'label_correspondence': {'PP': 'Poly(propylene) -> polypropylene label match',
                                 'PS': 'Poly(styrene) -> polystyrene label match',
                                 'PE': 'Do not map unspecific polyethylene to HDPE or LDPE'},
        'transfer_status': 'blocked',
        'blockers': ['107 locked input coordinates outside measured D4 range; extrapolation forbidden',
                     'physical signal representation and cross-dataset specimen independence unverified'],
        'external_evaluations_performed': 0}
    assert result['grid_points_below_d4'] + result['grid_points_above_d4'] > 0
    (output / 'd4_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    count_table = '\n'.join(['| Canonical label | Raw rows | Unique rows |', '| --- | --- | --- |'] +
        [f'| {name} | {row.raw_count} | {row.exact_unique_count} |' for name, row in counts.iterrows()])
    text = f'''# D4 dataset audit — frozen-M1 transfer blocked

Audited the CSV itself before experiments. Source SHA256: `{d4.report['source_sha256']}`.

## Structure and labels

The file has {len(records)} spectrum rows, {len(header)} columns, and one header row. Each sample is a row, not a column. `Nom ` is a sample/record name; `Interpretation ` is the textual interpretation. The next {len(axis)} numeric header columns are the spectral axis. `polymerID` is an integer category code (1–11), matching the textual categories; it is not an independent specimen ID. All rows have the same width. Metadata columns are excluded from spectra.

There are {len(counts)} classes and {len(d4.x)} distinct numerical spectra after removal of five exact duplicate records. The five repeated spectra also repeat their record identifiers. No cross-label exact duplicate conflict was found by the existing adapter.

{count_table}

## Axis and physical signal

Native axis: descending {axis.max():.5f} to {axis.min():.5f}, {len(axis)} coordinates, median absolute interval {np.median(np.abs(np.diff(axis))):.5f}; small interval variations reflect decimal export rounding. Numeric scale is consistent with conventional FTIR wavenumbers in cm^-1, but the file does not explicitly declare units. There is no supplied instrument, acquisition, or physical-signal metadata; the local `dataset/data.md` only contains `#dataset`.

There are no nonfinite spectral cells or missing-width rows. Numerical values range from {np.nanmin(values):.8f} to {np.nanmax(values):.8f}; {(values < 0).sum()} cells are negative and every spectrum has maximum 100. These facts support describing a normalized numerical signal. They do not establish absorbance or physical transmittance. In particular, negative values preclude blindly applying a physical transmittance logarithm. Neither the filename nor maximum 100 establishes percent transmittance. The existing adapter's default no-flip flag is a software convention, not physical provenance.

## Specimen and cross-dataset identity audit

Exact duplicate rows and record IDs were audited; the retained adapter yields unique record IDs. Names contain recurring prefixes and suffix patterns that may encode collection/sample/measurement information, but no metadata verifies which substrings identify physical specimens. No independent-specimen claim is justified.

C4 and D4 share {len(shared_ids)} exact retained row-ID strings and zero exact native spectrum+axis hashes. Their native vectors have different lengths ({len(c4.wave)} vs {len(d4.wave)}) and axes; zero native hashes does not rule out resampling, renormalization, repeated measurements or common specimens. No verified cross-file specimen association was supplied. Therefore cross-dataset independence and transformed duplicates remain unresolved. No speculative grouping or signal conversion was imposed to manufacture an independent test.

## Label correspondence with C4

`Poly(propylene)` corresponds by label semantics to C4 PP; `Poly(styrene)` to C4 PS. This is a textual polymer-name match, not a proof of identical material grade or acquisition conditions. D4 `Poly(ethylene)` is unspecific PE and cannot be assigned to either HDPE or LDPE. Keep `Poly(ethylene) + fouling`, `Poly(ethylene) like`, PEVA, ethylene propylene rubber, cellulose acetate, PA and the two morphotype labels separate. D4 has no verified PET/PVC label counterpart here.

## Locked input compatibility and decision

The saved M1 input grid contains {len(grid)} coordinates spanning {grid.min():.4f}–{grid.max():.4f}. D4 covers only {axis.min():.5f}–{axis.max():.5f}. **{(grid < axis.min()).sum()} coordinates lie below D4 and {(grid > axis.max()).sum()} above it: 107 coordinates are unmeasured.** Cubic interpolation inside D4 cannot construct those inputs without extrapolation, padding, or changing the locked input protocol.

**Stop both within-D4 frozen-M1 adaptation and C4-to-D4 transfer.** No external accuracy is computed. The encoder, grid, crop, legacy baseline/normalization order and checkpoints remain unchanged. Do not pad, extrapolate, silently reverse the signal convention, or retrain an encoder to bypass this blocker. Signal semantics are additionally unresolved, so even resolving coverage alone would not establish physical compatibility.

Count feasibility, if a future compatible independent dataset is provided: the smallest retained class has 10 spectra, so an all-11-class 10/20-shot task cannot also reserve a disjoint query per class. This is a dataset audit observation, not an executed adaptation experiment.

C4 baseline and prototype diagnostics can continue because their original input protocol is valid. `d4_transfer_metrics.csv` records explicit blocked statuses with empty accuracy/F1 fields; those rows are not measured performance.
'''
    (output / 'd4_dataset_audit.md').write_text(text, encoding='utf-8')
    pd.DataFrame([{'experiment': name, 'status': 'blocked_input_coverage', 'accuracy': np.nan, 'macro_f1': np.nan,
                   'reason': '107 locked coordinates outside measured range; signal semantics/specimen identity unresolved',
                   'evaluations_performed': 0} for name in ['within_d4_adaptation', 'c4_to_d4_transfer']]).to_csv(
        output / 'd4_transfer_metrics.csv', index=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    audit(p.parse_args().output)
