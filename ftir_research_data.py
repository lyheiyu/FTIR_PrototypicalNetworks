"""Explicit local FTIR adapters. Source families are conservative variant groups,
not verified instrument/specimen provenance. No automatic PE/HDPE/LDPE merging.
"""
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from PLS import airPLS

ROOT = Path(__file__).resolve().parent

# (filename, layout, variant family, default flip, trusted spectral interval)
REGISTRY = {
    'c4': ('deduplicated/FTIR_PLastics500_c4_deduplicated.csv', 'plastics', 'plastics', True, None),
    'c8': ('deduplicated/FTIR_PLastics500_c8_deduplicated.csv', 'plastics', 'plastics', True, None),
    'plastics_full_c4': ('FTIR_PLASTIC_c4.csv', 'alternating', 'plastics', True, None),
    'plastics_full_c8': ('FTIR_PLASTIC_c8.csv', 'alternating', 'plastics', True, None),
    'd4': ('D4_4_publication11.csv', 'd4', 'd4', False, None),
    'second': ('new_SecondDataset2.csv', 'second', 'second', True, None),
    'jung': ('216_2018_1156_MOESM2_ESM modified.csv', 'jung', 'jung', False, None),
    'jung_v2': ('216_2018_1156_MOESM2_ESM modified2.csv', 'jung', 'jung', False, None),
    'fourth_original': ('FourthDataset2.csv', 'fourth', 'fourth', True, None),
    'fourth_v3': ('FourthDataset3.csv', 'fourth', 'fourth', True, (675, 3999)),
    'fourth_v4': ('FourthDataset4.csv', 'fourth', 'fourth', True, (675, 3999)),
    'flopp_r': ('FourthdatasetFollp-r3.csv', 'fourth', 'fourth', True, None),
    'flopp_r_padded': ('FourthdatasetFollp-r.csv', 'fourth', 'fourth', True, (675, 3999)),
    'flopp_r_v2': ('FourthdatasetFollp-r2.csv', 'fourth', 'fourth', True, (675, 3999)),
    'flopp_e': ('FourthdatasetFoppl-e.csv', 'fourth', 'fourth', True, (675, 3999)),
}

ALIASES = {
    'POLY(PROPYLENE)': 'PP', 'POLYPROPYLENE': 'PP',
    'POLY(STYRENE)': 'PS', 'POLYSTYRENE': 'PS',
    'POLY(ETHYLENE)': 'PE', 'POLYETHYLENE': 'PE',
    'POLYETHYLENE TEREPHTHALATE': 'PET', 'POLYVINYLCHLORIDE': 'PVC',
    'POLYVINYL CHLORIDE': 'PVC', 'POLYCARBONATE': 'PC',
    'POLY(AMIDE)': 'PA', 'POLYAMIDE': 'PA', 'POLYURETHANE': 'PU',
}


def canonical_label(value, extra=None):
    value = ' '.join(value.strip().upper().split())
    return (extra or {}).get(value, ALIASES.get(value, value))


@dataclass
class Spectra:
    key: str
    wave: np.ndarray
    x: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    hashes: np.ndarray
    report: dict


def source_rows(path, layout):
    """Yield (row identifier, original label, wave values, spectral values)."""
    with path.open(encoding='latin-1', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if layout == 'plastics':
            wave = next(reader)[3:]
        for line, row in enumerate(reader, start=3 if layout == 'plastics' else 2):
            if len(row) != len(header):
                yield str(line), '', [], []
                continue
            if layout == 'plastics':
                yield row[0], row[1], wave, row[3:]
            elif layout == 'alternating':
                yield row[0], row[1], row[6::2], row[7::2]
            elif layout == 'd4':
                yield row[0], row[1], header[2:-1], row[2:-1]
            elif layout == 'second':
                yield f'csv_row_{line}', row[0], header[1:-1], row[1:-1]
            elif layout == 'jung':
                yield row[0], row[-2], header[1:-2], row[1:-2]
            elif layout == 'fourth':
                yield f'csv_row_{line}', row[-2], header[:-2], row[:-2]
            else:
                raise ValueError(layout)


def load_dataset(key, aliases=None):
    filename, layout, family, flip, valid_range = REGISTRY[key]
    path = ROOT / 'dataset' / filename
    kept, identifiers, labels, hashes, audit = [], [], [], [], []
    seen, wave, original_counts = {}, None, Counter()
    for row_id, original, w, values in source_rows(path, layout):
        label = canonical_label(original, aliases)
        original_counts[label or '<empty>'] += 1
        reason = ''
        try:
            current_wave = np.asarray(w, dtype=float)
            spectrum = np.asarray(values, dtype=float)
            if not label or len(spectrum) < 4 or not np.isfinite(spectrum).all():
                raise ValueError('empty label, short spectrum or non-finite values')
            order = np.argsort(current_wave)
            current_wave, spectrum = current_wave[order], spectrum[order]
            if valid_range:
                mask = (current_wave >= valid_range[0]) & (current_wave <= valid_range[1])
                current_wave, spectrum = current_wave[mask], spectrum[mask]
            if len(current_wave) < 4 or not np.isfinite(current_wave).all() or np.any(np.diff(current_wave) <= 0):
                raise ValueError('invalid wave axis')
            if wave is not None and not np.array_equal(wave, current_wave):
                raise ValueError('row-specific axis differs from first valid spectrum')
        except (ValueError, IndexError) as error:
            audit.append({'row_id': row_id, 'label': label, 'status': 'invalid', 'detail': str(error)})
            continue
        if wave is None:
            wave = current_wave
        spectrum[spectrum == 0] = 0  # canonicalize signed floating zero
        fingerprint = hashlib.sha256(wave.astype('<f8').tobytes() + spectrum.astype('<f8').tobytes()).hexdigest()
        if fingerprint in seen:
            first_id, first_label = seen[fingerprint]
            if first_label != label:
                raise ValueError(f'{key}: identical spectrum has conflicting labels {first_label}/{label}')
            reason = first_id
        else:
            seen[fingerprint] = (row_id, label)
            kept.append(spectrum)
            labels.append(label)
            identifiers.append(row_id)
            hashes.append(fingerprint)
        audit.append({'row_id': row_id, 'label': label,
                      'status': 'duplicate' if reason else 'retained', 'detail': reason})
    if not kept:
        raise ValueError(f'{key}: no valid spectra')
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f'{key}: source row identifiers are not unique')
    report = {'key': key, 'path': str(path), 'family': family,
              'family_is_verified_specimen_identity': False,
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'default_flip': flip, 'trusted_range': valid_range,
              'wave_min': float(wave.min()), 'wave_max': float(wave.max()), 'points': len(wave),
              'original_counts': dict(original_counts), 'unique_counts': dict(Counter(labels)),
              'statuses': dict(Counter(row['status'] for row in audit)), 'row_audit': audit}
    return Spectra(key, wave, np.asarray(kept), np.asarray(labels),
                   np.asarray(identifiers), np.asarray(hashes), report)


def preprocess(data, grid, cache_dir, baseline=True, flip=None):
    """Row-local preprocessing; no class, PCA, or target aggregate statistics."""
    flip = data.report['default_flip'] if flip is None else flip
    signature = json.dumps({'source': data.report['source_sha256'], 'labels': data.labels.tolist(),
                            'ids': data.ids.tolist(), 'grid': grid.tolist(), 'baseline': baseline,
                            'flip': flip, 'range': data.report['trusted_range'], 'version': 1}, sort_keys=True)
    digest = hashlib.sha256(signature.encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'{data.key}_{digest}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return saved['x']
    if grid.min() < data.wave.min() or grid.max() > data.wave.max():
        raise ValueError(f'{data.key}: grid exceeds measured range; extrapolation is forbidden')
    x = np.empty((len(data.x), len(grid)), dtype=np.float32)
    for i, row in enumerate(data.x):
        row = row.max() - row if flip else row.copy()
        if baseline and np.ptp(row) > 0:
            row -= airPLS(row)
        scale = np.max(np.abs(row))
        row = row / scale if scale else row
        x[i] = interp1d(data.wave, row, kind='cubic', bounds_error=True)(grid)
        if (i + 1) % 250 == 0:
            print(f'{data.key}: preprocessed {i + 1}/{len(x)}', flush=True)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite preprocessing result')
    np.savez_compressed(cache, x=x, signature=signature)
    return x


def inventory(output, keys):
    import pandas as pd
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for key in keys:
        print(f'Inspecting {key}', flush=True)
        data = load_dataset(key)
        report = data.report.copy()
        pd.DataFrame(report.pop('row_audit')).to_csv(output / f'{key}_rows.csv', index=False)
        (output / f'{key}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        for name, count in report['unique_counts'].items():
            records.append({'dataset': key, 'family': report['family'], 'class': name,
                            'unique_spectra': count, 'wave_min': report['wave_min'],
                            'wave_max': report['wave_max'], 'points': report['points']})
    pd.DataFrame(records).to_csv(output / 'dataset_classes.csv', index=False)
    return records
