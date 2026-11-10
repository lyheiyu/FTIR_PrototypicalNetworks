"""Versioned FTIR input primitives; compatible with the project's Python 3.8."""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator

SCHEMA_VERSION = 'ftir-input-v2.1'
METHODS = ['linear', 'cubic', 'pchip', 'bin_mean']


@dataclass
class SpectrumRecord:
    wavenumber_cm1: np.ndarray
    intensity: np.ndarray
    signal_type: str
    source_id: str
    spectrum_id: str
    specimen_id: Optional[str] = None
    device_id: Optional[str] = None
    batch_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def clean_axis(wave, intensity, duplicates='error'):
    wave, intensity = np.asarray(wave, dtype=float), np.asarray(intensity, dtype=float)
    if wave.ndim != 1 or wave.shape != intensity.shape or len(wave) < 4:
        raise ValueError('Expected aligned 1D arrays with at least four points')
    if not np.isfinite(wave).all() or not np.isfinite(intensity).all() or np.any(wave <= 0):
        raise ValueError('Non-finite data or nonpositive wavenumber')
    order = np.argsort(wave)
    wave, intensity = wave[order], intensity[order]
    if np.any(np.diff(wave) == 0):
        if duplicates != 'mean':
            raise ValueError('Duplicate wavenumbers: declare duplicates="mean" to aggregate')
        unique, inverse = np.unique(wave, return_inverse=True)
        intensity = np.bincount(inverse, weights=intensity) / np.bincount(inverse)
        wave = unique
    if len(wave) < 4:
        raise ValueError('Fewer than four unique wavenumbers')
    return wave, intensity


def parse_ftir_csv(path, schema_hint):
    """Explicit schema, never infer physical coordinates from column positions.

    vertical: wave_col, intensity_cols (one record per column).
    wide: wave_row, data_start_row, spectral_start_col, spectral_end_col,
          optional label_col/id_col. Column ends are Python-exclusive.
    Both require signal_type and axis_unit; defaults describe no hidden units.
    """
    path = Path(path)
    schema = dict(schema_hint or {})
    signal, unit = schema.get('signal_type'), schema.get('axis_unit')
    if signal not in ('absorbance', 'transmittance_percent', 'transmittance_fraction', 'legacy'):
        raise ValueError('An explicit signal_type is required')
    if unit not in ('cm-1', 'um', 'nm'):
        raise ValueError('Declare axis_unit as cm-1, um, or nm')
    with path.open(encoding=schema.get('encoding', 'utf-8-sig'), newline='') as handle:
        rows = list(csv.reader(handle, delimiter=schema.get('delimiter', ',')))
    parsed = []
    layout = schema.get('layout')
    if layout == 'vertical':
        start = schema.get('data_start_row', 1)
        wave_col = schema['wave_col']
        for column in schema['intensity_cols']:
            parsed.append((f'{path.stem}:column_{column}', '',
                           [row[wave_col] for row in rows[start:]],
                           [row[column] for row in rows[start:]]))
    elif layout == 'wide':
        first, last = schema['spectral_start_col'], schema.get('spectral_end_col')
        wave = rows[schema.get('wave_row', 0)][first:last]
        for index in range(schema.get('data_start_row', 1), len(rows)):
            row = rows[index]
            identifier = row[schema['id_col']] if 'id_col' in schema else f'csv_row_{index + 1}'
            label = row[schema['label_col']] if 'label_col' in schema else ''
            parsed.append((identifier, label, wave, row[first:last]))
    else:
        raise ValueError('Use explicit layout="vertical" or "wide"')
    result = []
    for identifier, label, wave, values in parsed:
        wave = np.asarray(wave, dtype=float)
        if unit != 'cm-1':
            if np.any(wave <= 0):
                raise ValueError('Wavelength must be positive')
            wave = (1e4 if unit == 'um' else 1e7) / wave
        wave, values = clean_axis(wave, values, schema.get('duplicates', 'error'))
        result.append(SpectrumRecord(wave, values, signal, str(path), str(identifier),
            specimen_id=schema.get('specimen_id'), device_id=schema.get('device_id'), batch_id=schema.get('batch_id'),
            metadata={'label': label, 'schema_version': SCHEMA_VERSION, 'schema': schema}))
    if len({record.spectrum_id for record in result}) != len(result):
        raise ValueError('Spectrum IDs must be unique')
    return result


def convert_signal(values, signal_type, legacy_flip=False, upper_policy='reject'):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite intensity')
    flags = []
    if signal_type == 'legacy':
        return (values.max() - values if legacy_flip else values.copy()), ['legacy_not_physical_absorbance']
    if signal_type == 'absorbance':
        return values.copy(), ['negative_absorbance'] if np.any(values < 0) else []
    if signal_type not in ('transmittance_percent', 'transmittance_fraction'):
        raise ValueError('Unknown signal semantics')
    transmittance = values / 100 if signal_type == 'transmittance_percent' else values
    if np.any(transmittance <= 0):
        raise ValueError('Log conversion requires strictly positive transmittance')
    if np.any(transmittance > 1):
        if upper_policy != 'flag':
            raise ValueError('Transmittance exceeds 100%; explicit upper_policy="flag" required to retain')
        flags.append('transmittance_above_100_retained_with_negative_absorbance')
    return -np.log10(transmittance), flags


def harmonize_spectrum(wave, values, target, method):
    wave, values = clean_axis(wave, values)
    target = np.asarray(target, dtype=float)
    if target.ndim != 1 or len(target) < 1 or not np.isfinite(target).all() or np.any(np.diff(target) <= 0):
        raise ValueError('Target grid must be finite and strictly increasing')
    if target[0] < wave[0] or target[-1] > wave[-1]:
        raise ValueError('Extrapolation is forbidden')
    if method == 'linear':
        output = np.interp(target, wave, values)
    elif method == 'cubic':
        # Preserve legacy scipy.interpolate.interp1d cubic implementation.
        output = interp1d(wave, values, kind='cubic', bounds_error=True, assume_sorted=True)(target)
    elif method == 'pchip':
        output = PchipInterpolator(wave, values, extrapolate=False)(target)
    elif method == 'bin_mean':
        if len(target) < 2:
            raise ValueError('bin_mean requires at least two target bins')
        if np.median(np.diff(target)) < np.median(np.diff(wave)) * (1 - 1e-6):
            raise ValueError('bin_mean is a coarse-grid candidate, not an upsampling method')
        # Averages of the piecewise-linear signal, not an ideal band-limit filter.
        # Endpoint bins have half width so no observations outside coverage are used.
        edges = np.r_[target[0], (target[1:] + target[:-1]) / 2, target[-1]]
        joined = np.unique(np.r_[wave, edges])
        sampled = np.interp(joined, wave, values)
        cumulative = np.r_[0, np.cumsum((sampled[1:] + sampled[:-1]) / 2 * np.diff(joined))]
        integral = np.interp(edges, joined, cumulative)
        output = np.diff(integral) / np.diff(edges)
    else:
        raise ValueError(method)
    if not np.isfinite(output).all():
        raise ValueError('Non-finite harmonized output')
    return output


def reconstruction_metrics(wave, values, method, seed=0):
    """Raw native observations: hold out first, do not fit baseline/normalization."""
    if method == 'bin_mean':
        return {'rmse': np.nan, 'nrmse': np.nan, 'status': 'not_point_interpolation'}
    wave, values = clean_axis(wave, values)
    rng = np.random.default_rng(seed)
    hidden = np.sort(rng.choice(np.arange(1, len(wave) - 1), max(1, len(wave) // 5), replace=False))
    retained = np.ones(len(wave), dtype=bool); retained[hidden] = False
    predicted = harmonize_spectrum(wave[retained], values[retained], wave[hidden], method)
    rmse = float(np.sqrt(np.mean((predicted - values[hidden])**2)))
    scale = np.percentile(values, 95) - np.percentile(values, 5)
    return {'rmse': rmse, 'nrmse': rmse / scale if scale else np.nan,
            'status': 'raw_native_points_no_baseline_or_normalization'}
