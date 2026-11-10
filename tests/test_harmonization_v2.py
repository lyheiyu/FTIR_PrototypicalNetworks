import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ftir_harmonization_v2 import (parse_ftir_csv, clean_axis, convert_signal,
                                   harmonize_spectrum, reconstruction_metrics)
from ftir_research_data import Spectra
from run_harmonization_v2 import run


class HarmonizationTests(unittest.TestCase):
    def test_explicit_vertical_schema_and_units(self):
        with tempfile.TemporaryDirectory(prefix='ftir_csv_v2_') as temp:
            path = Path(temp) / 'vertical.csv'
            path.write_text('wavelength,A,B\n10,1,2\n8,2,3\n6,3,4\n4,4,5\n')
            records = parse_ftir_csv(path, {'layout': 'vertical', 'signal_type': 'absorbance',
                'axis_unit': 'um', 'wave_col': 0, 'intensity_cols': [1, 2]})
            self.assertEqual(len(records), 2)
            np.testing.assert_allclose(records[0].wavenumber_cm1, [1000, 1250, 1e4 / 6, 2500])
            with self.assertRaises(ValueError):
                parse_ftir_csv(path, None)

    def test_wide_schema_keeps_real_axis_and_labels(self):
        with tempfile.TemporaryDirectory(prefix='ftir_wide_v2_') as temp:
            path = Path(temp) / 'wide.csv'
            path.write_text('label,1300,1200,1100,1000\nPP,4,3,2,1\n')
            record = parse_ftir_csv(path, {'layout': 'wide', 'signal_type': 'absorbance',
                'axis_unit': 'cm-1', 'spectral_start_col': 1, 'label_col': 0})[0]
            np.testing.assert_array_equal(record.wavenumber_cm1, [1000, 1100, 1200, 1300])
            np.testing.assert_array_equal(record.intensity, [1, 2, 3, 4])
            self.assertEqual(record.metadata['label'], 'PP')

    def test_duplicate_axis_is_explicit_and_invalid_values_fail(self):
        with self.assertRaises(ValueError):
            clean_axis([100, 200, 200, 300, 400], [1, 2, 4, 6, 8])
        wave, y = clean_axis([100, 200, 200, 300, 400], [1, 2, 4, 6, 8], duplicates='mean')
        np.testing.assert_allclose(y, [1, 3, 6, 8])
        with self.assertRaises(ValueError):
            clean_axis([100, 200, 300, 400], [1, np.nan, 3, 4])

    def test_signal_conversion_never_silently_clips(self):
        a, flags = convert_signal([100., 10., 1.], 'transmittance_percent')
        np.testing.assert_allclose(a, [0, 1, 2])
        with self.assertRaises(ValueError):
            convert_signal([110, 100], 'transmittance_percent')
        a, flags = convert_signal([110, 100], 'transmittance_percent', upper_policy='flag')
        self.assertLess(a[0], 0)
        self.assertTrue(flags)
        with self.assertRaises(ValueError):
            convert_signal([0, 20], 'transmittance_percent')

    def test_cubic_matches_legacy_and_no_method_extrapolates(self):
        wave = np.linspace(700, 3900, 60)
        y = np.sin(wave / 100)
        grid = np.linspace(700, 3900, 25)
        np.testing.assert_allclose(harmonize_spectrum(wave, y, grid, 'cubic'), interp1d(wave, y, kind='cubic')(grid))
        for method in ['linear', 'cubic', 'pchip', 'bin_mean']:
            with self.assertRaises(ValueError):
                harmonize_spectrum(wave, y, [699, 3900], method)

    def test_bin_mean_preserves_integral_on_its_explicit_bins(self):
        wave = np.linspace(100, 200, 101); y = wave**2
        grid = np.linspace(100, 200, 11)
        result = harmonize_spectrum(wave, y, grid, 'bin_mean')
        edges = np.r_[grid[0], (grid[1:] + grid[:-1]) / 2, grid[-1]]
        self.assertAlmostEqual(float(np.sum(result * np.diff(edges))), float(np.trapz(y, wave)), places=7)
        with self.assertRaises(ValueError):
            harmonize_spectrum(wave, y, np.linspace(100, 200, 1000), 'bin_mean')

    def test_reconstruction_holds_out_actual_observations(self):
        wave = np.linspace(100, 500, 40); y = 2 * wave + 1
        result = reconstruction_metrics(wave, y, 'linear')
        self.assertLess(result['rmse'], 1e-10)
        self.assertEqual(reconstruction_metrics(wave, y, 'bin_mean')['status'], 'not_point_interpolation')

    def test_development_run_excludes_novel_and_reuses_episode_ids(self):
        rng = np.random.default_rng(5)
        labels = np.repeat(np.array(['A', 'B', 'FUTURE']), 30)
        wave = np.linspace(700, 3900, 64)
        x = rng.normal(size=(90, 64)) + np.repeat([0., 1., 2.], 30)[:, None]
        data = Spectra('synthetic', wave, x, labels, np.arange(90).astype(str),
                       np.arange(90).astype(str), {'source_sha256': 'synthetic', 'default_flip': False})
        with tempfile.TemporaryDirectory(prefix='ftir_harmonization_test_') as temp:
            args = SimpleNamespace(dataset='c4', output=Path(temp) / 'run', base_classes=['A', 'B'],
                signal='legacy', transmittance_upper='reject', groups=None,
                methods=['linear', 'cubic', 'pchip', 'bin_mean'], grid_factors=[2, 4],
                seeds=[0], shots=[1, 5], queries=2, episodes=2, val_episodes=1,
                c_grid=[1.], fidelity_spectra=2, no_baseline=True)
            with patch('run_harmonization_v2.load_dataset', return_value=data):
                run(args)
            out = args.output
            index = pd.read_csv(out / 'sample_index.csv')
            self.assertEqual(set(index['class']), {'A', 'B'})
            episodes = json.loads((out / 'episodes.json').read_text())
            for episode in episodes:
                self.assertFalse(set(episode['support']) & set(episode['query']))
            a = pd.read_csv(out / 'linear_factor_2_seed0_5shot_predictions.csv')
            b = pd.read_csv(out / 'pchip_factor_2_seed0_5shot_predictions.csv')
            np.testing.assert_array_equal(a['index'], b['index'])
            self.assertEqual(len(pd.read_csv(out / 'development_metrics.csv')), 9 * 2 * 2)
            self.assertTrue((out / 'method_grid_development.pdf').exists())
            self.assertTrue((out / 'svm_tuning.csv').exists())
            fidelity = pd.read_csv(out / 'spectral_fidelity.csv')
            self.assertLess(fidelity.loc[fidelity.candidate.str.startswith('bin_mean'), 'area_relative_error'].max(), 1e-8)
            self.assertTrue((out / 'COMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main()
