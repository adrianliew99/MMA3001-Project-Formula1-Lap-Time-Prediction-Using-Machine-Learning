"""Offline numerical, leakage and frozen-artifact regression tests."""
import json
from pathlib import Path
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tools.notebook_contract import load_contract
from mma3001_f1.features import QualityConfig, summarize_telemetry, weather_at
from mma3001_f1.performance import repeated_timing, traced_peak_bytes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def api():
    return load_contract(ROOT / 'MMA3001_F1_Next_Lap_Prediction.ipynb')


@pytest.fixture(scope='module')
def clean(api):
    data, report = api['audit_pairs'](api['load_exports'](ROOT / 'f1_dataset'))
    assert report.excluded_rows.sum() == 0
    return data


@pytest.fixture
def toy(api):
    row = dict(Year=2023, Event='Test', Session='Race', Driver='AAA',
               PreviousLapNumber=4, TargetLapNumber=5, Stint=1,
               PreviousLapTime_s=90., NextLapTime_s=91., DeltaLapTime_s=1.,
               TyreAge_laps=4., TrackTemp_C=30., AvgSpeed_kmh=200.,
               MaxSpeed_kmh=300., FullThrottle_pct=50., Braking_pct=20., TyreCompound='SOFT')
    return pd.DataFrame([row])


@pytest.mark.parametrize('column,value,reason', [
    ('TargetLapNumber', 8, 'non_consecutive'),
    ('PreviousLapTime_s', 0., 'invalid_time'),
    ('NextLapTime_s', float('inf'), 'invalid_time'),
    ('Driver', None, 'missing_identifiers'),
    ('Stint', 0, 'invalid_stint'),
    ('TyreAge_laps', -1., 'invalid_tyre_age'),
    ('MaxSpeed_kmh', 401., 'invalid_speed'),
    ('Braking_pct', 101., 'invalid_percentages'),
    ('TrackTemp_C', 81., 'invalid_temperature'),
    ('DeltaLapTime_s', 2., 'wrong_delta_identity'),
])
def test_invalid_pairs_rejected(api, toy, column, value, reason):
    toy.loc[0, column] = value
    accepted, report = api['audit_pairs'](toy)
    assert accepted.empty
    assert report.set_index('reason').loc[reason, 'excluded_rows'] == 1


def test_duplicate_keeps_first_without_modifying_input(api, toy):
    original = toy.copy(deep=True)
    accepted, report = api['audit_pairs'](pd.concat([toy, toy], ignore_index=True))
    assert len(accepted) == 1
    assert report.set_index('reason').loc['duplicate_key', 'excluded_rows'] == 1
    pd.testing.assert_frame_equal(toy, original)


def test_optional_missing_inputs_are_prepared(api, toy):
    toy.loc[0, 'TyreAge_laps'] = np.nan
    toy.loc[0, 'TyreCompound'] = None
    accepted, _ = api['audit_pairs'](toy)
    X = api['prepare_model_inputs'](accepted, api['BASE_NUM'], api['BASE_CAT'])
    assert len(accepted) == 1
    assert X.TyreCompound.iloc[0] == 'Unknown'
    assert X.TyreAge_laps.dtype == np.dtype('float64')


def test_metrics_hand_calculated_and_delta_reconstruction(api, toy):
    second = toy.copy()
    second.NextLapTime_s = 93.
    second.DeltaLapTime_s = 3.
    frame = pd.concat([toy, second], ignore_index=True)
    scores = api['evaluate_split'](frame, np.array([2., 2.]))
    assert scores['MAE_s'] == 1.
    assert scores['RMSE_s'] == 1.
    assert scores['MAE_delta_s'] == 1.
    predicted = api['prediction_frame'](frame, np.array([2., 2.]))
    np.testing.assert_allclose(predicted.PredictedNextLapTime_s, [92., 92.])
    np.testing.assert_allclose(predicted.Residual_s, [-1., 1.])


def test_real_splits_disjoint_chronological_and_manifest(api, clean):
    parts = api['split_by_season'](clean)
    manifest = json.loads((ROOT / 'results/metrics/split_manifest.json').read_text())
    event_groups = []
    for name, part in zip(['train', 'validation', 'test'], parts):
        assert len(part) == manifest[name]['rows']
        assert sorted(part.Year.unique()) == manifest[name]['years']
        event_groups.append(set(zip(part.Year, part.Event, part.Session)))
    assert not any(event_groups[i] & event_groups[j] for i in range(3) for j in range(i + 1, 3))
    assert parts[0].Year.max() < parts[1].Year.min() < parts[2].Year.min()


def test_reject_overlapping_seasons(api, clean):
    function = api['split_by_season']
    old = function.__globals__['VAL_YEARS']
    try:
        function.__globals__['VAL_YEARS'] = (2023,)
        with pytest.raises((AssertionError, ValueError)):
            function(clean)
    finally:
        function.__globals__['VAL_YEARS'] = old


def test_preprocessing_training_only_unknown_categories_and_allowlist(api, toy):
    train = pd.concat([toy] * 3, ignore_index=True)
    train.PreviousLapTime_s = [90., 91., 92.]
    train.DeltaLapTime_s = [1., 0., -1.]
    numeric, categorical = api['FEATURE_SETS']['B_context']
    pipeline = api['make_pipeline'](LinearRegression(), numeric, categorical)
    X = api['prepare_model_inputs'](train, numeric, categorical)
    pipeline.fit(X, train.DeltaLapTime_s)
    scaler = pipeline.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['scale']
    np.testing.assert_allclose(scaler.mean_[0], 91.)
    before = scaler.mean_.copy()
    future = toy.assign(PreviousLapTime_s=150., Event='UNSEEN', TyreCompound='UNSEEN')
    future['NextLapTime_s'] = -999.  # Forbidden target must not enter predictors.
    pred = pipeline.predict(api['prepare_model_inputs'](future, numeric, categorical))
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(before, scaler.mean_)
    forbidden = {'NextLapTime_s', 'DeltaLapTime_s', 'TargetLapNumber', 'Driver', 'Year', 'Stint', 'SourceFile'}
    for numbers, categories in api['FEATURE_SETS'].values():
        assert not forbidden.intersection(numbers + categories)


def test_saved_holdout_matches_frozen_selection_and_all_rows(api, clean):
    prediction = pd.read_csv(ROOT / 'results/predictions/FINAL_holdout_2025_predictions.csv')
    metrics = json.loads((ROOT / 'results/metrics/FINAL_holdout_2025_metrics.json').read_text())
    grid = pd.read_csv(ROOT / 'results/metrics/validation_grid.csv').sort_values(['MAE_s', 'RMSE_s'])
    assert grid.iloc[0]['model'] == metrics['selected_model'][1] == 'Persistence'
    assert len(prediction) == metrics['observations'] == 20662
    assert set(prediction.Year) == {2025}
    held = clean[clean.Year.eq(2025)].reset_index(drop=True)
    for col in api['KEYS'] + ['PreviousLapTime_s', 'NextLapTime_s']:
        pd.testing.assert_series_equal(prediction[col].reset_index(drop=True), held[col], check_dtype=False)
    np.testing.assert_array_equal(prediction.PredictedDeltaLapTime_s, 0.)
    np.testing.assert_allclose(prediction.PredictedNextLapTime_s, prediction.PreviousLapTime_s)
    scores = api['evaluate_split'](held, prediction.PredictedDeltaLapTime_s.to_numpy())
    for key, value in metrics['scores'].items():
        assert scores[key] == pytest.approx(value, abs=1e-12)
    np.testing.assert_allclose(prediction.AbsoluteError_s,
                               abs(prediction.NextLapTime_s - prediction.PredictedNextLapTime_s), atol=1e-12)


def test_saved_validation_persistence(api, clean):
    val = api['split_by_season'](clean)[1]
    scores = api['evaluate_split'](val, np.zeros(len(val)))
    grid = pd.read_csv(ROOT / 'results/metrics/validation_grid.csv')
    saved = grid[grid.model.eq('Persistence')].iloc[0]
    assert scores['MAE_s'] == pytest.approx(saved.MAE_s, abs=1e-12)


def test_grouped_error_reports_account_for_every_prediction():
    prediction = pd.read_csv(ROOT / 'results/predictions/FINAL_holdout_2025_predictions.csv')
    overall = np.mean(abs(prediction.NextLapTime_s - prediction.PredictedNextLapTime_s))
    for filename in ['circuit_errors_2025.csv', 'driver_errors_2025.csv', 'compound_errors_2025.csv', 'race_stage_errors_2025.csv']:
        report = pd.read_csv(ROOT / 'results/metrics/engineering_error_analysis' / filename)
        assert report.Observations.sum() == len(prediction)
        assert np.average(report.MAE_s, weights=report.Observations) == pytest.approx(overall, abs=1e-12)


def test_telemetry_known_integrals_and_gap_rejection():
    telemetry = pd.DataFrame({'SessionTime': pd.to_timedelta([0., .5, 1.], unit='s'),
                              'Speed': [100., 200., 300.], 'Throttle': [100., 0., 100.],
                              'Brake': [False, True, False]})
    output = summarize_telemetry(telemetry, 0., 1., QualityConfig())
    assert output['AvgSpeed_kmh'] == 200.
    assert output['FullThrottle_pct'] == output['Braking_pct'] == 50.
    with pytest.raises(ValueError, match='coverage'):
        summarize_telemetry(telemetry, 0., 2., QualityConfig())
    with pytest.raises(ValueError, match='unordered'):
        summarize_telemetry(telemetry.iloc[[1, 0, 2]], 0., 1., QualityConfig())


def test_weather_does_not_look_ahead_and_rejects_staleness():
    weather = pd.DataFrame({'Time': pd.to_timedelta([120, 0, 60], unit='s'), 'TrackTemp': [99., 20., 30.]})
    assert weather_at(weather, 90., QualityConfig())['TrackTemp_C'] == 30.
    with pytest.raises(ValueError, match='stale'):
        weather_at(weather, 1000., QualityConfig())


def test_timing_runs_callable_and_reports_real_samples():
    calls = []
    timing = repeated_timing(lambda: calls.append(1), repeats=5, warmups=2)
    assert len(calls) == 7
    assert len(timing['samples_s']) == 5
    assert timing['min_s'] <= timing['median_s'] <= timing['p95_s'] <= timing['max_s']
    assert timing['min_s'] > 0
    with pytest.raises(ValueError):
        repeated_timing(lambda: None, repeats=1)


def test_memory_cleanup_on_exception():
    def failure():
        raise ValueError('expected')
    with pytest.raises(ValueError):
        traced_peak_bytes(failure)
    assert not tracemalloc.is_tracing()
    assert traced_peak_bytes(lambda: bytearray(10000)) >= 10000


def test_all_frozen_result_bytes_remain_unchanged():
    import hashlib
    audit = json.loads((ROOT / 'reports/frozen_results_audit.json').read_text())
    for path, expected in audit['protected_files_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
