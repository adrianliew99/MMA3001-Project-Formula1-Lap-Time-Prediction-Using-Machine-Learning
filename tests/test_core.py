"""Offline behavioral tests using small, deterministic examples."""
import numpy as np
import pandas as pd
import pytest

from mma3001_f1.config import ProjectConfig
from mma3001_f1.data import prepare_laps, audit_laps, seconds
from mma3001_f1.features import QualityConfig, summarize_telemetry, weather_at, feature_columns
from mma3001_f1.ingestion import atomic_json, validate_export, request_identity
from mma3001_f1.modeling import make_examples, split_weekends, score, train_baseline, evaluate


@pytest.fixture
def laps():
    return pd.DataFrame({
        'Driver': ['VER'] * 5, 'LapNumber': range(2, 7),
        'LapTime': pd.to_timedelta([90, 91, 92, 93, 94], unit='s'),
        'PitInTime': pd.Series([pd.NaT] * 5, dtype='timedelta64[ns]'),
        'PitOutTime': pd.Series([pd.NaT] * 5, dtype='timedelta64[ns]'),
        'TrackStatus': ['1'] * 5, 'IsAccurate': [True] * 5,
        'Deleted': [False] * 5, 'FastF1Generated': [False] * 5,
        'Compound': ['MEDIUM'] * 5, 'TyreLife': range(2, 7), 'Stint': [1] * 5,
        'EventDate': ['2024-03-01'] * 5,
    })


@pytest.mark.parametrize('kwargs', [
    {'seasons': (2017,)}, {'seed': -1}, {'max_sessions': 0},
    {'drivers': ('ver',)}, {'raw_dir': '../escape'}, {'raw_dir': 'C:/escape'},
    {'seasons': (2024, 2024)}, {'circuits': ('Bahrain',) * 2},
])
def test_config_rejects_invalid_scope(kwargs):
    with pytest.raises(ValueError):
        ProjectConfig(**kwargs)


def test_config_resolves_without_creating_directories(tmp_path):
    paths = ProjectConfig().paths(tmp_path)
    assert all(p.is_relative_to(tmp_path) and not p.exists() for p in paths.values())


def test_seconds_preserves_days_and_missing_values():
    result = seconds(pd.Series(['1 days 00:00:01', None]))
    assert result.iloc[0] == 86401
    assert np.isnan(result.iloc[1])


def test_prepare_preserves_source_and_audit_counts(laps):
    original = laps.copy(deep=True)
    laps.loc[1, 'Deleted'] = True
    prepared = prepare_laps(laps, '2024-01-R')
    assert len(prepared) == 5
    assert prepared.SourceRow.tolist() == list(range(5))
    assert prepared.Eligible.tolist() == [True, False, True, True, True]
    assert audit_laps(prepared)['eligible_rows'] == 4
    assert 'SourceRow' not in laps
    pd.testing.assert_series_equal(laps.LapTime, original.LapTime)


@pytest.mark.parametrize('column,value,flag', [
    ('TrackStatus', '4', 'FlagInterrupted'), ('Compound', 'WET', 'FlagUnsupportedTyre'),
    ('IsAccurate', False, 'FlagInaccurate'), ('LapNumber', 0, 'FlagInvalidID'),
    ('LapTime', pd.Timedelta(0), 'FlagInvalidTime'),
    ('PitInTime', pd.Timedelta(seconds=100), 'FlagPit'),
])
def test_quality_flags_exclude_bad_laps(laps, column, value, flag):
    laps.loc[0, column] = value
    result = prepare_laps(laps, '2024-01-R')
    assert result.loc[0, flag]
    assert not result.loc[0, 'Eligible']


def test_missing_quality_is_not_assumed_clean(laps):
    result = prepare_laps(laps.drop(columns='Deleted'), '2024-01-R')
    assert result.FlagUnknownQuality.all()
    assert not result.Eligible.any()


def test_duplicate_laps_are_flagged(laps):
    result = prepare_laps(pd.concat([laps, laps.iloc[[0]]]), '2024-01-R')
    assert result.FlagDuplicate.sum() == 2


def test_telemetry_integrates_and_excludes_outside_samples():
    telemetry = pd.DataFrame({
        'SessionTime': pd.to_timedelta([-1, 0, 1, 2, 3], unit='s'),
        'Speed': [440, 100, 200, 300, 440],
        'Throttle': [0, 100, 50, 100, 0], 'Brake': [0, 0, 1, 0, 0],
    })
    result = summarize_telemetry(telemetry, 0, 2, QualityConfig())
    assert result['AvgSpeed_kmh'] == 200
    assert result['MaxSpeed_kmh'] == 300
    assert result['FullThrottle_pct'] == 50
    assert result['Braking_pct'] == 50
    assert result['SpeedCoverage'] == 1
    telemetry.loc[2, 'Speed'] = np.nan
    with pytest.raises(ValueError, match='coverage'):
        summarize_telemetry(telemetry, 0, 2, QualityConfig())


@pytest.mark.parametrize('times', [[0, 0, 1], [0, 2, 1], [0, 2, 4]])
def test_telemetry_rejects_duplicate_unordered_or_gapped_samples(times):
    frame = pd.DataFrame({'SessionTime': pd.to_timedelta(times, unit='s'),
                          'Speed': 100, 'Throttle': 100, 'Brake': 0})
    with pytest.raises(ValueError):
        summarize_telemetry(frame, 0, 4, QualityConfig())


def test_weather_uses_latest_past_observation():
    weather = pd.DataFrame({'Time': pd.to_timedelta([30, 10, 20], unit='s'),
                            'TrackTemp': [99, 30, 35]})
    assert weather_at(weather, 25, QualityConfig()) == {
        'TrackTemp_C': 35, 'WeatherTime_s': 20, 'WeatherAge_s': 5}
    with pytest.raises(ValueError, match='no past'):
        weather_at(weather, 5, QualityConfig())
    with pytest.raises(ValueError, match='stale'):
        weather_at(weather, 200, QualityConfig())


def test_predictor_allowlist_excludes_targets():
    assert not {'NextLapTime_s', 'DeltaLapTime_s', 'TargetLapNumber',
                'TargetTyreAge_laps', 'TargetLapEnd_s'} & set(feature_columns())


def test_examples_use_only_consecutive_past_laps(laps):
    prepared = prepare_laps(laps, '2024-01-R')
    result = make_examples(prepared, history=3)
    assert result[['Lag1', 'Lag2', 'Lag3', 'TargetSeconds']].iloc[0].tolist() == [92, 91, 90, 93]
    prepared.loc[2, 'Eligible'] = False
    assert make_examples(prepared, history=3).empty


@pytest.mark.parametrize('change', ['gap', 'stint', 'duplicate'])
def test_examples_reject_invalid_windows(laps, change):
    prepared = prepare_laps(laps, '2024-01-R')
    if change == 'duplicate':
        with pytest.raises(ValueError, match='Duplicate'):
            make_examples(pd.concat([prepared, prepared.iloc[[0]]]))
    else:
        prepared.loc[2, 'LapNumber' if change == 'gap' else 'Stint'] = 10
        assert make_examples(prepared, history=3).empty


def test_weekend_split_enforces_chronology_and_disjointness():
    frame = pd.DataFrame({'SessionID': ['2024-01-R', '2024-02-R', '2025-01-R'],
                          'EventDate': ['2024-03-01', '2024-04-01', '2025-03-01']})
    ids = [[x] for x in frame.SessionID]
    assert [len(p) for p in split_weekends(frame, *ids)] == [1, 1, 1]
    with pytest.raises(ValueError, match='overlap'):
        split_weekends(frame, ids[0], ['2024-01-S'], ids[2])
    with pytest.raises(ValueError, match='before'):
        split_weekends(frame, *ids[::-1])
    with pytest.raises(ValueError, match='no usable'):
        split_weekends(frame, ids[0], ids[1], ['2026-01-R'])


def test_score_matches_hand_calculation():
    assert score([10, 20], [13, 16]) == pytest.approx({'MAE_seconds': 3.5, 'RMSE_seconds': np.sqrt(12.5)})


def test_baseline_fits_scaler_only_on_training_and_evaluates_same_rows(laps):
    train = make_examples(prepare_laps(laps, '2024-01-R'), history=1)
    validation = train.copy()
    validation['Lag1'] += 100
    model, features, grid = train_baseline(train, validation)
    np.testing.assert_allclose(model.steps[0][1].mean_, train[features].mean())
    assert model.steps[1][1].alpha == grid.loc[grid.MAE_seconds.idxmin(), 'alpha']
    metrics, predictions = evaluate(model, features, validation)
    assert metrics.rows.tolist() == [len(validation)] * 2
    np.testing.assert_allclose(predictions.PreviousLap, validation.Lag1)


def test_failed_json_write_preserves_existing_file(tmp_path):
    path = tmp_path / 'manifest.json'
    atomic_json(path, {'status': 'complete'})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {'invalid': float('nan')})
    assert path.read_bytes() == before


def test_missing_export_manifest_cannot_be_reused(tmp_path):
    valid, reason, manifest = validate_export(tmp_path, 'race', request_identity(2024, 'Bahrain'))
    assert not valid
    assert 'missing' in reason
    assert manifest is None
