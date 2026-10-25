"""Recompute saved holdout results independently; never retrain on holdout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / 'results/metrics'
pytestmark = pytest.mark.artifact


@pytest.fixture(scope='module')
def predictions():
    return pd.read_csv(ROOT / 'results/predictions/FINAL_holdout_2025_predictions.csv')


def test_holdout_identity_and_error_columns(predictions):
    p = predictions
    assert not p.empty
    assert p.Year.eq(2025).all()
    assert p.TargetLapNumber.eq(p.PreviousLapNumber + 1).all()
    assert not p.duplicated(['Year', 'Event', 'Session', 'Driver', 'TargetLapNumber']).any()
    columns = ['NextLapTime_s', 'PreviousLapTime_s', 'PredictedNextLapTime_s',
               'PredictedDeltaLapTime_s', 'Residual_s', 'AbsoluteError_s']
    assert np.isfinite(p[columns].to_numpy()).all()
    np.testing.assert_allclose(p.Residual_s, p.NextLapTime_s - p.PredictedNextLapTime_s, atol=1e-10, rtol=0)
    np.testing.assert_allclose(p.AbsoluteError_s, np.abs(p.Residual_s), atol=1e-10, rtol=0)
    np.testing.assert_allclose(p.PredictedNextLapTime_s, p.PreviousLapTime_s + p.PredictedDeltaLapTime_s, atol=1e-10, rtol=0)


def test_saved_overall_metrics_match_predictions(predictions):
    saved = json.loads((METRICS / 'FINAL_holdout_2025_metrics.json').read_text())
    p = predictions
    residual = p.NextLapTime_s - p.PredictedNextLapTime_s
    assert saved['observations'] == len(p)
    assert saved['holdout_year'] == 2025
    actual = {
        'MAE_s': np.abs(residual).mean(), 'RMSE_s': np.sqrt(np.mean(residual ** 2)),
        'R2_next_lap': 1 - np.sum(residual ** 2) / np.sum((p.NextLapTime_s - p.NextLapTime_s.mean()) ** 2),
        'MAE_delta_s': np.abs(p.NextLapTime_s - p.PreviousLapTime_s - p.PredictedDeltaLapTime_s).mean(),
    }
    assert saved['scores'] == pytest.approx(actual, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize('filename,key', [
    ('holdout_driver_errors.csv', 'Driver'),
    ('holdout_compound_errors.csv', 'TyreCompound'),
    ('holdout_circuit_errors.csv', 'Event'),
])
def test_group_reports_match_recomputed_errors(predictions, filename, key):
    p = predictions.copy()
    p['error'] = p.NextLapTime_s - p.PredictedNextLapTime_s
    actual = p.groupby(key).error.agg(
        Observations='size', MAE_s=lambda x: np.abs(x).mean(),
        Bias_s='mean', RMSE_s=lambda x: np.sqrt(np.mean(x ** 2))).sort_index()
    saved = pd.read_csv(METRICS / filename).set_index(key).sort_index()
    pd.testing.assert_frame_equal(saved[actual.columns], actual, check_dtype=False, rtol=1e-10, atol=1e-10)
    # The saved groupby summaries exclude rows with a missing group label.
    # Those rows still belong to the overall holdout evaluation, so account
    # for them explicitly rather than requiring every tyre label to be known.
    missing_labels = p[key].isna().sum()
    assert saved.Observations.sum() == p[key].notna().sum()
    assert saved.Observations.sum() + missing_labels == len(p)


def test_metric_verification_table_agrees_with_raw_predictions(predictions):
    table = pd.read_csv(METRICS / 'holdout_metric_verification.csv')
    mae = np.abs(predictions.NextLapTime_s - predictions.PredictedNextLapTime_s).mean()
    assert len(table) == 4
    np.testing.assert_allclose(table.MAE_s, mae, rtol=0, atol=1e-10)
    np.testing.assert_allclose(table.Difference_s, table.MAE_s - mae, rtol=0, atol=1e-10)
