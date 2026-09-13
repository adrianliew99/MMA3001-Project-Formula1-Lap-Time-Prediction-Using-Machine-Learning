"""Synthetic fixtures verify logic; these values are NOT real F1 results."""
import numpy as np
import pandas as pd
import pytest
from mma3001_f1.data import seconds, prepare_laps, audit_laps
from mma3001_f1.modeling import make_examples, split_weekends, train_baseline, evaluate


def fixture():
    n = 8
    return pd.DataFrame({"Driver": ["AAA"]*n, "LapNumber": range(1,n+1),
        "LapTime": pd.to_timedelta(np.arange(90,90+n), unit="s"),
        "PitInTime": pd.Series([pd.NaT]*n, dtype="timedelta64[ns]"), "PitOutTime": pd.Series([pd.NaT]*n, dtype="timedelta64[ns]"),
        "TrackStatus": ["1"]*n, "IsAccurate": [True]*n, "Deleted": [False]*n,
        "FastF1Generated": [False]*n, "Compound": ["SOFT"]*n,
        "TyreLife": range(1,n+1), "Stint": [1]*n})


def examples(sid="2024-01-R", date="2024-03-02"):
    d = prepare_laps(fixture(), sid)
    d["EventDate"] = date
    return d


def test_total_seconds_and_missing():
    x = seconds(pd.Series([pd.Timedelta(days=1, seconds=2), pd.NaT]))
    assert x.iloc[0] == 86402 and np.isnan(x.iloc[1])


def test_duplicate_flags_preserve_source():
    raw = fixture()
    raw.loc[1, "LapNumber"] = 1
    d = prepare_laps(raw, "fixture")
    assert len(d) == len(raw) and d.FlagDuplicate.sum() == 2
    assert "Eligible" not in raw


def test_missing_quality_is_not_clean():
    d = prepare_laps(fixture().drop(columns="Deleted"), "fixture")
    assert d.FlagUnknownQuality.all() and not d.Eligible.any()


def test_missing_and_negative_target_and_pit():
    raw = fixture()
    raw.loc[0, "LapTime"] = pd.NaT
    raw.loc[1, "LapTime"] = pd.Timedelta(seconds=-1)
    raw.loc[2, "PitInTime"] = pd.Timedelta(seconds=100)
    d = prepare_laps(raw, "fixture")
    assert d.FlagInvalidTime.sum() == 2 and d.FlagPit.sum() == 1
    assert audit_laps(d)["eligible_rows"] == 5


def test_windows_no_gaps_no_stint_crossing():
    d = examples()
    assert len(make_examples(d, 3)) == 5
    d.loc[3, "Eligible"] = False
    assert make_examples(d, 3).TargetLap.tolist() == [8]
    d = examples().drop(index=3)
    assert make_examples(d, 3).TargetLap.tolist() == [8]
    d = examples()
    d.loc[4:, "Stint"] = 2
    assert make_examples(d, 3).TargetLap.tolist() == [4,8]


def test_features_do_not_use_target_time():
    d = examples()
    a = make_examples(d, 3).iloc[0]
    d.loc[3, "LapTimeSeconds"] += 100
    b = make_examples(d, 3).iloc[0]
    assert a.Lag1 == b.Lag1 and a.CurrentTyreLife == b.CurrentTyreLife
    assert a.TargetSeconds != b.TargetSeconds


def test_split_and_train_smoke():
    e = pd.concat([make_examples(examples(sid, date)) for sid,date in
                   [("2024-01-R", "2024-03-02"), ("2024-02-R", "2024-03-09"), ("2024-03-R", "2024-03-24")]])
    ids = [[f"2024-0{i}-R"] for i in range(1,4)]
    tr,va,te = split_weekends(e,*ids)
    model, features, sensitivity = train_baseline(tr,va)
    metrics,preds = evaluate(model,features,te)
    assert len(sensitivity) == 3 and len(preds) == len(te)
    assert np.isfinite(metrics.MAE_seconds).all()
    assert np.allclose(model[0].mean_, tr[features].mean())
    with pytest.raises(ValueError, match="overlap"):
        split_weekends(e, ids[0], ids[0], ids[2])
    with pytest.raises(ValueError, match="before"):
        split_weekends(e, ids[1], ids[0], ids[2])
