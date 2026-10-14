"""Optional next-lap baseline scaffold. Whole chronological weekends stay separate."""
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_examples(laps, history=3):
    """Use consecutive clean laps within one driver/stint to predict the next lap.

    Eligibility of the target is retrospective: results describe uninterrupted dry
    running, not the ability to forecast future incidents or pit stops.
    """
    if not isinstance(history, int) or history < 1:
        raise ValueError("history must be a positive integer")
    if laps.duplicated(["SessionID", "Driver", "LapNumber"]).any():
        raise ValueError("Duplicate lap identifiers must be resolved before modeling")
    rows = []
    for (_, driver), g in laps.groupby(["SessionID", "Driver"]):
        g = g.sort_values("LapNumber")
        for end in range(history, len(g)):
            block = g.iloc[end-history:end+1]
            if not block.Eligible.all() or block.Stint.nunique() != 1 or not np.all(np.diff(block.LapNumber) == 1):
                continue
            past, target = block.iloc[:-1], block.iloc[-1]
            row = {"SessionID": target.SessionID, "Driver": driver, "TargetLap": target.LapNumber,
                   "EventDate": target.EventDate, "TargetSeconds": target.LapTimeSeconds,
                   "CurrentTyreLife": past.iloc[-1].TyreLife, "CurrentLap": past.iloc[-1].LapNumber}
            row.update({f"Lag{i+1}": float(past.iloc[-i-1].LapTimeSeconds) for i in range(history)})
            rows.append(row)
    return pd.DataFrame(rows)


def split_weekends(examples, train_ids, validation_ids, test_ids):
    """Reject overlap, missing sessions, and nonchronological partitions."""
    groups = [list(x) for x in [train_ids, validation_ids, test_ids]]
    if any(not x for x in groups):
        raise ValueError("Set nonempty training, validation and test race IDs")
    # Remove session suffix: race and sprint from one weekend cannot cross sets.
    weekends = [{sid.rsplit("-", 1)[0] for sid in ids} for ids in groups]
    if any(weekends[i] & weekends[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Race weekends overlap between partitions")
    parts = []
    for ids in groups:
        if not set(ids).issubset(set(examples.SessionID)):
            raise ValueError("A configured session has no usable examples")
        parts.append(examples[examples.SessionID.isin(ids)].copy())
    dates = [pd.to_datetime(p.EventDate) for p in parts]
    if not (dates[0].max() < dates[1].min() and dates[1].max() < dates[2].min()):
        raise ValueError("Require training before validation before final testing")
    return tuple(parts)


def score(y, prediction):
    """MAE is mean absolute error; RMSE gives larger errors more weight, in seconds."""
    return {"MAE_seconds": float(mean_absolute_error(y, prediction)),
            "RMSE_seconds": float(np.sqrt(mean_squared_error(y, prediction)))}


def train_baseline(train, validation, alphas=(0.1, 1.0, 10.0)):
    """Fit scaling on training only; choose ridge penalty on validation MAE only.

    Ridge minimizes squared errors plus alpha times squared coefficients. Larger
    alpha discourages large coefficients. Test data is deliberately not an input.
    """
    if not alphas or any(not np.isfinite(a) or a < 0 for a in alphas):
        raise ValueError("Supply finite nonnegative ridge alphas")
    features = sorted(c for c in train if c.startswith("Lag")) + ["CurrentTyreLife", "CurrentLap"]
    records, fitted = [], []
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
        start = perf_counter()
        model.fit(train[features], train.TargetSeconds)
        elapsed = perf_counter()-start
        pred = model.predict(validation[features])
        records.append({"alpha": alpha, "fit_seconds": elapsed, **score(validation.TargetSeconds, pred)})
        fitted.append(model)
    table = pd.DataFrame(records)
    best = int(table.MAE_seconds.argmin())
    return fitted[best], features, table


def evaluate(model, features, test):
    """Compare frozen ridge and previous-lap baseline on exactly the same rows."""
    out = test[["SessionID", "Driver", "TargetLap", "TargetSeconds"]].copy()
    start = perf_counter()
    out["Ridge"] = model.predict(test[features])
    elapsed = perf_counter()-start
    out["PreviousLap"] = test.Lag1.to_numpy()
    metrics = pd.DataFrame([{ "model": name, "rows": len(test), **score(test.TargetSeconds, out[name])} for name in ["PreviousLap", "Ridge"]])
    metrics["prediction_seconds"] = [np.nan, elapsed]
    return metrics, out
