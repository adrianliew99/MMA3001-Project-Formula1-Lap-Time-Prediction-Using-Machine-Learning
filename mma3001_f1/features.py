"""Phase 2 clean-lap features and auditable timing checks, separate from ingestion.

All targets and target quality checks define a retrospective population. They are
never part of the explicit predictor allowlist.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import prepare_laps, audit_laps

CONTEXT_FEATURES = ["PreviousLapTime_s", "TyreAge_laps", "TrackTemp_C", "PreviousLapNumber"]
TELEMETRY_FEATURES = ["AvgSpeed_kmh", "MaxSpeed_kmh", "FullThrottle_pct", "Braking_pct"]
CATEGORIES = ["TyreCompound", "Event"]
PAIR_KEYS = ["SessionID", "Driver", "PreviousLapNumber", "TargetLapNumber"]


@dataclass(frozen=True)
class QualityConfig:
    """A priori sampling/availability limits; not selected using prediction error."""
    max_gap_s: float = 1.0
    min_coverage: float = 0.95
    max_weather_age_s: float = 120.0
    timing_tolerance_s: float = 0.1

    def __post_init__(self):
        if not (np.isfinite(self.max_gap_s) and self.max_gap_s > 0):
            raise ValueError("max_gap_s must be finite and positive")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        if not np.isfinite(self.max_weather_age_s) or self.max_weather_age_s <= 0:
            raise ValueError("max_weather_age_s must be positive")
        if not np.isfinite(self.timing_tolerance_s) or self.timing_tolerance_s < 0:
            raise ValueError("timing_tolerance_s must be nonnegative")


def elapsed(value) -> float:
    """Session-relative timedelta to total seconds, retaining missing values."""
    return float(pd.to_timedelta(value).total_seconds()) if pd.notna(value) else float("nan")


def clean_laps(raw: pd.DataFrame, session_id: str, config: QualityConfig) -> pd.DataFrame:
    """Reuse existing quality flags and additionally validate tyre/stint/clock fields."""
    if "SessionID" in raw and not raw.SessionID.eq(session_id).all():
        raise ValueError("Raw session identity mismatch")
    required = ["LapStartTime", "Time", "TyreLife", "Stint"]
    if set(required) - set(raw):
        raise ValueError(f"Missing raw timing/context fields: {sorted(set(required) - set(raw))}")
    normalized = raw.copy()
    for name in ("TyreLife", "Stint", "LapNumber"):
        normalized[name] = pd.to_numeric(normalized[name], errors="coerce")
    frame = prepare_laps(normalized, session_id)
    for name in ("TyreLife", "Stint"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["LapStart_s"] = pd.to_timedelta(frame.LapStartTime).dt.total_seconds()
    frame["LapEnd_s"] = pd.to_timedelta(frame.Time).dt.total_seconds()
    frame["FlagOpeningLap"] = frame.LapNumber.le(1)
    frame["FlagInvalidTyreAge"] = (~np.isfinite(frame.TyreLife) | frame.TyreLife.lt(1)
                                   | frame.TyreLife.mod(1).ne(0))
    frame["FlagInvalidStint"] = (~np.isfinite(frame.Stint) | frame.Stint.lt(1)
                                 | frame.Stint.mod(1).ne(0))
    frame["FlagClock"] = (~np.isfinite(frame.LapStart_s) | ~np.isfinite(frame.LapEnd_s)
                          | frame.LapStart_s.lt(0) | frame.LapEnd_s.le(frame.LapStart_s)
                          | (frame.LapEnd_s - frame.LapStart_s - frame.LapTimeSeconds).abs().gt(config.timing_tolerance_s))
    # Unexpected boolean representations are unknown, not clean.
    for name in ("IsAccurate", "Deleted", "FastF1Generated"):
        frame["FlagUnknownQuality"] |= ~frame[name].isin([True, False])
    frame["Eligible"] = ~frame[[c for c in frame if c.startswith("Flag")]].any(axis=1)
    return frame


def summarize_telemetry(telemetry: pd.DataFrame, start: float, end: float,
                        config: QualityConfig) -> dict:
    """Integrate in-lap observations only; never extrapolate across lap boundaries.

    Speed uses trapezoids; throttle/brake indicators use left-endpoint hold on
    valid intervals. Both endpoints must be valid, and gaps > max_gap_s contribute
    no duration. Each channel must independently cover min_coverage of the lap.
    """
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("invalid lap interval")
    required = {"SessionTime", "Speed", "Throttle", "Brake"}
    if required - set(telemetry):
        raise ValueError("telemetry missing required channels")
    t = pd.to_timedelta(telemetry.SessionTime).dt.total_seconds().to_numpy()
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("telemetry timestamps missing, repeated or unordered")
    inside = (t >= start) & (t <= end)
    table = telemetry.loc[inside]
    t = t[inside]
    if len(t) < 3:
        raise ValueError("too few in-lap telemetry samples")
    dt = np.diff(t)
    gap_ok = (dt > 0) & (dt <= config.max_gap_s)
    coverage, means, integrals = {}, {}, {}
    bounds = {"Speed": (0, 450), "Throttle": (0, 100), "Brake": (0, 1)}
    for name, (lower, upper) in bounds.items():
        values = pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(values) & (values >= lower) & (values <= upper)
        valid = gap_ok & good[:-1] & good[1:]
        duration = float(dt[valid].sum())
        coverage[name] = duration / (end - start)
        if coverage[name] < config.min_coverage:
            raise ValueError(f"{name} coverage below minimum: {coverage[name]:.6f}")
        if name == "Speed":
            interval_values = (values[:-1] + values[1:]) / 2
        else:
            interval_values = (values[:-1] >= (95 if name == "Throttle" else 0.5)).astype(float)
        integrals[name] = float(np.dot(interval_values[valid], dt[valid]))
        means[name] = integrals[name] / duration
    speed = pd.to_numeric(table.Speed, errors="coerce")
    speed = speed[np.isfinite(speed) & speed.between(0, 450)]
    return {
        "AvgSpeed_kmh": means["Speed"], "MaxSpeed_kmh": float(speed.max()),
        "FullThrottle_pct": means["Throttle"] * 100, "Braking_pct": means["Brake"] * 100,
        "TelemetryStart_s": float(t[0]), "TelemetryEnd_s": float(t[-1]),
        "TelemetrySamples": len(t), "TelemetryMaxGap_s": float(dt.max()),
        "SpeedCoverage": coverage["Speed"], "ThrottleCoverage": coverage["Throttle"],
        "BrakeCoverage": coverage["Brake"], "SpeedIntegral_kmh_s": integrals["Speed"],
        "FullThrottleTime_s": integrals["Throttle"], "BrakingTime_s": integrals["Brake"],
    }


def weather_at(weather: pd.DataFrame, prediction_time: float, config: QualityConfig) -> dict:
    """Take the latest observation at/before prediction, regardless of input order."""
    if not {"Time", "TrackTemp"}.issubset(weather):
        raise ValueError("weather fields unavailable")
    if not np.isfinite(prediction_time):
        raise ValueError("invalid weather query time")
    table = weather.copy()
    table["WeatherTime_s"] = pd.to_timedelta(table.Time).dt.total_seconds()
    table = table.loc[np.isfinite(table.WeatherTime_s) & table.WeatherTime_s.le(prediction_time)]
    if table.empty:
        raise ValueError("no past weather observation")
    latest = table.loc[table.WeatherTime_s.eq(table.WeatherTime_s.max())]
    temperatures = pd.to_numeric(latest.TrackTemp, errors="coerce")
    if temperatures.isna().any() or temperatures.nunique() != 1:
        raise ValueError("missing or conflicting latest weather")
    temperature = float(temperatures.iloc[0])
    stamp = float(latest.WeatherTime_s.iloc[0])
    age = prediction_time - stamp
    if age > config.max_weather_age_s or not np.isfinite(temperature) or not -20 <= temperature <= 100:
        raise ValueError("stale or invalid weather")
    return {"TrackTemp_C": temperature, "WeatherTime_s": stamp, "WeatherAge_s": age}


def build_pairs(raw, weather, metadata, telemetry_loader, drivers, config=QualityConfig()):
    """Build within-driver consecutive pairs; loader receives previous SourceRow only."""
    prepared = clean_laps(raw, metadata["SessionID"], config)
    selected = prepared.loc[prepared.Driver.isin(drivers)].copy()
    if not set(drivers).issubset(set(selected.Driver)):
        raise ValueError("requested driver missing from source")
    rows, rejected, telemetry_sources = [], [], []
    for driver, group in selected.groupby("Driver", sort=True):
        group = group.sort_values("LapNumber", kind="stable")
        for i in range(len(group) - 1):
            previous, target = group.iloc[i], group.iloc[i + 1]
            base = {"SessionID": metadata["SessionID"], "Driver": driver,
                    "PreviousLapNumber": previous.LapNumber, "TargetLapNumber": target.LapNumber,
                    "PreviousSourceRow": int(previous.SourceRow), "TargetSourceRow": int(target.SourceRow)}
            reasons = []
            for role, lap in (("previous", previous), ("target", target)):
                reasons.extend(f"{role}:{c}" for c in prepared if c.startswith("Flag") and bool(lap[c]))
            if target.LapNumber != previous.LapNumber + 1:
                reasons.append("nonconsecutive")
            if target.Stint != previous.Stint:
                reasons.append("stint_transition")
            if target.Compound != previous.Compound:
                reasons.append("compound_transition")
            if target.TyreLife != previous.TyreLife + 1:
                reasons.append("tyre_age_transition")
            if previous.LapEnd_s > target.LapStart_s or abs(target.LapStart_s - previous.LapEnd_s) > config.timing_tolerance_s:
                reasons.append("lap_clock_discontinuity")
            if reasons:
                rejected.append({**base, "Reasons": "|".join(reasons)})
                continue
            try:
                telemetry = telemetry_loader(int(previous.SourceRow))
                source = pd.DataFrame(telemetry).copy()
                source["SessionID"], source["Driver"], source["LapNumber"] = metadata["SessionID"], driver, previous.LapNumber
                source["SourceRow"] = int(previous.SourceRow)
                telemetry_sources.append(source)
                features = summarize_telemetry(telemetry, previous.LapStart_s, previous.LapEnd_s, config)
            except Exception as exc:
                rejected.append({**base, "Reasons": f"telemetry:{type(exc).__name__}:{exc}"})
                continue
            try:
                weather_features = weather_at(weather, previous.LapEnd_s, config)
            except ValueError as exc:
                rejected.append({**base, "Reasons": f"weather:{exc}"})
                continue
            rows.append({
                **metadata, **base, "Session": "Race", "Stint": int(previous.Stint),
                "TargetStint": int(target.Stint), "TyreCompound": previous.Compound,
                "TargetCompound": target.Compound, "TyreAge_laps": float(previous.TyreLife),
                "TargetTyreAge_laps": float(target.TyreLife),
                "PreviousLapTime_s": float(previous.LapTimeSeconds),
                "NextLapTime_s": float(target.LapTimeSeconds),
                "DeltaLapTime_s": float(target.LapTimeSeconds - previous.LapTimeSeconds),
                "PreviousLapStart_s": float(previous.LapStart_s), "PredictionTime_s": float(previous.LapEnd_s),
                "TargetLapStart_s": float(target.LapStart_s), "TargetLapEnd_s": float(target.LapEnd_s),
                **features, **weather_features,
            })
    dataset = pd.DataFrame(rows)
    if not dataset.empty:
        dataset = dataset.sort_values(PAIR_KEYS).reset_index(drop=True)
        validate_dataset(dataset, config)
    rejection_table = pd.DataFrame(rejected, columns=PAIR_KEYS + ["PreviousSourceRow", "TargetSourceRow", "Reasons"])
    lap_audit = audit_laps(prepared, raw.columns)
    lap_audit["selected_driver_rows"] = len(selected)
    lap_audit["selected_eligible_laps"] = int(selected.Eligible.sum())
    return {"dataset": dataset, "laps": prepared, "rejections": rejection_table,
            "telemetry": pd.concat(telemetry_sources, ignore_index=True) if telemetry_sources else pd.DataFrame(),
            "lap_audit": lap_audit}


def validate_dataset(data: pd.DataFrame, config=QualityConfig()) -> None:
    """Fail on schema, identity, range or feature-time violations."""
    required = PAIR_KEYS + ["Year", "Event", "RoundNumber", "Stint", "TargetStint", "TyreCompound",
        "TargetCompound", "TyreAge_laps", "TargetTyreAge_laps", "PreviousLapTime_s", "NextLapTime_s",
        "DeltaLapTime_s", "PreviousLapStart_s", "PredictionTime_s", "TargetLapStart_s", "TargetLapEnd_s",
        "TelemetryStart_s", "TelemetryEnd_s", "WeatherTime_s", "WeatherAge_s", "TrackTemp_C",
        "SpeedCoverage", "ThrottleCoverage", "BrakeCoverage"] + TELEMETRY_FEATURES
    if data.empty or set(required) - set(data):
        raise ValueError("empty dataset or required schema fields missing")
    if data[required].isna().any().any():
        raise ValueError("missing required values")
    numeric = [c for c in required if c not in ("SessionID", "Event", "Driver", "TyreCompound", "TargetCompound")]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite numeric values")
    if data.duplicated(PAIR_KEYS).any():
        raise ValueError("duplicate pairs")
    def require(condition, message):
        if not bool(condition.all()):
            raise ValueError(message)
    for column in ("Year", "RoundNumber", "PreviousLapNumber", "TargetLapNumber", "Stint", "TargetStint",
                   "TyreAge_laps", "TargetTyreAge_laps"):
        require(data[column].gt(0) & data[column].mod(1).eq(0), "invalid integer/context value")
    sid = data.Year.astype(int).astype(str) + "-" + data.RoundNumber.astype(int).astype(str).str.zfill(2) + "-R"
    require(data.SessionID.eq(sid), "session identity mismatch")
    require(data.TargetLapNumber.eq(data.PreviousLapNumber + 1), "nonconsecutive pairs")
    require(data.Stint.eq(data.TargetStint), "cross-stint pair")
    require(data.TyreCompound.isin(["SOFT", "MEDIUM", "HARD"]) & data.TyreCompound.eq(data.TargetCompound), "compound mismatch")
    require(data.TargetTyreAge_laps.eq(data.TyreAge_laps + 1), "tyre age mismatch")
    require(data.PreviousLapTime_s.gt(0) & data.NextLapTime_s.gt(0), "invalid lap times")
    if not np.allclose(data.NextLapTime_s - data.PreviousLapTime_s, data.DeltaLapTime_s, atol=1e-8, rtol=0):
        raise ValueError("delta identity mismatch")
    require(data.TelemetryStart_s.ge(data.PreviousLapStart_s) & data.TelemetryEnd_s.le(data.PredictionTime_s),
            "telemetry outside completed lap")
    require(data.TelemetryEnd_s.gt(data.TelemetryStart_s), "invalid telemetry interval")
    require(data.PredictionTime_s.le(data.TargetLapStart_s), "feature after target start")
    require((data.TargetLapStart_s - data.PredictionTime_s).abs().le(config.timing_tolerance_s), "lap clock gap")
    require((data.PredictionTime_s - data.PreviousLapStart_s - data.PreviousLapTime_s).abs().le(config.timing_tolerance_s), "previous clock mismatch")
    require((data.TargetLapEnd_s - data.TargetLapStart_s - data.NextLapTime_s).abs().le(config.timing_tolerance_s), "target clock mismatch")
    require(data.WeatherTime_s.le(data.PredictionTime_s), "weather look-ahead")
    require(data.WeatherAge_s.between(0, config.max_weather_age_s), "stale weather")
    if not np.allclose(data.PredictionTime_s - data.WeatherTime_s, data.WeatherAge_s, atol=1e-8, rtol=0):
        raise ValueError("weather age mismatch")
    for name in ("SpeedCoverage", "ThrottleCoverage", "BrakeCoverage"):
        require(data[name].between(config.min_coverage, 1 + 1e-9), "insufficient telemetry coverage")
    for name in ("FullThrottle_pct", "Braking_pct"):
        require(data[name].between(0, 100), "percentage outside [0,100]")
    require(data.AvgSpeed_kmh.between(0, 450) & data.MaxSpeed_kmh.between(0, 450)
            & data.AvgSpeed_kmh.le(data.MaxSpeed_kmh + 1e-9), "invalid speed")
    require(data.TrackTemp_C.between(-20, 100), "invalid temperature")


def feature_columns(telemetry=True) -> list[str]:
    """Only information describing the completed lap or known event is a predictor."""
    return CONTEXT_FEATURES + (TELEMETRY_FEATURES if telemetry else []) + CATEGORIES


def make_preprocessor(telemetry=True) -> Pipeline:
    """Return an UNFITTED pipeline; caller must fit only on training races."""
    numeric = CONTEXT_FEATURES + (TELEMETRY_FEATURES if telemetry else [])
    numerical = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([("columns", ColumnTransformer(
        [("numeric", numerical, numeric), ("categorical", categorical, CATEGORIES)], remainder="drop"))])
