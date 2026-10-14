"""Export real FastF1 session data and a leakage-aware next-lap ML dataset.

Install from the repository root: python -m pip install -e .
Usage: python fastf1_export_dataset.py --year 2024 --event Bahrain --drivers VER HAM
       python fastf1_export_dataset.py --year 2024 --event Bahrain
       python -m dataset.fastf1export_dataset -- 

All telemetry features describe the PREVIOUS completed lap, not the target lap.
Use --offline --cache-dir PATH for cache-only extraction; otherwise provider access may occur.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1
from mma3001_f1.ingestion import (ML_COLUMNS, atomic_csv, atomic_json, build_manifest, check_tables, request_identity, slug, utc_now, validate_export)


def good_lap(lap):
    """Keep recorded green-flag, non-pit laps with credible lap times."""
    try:
        return (
            pd.notna(lap["LapTime"])
            and pd.notna(lap["LapNumber"])
            and lap["LapTime"].total_seconds() > 0
            and lap["IsAccurate"] == True
            and pd.isna(lap["PitInTime"])
            and pd.isna(lap["PitOutTime"])
            and str(lap["TrackStatus"]) == "1"
        )
    except (KeyError, TypeError, AttributeError):
        return False


def weighted_mean(values, weights):
    """Mean weighted by time between samples, ignoring invalid samples."""
    valid = values.notna() & np.isfinite(values) & (weights > 0)
    if not valid.any():
        return np.nan
    w = weights[valid]
    return float((values[valid] * w).sum() / w.sum())


def telemetry_features(lap):
    """Summarise only the car telemetry belonging to this completed lap."""
    tel = lap.get_car_data()
    if len(tel) < 3:
        return None, None

    t = pd.to_timedelta(tel["Time"]).dt.total_seconds()
    dt = t.diff().fillna(0).clip(lower=0)
    if dt.sum() <= 0:
        return None, None

    speed = pd.to_numeric(tel["Speed"], errors="coerce")
    throttle = pd.to_numeric(tel["Throttle"], errors="coerce")
    brake = pd.to_numeric(tel["Brake"], errors="coerce")

    stats = {
        "AvgSpeed_kmh": weighted_mean(speed, dt),
        "MaxSpeed_kmh": float(speed.max()),
        "FullThrottle_pct": 100 * weighted_mean((throttle >= 95).astype(float), dt),
        "Braking_pct": 100 * weighted_mean((brake >= 0.5).astype(float), dt),
    }
    # The raw example is labelled by the lap from which it came.
    return stats, tel


def recent_track_temperature(weather, previous_lap_end):
    """Use only the latest weather reading at or before lap completion."""
    if weather is None or weather.empty or pd.isna(previous_lap_end):
        return np.nan
    past = weather.loc[weather["Time"] <= previous_lap_end]
    if past.empty:
        return np.nan
    val = past.iloc[-1]["TrackTemp"]
    return float(val) if pd.notna(val) else np.nan


def extract_session(session, identity):
    """Build legacy pair features, retaining exclusion counts and source sample."""
    raw = session.laps.copy()
    raw["LapTimeSeconds"] = raw["LapTime"].dt.total_seconds()
    round_number = int(session.event.RoundNumber)
    session_id = f"{identity['year']}-{round_number:02d}-R"
    raw["SessionID"] = session_id
    raw["Year"] = identity["year"]
    raw["Event"] = identity["event"]
    drivers = identity["drivers"] or sorted(raw["Driver"].dropna().unique().tolist())
    rows, sample = [], None
    diagnostics = {"candidate_pairs": 0, "excluded_quality": 0, "excluded_gap": 0,
                   "excluded_stint": 0, "excluded_telemetry": 0, "accepted_pairs": 0,
                   "telemetry_errors": [], "per_driver_pairs": {}}
    for driver in drivers:
        laps = session.laps.loc[session.laps["Driver"] == driver].sort_values("LapNumber")
        count = 0
        for i in range(len(laps) - 1):
            prev, nxt = laps.iloc[i], laps.iloc[i + 1]
            diagnostics["candidate_pairs"] += 1
            if not (good_lap(prev) and good_lap(nxt)):
                diagnostics["excluded_quality"] += 1
                continue
            if nxt["LapNumber"] != prev["LapNumber"] + 1:
                diagnostics["excluded_gap"] += 1
                continue
            if pd.isna(prev["Stint"]) or pd.isna(nxt["Stint"]) or nxt["Stint"] != prev["Stint"]:
                diagnostics["excluded_stint"] += 1
                continue
            try:
                features, telemetry = telemetry_features(prev)
            except Exception as exc:
                diagnostics["excluded_telemetry"] += 1
                diagnostics["telemetry_errors"].append(
                    {"driver": driver, "lap": int(prev["LapNumber"]), "error": str(exc)})
                continue
            if features is None:
                diagnostics["excluded_telemetry"] += 1
                continue
            previous_time = prev["LapTime"].total_seconds()
            next_time = nxt["LapTime"].total_seconds()
            rows.append({
                "Year": identity["year"], "Event": identity["event"], "Session": "Race",
                "Driver": driver, "PreviousLapNumber": int(prev["LapNumber"]),
                "TargetLapNumber": int(nxt["LapNumber"]), "Stint": prev["Stint"],
                "PreviousLapTime_s": previous_time, "TyreCompound": prev["Compound"],
                "TyreAge_laps": prev["TyreLife"],
                "TrackTemp_C": recent_track_temperature(session.weather_data, prev["Time"]),
                **features, "NextLapTime_s": next_time, "DeltaLapTime_s": next_time - previous_time,
                "SessionID": session_id, "RoundNumber": round_number,
            })
            count += 1
            if sample is None:
                sample = telemetry
        diagnostics["per_driver_pairs"][driver] = count
        print(f"{driver}: {count} consecutive clean-lap pairs")
    diagnostics["accepted_pairs"] = len(rows)
    frames = {"all_laps": raw, "weather": pd.DataFrame(session.weather_data),
              "race_results": pd.DataFrame(session.results),
              "next_lap_ml": pd.DataFrame(rows, columns=ML_COLUMNS + ["SessionID", "RoundNumber"])}
    if sample is not None:
        frames["example_raw_telemetry"] = sample
    return session_id, frames, diagnostics


def main(argv=None):
    """Export one session; do not reuse partial files or silently overwrite sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--event", default="Bahrain")
    parser.add_argument("--drivers", nargs="*", default=None)
    parser.add_argument("--output", default="f1_export")
    parser.add_argument("--cache-dir", type=Path, help="Reuse an existing FastF1 cache")
    parser.add_argument("--offline", action="store_true", help="Only use cached provider responses; never download")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this event's existing exports")
    args = parser.parse_args(argv)
    identity = request_identity(args.year, args.event, args.drivers)
    if args.year < 2018 or not slug(args.event):
        parser.error("Use a FastF1-era season >=2018 and a nonempty event name")
    if any(len(d) != 3 or not d.isalpha() for d in identity["drivers"]):
        parser.error("Drivers must be three-letter codes")
    out = Path(args.output).resolve()
    prefix = f"{args.year}_{slug(args.event)}"
    valid, reason, manifest = validate_export(out, prefix, identity)
    if valid and not args.overwrite:
        print(f"Verified cached export: {manifest['session_id']}")
        return 0
    if not args.overwrite and out.exists() and list(out.glob(f"{prefix}_*")):
        print(f"Incomplete/stale export: {reason}. Choose a fresh --output or explicitly --overwrite.")
        return 1
    cache = (args.cache_dir or out / "fastf1_cache").resolve()
    if args.offline and not cache.is_dir():
        print(f"Offline cache directory does not exist: {cache}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    manifest_path = out / f"{prefix}_manifest.json"
    attempt = {"status": "running", "request": identity, "started_utc": started,
               "source_mode": "offline_cache" if args.offline else "provider_with_cache"}
    atomic_json(manifest_path, attempt)
    try:
        fastf1.Cache.enable_cache(str(cache))
        fastf1.Cache.offline_mode(args.offline)
        session = fastf1.get_session(args.year, args.event, "R")
        session.load(telemetry=True, weather=True, messages=False)
        if session.laps.empty:
            raise RuntimeError("FastF1 returned no laps")
        session_id, frames, diagnostics = extract_session(session, identity)
        errors = check_tables(frames, identity, session_id)
        if errors:
            raise ValueError("; ".join(errors))
        for kind, frame in frames.items():
            atomic_csv(out / f"{prefix}_{kind}.csv", frame)
        manifest = build_manifest(out, prefix, identity, session_id, frames,
                                  attempt["source_mode"], diagnostics, started)
        manifest["provider_event"] = str(session.event.EventName)
        atomic_json(manifest_path, manifest)
        print(f"Completed {session_id}: {len(frames['next_lap_ml'])} pairs; manifest {manifest_path}")
        return 0
    except Exception as exc:
        attempt.update(status="failed", failed_utc=utc_now(),
                       error=f"{type(exc).__name__}: {exc}")
        atomic_json(manifest_path, attempt)
        print(f"EXPORT FAILED: {attempt['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
