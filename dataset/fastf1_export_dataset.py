"""Export real FastF1 session data and a leakage-aware next-lap ML dataset.

Install: python -m pip install --upgrade fastf1 pandas numpy
Usage: python fastf1_export_dataset.py --year 2024 --event Bahrain --drivers VER HAM
       python fastf1_export_dataset.py --year 2024 --event Bahrain

All telemetry features describe the PREVIOUS completed lap, not the target lap.
Requires Internet access on the machine running this script.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--event", default="Bahrain", help="Event, e.g. Bahrain or Monaco")
    parser.add_argument("--drivers", nargs="*", default=None,
                        help="Optional driver abbreviations for quick test: VER HAM")
    parser.add_argument("--output", default="f1_export")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "fastf1_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))

    print(f"Downloading {args.year} {args.event} race (first run may take time)...")
    session = fastf1.get_session(args.year, args.event, "R")
    session.load(telemetry=True, weather=True, messages=False)
    if session.laps.empty:
        raise RuntimeError("No laps returned; check event name and upstream availability.")

    prefix = f"{args.year}_{args.event.lower().replace(' ', '_')}"
    raw = session.laps.copy()
    raw["LapTimeSeconds"] = raw["LapTime"].dt.total_seconds()
    raw.to_csv(out / f"{prefix}_all_laps.csv", index=False)
    session.weather_data.to_csv(out / f"{prefix}_weather.csv", index=False)
    session.results.to_csv(out / f"{prefix}_race_results.csv", index=False)
    print(f"Exported {len(raw)} raw lap records")

    drivers = [str(d).upper() for d in args.drivers] if args.drivers else sorted(
        raw["Driver"].dropna().unique().tolist()
    )
    rows = []
    example_saved = False
    for driver in drivers:
        dr = session.laps.loc[session.laps["Driver"] == driver].sort_values("LapNumber")
        count = 0
        for i in range(len(dr) - 1):
            prev, nxt = dr.iloc[i], dr.iloc[i + 1]
            if not (good_lap(prev) and good_lap(nxt)):
                continue
            # Avoid pairing laps separated by pit stops, gaps or new tyre stints.
            if nxt["LapNumber"] != prev["LapNumber"] + 1:
                continue
            if nxt["Stint"] != prev["Stint"]:
                continue

            try:
                features, tel = telemetry_features(prev)
            except Exception as exc:
                print(f"Skipping {driver} lap {prev['LapNumber']} telemetry: {exc}")
                continue
            if features is None:
                continue

            prev_time = prev["LapTime"].total_seconds()
            next_time = nxt["LapTime"].total_seconds()
            row = {
                "Year": args.year,
                "Event": args.event,
                "Session": "Race",
                "Driver": driver,
                "PreviousLapNumber": int(prev["LapNumber"]),
                "TargetLapNumber": int(nxt["LapNumber"]),
                "Stint": prev["Stint"],
                "PreviousLapTime_s": prev_time,
                "TyreCompound": prev["Compound"],
                "TyreAge_laps": prev["TyreLife"],
                "TrackTemp_C": recent_track_temperature(session.weather_data, prev["Time"]),
                **features,
                "NextLapTime_s": next_time,
                "DeltaLapTime_s": next_time - prev_time,
            }
            rows.append(row)
            count += 1
            if not example_saved:
                tel.to_csv(out / f"{prefix}_example_raw_telemetry.csv", index=False)
                example_saved = True
        print(f"{driver}: {count} consecutive valid green-flag training pairs")

    if not rows:
        print("No clean next-lap pairs found. Raw lap, result and weather CSVs were still saved.")
        print("Try a different event or examine IsAccurate and TrackStatus filtering.")
        return

    df = pd.DataFrame(rows)
    dest = out / f"{prefix}_next_lap_ml.csv"
    df.to_csv(dest, index=False)
    print(f"\nSaved {len(df)} ML examples to: {dest.resolve()}")
    print("Columns:", ", ".join(df.columns))
    print(df.head(5).to_string(index=False))
    print("\nImportant: Do not randomly split adjacent laps or races for testing.")
    print("Use separate race events or seasons for validation/test. Do not commit")
    print("large data/cache files publicly without checking data licensing.")


if __name__ == "__main__":
    main()
