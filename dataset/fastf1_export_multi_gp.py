"""Batch-export multiple Grand Prix races for the MMA3001 F1 ML project.

REQUIRES: fastf1_export_dataset.py in the same folder.

Install:
    python -m pip install --upgrade fastf1 pandas numpy

Quick test: 3 circuits x 2 drivers x 1 year:
    python fastf1_export_multi_gp.py --years 2024 --events Bahrain Monaco Japan --drivers VER HAM

More history from the same tracks:
    python fastf1_export_multi_gp.py --years 2023 2024 2025 --events Bahrain Monaco Japan

All Grands Prix from 2023-2025 (slow, potentially substantial data):
    python fastf1_export_multi_gp.py --years 2023 2024 2025

This script downloads REAL historical data only when run on a computer with
network access, and produces merged CSV files under f1_multi_export/.
It uses the original extractor's leakage-aware previous-lap features.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import fastf1
import pandas as pd


def slug(value):
    """Keep filename formatting consistent with the original exporter."""
    return str(value).lower().replace(" ", "_")


def get_races(year, filters):
    """Return scheduled race rounds, optionally filtered by country/location/name."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    races = []
    for _, event in schedule.iterrows():
        round_number = pd.to_numeric(event.get("RoundNumber"), errors="coerce")
        if pd.isna(round_number) or int(round_number) < 1:
            continue
        name = str(event.get("EventName", "")).strip()
        if not name or name.lower() == "nan":
            continue
        searchable = " ".join(
            str(event.get(field, "")) for field in ("EventName", "Country", "Location")
        ).lower()
        if filters and not any(token.lower() in searchable for token in filters):
            continue
        races.append((int(round_number), name))
    return races


def concat_files(entries, suffix, destination):
    """Concatenate race exports with year/event identifiers; ignore missing files."""
    frames = []
    for entry in entries:
        path = entry["folder"] / f'{entry["prefix"]}_{suffix}.csv'
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
            if frame.empty:
                continue
            if "Year" not in frame.columns:
                frame.insert(0, "Year", entry["year"])
            if "Event" not in frame.columns:
                frame.insert(1, "Event", entry["event"])
            if "RoundNumber" not in frame.columns:
                frame.insert(2, "RoundNumber", entry["round"])
            frames.append(frame)
        except Exception as exc:
            print(f'WARNING: Could not read {path}: {exc}')
    if not frames:
        print(f'No data available to create {destination.name}')
        return 0
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(destination, index=False)
    print(f'SAVED {destination}: {len(result):,} records, {len(frames)} race exports')
    return len(result)


def main():
    parser = argparse.ArgumentParser(description="Download and combine many FastF1 races")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025],
                        help="Seasons to extract, e.g. --years 2023 2024 2025")
    parser.add_argument("--events", nargs="*", default=[],
                        help="Optional event filters, e.g. Bahrain Monaco Japan; omit for ALL GPs")
    parser.add_argument("--drivers", nargs="*", default=None,
                        help="Optional abbreviations for trial runs, e.g. VER HAM")
    parser.add_argument("--max-races-per-year", type=int, default=None,
                        help="Download at most this many selected races in each season")
    parser.add_argument("--output", default="f1_multi_export", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Re-export existing race CSVs")
    args = parser.parse_args()

    if args.max_races_per_year is not None and args.max_races_per_year < 1:
        parser.error("--max-races-per-year must be a positive integer")

    extractor = Path(__file__).with_name("fastf1_export_dataset.py")
    if not extractor.exists():
        parser.error("Place fastf1_export_dataset.py beside fastf1_export_multi_gp.py")

    # Keep small driver trials separate from full-driver data: no accidental mixing.
    driver_codes = sorted(set(d.upper() for d in args.drivers)) if args.drivers else []
    subset_name = "drivers_" + "_".join(driver_codes) if driver_codes else "all_drivers"
    root = Path(args.output).resolve() / subset_name
    races_folder = root / "races"
    races_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output: {root}")
    print("Retrieving event schedules...")

    selected = []
    for year in args.years:
        try:
            races = get_races(year, args.events)
        except Exception as exc:
            print(f"WARNING: Could not obtain {year} schedule: {exc}")
            continue
        if args.max_races_per_year is not None:
            races = races[:args.max_races_per_year]
        print(f"{year}: {len(races)} matching races")
        for round_number, event_name in races:
            selected.append({"year": year, "round": round_number, "event": event_name,
                             "prefix": f"{year}_{slug(event_name)}", "folder": races_folder})

    if not selected:
        raise SystemExit("No matching races. Check season, --events spelling and internet access.")

    print(f"Processing {len(selected)} race sessions. Large selections may take a long time.")
    manifest = []
    for i, entry in enumerate(selected, start=1):
        year, event = entry["year"], entry["event"]
        prefix = entry["prefix"]
        target = races_folder / f"{prefix}_next_lap_ml.csv"
        all_laps = races_folder / f"{prefix}_all_laps.csv"
        print(f"\n[{i}/{len(selected)}] {year} round {entry['round']}: {event}", flush=True)
        if not args.overwrite and (target.exists() or all_laps.exists()):
            status = "cached"
            print("Existing export found; skipping. Add --overwrite to regenerate.")
        else:
            cmd = [sys.executable, str(extractor), "--year", str(year),
                   "--event", event, "--output", str(races_folder)]
            if driver_codes:
                cmd.extend(["--drivers", *driver_codes])
            try:
                subprocess.run(cmd, check=True)
                status = "success" if all_laps.exists() else "no_output"
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: Export failed for {year} {event}: exit code {exc.returncode}")
                status = "failed"
        records = 0
        if target.exists():
            try:
                records = sum(1 for _ in target.open(encoding="utf-8")) - 1
            except OSError:
                records = 0
        manifest.append({"Year": year, "RoundNumber": entry["round"],
                         "Event": event, "Status": status, "MLRows": max(records, 0)})

    pd.DataFrame(manifest).to_csv(root / "export_summary.csv", index=False)
    concat_files(selected, "next_lap_ml", root / "combined_next_lap_ml.csv")
    concat_files(selected, "all_laps", root / "combined_all_laps.csv")
    concat_files(selected, "weather", root / "combined_weather.csv")
    concat_files(selected, "race_results", root / "combined_race_results.csv")
    print("\nFinished. Review export_summary.csv for missing or failed races.")
    print("Keep test races or seasons held out when training models.")
    print("Check timing-data licensing before publishing raw CSVs on GitHub.")


if __name__ == "__main__":
    main()
