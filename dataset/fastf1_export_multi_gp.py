"""Reliable multi-Grand-Prix exporter for the MMA3001 F1 next-lap project.

Run from the repository root with::

    python -m dataset.fastf1_export_multi_gp --years 2024 --events Bahrain \
        --drivers VER HAM --max-sessions 1 --dry-run
    python -m dataset.fastf1_export_multi_gp --years 2022 2023 2024 2025 --events --drivers --max-sessions 100 --output .\

The child exporter runs as a MODULE with cwd set to the repository root,
so sibling package ``mma3001_f1`` can be imported without PYTHONPATH hacks.

Only completed exports with nonempty valid next-lap CSVs enter aggregates.
Never deletes old exports. Each requested race gets its own output folder.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import fastf1
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR_MODULE = "dataset.fastf1_export_dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "f1_multi_export_v2"
DEFAULT_CACHE = PROJECT_ROOT / "f1_export" / "fastf1_cache"
LOG = logging.getLogger("mma3001.batch_export")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def slug(text: str) -> str:
    """Create an ASCII-safe directory token for a race name."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save a machine-readable manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically save CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def get_schedule(year: int, retries: int = 2) -> pd.DataFrame:
    """Fetch the real event schedule, retrying transient provider errors."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            if schedule is not None and not schedule.empty and "EventName" in schedule:
                return schedule
            last_error = RuntimeError(f"Schedule for {year} is empty or lacks EventName")
        except Exception as exc:  # provider errors are variable across FastF1 versions
            last_error = exc
        if attempt < retries:
            LOG.warning("Schedule %d attempt %d failed: %s", year, attempt + 1, last_error)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Cannot resolve schedule for {year}: {last_error}")


def get_races(year: int, filters: list[str]) -> list[dict[str, Any]]:
    """Resolve user-supplied venue/country/name filters to race rounds."""
    schedule = get_schedule(year)
    resolved: list[dict[str, Any]] = []
    matched: set[str] = set()
    for _, event in schedule.iterrows():
        round_no = pd.to_numeric(event.get("RoundNumber"), errors="coerce")
        if pd.isna(round_no) or round_no < 1 or float(round_no) % 1 != 0:
            continue
        event_name = str(event.get("EventName", "")).strip()
        if not event_name or event_name.lower() == "nan":
            continue
        searchable = " ".join(str(event.get(k, "")) for k in ("EventName", "Country", "Location")).lower()
        hits = {f for f in filters if f.casefold() in searchable.casefold()}
        if filters and not hits:
            continue
        matched |= hits
        resolved.append({"year": int(year), "round": int(round_no), "event": event_name})
    unknown = set(filters) - matched
    if unknown:
        raise ValueError(f"Unmatched event filters for {year}: {sorted(unknown)}")
    return sorted(resolved, key=lambda item: (item["year"], item["round"]))


def race_folder(output: Path, entry: dict[str, Any]) -> Path:
    """Return a unique destination for one race's CSVs and manifest."""
    return output / "races" / f"{entry['year']}_{entry['round']:02d}_{slug(entry['event'])}"


def validate_race(folder: Path, entry: dict[str, Any], drivers: list[str]) -> tuple[bool, str, Path | None, int]:
    """Verify completion manifest and nonempty ML CSV before reuse/merging.

    Also validate Year, Event, and selected Driver identifiers in the CSV to
    prevent cross-request reuse. The extraction manifest is kept untouched.
    """
    manifests = sorted(folder.glob("*_manifest.json"))
    if len(manifests) != 1:
        return False, f"expected one completion manifest, found {len(manifests)}", None, 0
    try:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read manifest: {exc}", None, 0
    if str(manifest.get("status", "")).lower() != "complete":
        return False, "manifest status is not complete", None, 0
    csvs = sorted(folder.glob("*_next_lap_ml.csv"))
    if len(csvs) != 1:
        return False, f"expected one ML CSV, found {len(csvs)}", None, 0
    try:
        frame = pd.read_csv(csvs[0], low_memory=False)
    except Exception as exc:
        return False, f"cannot read ML CSV: {exc}", None, 0
    required = {"Year", "Event", "Driver", "PreviousLapTime_s", "NextLapTime_s"}
    absent = required.difference(frame.columns)
    if absent:
        return False, f"missing ML columns: {sorted(absent)}", None, 0
    if frame.empty:
        return False, "ML CSV contains zero lap pairs", None, 0
    years = pd.to_numeric(frame["Year"], errors="coerce")
    if not years.eq(entry["year"]).all():
        return False, "Year column disagrees with requested race", None, 0
    if not frame["Event"].fillna("").astype(str).str.casefold().eq(entry["event"].casefold()).all():
        return False, "Event column disagrees with requested race", None, 0
    if drivers and not frame["Driver"].dropna().astype(str).str.upper().isin(drivers).all():
        return False, "Driver column has unrequested drivers", None, 0
    if (pd.to_numeric(frame["NextLapTime_s"], errors="coerce") <= 0).any():
        return False, "nonpositive next-lap time", None, 0
    # Do not require every requested driver to have valid pairs; DNFs can prevent that.
    return True, "verified complete export", csvs[0], len(frame)


def export_one(entry: dict[str, Any], drivers: list[str], output: Path, cache: Path,
               offline: bool, overwrite: bool) -> dict[str, Any]:
    """Run one extractor process, safely resuming only verified existing races."""
    folder = race_folder(output, entry)
    ok, reason, csv_path, rows = validate_race(folder, entry, drivers)
    if ok and not overwrite:
        return {**entry, "Status": "cached", "MLRows": rows, "Reason": reason,
                "CSV": str(csv_path), "Folder": str(folder)}
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        return {**entry, "Status": "incomplete", "MLRows": 0,
                "Reason": f"{reason}; retry with --overwrite or a different --output",
                "CSV": "", "Folder": str(folder)}
    folder.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", EXTRACTOR_MODULE,
               "--year", str(entry["year"]), "--event", entry["event"],
               "--output", str(folder), "--cache-dir", str(cache)]
    if drivers:
        command.extend(["--drivers", *drivers])
    if offline:
        command.append("--offline")
    if overwrite:
        command.append("--overwrite")
    LOG.info("Launching %s", " ".join(command))
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except OSError as exc:
        return {**entry, "Status": "failed", "MLRows": 0,
                "Reason": str(exc), "CSV": "", "Folder": str(folder)}
    ok, reason, csv_path, rows = validate_race(folder, entry, drivers)
    successful = completed.returncode == 0 and ok
    return {**entry, "Status": "success" if successful else "failed",
            "MLRows": rows if successful else 0,
            "Reason": reason if successful else f"extractor exit {completed.returncode}; {reason}",
            "CSV": str(csv_path) if successful else "", "Folder": str(folder)}


def combine_exports(records: list[dict[str, Any]], output: Path) -> int:
    """Combine verified ML CSVs and optional all-laps CSVs, never duplicating races."""
    accepted = [r for r in records if r["Status"] in ("success", "cached")]
    frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    for record in accepted:
        ml = pd.read_csv(record["CSV"], low_memory=False)
        ml["RoundNumber"] = record["round"]
        ml["SessionID"] = f"{record['year']}-{record['round']:02d}-R"
        frames.append(ml)
        raw_files = sorted(Path(record["Folder"]).glob("*_all_laps.csv"))
        if len(raw_files) == 1:
            raw = pd.read_csv(raw_files[0], low_memory=False)
            raw["Year"] = record["year"]
            raw["Event"] = record["event"]
            raw_frames.append(raw)
    # Do not write a misleading empty aggregate if not a single race completed.
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("Year", "Event", "Session", "Driver", "PreviousLapNumber", "TargetLapNumber") if c in combined]
    if keys and combined.duplicated(keys).any():
        raise ValueError("Duplicate lap pairs in aggregate; inspect overlapping source exports")
    write_csv(output / "combined_next_lap_ml.csv", combined)
    if raw_frames:
        write_csv(output / "combined_all_laps.csv", pd.concat(raw_frames, ignore_index=True))
    return len(combined)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI preserving existing argument names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--events", nargs="*", default=["Bahrain", "Monaco", "Japan"],
                        help="Name, country or location filters; empty --events means all races")
    parser.add_argument("--drivers", nargs="*", default=["VER", "HAM"],
                        help="Three-letter driver codes; empty --drivers means all drivers")
    parser.add_argument("--max-races-per-year", type=int, default=30)
    parser.add_argument("--max-sessions", type=int, default=3,
                        help="Explicit upper bound on number of race sessions to process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Permit per-race exporter to replace its own incomplete artifacts")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve the download budget, execute exports and summarize real outcomes."""
    args = build_parser().parse_args(argv)
    if args.max_sessions < 1 or args.max_races_per_year < 1:
        raise SystemExit("--max-sessions and --max-races-per-year must be positive")
    drivers = sorted({d.upper() for d in args.drivers})
    filters = list(dict.fromkeys(args.events))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selections: list[dict[str, Any]] = []
    try:
        for year in sorted(set(args.years)):
            selections.extend(get_races(year, filters)[:args.max_races_per_year])
    except (RuntimeError, ValueError) as exc:
        print(f"Schedule resolution failed: {exc}", file=sys.stderr)
        return 2
    if not selections:
        print("No matching races were resolved", file=sys.stderr)
        return 2
    if len(selections) > args.max_sessions:
        print(f"Resolved {len(selections)} sessions but limit is {args.max_sessions}. "
              "Raise --max-sessions explicitly or reduce --events/--years.", file=sys.stderr)
        return 2
    print(f"Resolved {len(selections)} sessions (limit {args.max_sessions}); drivers={drivers or 'ALL'}", flush=True)
    for item in selections:
        print(f"  {item['year']} round {item['round']}: {item['event']}", flush=True)
    if args.dry_run:
        print("Dry run only: no event data was downloaded or written.")
        return 0
    output = args.output.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "batch_manifest.json"
    records: list[dict[str, Any]] = []
    write_json(run_path, {"status": "running", "started_utc": utc_now(), "selection": selections,
                          "drivers": drivers, "offline": args.offline})
    for entry in selections:
        result = export_one(entry, drivers, output, cache, args.offline, args.overwrite)
        records.append(result)
        print(f"{entry['year']} {entry['event']}: {result['Status']}; {result['Reason']}", flush=True)
        write_csv(output / "export_summary.csv", pd.DataFrame(records))
    try:
        rows = combine_exports(records, output)
        merge_error = None
    except Exception as exc:
        rows, merge_error = 0, str(exc)
        LOG.error("Aggregate failed: %s", exc)
    good = sum(record["Status"] in ("success", "cached") for record in records)
    completed = good == len(selections) and merge_error is None and rows > 0
    write_json(run_path, {"status": "complete" if completed else "incomplete",
                          "finished_utc": utc_now(), "selection": selections, "drivers": drivers,
                          "completed_sessions": good, "requested_sessions": len(selections),
                          "combined_ml_rows": rows, "merge_error": merge_error,
                          "records": records})
    print(f"Batch {'complete' if completed else 'incomplete'}: {good}/{len(selections)} "
          f"usable sessions; combined ML rows={rows:,}; {output}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
