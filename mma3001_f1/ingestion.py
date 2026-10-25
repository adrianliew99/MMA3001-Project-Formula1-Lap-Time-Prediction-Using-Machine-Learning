"""Export integrity and provenance utilities; no provider access in this module."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
EXPORTER_VERSION = "phase1-v1"
ML_COLUMNS = [
    "Year", "Event", "Session", "Driver", "PreviousLapNumber", "TargetLapNumber",
    "Stint", "PreviousLapTime_s", "TyreCompound", "TyreAge_laps", "TrackTemp_C",
    "AvgSpeed_kmh", "MaxSpeed_kmh", "FullThrottle_pct", "Braking_pct",
    "NextLapTime_s", "DeltaLapTime_s",
]
REQUIRED = {
    "all_laps": ["Driver", "LapNumber", "LapTimeSeconds"],
    "weather": ["Time", "TrackTemp"],
    "race_results": ["DriverNumber"],
    "next_lap_ml": ML_COLUMNS + ["SessionID", "RoundNumber"],
}


def utc_now() -> str:
    """Return the actual UTC export time, not an inferred source retrieval time."""
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    """Create a safe filename component, preserving normal legacy event slugs."""
    return re.sub(r"[^a-z0-9_-]+", "_", str(value).lower()).strip("_")


def request_identity(year: int, event: str, drivers=None) -> dict:
    """Normalize scope; an empty driver list means all recorded drivers."""
    return {"year": int(year), "event": str(event), "session": "R",
            "drivers": sorted(set(str(d).upper() for d in (drivers or [])))}


def sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation_hash() -> str:
    """Invalidate exports when extraction/integrity implementation changes."""
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "dataset/fastf1_export_dataset.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    """Replace one JSON document atomically; publish completion only at the end."""
    _atomic_write(path, lambda stream: json.dump(payload, stream, indent=2, allow_nan=False))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Publish a CSV without exposing a partially written file."""
    _atomic_write(path, lambda stream: frame.to_csv(stream, index=False))


def _atomic_write(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    # Keep a failed write for diagnosis; no deletion of existing user files.
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        writer(stream)
    os.replace(name, path)


def inspect_frame(frame: pd.DataFrame, required: list[str]) -> dict:
    """Report availability and missingness; this is not Phase 2 feature QA."""
    return {"rows": len(frame), "columns": list(frame.columns),
            "missing_columns": sorted(set(required) - set(frame.columns)),
            "missing_values": {str(c): int(frame[c].isna().sum()) for c in frame},
            "drivers": sorted(frame["Driver"].dropna().astype(str).unique().tolist()) if "Driver" in frame else []}


def check_tables(frames: dict, identity: dict, session_id: str) -> list[str]:
    """Require usable tables and coherent event/pair identifiers before completion."""
    errors = []
    for kind, required in REQUIRED.items():
        frame = frames.get(kind)
        if frame is None or frame.empty:
            errors.append(f"{kind}: absent or empty")
        elif set(required) - set(frame):
            errors.append(f"{kind}: missing columns {sorted(set(required) - set(frame))}")
    if errors:
        return errors
    ml = frames["next_lap_ml"]
    if not ml["Year"].eq(identity["year"]).all() or not ml["Event"].eq(identity["event"]).all():
        errors.append("ML event/year does not match requested scope")
    if not ml["Session"].eq("Race").all() or not ml["SessionID"].eq(session_id).all():
        errors.append("ML session identity mismatch")
    try:
        year, round_number, session = session_id.split("-")
        if int(year) != identity["year"] or session != "R" or int(round_number) < 1:
            raise ValueError("invalid session ID")
        if not pd.to_numeric(ml["RoundNumber"], errors="coerce").eq(int(round_number)).all():
            errors.append("ML round number mismatch")
    except (ValueError, AttributeError):
        errors.append("invalid session ID")
    expected = set(identity["drivers"])
    observed = set(ml["Driver"].dropna().astype(str))
    if ml["Driver"].isna().any() or (expected and observed != expected):
        errors.append("requested drivers do not all have usable ML pairs or contain unexpected drivers")
    if not observed.issubset(set(frames["all_laps"]["Driver"].dropna().astype(str))):
        errors.append("ML drivers absent from raw laps")
    previous = pd.to_numeric(ml["PreviousLapNumber"], errors="coerce")
    target = pd.to_numeric(ml["TargetLapNumber"], errors="coerce")
    if not (previous.gt(0) & previous.mod(1).eq(0) & target.eq(previous + 1)).all():
        errors.append("nonconsecutive/invalid lap identifiers")
    times = ml[["PreviousLapTime_s", "NextLapTime_s", "DeltaLapTime_s"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(times.to_numpy()).all() or not times.iloc[:, :2].gt(0).all().all():
        errors.append("nonfinite/nonpositive lap times")
    elif not np.allclose(times.NextLapTime_s - times.PreviousLapTime_s, times.DeltaLapTime_s, rtol=0, atol=1e-8):
        errors.append("delta target identity failed")
    if ml.duplicated(["SessionID", "Driver", "PreviousLapNumber", "TargetLapNumber"]).any():
        errors.append("duplicate lap pairs")
    return errors


def build_manifest(folder: Path, prefix: str, identity: dict, session_id: str,
                   frames: dict, source_mode: str, diagnostics: dict, started: str) -> dict:
    """Describe actual exported files; cache-source download timestamps stay unknown."""
    errors = check_tables(frames, identity, session_id)
    if errors:
        raise ValueError("; ".join(errors))
    artifacts = {}
    for kind, frame in frames.items():
        path = folder / f"{prefix}_{kind}.csv"
        artifacts[kind] = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                           **inspect_frame(frame, REQUIRED.get(kind, []))}
    return {"schema_version": SCHEMA_VERSION, "exporter_version": EXPORTER_VERSION,
            "implementation_sha256": implementation_hash(), "status": "complete",
            "request": identity, "session_id": session_id, "started_utc": started,
            "completed_utc": utc_now(), "source": "FastF1 historical F1 timing data",
            "source_mode": source_mode, "original_provider_download_utc": None,
            "versions": {p: version(p) for p in ("fastf1", "numpy", "pandas")},
            "artifacts": artifacts, "diagnostics": diagnostics,
            "limitations": "Legacy clean-lap and telemetry rules retained; full feature/leakage QA is Phase 2. Target eligibility is retrospective."}


def validate_export(folder: Path, prefix: str, identity: dict) -> tuple[bool, str, dict | None]:
    """Accept reuse only with a matching complete manifest and verified artifacts."""
    path = folder / f"{prefix}_manifest.json"
    if not path.is_file():
        return False, "completion manifest missing (legacy/partial export)", None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            return False, "manifest status is not complete", manifest
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("exporter_version") != EXPORTER_VERSION:
            return False, "manifest version mismatch", manifest
        if manifest.get("implementation_sha256") != implementation_hash():
            return False, "extraction implementation changed", manifest
        if manifest.get("request") != identity:
            return False, "requested event/driver scope differs", manifest
        frames = {}
        for kind in REQUIRED:
            info = manifest["artifacts"][kind]
            artifact = folder / f"{prefix}_{kind}.csv"
            if info["path"] != artifact.name or not artifact.is_file():
                return False, f"{kind}: file missing/path mismatch", manifest
            if sha256(artifact) != info["sha256"]:
                return False, f"{kind}: checksum mismatch", manifest
            frame = pd.read_csv(artifact, low_memory=False)
            if len(frame) != info["rows"] or list(frame.columns) != info["columns"]:
                return False, f"{kind}: rows/schema mismatch", manifest
            frames[kind] = frame
        # Optional telemetry sample must also remain intact when recorded.
        sample = manifest["artifacts"].get("example_raw_telemetry")
        if sample:
            artifact = folder / f"{prefix}_example_raw_telemetry.csv"
            if sample["path"] != artifact.name or sha256(artifact) != sample["sha256"]:
                return False, "telemetry sample checksum/path mismatch", manifest
        errors = check_tables(frames, identity, manifest["session_id"])
        if errors:
            return False, "; ".join(errors), manifest
        return True, "complete manifest and artifacts verified", manifest
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return False, f"invalid export: {type(exc).__name__}: {exc}", None
