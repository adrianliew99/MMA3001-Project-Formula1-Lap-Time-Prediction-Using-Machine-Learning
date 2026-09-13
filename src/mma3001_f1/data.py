"""Retrieve source laps, preserve provenance, and audit explicit quality rules."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import json
import re
import numpy as np
import pandas as pd

KEYS = ["SessionID", "Driver", "LapNumber"]
QUALITY = ["PitInTime", "PitOutTime", "TrackStatus", "IsAccurate", "Deleted", "FastF1Generated"]


def seconds(values):
    """Convert timedelta-like values to total seconds; missing values stay NaN."""
    return pd.to_timedelta(values, errors="raise").dt.total_seconds()


def prepare_laps(raw, session_id):
    """Copy source rows, add seconds and flags; never silently remove observations.

    Conservative eligibility requires all quality fields to be present and known.
    TrackStatus '1' means only green running. Dry compounds are SOFT/MEDIUM/HARD.
    """
    d = pd.DataFrame(raw).copy().reset_index(drop=True)
    for col in ["Driver", "LapNumber", "LapTime"]:
        if col not in d:
            raise ValueError(f"Required source field absent: {col}")
    d["SourceRow"] = np.arange(len(d))
    d["SessionID"] = session_id
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        d[col + "Seconds"] = seconds(d[col]) if col in d else np.nan
    for col in QUALITY + ["Compound", "TyreLife", "Stint"]:
        if col not in d:
            d[col] = pd.NA
    d["FlagDuplicate"] = d.duplicated(KEYS, keep=False)
    d["FlagInvalidID"] = d[KEYS].isna().any(axis=1) | d.LapNumber.le(0) | d.LapNumber.mod(1).ne(0)
    d["FlagInvalidTime"] = ~np.isfinite(d.LapTimeSeconds) | d.LapTimeSeconds.le(0)
    d["FlagPit"] = d.PitInTime.notna() | d.PitOutTime.notna()
    d["FlagInterrupted"] = d.TrackStatus.astype("string").ne("1").fillna(True)
    d["FlagInaccurate"] = ~d.IsAccurate.eq(True).fillna(False)
    d["FlagDeleted"] = d.Deleted.eq(True).fillna(False)
    d["FlagGenerated"] = d.FastF1Generated.eq(True).fillna(False)
    unknown = pd.Series(False, index=d.index)
    for col in ["TrackStatus", "IsAccurate", "Deleted", "FastF1Generated"]:
        unknown |= d[col].isna()
    if any(col not in raw.columns for col in ["PitInTime", "PitOutTime"]):
        unknown[:] = True
    d["FlagUnknownQuality"] = unknown
    d["FlagUnsupportedTyre"] = ~d.Compound.isin(["SOFT", "MEDIUM", "HARD"]) | d.TyreLife.isna() | d.TyreLife.lt(0) | d.Stint.isna()
    flags = [c for c in d if c.startswith("Flag")]
    d["Eligible"] = ~d[flags].any(axis=1)
    return d


def audit_laps(d, source_columns=None):
    """Return JSON-safe counts, overlapping flags, and sequential exclusion effects."""
    keep = pd.Series(True, index=d.index)
    effects = []
    for flag in [c for c in d if c.startswith("Flag")]:
        before = int(keep.sum())
        keep &= ~d[flag]
        effects.append({"rule": flag, "flagged": int(d[flag].sum()), "newly_excluded": before-int(keep.sum()), "remaining": int(keep.sum())})
    return {"rows": len(d), "drivers": int(d.Driver.nunique()), "eligible_rows": int(d.Eligible.sum()),
            "source_columns": list(source_columns if source_columns is not None else d.columns),
            "unavailable_quality_fields": [c for c in QUALITY if source_columns is not None and c not in source_columns],
            "missing": {c: int(d[c].isna().sum()) for c in d.columns},
            "nonpositive_durations": {c: int(d[c].le(0).sum()) for c in d if c.endswith("Seconds")},
            "lap_coverage": d.groupby("Driver").LapNumber.agg(["min", "max", "count", "nunique"]).to_dict("index"),
            "exclusion_waterfall": effects}


def collect_session(root, year=2024, event="Bahrain", session="R"):
    """Load only laps and optional weather; export source CSV, audit, and manifest.

    Raises on failed/empty lap retrieval. No synthetic fallback is permitted.
    FastF1 caching reduces repeat network requests. Weather is not joined to laps.
    """
    import fastf1
    root = Path(root)
    for folder in ["cache", "data/raw", "data/processed", "reports", "results/figures", "results/tables"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(root / "cache"))
    s = fastf1.get_session(year, event, session)
    s.load(laps=True, telemetry=False, weather=True, messages=True)
    raw = pd.DataFrame(s.laps).copy()
    if raw.empty:
        raise RuntimeError("FastF1 returned no laps; no dataset was generated.")
    sid = f"{year}-{int(s.event.RoundNumber):02d}-{session}"
    raw.to_csv(root / f"data/raw/{sid}_laps.csv", index=False)
    prepared = prepare_laps(raw, sid)
    prepared["Year"] = year
    prepared["Event"] = s.event.EventName
    prepared["Session"] = session
    prepared["EventDate"] = str(s.event.EventDate)
    prepared.to_csv(root / f"data/processed/{sid}_laps.csv", index=False)
    weather = getattr(s, "weather_data", None)
    weather_rows = 0
    if weather is not None and len(weather):
        weather_rows = len(weather)
        weather.to_csv(root / f"data/raw/{sid}_weather.csv", index=False)
    audit = audit_laps(prepared, raw.columns)
    (root / f"reports/{sid}_quality.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {"source": "FastF1 / F1 timing data", "session_id": sid, "year": year, "event": str(s.event.EventName),
                "session": session, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "versions": {p: version(p) for p in ["fastf1", "pandas", "numpy"]},
                "rows": len(raw), "weather_rows": weather_rows,
                "transformations": "Copy all rows; SourceRow zero-based; timedelta total_seconds; explicit conservative quality flags; no weather join.",
                "quality_report": f"reports/{sid}_quality.json"}
    (root / f"data/raw/{sid}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return prepared, audit, manifest


def plot_pilot(d, output, drivers=("VER", "LEC")):
    """Save pace and stint figures; flagged laps stay visible in the pace plot."""
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(drivers), 1, figsize=(11, 3.3*len(drivers)), squeeze=False, sharex=True)
    for ax, driver in zip(axes.flat, drivers):
        g = d[d.Driver.eq(driver)].sort_values("LapNumber")
        ax.plot(g.LapNumber, g.LapTimeSeconds, ".-", linewidth=1, label="All recorded laps")
        bad = g[~g.Eligible]
        ax.scatter(bad.LapNumber, bad.LapTimeSeconds, marker="x", color="crimson", label="Outside proposed clean population", zorder=3)
        ax.set(title=driver, ylabel="Lap time (s)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Original lap number")
    fig.suptitle(f"{d.SessionID.iloc[0]} | Race pace and quality flags")
    fig.tight_layout()
    fig.savefig(output / "lap_times.png", dpi=150)
    fig2, ax = plt.subplots(figsize=(11, 5))
    palette = {"SOFT": "#e74c3c", "MEDIUM": "#e4b400", "HARD": "#8b99aa", "INTERMEDIATE": "#38a169", "WET": "#3182ce"}
    names = sorted(d.Driver.dropna().unique())
    for y, driver in enumerate(names):
        for (_, compound), g in d[d.Driver.eq(driver)].groupby(["Stint", "Compound"]):
            ax.barh(y, g.LapNumber.max()-g.LapNumber.min()+1, left=g.LapNumber.min()-.5,
                    color=palette.get(compound, "black"), height=.7, edgecolor="white", linewidth=.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=palette.get(c, "black"), label=c) for c in sorted(d.Compound.dropna().unique())], loc="upper left", bbox_to_anchor=(1, 1))
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Lap number", title="Recorded tyre stints (span does not imply complete lap coverage)")
    fig2.tight_layout()
    fig2.savefig(output / "tyre_stints.png", dpi=150)
    return fig, fig2
