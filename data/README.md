# Data provenance and units

FastF1 accesses F1 timing data. `raw/<year>-<round>-<session>_laps.csv` preserves the provider lap table as CSV. Timedeltas are strings in that export; prefer `collect_session` to rebuild prepared tables instead of guessing CSV types. The corresponding manifest records retrieval UTC, versions, event and transformations. The FastF1 cache is excluded from Git. Raw exports are replaced on explicit reruns of the same session; archive a version separately if you need an immutable research snapshot.

Processed files preserve source row order and add `SourceRow` (zero-based source position), `SessionID`, event metadata, seconds, flags and `Eligible`. The tracing key is SessionID + Driver + original LapNumber; duplicates are flagged on every affected row. No row is removed from the processed export.

| Field | Meaning / units |
|---|---|
| Driver, LapNumber | Provider driver abbreviation; original lap count |
| LapTimeSeconds, Sector1TimeSeconds … Sector3TimeSeconds | Total elapsed seconds; missing stays missing |
| Compound | Reported tyre compound; proposed population allows SOFT/MEDIUM/HARD |
| TyreLife | Reported tyre age in laps, not physical wear or remaining life |
| Stint | Provider stint identifier |
| PitInTime / PitOutTime | Session-relative timing of pit events; missing normally means no event |
| TrackStatus | Provider concatenated status codes; only exactly '1' accepted as green |
| IsAccurate / Deleted / FastF1Generated | Provider quality indicators, retained alongside explicit flags |
| Eligible | Conservative analysis population, not proof of independent accuracy |
| weather Time | Session-relative observation timestamp; no weather join implemented |

Quality JSON reports list source columns, unavailable quality fields, missing counts, duplicate flags, duration checks, coverage and sequential exclusions. Counts of different flags overlap. Weather availability does not establish lap-time alignment. Training windows additionally require consecutive original lap numbers and one stint for both history and target.

Underlying data ownership and reuse terms are distinct from FastF1's software licence. Use locally for feasibility, cite FastF1 and timing sources, and check applicable provider terms with the teaching team before redistribution. No broad dataset redistribution licence is asserted.
