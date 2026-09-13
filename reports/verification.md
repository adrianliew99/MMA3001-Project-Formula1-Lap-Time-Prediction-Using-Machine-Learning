# Executed verification and pilot findings

Date: 2026-09-13. Environment: Windows, Python 3.12.14, FastF1 3.8.3, pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1. Full versions: `requirements-lock.txt`.

## Actual checks

- `uv --cache-dir .uv-cache pip install --python .venv-project/Scripts/python.exe -e .`: succeeded after dependencies were downloaded into the local project environment.
- `.venv-project/Scripts/python.exe -m pytest --junitxml=reports/tests.xml -q`: **7 passed**. Tests cover total seconds including >24 hours, missing/negative durations, preserved source rows, duplicates, missing quality fields, pit exclusions, consecutive same-stint histories, target-time leakage, chronological split rejection, training-only scaling and baseline metric execution. Values are synthetic fixtures, not research results.
- `.venv-project/Scripts/python.exe scripts/run_pilot.py`: successful real Bahrain 2024 race retrieval, source/prepared CSVs, manifest, weather export, quality JSON and two figures.
- `.venv-project/Scripts/python.exe scripts/build_notebook.py`: generated 18 cells (9 code cells).
- `.venv-project/Scripts/python.exe scripts/execute_notebook.py`: all **9 code cells executed successfully**, saving real outputs inside the notebook. Training and final test cells correctly report disabled status. Training code was exercised separately by the synthetic unit test, not by a real multi-race experiment.
- `.venv-project/Scripts/python.exe -m pdoc mma3001_f1 -o docs/html`: generated HTML documentation.
- `nbformat.validate`: passed. Race-pace, tyre-stint and least-squares PNGs visually inspected.

The first synthetic fixture used an ambiguous pit-time dtype, producing a pandas warning; it was corrected to explicit timedelta. NumPy/pandas timedelta deprecation warnings and Windows Jupyter event-loop/kernel notices were observed; execution still completed. No claim of a warning-free dependency stack is made. Jupyter's browser interface launch was not manually tested; the same installed kernel executed the notebook successfully.

## Real data findings

| Measure | Observed |
|---|---:|
| Lap rows | 1,129 |
| Drivers | 20 |
| Eligible laps under proposed rules | 988 |
| Three-lap history + next-lap candidates | 744 |
| Weather observations, not joined | 157 |
| Missing lap times | 2 |
| Duplicate lap identifiers | 0 |
| Pit-related flagged laps | 86 |
| Non-green flagged laps | 42 |
| Inaccurate flagged laps | 105 |
| Deleted flagged laps | 20 |
| Generated laps | 0 |
| Unavailable requested quality fields | 0 |

Flag counts overlap. Sequential exclusions remove 2 invalid times, 84 additional pit laps, 37 additional non-green laps and 18 additional deleted laps, leaving 988. See the JSON audit for full missingness and coverage.

The pace charts show large excursions around pit laps and distinct pace levels between stints. The numerical example for VER's first stint yields an approximately +0.0315 s per tyre-lap association in the selected clean data. This is descriptive only: fuel/traffic/management are not controlled, and there is no independent physical-wear validation.

## Remaining scope

Phase 1 data feasibility works locally. Teaching-team approval, final topic/model selection, multi-race roles, actual predictive training, untouched final evaluation, GitHub publication and the student's own interpretation/reflection remain outstanding. The optional ridge pipeline and its parameters are ready for the next stage; no CNN result or generalisable performance claim exists.
