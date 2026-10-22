# MMA3001: Formula 1 Next-Lap-Time Prediction

Predict lap n+1 duration at the end of lap n, conditional on two eligible consecutive
laps in the same stint. Compare persistence, linear regression, decision trees and
random forests with context and previous-lap telemetry features. No CNN is used in
the current validated experiment.

The engineering use is short-horizon race-pace estimation and assessment of whether
added model complexity improves prediction. This is a retrospective clean-target
study, not a validated pit-strategy optimiser or an incident/neutralisation forecast.
The evaluated exports include intermediate tyres and unknown compounds; they are
not a strictly dry-tyre population.

## Frozen results

| Partition | Seasons | Lap pairs |
|---|---|---:|
| Training | 2022-2023 | 35,872 |
| Validation and selection | 2024 | 21,394 |
| Independent holdout | 2025 | 20,662 |

Persistence (`next = previous`) won on 2024 validation MAE **0.464507 s**. Its frozen
2025 MAE is **0.386217 s**, RMSE **0.679810 s**, and pooled next-lap R² **0.995753**.
High pooled R² partly reflects large between-circuit lap-time differences; it does
not prove accurate prediction of lap-to-lap changes. Do not tune using 2025 errors.
The best telemetry tree has validation MAE 0.469607 s and RMSE 0.879105 s: lower
RMSE but higher MAE than persistence. The selection criterion remains MAE.

## Setup

Use Python 3.12 from the repository root. A virtual environment is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt -c constraints-reproduced.txt
python -m pytest --junitxml=reports/pytest.xml
```

`requirements.txt` gives supported version ranges. `constraints-reproduced.txt`
captures the measured Linux audit environment, not a universal lock file. The
original validated notebook used Windows/Python 3.12.10, pandas 2.3.3 and sklearn
1.8.0. The audit uses Python 3.12.14/pandas 2.2.3/sklearn 1.8.0; see reproduction
and benchmark manifests for evidence of numerical agreement. FastF1 is needed
only for exporter commands; testing and modelling use existing CSVs offline.

## Review and reproduce safely

```bash
python -m tools.reproduce
python -m tools.benchmark
python -m tools.build_docs
python -m tools.build_reports
```

The reproduction command executes the notebook's code in an isolated temporary
root and compares the regenerated holdout predictions with the frozen record.
It blocks network connections and checks that the original results remain
unchanged. It is a numerical repeat, not a new independent holdout evaluation.
The benchmark fits existing validation-selected representatives on 2022-23 and
measures 2024 only; it never retunes on 2025. Timings are machine-dependent.

To explore interactively, launch `python -m jupyterlab` and open
`MMA3001_F1_Next_Lap_Prediction.ipynb`. The historical notebook has
`RUN_FINAL_TEST=True` and writes under `results/`; use the isolated command above
for review, or first work in a separate clone/output root. Network downloading
remains disabled by default. Section 17 reads the separate performance evidence.

## Input and output contract

| Fields | Format and meaning | Handling |
|---|---|---|
| Year, Event, Session, Driver | CSV identifiers for a race and driver | Missing keys rejected; missing Session defaults to R |
| PreviousLapNumber, TargetLapNumber, Stint | Lap/stint identifiers, laps | Consecutive pair required; duplicates keep first; stint range 1-100 |
| PreviousLapTime_s, NextLapTime_s | Duration, seconds, 30-300 | Invalid/missing rejected; next lap is target only |
| DeltaLapTime_s | Next minus previous, seconds | Supplied mismatch rejected; otherwise reconstructed |
| TyreAge_laps, TyreCompound | Prior completed lap tyre age and label | Age outside 0-100 rejected; missing inputs imputed within training pipeline |
| TrackTemp_C | Prior observation, Celsius, -10 to 80 | Invalid present values rejected; missing imputed |
| AvgSpeed_kmh, MaxSpeed_kmh | Prior lap speed, km/h, 0-400 | Invalid present values rejected; missing imputed |
| FullThrottle_pct, Braking_pct | Prior lap summary, percent, 0-100 | Invalid present values rejected; missing imputed |
| PredictedDeltaLapTime_s | Predicted pace change, seconds | Added to previous lap time |
| PredictedNextLapTime_s | Main predicted duration, seconds | Residual = actual minus predicted; absolute error in seconds |

These describe the existing audit rules, not guarantees of exhaustive invalid-input
rejection. Fractional lap/stint identifiers and some implausible channel combinations
are not explicitly rejected by the notebook audit. Upstream export continuity and
quality flags support the historical dataset; deployment needs stronger validation.
Only the feature allowlist enters fitted pipelines. Target values/identifiers, year,
driver, source path and stint do not enter predictors. Preprocessors fit training
only. Unknown categories are handled by the encoder. Retrospective target quality
filtering cannot identify a future pit stop or incident at prediction time.

## Repository map

- `MMA3001_F1_Next_Lap_Prediction.ipynb`: assessed workflow, saved outputs and Section 17.
- `dataset/`: existing single-race and batch FastF1 exporters.
- `mma3001_f1/`: ingestion, integrity and earlier pilot helpers; `performance.py` adds timing utilities. `modeling.py` is an earlier scaffold, not the final notebook model.
- `f1_dataset/`: existing real exported CSVs, 92 races over 2022-25; no download needed for review.
- `results/`: frozen validation grid, final holdout predictions/metrics, engineering diagnostics and figures.
- `tests/`: offline regression and numerical tests using actual notebook definitions.
- `tools/`: isolated reproduction, repeated benchmarking, documentation generation.
- `reports/`: JUnit test report, reproduction summary and performance measurements. Raw local execution logs are ignored.
- `docs/html/index.html`: generated package/tool API; `docs/html/notebook.html`: active notebook functions.
- `docs/SUBMISSION_READINESS.md`: evidence-based assessment and final submission checklist.

Open `docs/html/notebook.html` in a browser to inspect active function docstrings
and source. Regenerate documentation after source changes. CI tests each push/PR
and uploads the JUnit report plus generated HTML. Local success does not mean
GitHub Actions has already run.

## Data provenance and scope

Data: [FastF1](https://github.com/theOehrly/Fast-F1), historical Formula 1 timing
and telemetry, exported to local CSV. Existing manifests record export provenance.
FastF1 software licensing does not transfer ownership of upstream timing data.
See `LICENSE` for the current no-grant notice and unresolved distribution terms.
The existing dataset is retained unchanged; do not assume this audit establishes
permission to redistribute it. Raw per-lap telemetry is sampled rather than fully
archived, so aggregate CSVs cannot independently prove every telemetry summary.

## AI use

The prior README lists ChatGPT, Cursor Pro and Google Gemini as development aids.
This continuation used ChatGPT/Codex for repository inspection, automated tests,
performance tools and submission documentation. Checks include hand-computable
numerical tests, frozen-artifact reconciliation and isolated notebook execution.
The student must confirm the earlier tool identities, actual contribution level,
important decisions, corrections and learning in their own final AI reflection.
No unverified percentage of AI contribution is claimed here.
