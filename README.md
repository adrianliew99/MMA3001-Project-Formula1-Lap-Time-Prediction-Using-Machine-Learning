# Formula 1 Next-Lap-Time Prediction

**MMA3001 — Numerical Methods and Machine Learning**  
**Project:** Comparative Evaluation of Machine Learning Models for Formula 1 Lap Time Prediction and Race Strategy Analysis
##### Can the lap time and sector trend, recent race history, tyre age and strategy predict the next lap time, how accurate it can be estimated 🤔


## About this project
This repositories records the development of individual project for Monash University School of Engineering study unit MMA3001: Numerical Methods and Machine Learning. It will bring together the project explanation, data preparation and process, computer code, evaluation, further improvement and discussion & findings. 

## Project Description
Formula 1 lap times depend on vehicle speed, throttle and braking behaviour, tyre conditions, track temperature and race circumstances. This project investigates whether machine-learning regression models can predict a driver's next lap time using information available at the end of the previous completed lap.
The primary deliverable is a reproducible comparison of simple and advanced prediction methods. A potential second stage will use lap-pace and tyre-age estimates to compare hypothetical pit-stop strategies. Race winner and DNF probability models are possible future extensions, not current outputs.

### Engineering Problem
Can historical Formula 1 lap timing, telemetry and racing conditions improve next-lap time predictions over simple baselines, and can these predictions inform tyre-strategy analysis?

## Input and Output
*Input*: Recent race finished position, recent lap time, tyre age and selection
*Output*: Next lap time prediction
*Validation*: Comparison with labelled assessment and how other factor influence

### Primary model inputs — lap `n`

| Field | Unit / role | Description |
|---|---|---|
| `PreviousLapTime_s` | seconds | Observed last completed lap |
| `PreviousLapNumber` | count | Index of last completed lap |
| `TyreCompound` | category | Known tyre compound |
| `TyreAge_laps` | laps | Tyre age at end of prior lap |
| `TrackTemp_C` | °C | Track temperature associated with previously completed lap |
| `Event` | category | Circuit / event context |
| `AvgSpeed_kmh` | km/h | Previous-lap time-weighted mean speed |
| `MaxSpeed_kmh` | km/h | Previous-lap maximum speed |
| `FullThrottle_pct` | % | Previous-lap high-throttle fraction |
| `Braking_pct` | % | Previous-lap braking fraction |

`Year`, `Session`, `Driver`, `Stint`, `TargetLapNumber` and identifiers are retained for audit, split and reporting. They are **not all model features**. Telemetry is used for a distinct ablation rather than being assumed necessary for every model.

### Target and exported prediction fields

| Field | Unit | Meaning |
|---|---|---|
| `DeltaLapTime_s` | seconds | Actual difference `NextLapTime_s - PreviousLapTime_s` |
| `NextLapTime_s` | seconds | Observed target-lap duration |
| `PredictedDeltaLapTime_s` | seconds | Modelled lap-to-lap change |
| `PredictedNextLapTime_s` | seconds | Reconstructed prediction |
| `Residual_s` | seconds | Actual minus predicted duration |
| `AbsoluteError_s` | seconds | Absolute residual |

The notebook checks identifiers, lap continuity, duplicates, plausibility and target arithmetic. Missing numeric data are imputed **after fitting the preprocessing pipeline on training data only**; unknown categorical values are handled by one-hot encoding with unknown-category support. Invalid/ineligible pairs are excluded and reported. Model inputs are transformed to scikit-learn-compatible numeric and categorical types to avoid pandas `pd.NA` errors.

**Scope qualification:** Clean-lap eligibility is determined retrospectively. This experiment estimates the next lap *conditional on clean, consecutive racing laps*; it cannot guarantee prediction quality for the next arbitrary lap or for events not known in advance (pit stops, neutralisations, retirements).

## Version
Current version: v2.0

## Data Source
Primary Source is obtained through [FastF1]https://github.com/theOehrly/Fast-F1, an open-source Python providing access to previous Formula 1 racing data

## Dataset Description
Data exported: All Drivers at all circuits for Year 2022, 2023, 2024 & 2025\
The retrieved data is exported into .csv dataset format\
**Dataset type:** Historical Formula 1 lap timing and session data

### Extracted Features

| Category | Features | Description |
|---|---|---|
| Driver & Team | `Driver`, `DriverNumber`, `Team`, `Position` | Driver identification, team affiliation, and recorded position |
| Lap Information | `LapNumber`, `LapTime`, `LapTimeSeconds`, `Time`, `LapStartTime`, `LapStartDate` | Lap sequence, lap duration, and timing information |
| Sector Timing | `Sector1Time`, `Sector2Time`, `Sector3Time`, `Sector1SessionTime`, `Sector2SessionTime`, `Sector3SessionTime` | Individual sector durations and corresponding session timestamps |
| Speed Measurements | `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST` | Recorded speeds at intermediate points, finish line, and speed trap (km/h) |
| Tyre & Stint | `Compound`, `TyreLife`, `FreshTyre`, `Stint` | Tyre compound, tyre age, tyre condition, and stint number |
| Pit Stop Information | `PitInTime`, `PitOutTime` | Session timestamps associated with pit entry and exit |
| Race & Track Conditions | `TrackStatus`, `IsPersonalBest` | Track status codes and personal-best lap indicator |
| Data Quality | `IsAccurate`, `Deleted`, `DeletedReason`, `FastF1Generated` | Flags describing lap accuracy, deletion status, and generated records |
| Session Metadata | `SessionID`, `Year`, `Event` | Session identification, season, and Grand Prix event |

**References:**
- [FastF1 GitHub Repository](https://github.com/theOehrly/Fast-F1)
- [FastF1 Official Documentation](https://docs.fastf1.dev/)

FastF1 is used to retrieve racing data. The original timing data is not owned by this project. Any additional preprocessing, feature engineering, and machine learning implementation will be documented separately.

### Dataset Export

The successful run selected all Grand Prix races and all drivers across 2022–2025. In the current batch CLI, passing `--events` and `--drivers` without values explicitly selects all events and all drivers:

```powershell
python -m dataset.fastf1_export_multi_gp --years 2022 2023 2024 2025 --events --drivers --max-sessions 100 --output .\f1_dataset
```

### Dataset Split
Drivers - All Drivers
Circuits - All Circuits

| Data type | Year |
|---|---|
| Training seasons | 2022 & 2023 |
| Validation season | 2024 |
| Frozen independent test | 2025 |

## Repository layout

```text
mmaproject/
├── dataset/
│   ├── fastf1_export_dataset.py      # Single Grand Prix exporter
│   └── fastf1_export_multi_gp.py     # Multi-season resumable batch exporter
├── mma3001_f1/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── ingestion.py
│   └── modeling.py
├── MMA3001_F1_Next_Lap_Prediction.ipynb
├── requirements.txt
├── f1_dataset/                       # Generated locally; generally excluded from Git
│   ├── races/
│   ├── batch_manifest.json
│   ├── export_summary.csv
│   └── combined_next_lap_ml.csv
├── f1_export/                        # Optional local FastF1 cache and single-race exports
│   └── fastf1_cache/
└── results/                          # Generated analysis outputs
    ├── figures/
    ├── metrics/
    ├── models/
    └── predictions/
``` 

## Model comparison and experimental design

| Variant | Input information | Method |
|---|---|---|
| A — persistence | Previous completed lap time | Predict zero lap-time change |
| B — context | Previous pace, tyre, temperature, lap index, event | Linear regression / tree / forest |
| C — telemetry | All B features plus previous-lap speed, throttle and braking | Linear regression / tree / forest |

For trained models the response is `DeltaLapTime_s`. Predictions are reconstructed as `PreviousLapTime_s + predicted change` so all methods are evaluated against **identical next-lap targets**.

- Split by complete seasons: **train 2022–2023; validate 2024; hold out 2025**. Never randomly split individual race laps.
- Fit encoders, scalers, imputers and regressors on training rows; select hyperparameters and feature family using validation **MAE**.
- Persistence is eligible to win the selection process.
- Compare MAE, RMSE, next-lap-time R² and runtimes; telemetry context comparisons are **predictive**, not causal tests.
- Do not tune on the 2025 holdout after revealing its scores.

### Current observed results

| Metric | Selected persistence model on 2025 |
|---|---:|
| Mean absolute error | **0.3862 s** |
| Root mean squared error | **0.6798 s** |
| R² of next-lap time | **0.9958** |
| Mean absolute delta error | **0.3862 s** |
| MAE improvement over persistence | **0.00%** |

The same baseline and selected-model rows are intentionally identical: validation-based selection picked the baseline. This is a meaningful negative complexity result and should **not** be replaced by an ML model merely to claim improvement.


### AI acknowledgement
LLMs model (Chatgpt 6 Astra, Cursor Pro, Google Gemini3 Plus) was used in the project development, suggesting evaluation 
