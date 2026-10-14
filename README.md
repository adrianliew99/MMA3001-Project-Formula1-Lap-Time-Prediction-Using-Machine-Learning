# MMA3001 Formula 1 next-lap prediction notebook

This is a standalone, documented Jupyter notebook for chronological model comparison using **real FastF1 next-lap ML CSV exports**. It does not overwrite the user's existing repository or scripts.

## Getting started on Windows 11

1. Copy `MMA3001_F1_Next_Lap_Prediction.ipynb` into the MMA3001 repository (for example `notebooks/`).
2. In the project's activated Python 3.12 virtual environment, run `python -m pip install -r requirements.txt`.
3. Run `python -m ipykernel install --user --name mma3001 --display-name "MMA3001 (Python 3.12)"`, if your environment is not listed in VS Code.
4. Open the notebook and change `DATA_DIR` to a directory containing one or more `*next_lap_ml*.csv` files.
5. First import a 2024 Bahrain export to inspect the schema and QA. For cross-season regression, obtain at least train and validation seasons (defaults: 2022–23 train, 2024 validation, 2025 test).
6. Run cells sequentially. Leave `RUN_FINAL_TEST=False` during development. Turn it on only once after selecting your approach using validation.

**Important:** A single 2024 export is not sufficient for the default chronological train/validation split. Do not use random-lap holdouts as a workaround. Change the year configuration only to other truly distinct chronological seasons for which you have data.

## Notebook capabilities

- Optional explicitly enabled one-event download via FastF1; existing exporters are preferred.
- Export inventory, validation and exclusion reasons for malformed pairs.
- Persistence, Linear Regression, Decision Tree and Random Forest regressors.
- Preprocessing isolated by training, chronological split, validation hyperparameter selection.
- Comparison of context-only versus context plus previous-lap telemetry (same rows).
- Prediction CSVs, model pipeline, split manifest, quality report, metrics and plots under `results/`.
- Locked final holdout and documentation for scientific interpretation.

## CUDA

Not required: the four classical scikit-learn algorithms operate on CPU, even when an RTX 4060 is present. The code checks for CUDA through PyTorch only if already installed; installing PyTorch does not accelerate these scikit-learn models.

## Known limitations

- The upstream scripts were not provided, so lap cleanliness, stint transitions and weather join timestamps cannot be independently verified from CSV alone. Audit them in original FastF1 records.
- Historical target-lap clean filtering means this is conditional retrospective evaluation, **not** a prediction of all future race laps.
- The optional fallback acquisition only emits basic fields (no car telemetry or timestamp-verified weather).
- The notebook does not include tyre-stop optimization or live inference, which the planning brief lists as later scope.
- No real event data was available in the provided attachments, so model performance cannot be asserted until you execute using verified historic CSVs.

## Tests performed in preparation

- `nbformat.validate` and `ast.parse` on every executable cell: passed.
- Full offline execution of the notebook against a **synthetic fixture** spanning 2022–25: passed. These toy observations are **not historical F1 results** and are not included in this handover.
- Actual FastF1 API downloads and real-race benchmark results: **not performed**.

## Submission items remaining

Provide repository README, HTML documentation, dedicated pytest test suite, licences subject to user preferences and FastF1 terms, approved engineering framing, comprehensive report and AI-use reflection. See uploaded official MMA3001 project brief for the marking rubric.
