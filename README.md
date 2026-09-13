# MMA3001: F1 data and model laboratory

Start with **notebooks/01_f1_project_lab.ipynb**. It walks through real Bahrain 2024 lap data, missing values and quality flags, race-pace/stint plots, a numerical least-squares example, and optional next-lap baseline training with editable parameters.

This is a preparation workspace. The final title, teaching-team approval and CNN choice remain open. The baseline training section is implemented but disabled by default; choose additional chronological race weekends and obtain actual topic approval before using it. No neural network is installed or claimed trained.

## Open on this Windows computer

In PowerShell:

```powershell
cd C:\Users\Admin\Documents\MMA3001\Project
.\.venv-project\Scripts\python.exe -m jupyterlab notebooks/01_f1_project_lab.ipynb
```

Choose **Python (MMA3001)**, then run cells in order. Change the control-panel cell to select year, event, drivers, history length, ridge penalties and race roles. The delivered notebook includes pilot outputs. Rerunning retrieval may need internet even with cached data.

The original `.venv` is broken because its base Python executable is unavailable. It was preserved; use `.venv-project`. This environment uses the available Codex bundled Python 3.12.14. If moving the project or removing that runtime, recreate an environment using your own installed Python.

## Recreate on another Windows installation

Install Python 3.12 with the Python launcher, then:

```powershell
cd C:\Users\Admin\Documents\MMA3001\Project
py -3.12 -m venv .venv-project
.\.venv-project\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-project\Scripts\python.exe -m pip install --no-deps -e .
.\.venv-project\Scripts\python.exe -m ipykernel install --prefix .venv-project --name mma3001 --display-name "Python (MMA3001)"
.\.venv-project\Scripts\python.exe -m jupyterlab notebooks/01_f1_project_lab.ipynb
```

`requirements-lock.txt` records the tested Windows/Python environment. `pip install -e .` alone installs compatible declared dependencies when an exact lock is unsuitable on another OS. The `py` recreation commands are portability instructions; they were not executed here because the Python launcher is absent.

## Repeat the checks and outputs

```powershell
.\.venv-project\Scripts\python.exe scripts/run_pilot.py --year 2024 --event Bahrain --session R
.\.venv-project\Scripts\python.exe -m pytest --junitxml=reports/tests.xml -q
.\.venv-project\Scripts\python.exe -m pdoc mma3001_f1 -o docs/html
.\.venv-project\Scripts\python.exe scripts/execute_notebook.py
```

See `reports/verification.md` for checks actually executed and real findings. Tests use labelled synthetic fixtures and need no network. The pilot command and notebook use real data and fail visibly if retrieval fails.

## Files and outputs

- `src/mma3001_f1`: reusable preparation, quality audit, figures, ordered examples and baseline training.
- `data/raw`, `data/processed`, `cache`: source exports, flagged tables and retrieval cache. See `data/README.md` for units and terms.
- `reports`: real data quality JSON, test XML and verification note.
- `results/figures`, `results/tables`: charts and summaries; optional training produces metrics, predictions and a saved pipeline under `results/models`.
- `docs/html/index.html`: generated function documentation.
- `docs/feasibility.md`: reviewable research proposal and alternatives.
- `docs/decisions_and_ai.md`: decisions, AI contributions and student reflection prompts.

The untouched `Fast-F1` folder is the provider's source checkout, not this project's implementation. New code is MIT licensed; assessment documents and data have separate terms. No remote repository has been published. Use meaningful local commits during development, then arrange the required accessible GitHub repository and zipped final submission. Confirm selection dates with the teaching team: the handoff reports Week 5/Week 6 disagreement between supplied sources.
