# Submission evidence and reproduction

The root README is maintained separately and was not changed for this work.

| Requirement | Repository evidence |
| --- | --- |
| README | `README.md` |
| HTML source documentation | `docs/html/index.html`, pdoc API pages and static main-notebook export |
| Licensing agreement | Root `LICENSE`: all rights reserved for original project materials; third-party rights retained |
| Source code and comments | Main notebook, `mma3001_f1/` and `dataset/` |
| Test functions | `tests/test_core.py` and `tests/test_holdout_artifacts.py` |
| Test report | `results/tests/report.html`, `junit.xml`, and `coverage/index.html` |

## Reproduce on Windows PowerShell

Use an installed Python 3.12 interpreter from the repository root:

```powershell
python -m venv .venv-tests
.venv-tests/Scripts/python.exe -m pip install -r requirements.txt -r requirements-test.txt -r requirements-docs.txt
./scripts/run_tests.ps1
.venv-tests/Scripts/python.exe scripts/build_docs.py
```

If PowerShell blocks `.ps1` files, run the same test command directly without
changing execution policy:

```powershell
.venv-tests/Scripts/python.exe -m pytest --html=results/tests/report.html --self-contained-html --junitxml=results/tests/junit.xml --cov=mma3001_f1 --cov-report=term-missing --cov-report=html:results/tests/coverage --cov-report=xml:results/tests/coverage.xml
```

On other platforms, use `.venv-tests/bin/python` and run pytest directly:

```sh
.venv-tests/bin/python -m pytest --html=results/tests/report.html --self-contained-html --junitxml=results/tests/junit.xml --cov=mma3001_f1 --cov-report=term-missing --cov-report=html:results/tests/coverage --cov-report=xml:results/tests/coverage.xml
.venv-tests/bin/python scripts/build_docs.py
```

The generated HTML and test reports are submission artifacts and should be
committed along with their source scripts. Open `docs/html/index.html` locally
to access the documentation and reports. The complete coverage directory is
needed for its styling and source links.

## Validation scope

Local verification on 25 October 2026 used Python 3.12.14: **40 tests passed**
with one NumPy/pandas timedelta deprecation warning. Package statement coverage
was **49%** (270 of 556 statements). `results/tests/environment.txt` records the
installed package versions for this run. Syntax parsing passed for all 12 Python
files and all 27 notebook code cells. The documentation build completed and all
local links in the documentation index and API pages resolved. Notebook export
reported missing alternative text on eight existing images.

The tests use deterministic examples and saved 2025 predictions. They check
configuration, lap quality, telemetry timing, past-only weather, consecutive-lap
examples, chronological splits, training-only preprocessing, numerical metrics,
atomic writes, and consistency of saved holdout summaries. They do not download
provider data or retrain the notebook. Coverage measures `mma3001_f1`, not the
notebook or the download scripts; passing tests do not establish complete
end-to-end notebook execution or live provider availability.

The saved predictions include 49 rows with missing tyre-compound labels.
The compound summary covers the 20,613 labelled rows; all 20,662 rows remain
in the overall holdout metrics. The group-summary test explicitly accounts
for missing labels while independently checking each saved group's metrics.

Documentation generation renders the saved notebook without executing its cells.
Existing outputs may therefore reflect an earlier run. The Python API pages
include function documentation and source; the notebook page includes its
markdown, inline comments, code, and saved outputs.

## Main branch delivery

At preparation time the checkout was `v2.3`, with pre-existing notebook and README
edits. A local `master` branch also existed; no local `main` branch existed.
These artifacts are working-tree changes, not proof that the remote default
branch contains the submission. Review and commit the intended files, merge
the tested revision into the repository's chosen main/default branch, and run
the same tests there before submission. Do not add local virtual environments,
caches, or unrelated experimental exports.

The workflow `.github/workflows/validation.yml` runs the tests and documentation
build on pushes and pull requests and uploads the generated evidence. Its remote
run status must be checked after pushing; a local pass does not imply a CI pass.
