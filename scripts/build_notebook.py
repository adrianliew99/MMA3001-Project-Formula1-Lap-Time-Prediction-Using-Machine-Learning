"""Rebuild the guided notebook source (execution is a separate step)."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# MMA3001 · Formula 1 data and model laboratory
**Start here:** run cells from top to bottom using the project's Python kernel. First downloads require internet; subsequent runs reuse the FastF1 cache.

This is a **Phase 1 feasibility pilot**, not a final approved topic or a trained CNN. It explores Bahrain 2024, audits the data and gives you a reusable baseline-training workspace. The final section stays off until your topic is approved and you configure separate race weekends.

**Engineering motivation:** support race-pace analysis by estimating the next lap's duration from information already observed. This cannot measure tyre wear, predict incidents, or produce a team's optimal strategy.

1. Settings and environment → 2. Real data → 3. Quality audit → 4. Visual analysis → 5. Numerical exploration → 6. Optional training → 7. Reflection.
""")
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from importlib.metadata import version
from mma3001_f1.data import collect_session, plot_pilot
from mma3001_f1.modeling import make_examples, split_weekends, train_baseline, evaluate, score

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
assert (ROOT / "pyproject.toml").exists(), "Launch Jupyter from the Project folder."
print("Python:", sys.executable)
display(pd.Series({p: version(p) for p in ["fastf1", "pandas", "numpy", "scikit-learn"]}, name="version"))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})''')
md("""## 1 · Your control panel
Change these values, then rerun from here. A history of 3 means three consecutive completed laps from the same driver and stint. `RIDGE_ALPHAS` controls coefficient regularisation: larger values discourage large fitted coefficients. This regression has no epochs, batch size or learning rate because it uses a direct numerical solver, not neural-network training.

Keep the explored Bahrain pilot in development/training data. Never use it as an untouched final test race. Choose additional races and record their roles **before** inspecting their outcomes.""")
code('''YEAR, EVENT, SESSION = 2024, "Bahrain", "R"
DRIVERS = ("VER", "LEC")
HISTORY_LAPS = 3
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
TOPIC_APPROVED = False  # Change only after actual teaching-team confirmation.
RUN_TRAINING = False
RUN_FINAL_TEST = False  # Enable only after validation choices are frozen.
# Tuples are (year, event, session). All sessions in a weekend must share one role.
TRAIN_RACES = [(2024, "Bahrain", "R")]
VALIDATION_RACES = []
TEST_RACES = []''')
md("""## 2 · Retrieve real observations
The source export is preserved in `data/raw`; converted fields and quality flags go to `data/processed`. Durations use **total seconds**, so minutes and days are handled correctly. Missing targets remain missing. Download failures raise an error; no invented sample race is substituted.

Weather is exported if available with its original session timestamps. It is not joined or used for prediction. A future weather join must use the last observation available at prediction time, with a justified maximum age.""")
code('''laps, quality, manifest = collect_session(ROOT, YEAR, EVENT, SESSION)
display(pd.Series(manifest))
display(laps[["SourceRow", "Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreLife", "Stint", "Eligible"]].head(12))''')
md("""## 3 · Audit before excluding
Raw rows are never dropped. The proposed clean population excludes missing/nonpositive times, invalid or duplicate identifiers, pit laps, non-green running, inaccurate/deleted/generated records, unknown quality and unsupported tyre context. Slow laps are **not** excluded by an arbitrary time cutoff. Counts below show both overlapping flags and each rule's additional effect.

Missing `PitInTime`/`PitOutTime` values normally mean no pit event; an entirely absent source column is instead treated as unknown. Provider quality flags are useful checks, not independent ground truth.""")
code('''display(pd.Series({k: quality[k] for k in ["rows", "drivers", "eligible_rows", "unavailable_quality_fields"]}))
display(pd.DataFrame(quality["exclusion_waterfall"]))
display(pd.Series(quality["missing"], name="missing_count").sort_values(ascending=False).to_frame())
display(pd.DataFrame.from_dict(quality["lap_coverage"], orient="index"))
display(pd.Series(quality["nonpositive_durations"], name="nonpositive_duration_count"))
clean = laps.loc[laps.Eligible].copy()
display(clean.groupby("Compound").LapTimeSeconds.agg(["count", "median", "mean", "std"]))''')
md("""## 4 · Race pace and tyre-stint analysis
The first chart includes all recorded laps and marks records outside the proposed population. The second shows compound and stint timing across drivers. Pace changes can also reflect fuel load, traffic, weather and driver management. Neither plot isolates a causal tyre effect; tyre age is not remaining tyre life.""")
code('''figures = plot_pilot(laps, ROOT / "results/figures" / manifest["session_id"], DRIVERS)
plt.show()
summary = clean.groupby(["Driver", "Stint", "Compound"]).agg(
    laps=("LapNumber", "count"), first_lap=("LapNumber", "min"), last_lap=("LapNumber", "max"),
    median_seconds=("LapTimeSeconds", "median"), spread_seconds=("LapTimeSeconds", "std"))
summary.to_csv(ROOT / "results/tables" / f"{manifest['session_id']}_stints.csv")
display(summary)''')
md("""## 5 · Numerical exploration: a straight-line fit
Least squares chooses the line with the smallest sum of squared residuals. `numpy.linalg.lstsq` solves the small system without explicitly inverting a matrix. We centre tyre age to improve conditioning. The slope is an **association in seconds per tyre lap**, not a physical wear rate. This within-stint demonstration is not withheld-data validation and does not select the final method.""")
code('''groups = [(key, g) for key,g in clean[clean.Driver.eq(DRIVERS[0])].groupby("Stint") if len(g) >= 5 and g.TyreLife.nunique() >= 2]
if groups:
    stint, sample = groups[0]
    x, y = sample.TyreLife.to_numpy(dtype=float), sample.LapTimeSeconds.to_numpy()
    A = np.column_stack([np.ones(len(x)), x-x.mean()])
    coef, _, rank, singular_values = np.linalg.lstsq(A, y, rcond=None)
    print(f"{DRIVERS[0]}, stint {stint}: slope = {coef[1]:.4f} s/tyre lap; matrix condition number = {np.linalg.cond(A):.2f}")
    order = np.argsort(x)
    plt.figure(figsize=(9,4))
    plt.scatter(x,y,label="Eligible observed laps")
    plt.plot(x[order],(A@coef)[order],label="Least-squares line",color="coral")
    plt.xlabel("Reported tyre age (laps)"); plt.ylabel("Lap time (s)")
    plt.title("Within-stint association: descriptive only"); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / "results/figures" / manifest["session_id"] / "stint_line.png", dpi=150)
    plt.show()
else:
    print("No stint has enough eligible observations for this demonstration.")''')
md("""## 6 · Prepare ordered next-lap examples
Prediction time is the end of the current lap. Inputs contain only previous completed lap times, current tyre age and current lap number. The target is next-lap seconds. Windows cannot cross drivers, missing lap numbers, excluded laps or stint boundaries.

We retrospectively require the target lap to be clean too. Consequently, the evaluation answers **how accurate predictions are during uninterrupted dry running**, not whether the model knows future incidents. Circuit/driver effects remain a limitation of this small baseline.""")
code('''pilot_examples = make_examples(laps, HISTORY_LAPS)
print("Pilot training-example candidates:", len(pilot_examples))
display(pilot_examples.head())''')
md("""### Optional training and parameter sensitivity
After topic approval, populate the three race lists. The code rejects overlapping weekends and enforces training → validation → test chronology. Scaling is learned on training rows only. Validation compares ridge penalties on the same examples; the final test is not downloaded until you explicitly enable it.

The baseline predicts that the next lap equals the last lap. MAE is the typical absolute error; RMSE penalises large misses more heavily. Both use seconds. One validation and one test race are minimum software requirements, not evidence of broad generalisation. A CNN would need more races, a justified sequence architecture and a separate comparison; no deep-learning framework is installed here.""")
code('''model = None
if RUN_TRAINING:
    if not TOPIC_APPROVED:
        raise ValueError("Record actual topic approval before substantial model development.")
    roles = [TRAIN_RACES, VALIDATION_RACES, TEST_RACES]
    if any(not r for r in roles):
        raise ValueError("Configure all three race roles before training.")
    # Resolve schedule IDs without loading test lap outcomes.
    import fastf1
    role_ids, role_dates = [], []
    for races in roles:
        sessions = [fastf1.get_session(*r) for r in races]
        role_ids.append([f"{r[0]}-{int(s.event.RoundNumber):02d}-{r[2]}" for r,s in zip(races,sessions)])
        role_dates.append([pd.Timestamp(s.event.EventDate) for s in sessions])
    weekend_sets = [{sid.rsplit("-",1)[0] for sid in ids} for ids in role_ids]
    if any(weekend_sets[i] & weekend_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Weekend roles overlap.")
    if not (max(role_dates[0]) < min(role_dates[1]) and max(role_dates[1]) < min(role_dates[2])):
        raise ValueError("Races must be ordered training, validation, test.")
    if manifest["session_id"].rsplit("-",1)[0] in weekend_sets[2]:
        raise ValueError("The explored pilot cannot be the final test.")
    def load_examples(races):
        return pd.concat([make_examples(collect_session(ROOT,*r)[0],HISTORY_LAPS) for r in races], ignore_index=True)
    train, validation = load_examples(TRAIN_RACES), load_examples(VALIDATION_RACES)
    for part,ids in zip([train,validation],role_ids[:2]):
        if part.empty or not set(ids).issubset(set(part.SessionID)):
            raise ValueError("A selected race has no eligible examples; inspect its quality report.")
    model, features, sensitivity = train_baseline(train, validation, RIDGE_ALPHAS)
    display(sensitivity)
    print("Previous-lap validation baseline:",score(validation.TargetSeconds,validation.Lag1))
    sensitivity.to_csv(ROOT / "results/tables/validation_alpha_sensitivity.csv", index=False)
    import joblib
    (ROOT / "results/models").mkdir(exist_ok=True)
    joblib.dump({"model":model,"features":features,"history":HISTORY_LAPS},ROOT / "results/models/ridge.joblib")
    settings = {"history":HISTORY_LAPS,"alphas":RIDGE_ALPHAS,"roles":role_ids,"selected_alpha":model[-1].alpha,
                "versions":{p:version(p) for p in ["numpy","pandas","scikit-learn","fastf1"]}}
    (ROOT / "results/models/training_settings.json").write_text(json.dumps(settings,indent=2))
else:
    print("Training scaffold ready. No model trained: topic and additional race roles remain open.")''')
code('''if RUN_FINAL_TEST:
    if model is None:
        raise ValueError("Run approved training first and freeze your validation choices.")
    final_test = load_examples(TEST_RACES)
    _,_,final_test = split_weekends(pd.concat([train,validation,final_test]),*role_ids)
    metrics, predictions = evaluate(model, features, final_test)
    display(metrics)
    predictions["absolute_error"] = abs(predictions.TargetSeconds-predictions.Ridge)
    display(predictions.groupby(["SessionID","Driver"]).absolute_error.agg(["count","mean","max"]))
    metrics.to_csv(ROOT / "results/tables/final_test_metrics.csv",index=False)
    predictions.to_csv(ROOT / "results/tables/final_test_predictions.csv",index=False)
else:
    print("Final test remains untouched. Do not tune settings after reviewing final-test scores.")''')
md("""## 7 · What to record and discuss
- Which laps were excluded, and how does this change the question your model answers?
- What information was available at prediction time? Why would target-lap sectors leak the answer?
- Does the validation gain over the previous-lap baseline justify added complexity?
- How might circuit differences, fuel load, traffic and pit strategy affect results?
- Record actual approval, dataset choices, your interpretation and any corrections in `docs/decisions_and_ai.md`.

Read `docs/feasibility.md` for the proposed scope. Tests and their report are in `tests/` and `reports/`; generated function documentation is in `docs/html/`.

Sources: [FastF1 documentation](https://docs.fastf1.dev/), [FastF1 source](https://github.com/theOehrly/Fast-F1), [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). The local assessment summary and handoff guide the deliverables. Package licensing does not automatically cover underlying F1 data.""")
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python (MMA3001)","language":"python","name":"mma3001"},"language_info":{"name":"python","version":"3.12"}})
(root / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, root / "notebooks/01_f1_project_lab.ipynb")
print("Notebook created:",len(cells),"cells")
