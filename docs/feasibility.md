# Proposal for teaching-team discussion

Status: preparation only; topic, final title, taught-method suitability and approval remain open.

Recommended question: **Can recent completed laps improve next-lap time estimates over simply repeating the previous lap during uninterrupted dry running?** This connects prediction to race-pace monitoring. An intended user is an engineering analyst assessing short-term pace; outputs are seconds, not strategic instructions or measured tyre wear.

Inputs: FastF1 lap tables, driver and weekend identifiers, original lap number, stint, compound, reported tyre age (laps), quality flags, completed lap durations (seconds). Output: next-lap duration in seconds. Nonpositive/missing times and unknown quality are ineligible, retained in the source and audit. Same-driver histories must be consecutive and inside one stint. Inputs end at the current lap; no future sectors, telemetry, weather or eventual stint length are used.

The implemented pilot is one development race. The optional training scaffold compares persistence (next lap equals last lap) with scaled ridge regression. Larger ridge alpha shrinks coefficients; validation selects it. The notebook also demonstrates centred least squares descriptively within a stint. Confirm that the selected approach meets the unit's taught-method requirement. A CNN is a possible later experiment: a 1D CNN learns local patterns along an ordered lap sequence, but its data requirements and extra complexity need justification. No CNN or final research model has been selected or trained.

Validation proposal: reserve whole later race weekends for validation and final testing, choosing roles before inspecting outcomes. The software requires three chronological partitions and prevents weekend overlap, including sprint/race overlap. Use more than the minimum one race per role for a defensible study. Fit preprocessing on training only, compare MAE/RMSE on identical held-out examples, report driver/weekend errors and measured training/prediction runtime. Alpha sensitivity is implemented; history-length sensitivity is a later option chosen using development data only. Keep the already explored pilot out of the final test. If final-test results inform changes, obtain a new untouched test set.

The clean target population is selected retrospectively. The model does not know future pit stops, flags or incidents. Results will be conditional on uninterrupted dry running. Fuel mass, traffic, tyre management and setup are not measured causal controls here. A cross-circuit model using only simple lap-history features may fail at different circuits; investigate those errors before adding complexity. Laps within races are dependent; raw example count is not the number of independent experiments.

Alternatives considered:

| Direction | Strength | Limitation / reason not to start there |
|---|---|---|
| Next-lap prediction | Clear units, timing, inexpensive baseline, ordered history supports possible CNN comparison | Requires multiple weekends and strict leakage controls |
| Tyre-age-associated pace | Accessible regression and stint plots | Fuel, traffic and strategy confound physical wear interpretations |
| Numerical telemetry analysis | Strong numerical-method focus with interpolation/smoothing comparisons | Needs channel/sampling study and justified independent validation; telemetry not collected yet |

Approval discussion: agree question, dataset usage terms, number of races, baseline/taught method, clean-target population and achievable scope. The supplied student summary states selection by end of Week 6; the handoff reports Week 5 in the full brief. That full brief is not present under its listed filename. Resolve dates with the teaching team/current Unit Overview Spreadsheet. GitHub publication and final assessment report/presentation remain future work.

References: [FastF1](https://docs.fastf1.dev/), [FastF1 source](https://github.com/theOehrly/Fast-F1), [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), local `MMA3001_Project_Brief_Student_Key_Points.pdf` and `CURSOR_PROJECT_HANDOFF.md`.
