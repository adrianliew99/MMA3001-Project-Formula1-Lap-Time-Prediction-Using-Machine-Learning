# MMA3001 submission readiness review

Reviewed against the supplied MMA3001 2026 Project Brief, especially pp. 4-9 and
assessment descriptors pp. 10-15. Source reviewed: v2.1, commit
e823d7e1856ea93c98576c790c4266acb39a3a88. The repository was renamed from
Using-a-1-Dimensional-CNN to Using-Machine-Learning. This continuation preserves
all pre-existing notebook cells and all tracked files under `results/`.

## Findings that affect the submission

1. The latest pushed notebook stopped at Section 16. Section 17 was not available
   to verify. Added a performance audit and separately generated measurements;
   earlier runtime evidence was Section 13, based on single-run timers.
2. Persistence is the validation-MAE winner: 0.46450673085911937 s, not a failed
   prerequisite to adding more models. C-feature depth-5/leaf-15 tree MAE is
   0.469607 s; its RMSE of 0.879105 s is below persistence's 1.051193 s. Explain
   the criterion and trade-off rather than changing the winner after seeing 2025.
3. Holdout predictions reconcile with the frozen metrics: 20,662 pairs, MAE
   0.3862170651437421 s, RMSE 0.6798099358511948 s, R² 0.9957528727036938.
   Tests verify pair identities against the input CSV, reconstruction, scores and
   weighted diagnostic totals. High pooled R² is not an incremental skill score.
4. `results/metrics/model_manifest.json` says `final_test_executed: false` despite
   completed final metrics and predictions. Preserve this historical file and
   cite the final metric/prediction files and new audit evidence as authoritative
   for completed testing. No fabricated execution date or fitted model is added.
5. The evaluated population includes 854 intermediate-tyre pairs and 49 unknown
   compounds. Do not describe all observations as dry laps. The new tests retain
   these rows and enforce accounting of every observation in grouped reports.
6. At review, `main` contained only a README. The brief requires functional source
   on `main`. A development branch or ZIP alone does not resolve this.
7. No finished 5-10 page written report or 5-minute presentation was found. The
   notebook's Section 16 is still a conclusion template. These require final
   student-authored interpretation and personal reflection.

## Section 17 verification

The old persistence `fit_seconds=0` and `predict_seconds=0` are placeholders.
They are not measured zero execution time. Original fit timings included both
training/validation input preparation before `.fit`; the new fit timer excludes
that preparation and should not be directly pooled with the old timer.

The new benchmark reports all seven representative method/feature combinations:
persistence plus the best validation configuration for each existing ML family
within B and C. Representative selection uses the saved validation grid only.
Accuracy is checked on the same 21,394 validation pairs, with fit on 35,872
training pairs. Two warm-ups and nine measured calls yield median/p95 batch and
one-row latency. Full pipeline preprocessing and next-lap reconstruction are
inside each prediction call; scoring, data loading and frame preparation are not.
Model training is timed once, clearly labelled.

Memory measures are input deep bytes, serialized model storage bytes and peak
traced prediction allocations from a separate call. They do not measure complete
native/process peak RAM, nor training peak RAM. No FLOPS or arithmetic intensity
is claimed: the brief asks for such estimates where appropriate, and this audit
does not justify misleading counts for path-dependent tree inference. Complexity
is linear in row count for persistence; trees add path traversal and forests add
multiple trees, with feature preprocessing and job scheduling overhead.

See `reports/performance/validation_performance.csv` and `environment.json` for
actual measurements, versions, source hashes and raw timing samples. CPU count,
OS and threading are recorded; shared-host load and short-run variability limit
transferability. A batch's time/row is throughput, not single-request latency.
These results justify retaining persistence under the predefined MAE criterion.

## Requirement-to-evidence matrix

| Brief requirement | Evidence/status | Remaining action |
|---|---|---|
| Mechanical/aerospace engineering context | Motorsport pace estimation, tyres, vehicle/track operating conditions | Report a specific intended user/decision; avoid validated strategy-optimisation claims |
| Defined inputs/outputs and invalid data | README units/ranges, audit rules, predictor allowlist | Explain limits of legacy validation, upstream quality checks and unknown future conditions |
| At least one taught numerical/ML technique | Linear regression, trees, forests, persistence; delta target | Explain objective/loss, preprocessing, settings and assumptions; confirm taught-technique alignment |
| Credible alternatives and selection | 19-row 2024 validation grid; two feature sets | State MAE criterion and why extra complexity did not win |
| Validation/verification | Chronological splits, frozen 2025 metrics, tests and reproduction | Include honest clean-target, circuit-confounding and source-lineage limitations |
| Sensitivity/optimisation/performance | Depth/leaf/tree-count grid, B/C ablation, repeated Section 17 benchmark | Distinguish grid sensitivity from formal robustness certification; no new model needed |
| GitHub and reproducibility | Existing commits, improved README, constraints snapshot, offline tools, CI config | Merge reviewed work onto functional `main`; check fresh clone and hosted CI |
| Inline comments/docstrings and HTML | Existing code plus generated package/tool and active notebook HTML | Student review completeness, especially exporter/API parameter and return contracts |
| Test functions and test report | Pytest suite, JUnit XML, readable HTML summary | Re-run on final commit; include report in submitted clone |
| Licensing agreement | LICENSE records no licence grant and third-party ownership | Author must choose code distribution terms and verify data rights before claiming this requirement complete |
| 5-10 page report | Not found | Write and export individual report; references/appendices excluded from 10-page cap |
| AI reflection | README records this continuation and prior declared tools | Confirm identities/contribution, student decisions, verified errors and learning in personal voice |
| Accessible link and zipped clone | Review branch and final package can support review | Final link must point to functional main; submit zipped clone on Moodle |
| Demo/interview | Not found | Book Week 14 appointment, prepare 5 min talk + 3 min unaided questions |

Dates are in the Unit Overview Spreadsheet, not these repository files. Do not
infer a deadline from the project-selection week (the two supplied PDFs differ
on that week). Use the original brief for assessment requirements.

## Efficient final report plan (8 body pages)

| Page | Content |
|---|---|
| 1 | Engineering problem, use case, scope and objectives |
| 2 | Data source, I/O units, eligibility and 92-race coverage |
| 3 | Delta target, persistence, regression/tree/forest methods and preprocessing |
| 4 | Chronological split, train-only transforms, verification and acceptance criteria |
| 5 | 2024 alternatives, B/C ablation, hyperparameter sensitivity, MAE/RMSE trade-off |
| 6 | Frozen 2025 scores and selected circuit/tyre/race-progress diagnostics |
| 7 | Repeated timing and memory evidence, reproducibility and engineering limits |
| 8 | Conclusions, lessons learned, AI use and critical reflection |

Append detailed tests, full validation grid, feature contract and supplementary
plots. Reference actual saved files/figures. Do not present these planning notes
as the finished assessed report.

## Five-minute demo plan

- 0:00-0:40: Engineering question and primary I/O.
- 0:40-1:25: Data, clean-target condition and chronological split.
- 1:25-2:20: Methods, validation comparison and persistence selection.
- 2:20-3:10: 2025 holdout and one engineering error example.
- 3:10-4:15: Repository, generated HTML, tests, reproduction and Section 17.
- 4:15-5:00: Main limitations, lesson learned and AI evaluation.

Be ready to explain delta reconstruction, why R² is high, MAE versus RMSE,
leakage, missing inputs, tyre/traffic/fuel confounding and why no CNN is needed.
Do not use AI to answer interview questions.
