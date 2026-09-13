# MMA3001 Project Handoff for Cursor

Prepared from the project discussion and supplied assessment documents on 13 September 2026.

## Your assignment

Help me develop my individual **MMA3001: Numerical Methods and Machine Learning** project at **Monash University Malaysia**. I am interested in **Formula 1 engineering and performance analysis** and want an organised, reproducible Python repository that I can understand, explain, and defend during assessment.

**Start by delivering Phase 1 below. Inspect the existing repository, make the necessary local changes, run the available checks, and report what actually works. A plan alone is not the requested first deliverable.**

The final title and research question are deliberately undecided. Build a small foundation for data exploration and a concrete proposal before substantial model development. Explain decisions in language that a reader with a business background could follow, while retaining the engineering purpose of the work.

## 1. Established context and open decisions

| Item | Current status |
| --- | --- |
| Unit and assessment | Individual MMA3001 project, worth 25% of the unit mark. |
| Preferred subject | Formula 1, with a mechanical or aerospace engineering application. |
| Current practical focus | Starting the repository and learning to access F1 data with Python. |
| FastF1 | I have reached the pip-install/getting-started stage. Verify the active Python environment and installed package before relying on it. |
| README | A plain-language starter `README.md` was drafted. Inspect and preserve my current version if present. |
| Earlier code | A `first_f1.py` example was provided for Bahrain 2024 race laps, caching, conversion to seconds, and CSV export. Its presence and successful execution on my machine have not been confirmed. |
| CNN | An interest to investigate. A 1D CNN for sequences was suggested; neither CNN nor a specific architecture has been selected. |
| Title, target, and final dataset | Not yet selected. |
| Teaching-team approval | Not established in the discussion. Record it only when there is actual confirmation. |
| Trained models and results | None confirmed. Do not describe suggested work as completed. |

The three candidate directions discussed were next-lap time prediction, tyre-age-associated race-pace analysis, and numerical analysis of telemetry. These are alternatives for proposal development, not three projects to implement.

## 2. How to work with me

- Explain what each step does, why it is needed, and what its output means. Define new terms briefly.
- Begin with a small working example, then introduce complexity when there is evidence it helps.
- Choose routine implementation details independently. Use the existing repository conventions when they are reasonable.
- Inspect existing instructions, files, Git status, and dependencies before editing. Preserve my work and any unrelated changes.
- Keep status labels accurate: planned, implemented, tested, or blocked. State which commands actually ran.
- Keep the first stage local. This handoff does not request remote publication, paid services, or changes to repository visibility.
- If an external dependency fails, complete the useful local work and state the specific remaining blocker. Do not fabricate downloaded data or successful results.
- Do not ask me to choose the final title before you deliver the repository and data-feasibility work below.

## 3. Assessment requirements to design around

Consult the unit-provided PDFs if they are present in the repository:

- `MMA3001_2026_Project_Brief(1).pdf`
- `MMA3001_Project_Brief_Student_Key_Points.pdf`

Equivalent filenames may be used locally. The requirements below are a working summary of those documents.

| Requirement | Evidence the completed project must provide |
| --- | --- |
| Engineering relevance | A clear mechanical/aerospace problem, motivation, intended user, use of the output, and justified scope. |
| Inputs and outputs | Formats, variables, units, valid domains, output meanings, and treatment of missing or unsupported inputs. |
| Computational method | At least one numerical and/or machine-learning technique taught in the unit; explain assumptions, implementation, parameters, and limitations. |
| Alternative approach | At least one credible baseline, evaluated fairly against the selected approach. |
| Verification and validation | Code checks plus an appropriate evaluation against withheld data, known answers, benchmarks, or other justified evidence. |
| Performance and improvement | Relevant sensitivity/optimisation work, timing/profiling, quantitative comparisons, and engineering trade-offs. FLOP estimates are appropriate only where useful. |
| Reproducibility | GitHub used during development, meaningful commits, a functional main branch, dependencies, and repeatable instructions. |
| Documentation | README, inline comments, function docstrings, and generated HTML source-code documentation. |
| Testing | Meaningful test functions and an actual test report. |
| Licensing and attribution | A code licence and appropriate treatment/citation of third-party data and methods. |
| AI reflection | Tools, uses, approximate contributions, reasons, checks, errors, student decisions, and effect on learning. |

The written report is 5–10 pages, with references and appendices excluded from the 10-page limit. Final submission includes an accessible repository link and a zipped clone through Moodle. The presentation is 5 minutes plus 3 minutes of questions in Week 14. Marks depend on the presentation and ability to answer questions about the work. AI cannot be used to answer those questions.

Self-selected topics and datasets need teaching-team approval before substantial work. Phase 1 is limited preparation and feasibility work for that discussion. Confirm the selected method satisfies the taught-method requirement; interest in CNN alone does not establish this.

The full brief says project selection by the end of Week 5; the student key points say Week 6. Flag this discrepancy and refer to the teaching team/current Unit Overview Spreadsheet for dates. Do not invent a calendar deadline.

## 4. Recommended initial technical scope

Use **FastF1 as the starting data-access tool** because it is the tool I am currently learning. Begin with **2024, Bahrain, Race (`R`)** as a configurable example. This is a pilot choice, not the final dataset or title.

Use Python, pandas, NumPy, and Matplotlib where needed. Add pytest for meaningful tests and a simple documentation generator such as pdoc for HTML code documentation. Reuse compatible installed dependencies and record the versions actually used. Defer deep-learning frameworks until a modelling direction is selected.

Create the FastF1 cache directory before enabling it. Keep downloads cached, make the selected year/event/session configurable, and load only the data required for the current task. Verify the installed FastF1 API instead of copying outdated calls without checking them.

Sources discussed:

| Resource | Role |
| --- | --- |
| [FastF1](https://github.com/theOehrly/Fast-F1) and [documentation](https://docs.fastf1.dev/) | First choice for Python access to session and lap data; telemetry can be explored later. |
| [OpenF1 documentation](https://openf1.org/docs/) | Alternative access or supplementary observations if a specific need arises. |
| [Jolpica documentation](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md) | Supplementary historical race information if needed. |

An earlier assistant check retrieved OpenF1 records for Bahrain 2024, session key `9472`: 1,129 lap records, 63 stint records, and 157 weather records. This was a historical availability check, not a file delivered into this repository or proof of a successful local FastF1 run. Verify access yourself. Different providers, parsing, versions, and exclusions can produce different counts; do not use these figures as hard-coded FastF1 test expectations.

## 5. Phase 1 — Deliver now

### A. Inspect and establish the repository

1. Read the existing README, project guides, instructions, code, and Git status.
2. Identify the Python interpreter and whether FastF1 imports successfully. Explain any environment repair in plain language.
3. Reuse a sensible existing structure, or establish the compact layout below. Create useful files as needed rather than filling the repository with empty modules.
4. Provide a reproducible dependency installation method and a suitable `.gitignore` for environments, caches, temporary output, and large downloaded data.
5. Update the README to describe what now exists, with tested setup and run commands. Leave the final title and method open.

| Suggested location | Purpose |
| --- | --- |
| `README.md` | Plain-language overview, status, navigation, and tested usage instructions. |
| `src/` | Reusable data-access, preparation, checking, and plotting functions. |
| `scripts/` | Simple entry points for collecting and exploring a chosen session. |
| `notebooks/` | Optional guided exploration using the reusable functions. |
| `data/raw/` | Preserved source exports, subject to usage terms. |
| `data/processed/` | Prepared tables with documented transformations and flags. |
| `data/README.md` | Sources, retrieval process, variable definitions, and data handling. |
| `cache/` | FastF1 cache, excluded from version control. |
| `results/figures/` and `results/tables/` | Reproducible charts and summaries. |
| `tests/` | Small, meaningful tests that run without a live data service. |
| `docs/` | Method notes, decisions, AI-use record, and generated HTML documentation. |
| `reports/` | Data-quality, test, feasibility, and eventual assessment reports. |

Use the simplest reliable project packaging and invocation scheme. The student should be able to follow the README on their machine without editing internal source paths. Verify the actual operating system; provide matching terminal instructions.

### B. Retrieve and prepare one real session

Build a reusable workflow that:

- Loads the selected race's lap data and saves a source export.
- Retains driver, event, session, and lap identifiers; lap and sector durations; compound, tyre age, and stint information where available.
- Preserves relevant quality information such as pit-in/out timing, track status, accuracy, deleted-lap, or generated-lap flags where the provider supplies them. Report unavailable fields instead of assuming their values.
- Converts durations with total seconds, preserving missing values. Uses explicit names such as `LapTimeSeconds` and documents units.
- Keeps original lap numbering and enough information to trace each prepared record back to its source.
- Records source, selected session, retrieval time, package versions, and transformation rules in a small manifest or data note.

Weather is useful exploratory context if accessible. Preserve its timestamps and explain any join to lap data. Any future forecasting features must use observations available by the prediction time. Additional telemetry collection is optional and should serve a specific exploratory question.

The minimum useful data deliverable is the lap workflow. If optional weather or telemetry cannot be retrieved, report that limitation and still finish the lap analysis.

### C. Audit quality before choosing exclusions

Produce a short report containing:

- Actual row and driver counts, lap coverage, and available columns.
- Missing-value counts for relevant variables and checks for duplicate identifiers.
- Checks for nonpositive durations and unsupported values.
- Counts of pit-related, interrupted, inaccurate, deleted, or otherwise questionable laps where identifiable.
- The proposed inclusion/exclusion rules and the effect of each rule.

Keep raw data intact. Prefer explicit flags and separate prepared outputs to silently deleting inconvenient observations. Do not fill missing targets with zero or remove all slow laps using an unexplained cutoff.

### D. Produce clear exploratory outputs

Create at least two readable figures from the retrieved data, for example:

1. Lap time in seconds against lap number for two selected drivers, with separate panels or clear labels and visible treatment of pit/interrupted laps.
2. A tyre-stint timeline or lap-time view within stints, showing compound and driver context.

Explain what each chart shows and what it cannot establish. A lap-time pattern does not, by itself, isolate the cause of performance changes.

### E. Verify, document, and make the next decision concrete

- Test duration conversion, missing-value preservation, duplicate/quality checks, and any cleaning or joining logic actually implemented. Use small, clearly labelled test fixtures; never present fixture values as real F1 results.
- Keep routine tests independent of the network. Run a separate real-data smoke check when access is available.
- Generate an actual test report and basic HTML documentation for implemented functions. Keep comments/docstrings accurate.
- Write a short feasibility note recommending one focused research question and comparing it with the other candidate directions below. Include inputs, output, baseline, validation plan, scope, data limitations, and coordinator-approval points.
- Record important decisions and AI contributions as the work happens.

### Phase 1 acceptance criteria

Phase 1 is complete when a reader can follow the README to retrieve the pilot session, obtain a prepared lap table and quality report, regenerate the two figures, run the meaningful tests, and inspect the documented code. The actual outputs and a concrete proposal must exist. If network access prevents retrieval, distinguish completed local work from the still-unverified data workflow and name the remaining acceptance criteria.

## 6. Candidate modelling directions — not yet selected

| Direction | Possible question and output | Suitable comparison |
| --- | --- | --- |
| Next-lap prediction | Can information available after the current lap predict the next lap's duration in seconds? | Previous-lap prediction and regression versus a small 1D CNN, if approved and justified. |
| Tyre-age-associated pace | How does observed pace vary across tyre stints after accounting for available context? | Simple versus more flexible regression; a simplified pit-timing simulation could be a later extension. |
| Numerical telemetry analysis | How do sampling, interpolation, and smoothing affect estimated braking or speed profiles? | Alternative numerical methods with explicit reconstruction and sensitivity checks. |

My preference is to explore whether CNN makes sense for an F1 problem. If a 1D CNN is investigated, explain that it learns patterns across an ordered sequence rather than an image. Do not describe it as the final model until the question, scope, and method have been selected.

After Phase 1, present a proposal that is ready for discussion with the teaching team. Continue into substantial baseline/model development when the chosen scope and required approval are established. A request for progress should not be treated as approval of a previously undecided topic.

## 7. Evaluation rules for later modelling

These are methodological requirements for the proposed workflow, not claims of completed experiments.

### If next-lap prediction is selected

- Define exactly when the prediction is made and what conditions it assumes. For example, information available after lap 15 predicts lap 16.
- Inputs must not include the target lap's completed sector times, full telemetry, future weather, eventual stint length, or other future outcomes.
- Build ordered histories for the same driver. If histories are defined within a stint, do not cross pit stops or concatenate nonconsecutive laps after filtering.
- Separate whole race weekends between training, validation, and final testing, with chronology appropriate to the intended use. Do not randomly split overlapping windows across sets.
- Fit preprocessing and tune models using training/validation data only. Keep final test races out of exploratory model selection; the already-explored pilot is development data.
- Account for differences between circuits, drivers/teams, and seasons. A single pilot race cannot establish general performance.
- Define the evaluation population, such as uninterrupted dry running. If future lap conditions are used to select evaluation cases retrospectively, state this explicitly rather than presenting the model as knowing future incidents.
- Compare methods on the same records and information. Report MAE/RMSE in seconds, relevant error breakdowns, and measured runtime.
- Select one or two justified sensitivity studies, such as history length or feature groups. Do not assume the CNN will outperform the baseline.

### For all directions

Reported tyre age is not remaining life or measured physical wear. Changes in pace may also reflect traffic, fuel load, weather, and driver management. Describe associations and model assumptions accurately.

If using telemetry, verify channel meanings and sampling characteristics. Interpolation cannot create new measured detail. Derived quantities need appropriate validation; for example, distance integrated from speed is not an independent reference for validating that same speed integration.

Document what the data does not support. Never invent unavailable measurements, independent ground truth, causal effects, or a real team's optimal strategy.

## 8. Documentation and delivery expectations

Keep the README approachable: explain the purpose, value, inputs, outputs, execution steps, findings, and limitations before detailed implementation notes. Mark incomplete work honestly and replace planning placeholders only when the relevant deliverable exists.

Maintain an AI-use record naming ChatGPT, Cursor, and other tools actually used; their tasks and approximate contributions; verification; errors; and student-owned decisions. Do not invent student review, personal reflections, or understanding. Prepare brief explanations that help me answer assessment questions myself.

Record the selected code licence and separate dataset usage terms. A software package's licence does not automatically license its underlying data. Keep large caches and data out of Git unless their inclusion is justified and permitted; provide retrieval instructions so the workflow remains reproducible.

At the end of each completed phase, report:

1. What was implemented and why.
2. Files created or changed.
3. Exact commands to run the work.
4. Checks actually executed and their outcomes.
5. Real findings, limitations, and any blockers.
6. The next decision or task, explained simply.

**Begin with Phase 1 now. Deliver a working, documented pilot and a reviewable proposal while keeping the final project title and model choice open.**
