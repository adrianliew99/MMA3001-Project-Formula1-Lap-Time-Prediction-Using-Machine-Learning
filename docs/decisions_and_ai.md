# Decisions and AI-use record

## 2026-09-13: initial build

- ChatGPT: user-supplied handoff preparation, as described by the user. Exact prior contribution is not independently known.
- Codex: generated nearly all new notebook, module, test and documentation text in this build; used to make a runnable starting workspace and explain the methods. Approximate contribution to these new files: 90–100% before student review. Cursor did not execute this build; its name appears in the supplied handoff only.
- Verified through synthetic offline tests, real FastF1 retrieval, notebook execution and chart inspection; see `reports/verification.md` for actual outcomes. Synthetic test values are never used as F1 findings.
- Existing `.venv` referenced an unavailable Python 3.12.10 executable. Created `.venv-project` using available Python 3.12.14, leaving the old environment and clean upstream `Fast-F1` checkout intact. The new environment relies on the bundled base interpreter on this machine; recreate it with a normal Python installation when moving machines.
- Initial dependency download was blocked by the sandbox network policy; an approved elevated download succeeded. Initial uv cache path was unusable, so cache was placed inside Project.
- Implementation decision: include race-control messages because deletion flags need that context; skip telemetry. Weather is retained separately, never silently joined.
- Implementation decision: a direct ridge solver provides an understandable baseline; neural-network epochs, batch size and learning rate are not relevant yet.
- Student-owned decisions still pending: research question, teaching-team approval, permitted dataset use, final race roles, final model, interpretation, and personal reflection. No student review or learning outcome is asserted.

## Your reflection after running the notebook

Record in your own words what you changed, what surprised you, errors you found, why the split avoids leakage, and what you can now explain without AI. Add approval details only when confirmed.
