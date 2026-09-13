"""Collect and plot a real session. Run from Project after editable installation."""
import argparse
from pathlib import Path
from mma3001_f1.data import collect_session, plot_pilot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--event", default="Bahrain")
    parser.add_argument("--session", default="R")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, audit, manifest = collect_session(root, args.year, args.event, args.session)
    plot_pilot(data, root / "results/figures" / manifest["session_id"])
    print({k: audit[k] for k in ["rows", "drivers", "eligible_rows"]})

