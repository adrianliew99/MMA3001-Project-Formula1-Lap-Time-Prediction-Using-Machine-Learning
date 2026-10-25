"""Generate HTML API references and a static notebook view without running cells.

Run from the repository root: python scripts/build_docs.py
Dependencies: requirements.txt and requirements-docs.txt.
"""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Build documentation using the active interpreter; propagate build errors."""
    output = ROOT / "docs" / "html"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pdoc", "mma3001_f1",
         "dataset/fastf1_export_dataset.py", "dataset/fastf1_export_multi_gp.py",
         "-o", str(output / "api")],
        cwd=ROOT, check=True,
    )
    # Export existing notebook content only; never retrain or download race data.
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "html",
         "--output", "notebook", "--output-dir", str(output),
         "MMA3001_F1_Next_Lap_Prediction.ipynb"],
        cwd=ROOT, check=True,
    )
    (output / "index.html").write_text('''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MMA3001 source documentation</title>
<style>body{font:18px/1.6 system-ui;max-width:850px;margin:3rem auto;padding:0 1rem}
a{color:#1257a6}li{margin:.7rem 0}</style></head><body>
<h1>MMA3001 F1 source documentation</h1>
<p>Reference documentation generated from the Python source and the saved notebook.</p>
<ul>
<li><a href="api/index.html">Python API reference</a>: configuration, data quality,
features, ingestion, modeling and race exporters, including source code.</li>
<li><a href="notebook.html">Main analysis notebook</a>: saved explanations, code,
and outputs. Cells were not executed during documentation generation.</li>
<li><a href="../../results/tests/report.html">Pytest test report</a></li>
<li><a href="../../results/tests/coverage/index.html">Python package coverage</a></li>
</ul><p>Coverage measures the Python package, not notebook cells or live provider
downloads. See <a href="../SUBMISSION.md">submission and reproduction notes</a>
for build commands and validation limits.</p></body></html>''', encoding="utf-8")
    print(f"Documentation: {output / 'index.html'}")


if __name__ == "__main__":
    main()
