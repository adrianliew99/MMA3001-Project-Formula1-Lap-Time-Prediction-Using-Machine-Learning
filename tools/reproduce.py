"""Reproduce all notebook code in an isolated temporary output directory.

The real CSV exports are read locally. New figures, metrics and predictions are
written under a temporary project root, never over the frozen results. A socket
guard blocks network connections in this process. No new hyperparameters are tried.
"""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import socket
import tempfile
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def protected_hashes():
    """Hash every committed result artifact before and after reproduction."""
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'results').rglob('*') if p.is_file() and not p.name.startswith('.')}


def main():
    """Execute unchanged notebook cells and compare selected scores to the record."""
    report_dir = ROOT / 'reports/reproduction'
    report_dir.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((ROOT / 'MMA3001_F1_Next_Lap_Prediction.ipynb').read_text())
    before = protected_hashes()
    log = io.StringIO()
    start = time.perf_counter()
    old_cwd, old_connect, old_connect_ex = Path.cwd(), socket.socket.connect, socket.socket.connect_ex
    def blocked(*args, **kwargs):
        raise RuntimeError('Network access forbidden during offline notebook reproduction')
    def display(*objects, **kwargs):
        for obj in objects:
            print(obj)
    namespace = {'__name__': '__main__', 'display': display}
    try:
        with tempfile.TemporaryDirectory(prefix='mma3001-reproduce-') as folder:
            folder = Path(folder)
            (folder / 'f1_dataset').symlink_to(ROOT / 'f1_dataset', target_is_directory=True)
            os.chdir(folder)
            socket.socket.connect = blocked
            socket.socket.connect_ex = blocked
            with redirect_stdout(log), redirect_stderr(log):
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), namespace)
                        plt.close('all')
            regenerated = pd.read_csv(folder / 'results/predictions/FINAL_holdout_2025_predictions.csv')
            original = pd.read_csv(ROOT / 'results/predictions/FINAL_holdout_2025_predictions.csv')
            pd.testing.assert_frame_equal(original, regenerated, check_exact=False, atol=1e-12, rtol=1e-12)
            regenerated_metrics = json.loads((folder / 'results/metrics/FINAL_holdout_2025_metrics.json').read_text())
            original_metrics = json.loads((ROOT / 'results/metrics/FINAL_holdout_2025_metrics.json').read_text())
            assert regenerated_metrics['selected_model'] == original_metrics['selected_model']
            for name, value in original_metrics['scores'].items():
                np.testing.assert_allclose(regenerated_metrics['scores'][name], value, atol=1e-12, rtol=0)
            selected = list(namespace['best_key'])
    finally:
        os.chdir(old_cwd)
        socket.socket.connect, socket.socket.connect_ex = old_connect, old_connect_ex
        (report_dir / 'execution.log').write_text(log.getvalue(), encoding='utf-8')
        if protected_hashes() != before:
            raise AssertionError('Protected result files changed')
    report = {'status': 'PASS', 'executed_code_cells': sum(c['cell_type'] == 'code' for c in notebook['cells']),
              'duration_s': time.perf_counter() - start, 'selected_key': selected,
              'predictions_equal_atol': 1e-12, 'original_results_unchanged': True,
              'network_connections_blocked': True,
              'scope': 'Numerical rerun, not a new holdout experiment; no retuning based on 2025.',
              'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
              'source_sha256': hashlib.sha256((ROOT / 'MMA3001_F1_Next_Lap_Prediction.ipynb').read_bytes()).hexdigest()}
    (report_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
