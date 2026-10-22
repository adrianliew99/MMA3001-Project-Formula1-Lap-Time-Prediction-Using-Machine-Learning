"""Run the Section 17 audit independently: python -m tools.benchmark.

Fixed representative comparisons come from each feature/model family's best
2024 validation MAE. Fit once on 2022-23, then repeatedly predict 2024 only.
Existing notebook outputs and final 2025 artifacts are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from mma3001_f1.performance import repeated_timing, traced_peak_bytes, serialized_bytes
from tools.notebook_contract import load_contract

ROOT = Path(__file__).resolve().parents[1]


def run(output, repeats=9):
    """Write validation-only timing, accuracy and memory evidence to a separate folder."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    notebook = ROOT / 'MMA3001_F1_Next_Lap_Prediction.ipynb'
    api = load_contract(notebook)
    clean, _ = api['audit_pairs'](api['load_exports'](ROOT / 'f1_dataset'))
    train, val, _ = api['split_by_season'](clean)
    grid_path = ROOT / 'results/metrics/validation_grid.csv'
    grid = pd.read_csv(grid_path)
    candidates = grid[grid.model.ne('Persistence')].sort_values(['MAE_s', 'RMSE_s'])
    representatives = candidates.groupby(['features', 'model'], sort=False).head(1)
    records, details = [], []
    for feature, family, params in [('A_persistence', 'Persistence', 'none')] + list(
            representatives[['features', 'model', 'params']].itertuples(index=False, name=None)):
        if family == 'Persistence':
            # Actual zero-delta allocation and next-lap reconstruction, not a literal zero timer.
            previous = val.PreviousLapTime_s.to_numpy(dtype=float)
            predict = lambda: previous + np.zeros(len(previous), dtype=float)
            predict_one = lambda: previous[:1] + np.zeros(1, dtype=float)
            model, fit_s, model_bytes = None, None, 0
            predicted_next = predict()
            delta = predicted_next - previous
            input_bytes = previous.nbytes
        else:
            numeric, categorical = api['FEATURE_SETS'][feature]
            X_train = api['prepare_model_inputs'](train, numeric, categorical)
            X_val = api['prepare_model_inputs'](val, numeric, categorical)
            regressor = dict(api['candidate_models']()[family])[params]
            model = api['make_pipeline'](regressor, numeric, categorical)
            target = train.DeltaLapTime_s.to_numpy(dtype=float)
            start = time.perf_counter()
            model.fit(X_train, target)
            fit_s = time.perf_counter() - start
            previous = val.PreviousLapTime_s.to_numpy(dtype=float)
            one_previous = previous[:1]
            one_X = X_val.iloc[:1]
            predict = lambda: previous + model.predict(X_val)
            predict_one = lambda: one_previous + model.predict(one_X)
            predicted_next = predict()
            delta = predicted_next - previous
            input_bytes = int(X_val.memory_usage(index=True, deep=True).sum())
            model_bytes = serialized_bytes(model)
        batch = repeated_timing(predict, repeats=repeats)
        single = repeated_timing(predict_one, repeats=repeats)
        peak = traced_peak_bytes(predict)
        scores = api['evaluate_split'](val, delta)
        records.append({'features': feature, 'model': family, 'params': params,
                        'rows': len(val), 'fit_once_s': fit_s,
                        'batch_median_s': batch['median_s'], 'batch_p95_s': batch['p95_s'],
                        'amortized_us_per_row': batch['median_s'] / len(val) * 1e6,
                        'single_row_median_ms': single['median_s'] * 1e3,
                        'single_row_p95_ms': single['p95_s'] * 1e3,
                        'input_deep_bytes': input_bytes, 'serialized_model_bytes': model_bytes,
                        'prediction_traced_peak_bytes': peak, **scores})
        details.append({'key': [feature, family, params], 'batch': batch, 'single': single})
    table = pd.DataFrame(records)
    table.to_csv(output / 'validation_performance.csv', index=False)
    evidence_paths = [notebook, grid_path, ROOT / 'f1_dataset/combined_next_lap_ml.csv']
    metadata = {
        'measured_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'cpu_model': platform.processor(), 'logical_cpu_count': os.cpu_count(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn']},
        'thread_environment': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'random_forest_n_jobs': -1,
        'train_years': [2022, 2023], 'validation_years': [2024], 'holdout_used': False,
        'rows_train': len(train), 'rows_validation': len(val), 'fit_repetitions': 1,
        'scope': 'End-to-end pipeline predict plus next-lap reconstruction; prepared inputs; no CSV loading, feature preparation, metrics, serialization or memory tracing in timed calls.',
        'memory_scope': 'DataFrame deep input bytes (persistence: NumPy bytes), pickle storage bytes, separate-call tracemalloc peak. No total/native peak RSS claim.',
        'limitations': ['Shared-host timings, not a real-time guarantee',
                       'p95 over few repetitions is descriptive, not a confidence bound',
                       'Single fit duration is not a repeated training benchmark',
                       'CPU tree/forest operation counts depend on paths and are not measured FLOPS; no roofline/arithmetic-intensity claim'],
        'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_paths},
        'timings': details,
    }
    (output / 'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/performance')
    parser.add_argument('--repeats', type=int, default=9)
    args = parser.parse_args()
    print(run(args.output, args.repeats).to_string(index=False))
