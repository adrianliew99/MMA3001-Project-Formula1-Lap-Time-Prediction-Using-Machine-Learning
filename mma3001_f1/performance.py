"""Repeated CPU prediction timing and explicitly scoped memory measurements.

Benchmarking uses training and validation only. It never selects a new model or
reads/writes final holdout predictions. Native allocations and process RSS are
not measured by tracemalloc; serialized bytes are not runtime memory.
"""
import pickle
from time import perf_counter_ns
import tracemalloc

import numpy as np


def repeated_timing(operation, repeats=9, warmups=2):
    """Time a callable after warm-up; return seconds and descriptive quantiles.

    Args:
        operation: No-argument callable that completes synchronously.
        repeats: At least three measured calls.
        warmups: Nonnegative number of unmeasured calls.

    Returns:
        Dictionary with median, p95, min, max, repeat count and raw seconds.

    Raises:
        ValueError: Invalid repeat or warm-up counts.
    """
    if type(repeats) is not int or repeats < 3 or type(warmups) is not int or warmups < 0:
        raise ValueError('repeats >= 3 and warmups >= 0 must be integers')
    for _ in range(warmups):
        operation()
    seconds = []
    for _ in range(repeats):
        start = perf_counter_ns()
        operation()
        seconds.append((perf_counter_ns() - start) / 1e9)
    return {'median_s': float(np.median(seconds)),
            'p95_s': float(np.percentile(seconds, 95)),
            'min_s': min(seconds), 'max_s': max(seconds),
            'repeats': repeats, 'samples_s': seconds}


def traced_peak_bytes(operation):
    """Measure a separate call's peak traced allocation, excluding native/RSS use.

    Raises RuntimeError if another caller owns a tracing session, to avoid
    resetting their measurements. Tracing is stopped even if the call raises.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError('tracemalloc is already active')
    tracemalloc.start()
    try:
        operation()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def serialized_bytes(model):
    """Return protocol-5 pickle size, a storage proxy only (never load untrusted pickles)."""
    return len(pickle.dumps(model, protocol=5))
