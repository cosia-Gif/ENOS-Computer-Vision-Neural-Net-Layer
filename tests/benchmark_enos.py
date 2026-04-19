"""
Performance + statistical benchmarks for the ENOS layer.

Three measurements:

1. Forward-pass wall time across a sweep of (batch, H, W, C) shapes.
2. Empirical scaling vs. the theoretical O(B*H*W*C) upper bound.
3. Output statistics (mean / max / sparsity) on natural-feature-map-like
   inputs at a range of thresholds — useful for deciding *where* to insert
   the layer and what threshold to use.

Run:
    python tests/benchmark_enos.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ENOSNet import ENOSLayer  # noqa: E402


@dataclass
class BenchResult:
    shape: tuple
    threshold: float
    mean_time_ms: float
    n_elements: int
    output_max: float
    output_mean: float
    sparsity: float  # fraction of mask == 0 in input


def _time(fn, repeats: int = 5) -> float:
    """Median wall time in seconds across `repeats` runs (excluding 1 warmup)."""
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def benchmark_shape_sweep() -> list[BenchResult]:
    rng = np.random.default_rng(42)
    layer = ENOSLayer(threshold=0.5)
    results: list[BenchResult] = []
    shapes = [
        (1, 14, 14, 8),
        (1, 28, 28, 16),
        (8, 28, 28, 16),
        (8, 56, 56, 32),
        (16, 56, 56, 32),
        (8, 112, 112, 32),
        (4, 224, 224, 16),
    ]
    for shape in shapes:
        x = rng.random(shape).astype(np.float32)
        t = _time(lambda: layer.forward(x))
        y = layer.forward(x)
        sparsity = float(np.mean((x > 0.5) == 0))
        results.append(
            BenchResult(
                shape=shape,
                threshold=0.5,
                mean_time_ms=t * 1000,
                n_elements=int(np.prod(shape)),
                output_max=float(y.max()),
                output_mean=float(y.mean()),
                sparsity=sparsity,
            )
        )
    return results


def benchmark_threshold_sweep() -> list[BenchResult]:
    rng = np.random.default_rng(43)
    shape = (4, 56, 56, 16)
    # Simulate ReLU-like activations (rectified normal distribution).
    raw = rng.standard_normal(shape).astype(np.float32)
    x = np.maximum(0.0, raw)
    # Normalize roughly to [0, 1].
    x /= max(1e-9, x.max())
    results: list[BenchResult] = []
    for t in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
        layer = ENOSLayer(threshold=t)
        wall = _time(lambda: layer.forward(x))
        y = layer.forward(x)
        sparsity = float(np.mean((x > t) == 0))
        results.append(
            BenchResult(
                shape=shape,
                threshold=t,
                mean_time_ms=wall * 1000,
                n_elements=int(np.prod(shape)),
                output_max=float(y.max()),
                output_mean=float(y.mean()),
                sparsity=sparsity,
            )
        )
    return results


def benchmark_vs_oracle_speedup() -> dict:
    """How much faster is the cumsum implementation than nested loops?"""
    from tests.test_enos_layer import enos_implementation_oracle

    rng = np.random.default_rng(44)
    shape = (1, 16, 16, 4)
    x = rng.random(shape).astype(np.float32)
    layer = ENOSLayer()
    t_vec = _time(lambda: layer.forward(x), repeats=5)
    t_oracle = _time(lambda: enos_implementation_oracle(x), repeats=2)
    return {
        "shape": shape,
        "vectorized_ms": t_vec * 1000,
        "oracle_ms": t_oracle * 1000,
        "speedup_x": t_oracle / max(t_vec, 1e-9),
    }


def _print_table(results: list[BenchResult], title: str) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'shape':<24}{'thr':>6}{'time_ms':>12}"
        f"{'elements':>14}{'sparsity':>12}{'out_max':>12}{'out_mean':>12}"
    )
    for r in results:
        print(
            f"{str(r.shape):<24}{r.threshold:>6.2f}{r.mean_time_ms:>12.3f}"
            f"{r.n_elements:>14d}{r.sparsity:>12.3f}{r.output_max:>12.2f}{r.output_mean:>12.3f}"
        )


def main() -> None:
    shape_results = benchmark_shape_sweep()
    _print_table(shape_results, "Shape sweep (threshold=0.5, uniform random input)")

    threshold_results = benchmark_threshold_sweep()
    _print_table(threshold_results, "Threshold sweep (4x56x56x16 ReLU-like input)")

    speedup = benchmark_vs_oracle_speedup()
    print("\n=== Vectorized vs nested-loop oracle ===")
    print(
        f"shape={speedup['shape']}  vectorized={speedup['vectorized_ms']:.3f} ms  "
        f"oracle={speedup['oracle_ms']:.3f} ms  speedup={speedup['speedup_x']:.1f}x"
    )


if __name__ == "__main__":
    main()
