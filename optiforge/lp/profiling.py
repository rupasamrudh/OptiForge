"""Profiling and Phase Timing Instrumentation for OptiForge Solvers.

Stage 8.38: Lightweight, non-intrusive phase profiler and data containers
to measure execution breakdown, component runtime shares, and theoretical
Amdahl maximum speedups across solver iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SimplexProfileStats:
    """Detailed phase timing and profiling metrics for solver executions.

    Attributes
    ----------
    total_time : float
        Total elapsed wall-clock time in seconds.
    iterations : int
        Number of simplex pivots or solver iterations.
    component_times : dict[str, float]
        Cumulative time spent in each solver phase (in seconds).
    component_counts : dict[str, int]
        Number of invocations of each solver phase.
    provider_stats : dict[str, int]
        Snapshot of triangular solve provider execution and fallback telemetry.
    extra_metrics : dict[str, Any]
        Optional solver-specific telemetry (e.g. node counts, refactorizations).
    """

    total_time: float = 0.0
    iterations: int = 0
    component_times: dict[str, float] = field(default_factory=dict)
    component_counts: dict[str, int] = field(default_factory=dict)
    provider_stats: dict[str, int] = field(default_factory=dict)
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def runtime_shares(self) -> dict[str, float]:
        """Compute runtime share f_i = T_i / T_total for each component."""
        if self.total_time <= 0:
            return {k: 0.0 for k in self.component_times}
        return {k: float(v / self.total_time) for k, v in self.component_times.items()}

    def amdahl_maximums(self) -> dict[str, float]:
        """Compute theoretical Amdahl speedup S_max_i = 1 / (1 - f_i)."""
        shares = self.runtime_shares()
        res = {}
        for k, f_i in shares.items():
            if f_i >= 1.0 - 1e-12:
                res[k] = float("inf")
            else:
                res[k] = float(1.0 / (1.0 - f_i))
        return res

    def bottleneck_ranking(self) -> list[tuple[str, float, float, int, float]]:
        """Return components sorted descending by runtime share:
        (name, time_sec, share_fraction, call_count, amdahl_max_speedup).
        """
        shares = self.runtime_shares()
        amdahl = self.amdahl_maximums()
        ranked = []
        for name, t_sec in self.component_times.items():
            f_i = shares.get(name, 0.0)
            s_max = amdahl.get(name, 1.0)
            count = self.component_counts.get(name, 0)
            ranked.append((name, t_sec, f_i, count, s_max))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def summary(self) -> str:
        """Format an ASCII report of component breakdown and Amdahl speedup limits."""
        lines = [
            f"Simplex Profile Summary (Total: {self.total_time * 1e3:.2f} ms, Iterations: {self.iterations})",
            f"{'Phase / Component':<24} | {'Time (ms)':>10} | {'Share (%)':>10} | {'Calls':>7} | {'Avg (us)':>9} | {'Amdahl S_max':>12}",
            "-" * 84,
        ]
        ranking = self.bottleneck_ranking()
        for name, t_sec, f_i, count, s_max in ranking:
            avg_us = (t_sec / count * 1e6) if count > 0 else 0.0
            lines.append(
                f"{name:<24} | {t_sec * 1e3:>10.2f} | {f_i * 100:>9.2f}% | {count:>7} | {avg_us:>9.2f} | {s_max:>11.2f}x"
            )
        return "\n".join(lines)


class SimplexProfiler:
    """Lightweight, non-intrusive phase timer for solver executions."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.component_times: dict[str, float] = {}
        self.component_counts: dict[str, int] = {}
        self._t0: float = 0.0

    def start(self) -> None:
        if self.enabled:
            self._t0 = time.perf_counter()

    def record(self, component: str, elapsed: float) -> None:
        if not self.enabled:
            return
        self.component_times[component] = self.component_times.get(component, 0.0) + elapsed
        self.component_counts[component] = self.component_counts.get(component, 0) + 1

    def finish(
        self,
        iterations: int,
        provider_stats: Optional[dict[str, int]] = None,
        extra_metrics: Optional[dict[str, Any]] = None,
    ) -> Optional[SimplexProfileStats]:
        if not self.enabled:
            return None
        total_time = time.perf_counter() - self._t0 if self._t0 > 0 else sum(self.component_times.values())
        return SimplexProfileStats(
            total_time=total_time,
            iterations=iterations,
            component_times=dict(self.component_times),
            component_counts=dict(self.component_counts),
            provider_stats=dict(provider_stats or {}),
            extra_metrics=dict(extra_metrics or {}),
        )
