from __future__ import annotations

import functools
import time

import numpy as np
from loguru import logger


class FunctionTimer:
    """Collects timing data for profiled functions."""

    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = {}

    def record(self, name: str, duration: float) -> None:
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

    def reset(self) -> None:
        """Clear all recorded timings."""
        self.timings.clear()

    def reset_records(self) -> None:
        """Clear all recorded timings."""
        for name in self.timings:
            self.timings[name] = []

    def get_stats(self, cte: float = 1.0) -> dict[str, dict[str, float]]:
        """Get summary statistics for all timed functions."""
        stats = {}
        for name, unnor_durations in self.timings.items():
            durations = np.array(unnor_durations) * cte
            stats[name] = {
                "count": len(durations),
                "total": durations.sum(),
                "mean": float(durations.mean()),
                "median": float(np.median(durations)),
                "min": min(durations),
                "max": max(durations),
                "std": float(np.std(durations)) if len(durations) > 1 else 0.0,
            }
        return stats

    def get_last_timings(self) -> dict[str, float]:
        """Get the most recent timing for each function."""
        return {name: durations[-1] for name, durations in self.timings.items() if durations}

    def summary(self) -> None:
        """Print a formatted summary of all timing statistics."""
        if not self.timings:
            logger.info("No timing data recorded.")
            return

        stats = self.get_stats()

        # Print header
        logger.info("\n" + "=" * 80)
        logger.info("Function Timer Statistics")
        logger.info("=" * 80)

        # Print statistics for each function
        for name, stat in stats.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Calls:      {stat['count']}")
            logger.info(f"  Total:      {stat['total']:.6f} s")
            logger.info(f"  Mean:       {stat['mean']:.6f} s")
            logger.info(f"  Median:     {stat['median']:.6f} s")
            logger.info(f"  Min:        {stat['min']:.6f} s")
            logger.info(f"  Max:        {stat['max']:.6f} s")
            if stat["count"] > 1:
                logger.info(f"  Std Dev:    {stat['std']:.6f} s")

        logger.info("\n" + "=" * 80)


def profile_method(timer: FunctionTimer, name: str):
    """Decorator to profile a method and record timing to a FunctionTimer."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - tic
            timer.record(name, duration)
            return result

        return wrapper

    return decorator
