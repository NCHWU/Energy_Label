"""Statistical significance: bootstrap confidence intervals for EPCA."""

import random
from typing import List, Tuple

from .schemas import TaskResult
from .scoring import epca


def bootstrap_epca_ci(
    results: List[TaskResult],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for EPCA.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = random.Random(seed)
    point = epca(results)
    n = len(results)

    samples = []
    for _ in range(n_bootstrap):
        resampled = rng.choices(results, k=n)
        samples.append(epca(resampled))

    samples.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    return point, samples[lo_idx], samples[hi_idx]


def bootstrap_epca_diff_ci(
    results_a: List[TaskResult],
    results_b: List[TaskResult],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the difference in EPCA (model_a - model_b).

    If the CI excludes 0, the difference is statistically significant at
    the given confidence level.

    Returns (point_diff, ci_lower, ci_upper).
    """
    rng = random.Random(seed)
    point_diff = epca(results_a) - epca(results_b)
    na, nb = len(results_a), len(results_b)

    diffs = []
    for _ in range(n_bootstrap):
        sa = rng.choices(results_a, k=na)
        sb = rng.choices(results_b, k=nb)
        diffs.append(epca(sa) - epca(sb))

    diffs.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    return point_diff, diffs[lo_idx], diffs[hi_idx]
