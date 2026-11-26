"""Sampling of the design space and the hold-out split (F15-S4). scipy.stats.qmc is imported inside the functions."""
from __future__ import annotations

import math
import warnings

import numpy as np

MAX_CONTINUOUS = 5              # continuous parameters of one surrogate (the plan: up to 5)
MIN_VALIDATION = 5              # smallest hold-out set the split helper makes (the plan: 20 %, at least 5); a surrogate that states
#                                 a 90 % band needs ⌈level / (1 − level)⌉ = 9 of them (calibrate.required_calibration)
VALIDATION_FRACTION = 0.2


def _unit(n: int, d: int, seed: int, method: str) -> np.ndarray:
    from scipy.stats import qmc
    if method == "sobol":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # "n is not a power of 2" – the balance property is not needed here
            return qmc.Sobol(d=d, scramble=True, rng=int(seed)).random(n)
    if method == "lhs":
        return qmc.LatinHypercube(d=d, rng=int(seed), optimization="random-cd").random(n)
    raise ValueError(f"Unknown sampling method '{method}' (lhs | sobol)")


def lhs_design(bounds: list[tuple[float, float]], n: int, seed: int = 0, method: str = "lhs") -> np.ndarray:
    """n space-filling points in the box `bounds` [(lo, hi), …] → array (n, d); deterministic for a given seed. method lhs = Latin
    hypercube (centred-discrepancy optimised), sobol = scrambled Sobol sequence."""
    d = len(bounds)
    if not 1 <= d <= MAX_CONTINUOUS:
        raise ValueError(f"The design needs 1–{MAX_CONTINUOUS} continuous parameters (got {d})")
    if n < 1:
        raise ValueError("The number of design points must be at least 1")
    lo = np.array([float(b[0]) for b in bounds])
    hi = np.array([float(b[1]) for b in bounds])
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo)):
        raise ValueError("Every parameter needs finite bounds with lo < hi")
    return lo + _unit(int(n), d, seed, method) * (hi - lo)


def assign_categories(categories: list, n: int, seed: int = 0) -> list:
    """Balanced assignment of n points to the categories (each appears floor(n / k) or ceil(n / k) times, shuffled by the seed)."""
    k = len(categories)
    if k == 0:
        raise ValueError("A categorical parameter needs at least one value")
    rng = np.random.default_rng(int(seed))
    out = [categories[i % k] for i in range(n)]
    rng.shuffle(out)
    return out


def validation_size(n: int, fraction: float = VALIDATION_FRACTION, minimum: int = MIN_VALIDATION) -> int:
    """Size of the hold-out set of n points: `fraction` of them, at least `minimum`."""
    return max(int(minimum), int(math.ceil(float(fraction) * n)))


def split_indices(n: int, fraction: float = VALIDATION_FRACTION, minimum: int = MIN_VALIDATION, seed: int = 0,
                  n_val: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(train, validation) indices of n points: a seeded random permutation, the last `validation_size` (or `n_val`) go to the hold-out
    set. The hold-out set is never used for the fit (it calibrates the error bars)."""
    n_val = validation_size(n, fraction, minimum) if n_val is None else int(n_val)
    perm = np.random.default_rng(int(seed)).permutation(n)
    return np.sort(perm[: n - n_val]), np.sort(perm[n - n_val:])


def candidate_pool(bounds: list[tuple[float, float]], n: int = 512, seed: int = 0) -> np.ndarray:
    """A dense Sobol pool in the box (candidates for the next points – adaptive refinement)."""
    return lhs_design(bounds, n, seed, method="sobol")
