"""Run-time tiers of the verification cases (verification.cases.CASES) – used by tests/test_verification.py (pytest markers)
and by verification.release_check (the fixed list of the release run).

- VERY_SLOW: cases of several minutes up to an hour (measured with verification.run_all on the development notebook, 8 cores) or
  needing the optional Project Chrono environment – pytest marker `very_slow` (not in `-m slow`).
- every other case: pytest marker `slow` (Gmsh + CalculiX, seconds to ~1.5 min each).
- RELEASE_CASES: the fixed, reasonably fast set of the release check (statics, heat, fatigue, printing, joints, eigen, beams).
"""
from __future__ import annotations

# measured durations [s] (verification/out, November 2026): KK4 ~3300 (impact, *DYNAMIC), K6 ~460-590 (bolt with contact joint),
# K2 ~250 (Hertz, mortar), KK2 ~190 (snap-fit, NLGEOM + friction), KK5 / MB4C / MB4R (Chrono; KK5 ~100, environment optional)
# TO2 / TO3 / TO4 (topology optimisation with manufacturing constraints: ~40 analyses of the optimisation + the same without the constraint) ~4-7 min each
# SU1 / SU2 / SU3 / SU4 (surrogate model over cheap 2D sweeps: 140 / 180 / 114 / 48 analyses) ~5-15 min each
VERY_SLOW = frozenset({"KK4", "K6", "K2", "KK2", "KK5", "MB4C", "MB4R", "TO2", "TO3", "TO4", "SU1", "SU2", "SU3", "SU4"})
NEEDS_CHRONO = frozenset({"KK5", "MB4C", "MB4R"})

RELEASE_CASES = (
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V7R", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7",
    "U1", "U2", "U3",
    "P1", "P2", "P3", "P4", "P5",
    "S1", "S2", "S3", "W1",
    "M1", "B1",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
)


def tier(key: str) -> str:
    """"very_slow" or "slow" (pytest marker name) of a verification case."""
    return "very_slow" if key in VERY_SLOW else "slow"
