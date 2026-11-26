"""Runs of the existing solver paths on COPIES of the study for the model checks (the study of the caller is never changed): statics, natural frequencies.

No new solver: `solve.run_study` (ccx / direct linear path, the program's own choice), `eigen.run_eigen` (direct modes or ccx), `eigen.is_free_body`.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from ._checks_common import CheckContext


class CheckCancelled(RuntimeError):
    pass


def cancelled(ctx: CheckContext) -> bool:
    return ctx.cancel is not None and ctx.cancel.is_set()


def sub_workdir(ctx: CheckContext, name: str) -> Path:
    p = Path(ctx.workdir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def statics_copy(study: Any, load_cases: list[Any]) -> Any:
    """A copy of the study that runs the given load cases only (no frequencies, buckling, dynamics, print creep): `run_study` then does statics and nothing else."""
    st = copy.deepcopy(study)
    st.load_cases = list(load_cases)
    st.modal.enabled = False
    st.buckling.enabled = False
    st.nlbuckling.enabled = False
    st.dynamics.enabled = False
    if getattr(st, "printing", None) is not None and getattr(st.printing, "creep", None) is not None:
        st.printing.creep.enabled = False
    need = {t.case for lc in st.load_cases for t in st.thermal_loads(lc) if t.case}
    st.thermal.cases = [c for c in st.thermal.cases if c.name in need]
    return st


def run_statics(ctx: CheckContext, load_cases: list[Any], name: str, study: Any | None = None) -> Any:
    """`solve.run_study` of the given load cases on the mesh of the checks → Results."""
    from ..solve import run_study
    st = statics_copy(ctx.study if study is None else study, load_cases)
    return run_study(st, sub_workdir(ctx, name), study_dir=ctx.study_dir, on_log=ctx.log, cancel=ctx.cancel, mesh=ctx.mesh)


def modal_copy(study: Any, modes: int, f_max: float | None = None, free: bool = False) -> Any:
    """A copy of the study that computes natural frequencies only. free = every support removed (a free body)."""
    st = statics_copy(study, [])
    st.thermal.cases = []
    st.modal.enabled = True
    st.modal.modes = int(modes)
    st.modal.f_min = 0.0
    st.modal.f_max = None if f_max is None else float(f_max)
    st.modal.verify_ccx = False
    if free:
        st.constraints = []
    return st


def run_modes(ctx: CheckContext, st: Any, name: str) -> Any:
    """The first ModeSet of kind "modal" of `eigen.run_eigen` (the engine choice and the fallbacks are its own)."""
    from ..eigen import run_eigen
    sets = run_eigen(st, ctx.mesh, sub_workdir(ctx, name), ctx.log, cancel=ctx.cancel)
    modal = [m for m in sets if m.kind == "modal"]
    if not modal:
        raise RuntimeError("The natural-frequency analysis returned no modal set")
    return modal[0]


def solver_threads(ctx: CheckContext) -> int | None:
    return ctx.options.threads


def e_max(study: Any, mesh: Any) -> float:
    """The largest Young's modulus of the materials that are in the mesh [MPa]."""
    vals = [float(study.material_of(p).E) for p in mesh.part_names]
    return max(vals) if vals else 1.0


def part_of_node(mesh: Any, node: int) -> str:
    """The name of a part that owns the node (first element that uses it)."""
    el = np.nonzero((mesh.tets == int(node)).any(axis=1))[0]
    return str(mesh.part_names[int(mesh.tet_part[el[0]])]) if len(el) else "?"
