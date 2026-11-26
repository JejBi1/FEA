"""Equation solver of ccx (F13 step 2): what the process gets (threads, PaStiX precision), what the writers put into the decks
(mixed decks, the dynamics reference, submodel, voxels), what the results record per procedure and what the user is told.
Fast – synthetic meshes, a fake process instead of ccx.
"""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from fea_core import config
from fea_core.study import LoadCase, Pressure
from test_loads import bar_mesh, bar_study


def _pretend(monkeypatch, exe: str, win: bool = True) -> None:
    """The ccx build that the rule looks at (config.CCX_EXE)."""
    monkeypatch.setattr(config, "CCX_EXE", Path("x") / exe)
    monkeypatch.setattr(config, "IS_WIN", win)


# ================================================================= the environment of the process

class _Out:
    def __iter__(self):
        return iter(["Job finished\n"])

    def close(self) -> None:
        pass


class _FakeProc:
    last: dict = {}

    def __init__(self, cmd, **kw):
        type(self).last = kw
        self.args, self.stdout = cmd, _Out()

    def poll(self):
        return 0

    def wait(self):
        return 0


@pytest.fixture
def fake_ccx(tmp_path, monkeypatch):
    """run_ccx with a fake process: returns a function job_text → the environment the process would get."""
    from fea_core.ccx import runner
    monkeypatch.setattr(runner, "subprocess", NS(Popen=_FakeProc, TimeoutExpired=Exception, PIPE=-1, STDOUT=-2))
    exe = tmp_path / "bin" / "ccx"
    exe.parent.mkdir()
    exe.write_text("x")
    work = tmp_path / "work"
    work.mkdir()

    def run(text: str, name: str = "job") -> dict:
        (work / f"{name}.inp").write_text(text)
        assert runner.run_ccx(work, name, exe=exe).finished
        return _FakeProc.last["env"]
    return run


def test_spooles_input_gets_one_equation_solver_thread_and_pastix_input_does_not(fake_ccx, monkeypatch):
    """Multi-threaded SPOOLES of ccx_static returns wrong results without a message (B1: 1.67 instead of 21.65) → one thread for every input
    that asks for it; PaStiX takes min(threads, the variable), so it must NOT get the variable – also not from the environment of the host."""
    monkeypatch.setenv("CCX_NPROC_EQUATION_SOLVER", "4")                 # a stray setting of the host reaches neither of the two
    spooles = fake_ccx("*STEP\n*STATIC, SOLVER=SPOOLES\n*END STEP\n")
    pastix = fake_ccx("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n")
    assert spooles["CCX_NPROC_EQUATION_SOLVER"] == "1"
    assert "CCX_NPROC_EQUATION_SOLVER" not in pastix
    mixed = fake_ccx("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n*STEP, PERTURBATION\n*FREQUENCY, SOLVER=SPOOLES\n4, 0.\n*END STEP\n")
    assert mixed["CCX_NPROC_EQUATION_SOLVER"] == "1"                     # one process → the SPOOLES step decides (PaStiX then runs on one thread)


def test_marked_binary_gets_the_threads_of_the_job_for_spooles(fake_ccx, tmp_path, monkeypatch):
    """A ccx that carries the serial-solve patch (the marker of `ccx -v`, installer/*/patches/ccx_2.23_spooles_serial_solve.patch) factorizes SPOOLES on the
    threads of the job; the binary of `fake_ccx` has no marker (one thread). PaStiX never gets the variable."""
    from fea_core.ccx import runner
    monkeypatch.delenv("FEA_SPOOLES_THREADS", raising=False)
    monkeypatch.delenv("FEA_THREADS", raising=False)                                                       # with FEA_THREADS the program sets the threads itself (test_threads_cap.py)
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert fake_ccx("*STEP\n*STATIC, SOLVER=SPOOLES\n*END STEP\n")["CCX_NPROC_EQUATION_SOLVER"] == "1"
    exe = tmp_path / "bin" / "ccx"
    exe.write_bytes(b"x" + runner.FACTOR_CHECK_MARKER + runner.ORDERED_MARKER)                           # the same file, now with the marker (the cache follows size and time)
    env = fake_ccx("*STEP\n*STATIC, SOLVER=SPOOLES\n*END STEP\n")
    assert env["CCX_NPROC_EQUATION_SOLVER"] == "6"
    assert "CCX_NPROC_EQUATION_SOLVER" not in fake_ccx("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n")


def test_pastix_precision_default_and_override(fake_ccx, monkeypatch):
    """The PaStiX of the Linux build computes in float unless told otherwise (the factorization of stiff models ends with NaN and repeats in double)."""
    monkeypatch.delenv("PASTIX_MIXED_PRECISION", raising=False)
    assert fake_ccx("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n")["PASTIX_MIXED_PRECISION"] == "0"
    monkeypatch.setenv("PASTIX_MIXED_PRECISION", "1")                    # an explicit setting of the user stays
    assert fake_ccx("*STEP\n*STATIC, SOLVER=PASTIX\n*END STEP\n")["PASTIX_MIXED_PRECISION"] == "1"


# ================================================================= decks

def _deck(tmp_path, st, **kw) -> str:
    from fea_core.ccx.inp import write_inp
    write_inp(st, bar_mesh(), tmp_path / "d.inp", **kw)
    return (tmp_path / "d.inp").read_text()


def _study():
    st = bar_study()
    st.load_cases = [LoadCase("L", [Pressure("z10", 0.2)])]
    return st


def test_prestressed_modal_deck_keeps_pastix_for_the_static_step(tmp_path, monkeypatch):
    """Mixed deck (prestressed modal analysis, Campbell): the static step PaStiX, the natural frequencies SPOOLES. Measured on Y9 (219 k DOF,
    one thread for the whole job because of the SPOOLES step): the static step 47–56 s with PaStiX against 62–71 s with SPOOLES, the whole
    deck 117–130 s against 126–129 s – the mixed deck stays."""
    from fea_core.ccx.runner import uses_spooles
    _pretend(monkeypatch, "ccx_static.exe")
    txt = _deck(tmp_path, _study(), procedure="modal", eigen_line="4, 0.", prestress=True)
    assert "*STATIC, SOLVER=PASTIX" in txt and "*FREQUENCY, SOLVER=SPOOLES" in txt
    assert txt.index("*STATIC, SOLVER=PASTIX") < txt.index("*STEP, PERTURBATION") < txt.index("*FREQUENCY, SOLVER=SPOOLES")
    assert uses_spooles(tmp_path / "d.inp")
    plain = _deck(tmp_path, _study(), procedure="modal", eigen_line="4, 0.")                  # natural frequencies alone: SPOOLES only
    assert "SOLVER=SPOOLES" in plain and "PASTIX" not in plain
    buckle = _deck(tmp_path, _study(), procedure="buckle", eigen_line="2")                    # buckling: PaStiX (B1 17.8 s : 22.5 s)
    assert "*BUCKLE, SOLVER=PASTIX" in buckle and "SPOOLES" not in buckle
    st = _study()
    st.solver.solver = "SPOOLES"                                                              # an explicit choice is kept everywhere
    assert "PASTIX" not in _deck(tmp_path, st, procedure="modal", eigen_line="4, 0.", prestress=True)
    st.solver.solver = "PASTIX"
    assert "SPOOLES" not in _deck(tmp_path, st, procedure="modal", eigen_line="4, 0.", prestress=True)


def test_decks_without_pastix_use_spooles(tmp_path, monkeypatch):
    """Linux package with ccx_spooles only (the licence decision on ccx_pastix): everything SPOOLES."""
    _pretend(monkeypatch, "ccx_spooles", win=False)
    txt = _deck(tmp_path, _study(), procedure="modal", eigen_line="4, 0.", prestress=True)
    assert "SOLVER=PASTIX" not in txt and txt.count("SOLVER=SPOOLES") == 2
    assert config.solver_name(config.AUTO_SOLVER, "static") == "SPOOLES" and config.solver_name("PASTIX", "static") == "SPOOLES"


def test_dynamics_reference_decks_follow_the_solver_rule(tmp_path, monkeypatch):
    """The ccx reference of the modal superposition (*STEADY STATE DYNAMICS, *MODAL DYNAMIC) wrote SOLVER=PARDISO for ever ("the PARDISO
    library is not linked" in ccx_static, verification case Y13) → the solver of the *FREQUENCY step of the same deck, never the removed one."""
    from fea_core.dynamics import etalon, response
    decks = {}

    def fake_write_inp(st, mesh, path, **kw):
        Path(path).write_text("*NODE\n1, 0, 0, 0\n*STEP\n*FREQUENCY, SOLVER=" + config.solver_name(st.solver.solver, "modal")
                              + ", STORAGE=YES\n4, 0.\n*END STEP\n")

    def fake_run_ccx(workdir, job, exe=None, **kw):
        decks[job] = (Path(workdir) / f"{job}.inp").read_text()
        return NS(finished=True, errors=[], returncode=0)
    term = {"exc": NS(kind="force", table=None), "F": np.array([[0.0, 0.0, 1.0]]), "scale": 1.0, "phase": 0.0, "kind": "force",
            "pts": None, "dir": np.array([0.0, 0.0, 1.0])}
    monkeypatch.setattr("fea_core.ccx.inp.write_inp", fake_write_inp)
    monkeypatch.setattr("fea_core.ccx.runner.run_ccx", fake_run_ccx)
    monkeypatch.setattr("fea_core.eigen._sub_study", lambda study, cases: study)
    monkeypatch.setattr(response, "build_terms", lambda *a, **k: (NS(items=[term]), None))
    monkeypatch.setattr(etalon, "read_print", lambda *a, **k: [])
    monkeypatch.setattr(etalon, "_compare", lambda *a, **k: {"max_rel_error": 0.0})

    def run(solver: str) -> dict:
        study = NS(solver=NS(solver=solver, exe=""),
                   dynamics=NS(damping=NS(kind="modal"), harmonic=NS(enabled=True, f_min=10.0, f_max=100.0, points=5, bias=1.0, excitation=[]),
                               transient=NS(enabled=True, excitation=[])))
        out = NS(probes=[{"nodes": np.array([0]), "w": np.array([1.0])}], harmonic={"peak_node": None},
                 transient={"t": np.linspace(0.0, 1.0, 11), "ub": np.zeros((11, 3))})
        decks.clear()
        etalon.run_etalon(study, None, NS(K=4), np.zeros(4), out, tmp_path, lambda s: None)
        return dict(decks)
    _pretend(monkeypatch, "ccx_static.exe")
    d = run(config.AUTO_SOLVER)
    assert "*STEADY STATE DYNAMICS, SOLVER=SPOOLES" in d["etalon_harmonic"] and "*MODAL DYNAMIC, SOLVER=SPOOLES" in d["etalon_transient"]
    assert "PARDISO" not in d["etalon_harmonic"] + d["etalon_transient"]
    d = run("PASTIX")                                                      # an explicit choice goes to every step of the deck
    assert "*STEADY STATE DYNAMICS, SOLVER=PASTIX" in d["etalon_harmonic"] and "SOLVER=SPOOLES" not in d["etalon_harmonic"]


def test_other_writers_take_the_requested_solver(tmp_path, monkeypatch):
    """Submodel and voxel decks ignored Study.solver.solver (always the automatic one)."""
    from fea_core.printing import voxel as V
    from fea_core.study import Material
    from fea_core.submodel import write_submodel_inp
    from fea_core.submodel_run import _write_inp as write_submodel_run
    from test_voxel import _gen
    from fea_core.printing import PrintSpec
    _pretend(monkeypatch, "ccx_static.exe")
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tets = np.array([[0, 1, 2, 3]])
    mat = Material("m", 210000.0, 0.3)
    for req, want in ((config.AUTO_SOLVER, "PASTIX"), ("SPOOLES", "SPOOLES"), ("PASTIX", "PASTIX")):
        p = tmp_path / f"sub_{req}.inp"
        write_submodel_inp(coords, tets, mat, np.array([0, 1]), np.zeros((2, 3)), p, solver=req)
        assert f"*STATIC, SOLVER={want}" in p.read_text()
    m = bar_mesh()
    mats = {m.part_names[0]: Material("m", 210000.0, 0.3)}
    for req, want in ((config.AUTO_SOLVER, "PASTIX"), ("SPOOLES", "SPOOLES")):
        p = tmp_path / f"run_{req}.inp"
        write_submodel_run(p, m, mats, {0: [0.0, 0.0, 0.0]}, np.zeros((len(m.nodes), 3)), "t", req)
        assert f"*STATIC, SOLVER={want}" in p.read_text()
    g = _gen(PrintSpec("A", walls=1, top_layers=0, bottom_layers=0, infill=1.0), np.array([1.0, 1.0, 0.4]))
    mv = V.build_model(g)
    bins = V._angle_bins(g.angle[tuple(mv["ijk"].T)], 36)
    axes = V.bin_axes(np.eye(3), 36)
    for req, want in ((config.AUTO_SOLVER, "PASTIX"), ("SPOOLES", "SPOOLES")):
        p = tmp_path / f"vox_{req}.inp"
        V.write_inp(p, mv["nodes"], mv["hexes"], bins, axes, np.eye(6) * 1000, np.arange(4), [np.zeros((4, 3))], solver=req)
        assert f"*STATIC, SOLVER={want}" in p.read_text()


# ================================================================= what the results record

def _thermal_study():
    from fea_core.thermal.model import ThermalCase
    st = _study()
    st.thermal.cases = [ThermalCase("T")]
    return st


def test_procedures_of_a_run_and_the_recorded_solvers(monkeypatch):
    """Results.info["ccx_solver"] used to be the solver of the static step even where heat, natural frequencies and shells ran with another."""
    from fea_core.solve import ccx_procedures, record_ccx_solvers, solver_label
    _pretend(monkeypatch, "ccx_static.exe")
    st = _study()
    assert ccx_procedures(st, True) == ["static"] and ccx_procedures(st, False) == []        # NGSolve mechanics: no ccx static step
    st = _thermal_study()
    st.modal.enabled = True
    st.buckling.enabled = True
    st.dynamics.enabled = True
    assert ccx_procedures(st, True) == ["static", "heat", "modal", "buckle"]                  # dynamics: modal basis + static correction (static is there)
    assert ccx_procedures(st, False) == ["heat", "modal", "buckle", "static"]
    res = NS(info={})
    record_ccx_solvers(st, res, ccx_procedures(st, True))
    assert res.info["ccx_solvers"] == {"static": "PASTIX", "heat": "PASTIX", "modal": "SPOOLES", "buckle": "PASTIX"}
    assert res.info["ccx_solver"] == "PASTIX (static/heat/buckle), SPOOLES (modal)"
    st.solver.solver = "SPOOLES"
    record_ccx_solvers(st, res, ccx_procedures(st, True))
    assert set(res.info["ccx_solvers"].values()) == {"SPOOLES"} and res.info["ccx_solver"] == "SPOOLES"
    assert solver_label({}) == "" and solver_label({"static": "PASTIX"}) == "PASTIX"
    _pretend(monkeypatch, "ccx_spooles", win=False)                                           # the Linux package: all SPOOLES
    st.solver.solver = config.AUTO_SOLVER
    record_ccx_solvers(st, res, ccx_procedures(st, True))
    assert res.info["ccx_solver"] == "SPOOLES"


def test_nonlinear_mortar_is_recorded_as_its_own_procedure(monkeypatch):
    """Mortar contact where ccx solves it (Linux): SPOOLES for the mechanics, the other procedures as usual."""
    from fea_core.solve import ccx_procedures, record_ccx_solvers
    from fea_core.contacts.model import ContactPair
    from fea_core.study import Study
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictionless")]
    st.contact_settings.method = "mortar"
    monkeypatch.setenv("FEA_MORTAR", "1")
    _pretend(monkeypatch, "ccx_pastix", win=False)
    assert isinstance(st, Study) and ccx_procedures(st, True) == ["mortar"]
    res = NS(info={})
    record_ccx_solvers(st, res, ["mortar", "heat"])
    assert res.info["ccx_solvers"] == {"mortar": "SPOOLES", "heat": "PASTIX"}
    monkeypatch.setenv("FEA_MORTAR", "0")                                                     # substituted by surface-to-surface → PaStiX
    assert ccx_procedures(st, True) == ["static"]


def test_ccx_solvers_survive_the_results_file(tmp_path):
    from fea_core.beams.run import empty_mesh
    from fea_core.results import Results
    info = {"backend": "ccx", "ccx_solver": "PASTIX (static), SPOOLES (modal)", "ccx_solvers": {"static": "PASTIX", "modal": "SPOOLES"}}
    Results(empty_mesh(), [], [], info).save(tmp_path / "results.npz")
    back = Results.load(tmp_path / "results.npz").info
    assert back["ccx_solvers"] == info["ccx_solvers"] and back["ccx_solver"] == info["ccx_solver"]


def test_the_summary_reports_the_protocol_and_the_ccx_solvers_side_by_side():
    """A result of the direct solver (own assembly + rslab) that also ran CalculiX jobs (natural frequencies, heat) carries both records: the
    protocol of the linear solver (Results.info["solver"]) and the equation solver of each ccx procedure (Results.info["ccx_solvers"])."""
    from fea_core.beams.run import empty_mesh
    from fea_core.mcp.diagnostics import results_summary
    from fea_core.results import Results
    protocol = {"backend": "ngs_direct", "engine": "rslab 0.1", "ndof": 1000, "reason": "the direct solver fits into the free memory"}
    info = {"backend": "ngs_direct", "solver": protocol, "ccx_solvers": {"modal": "SPOOLES", "heat": "PASTIX"}, "ccx_solver": "SPOOLES (modal), PASTIX (heat)"}
    res = Results(empty_mesh(), [], [], info)
    sess = NS(study=NS(name="x"), require_results=lambda: res)
    d = results_summary(sess)
    assert d["backend"] == "ngs_direct" and d["solver"]["engine"] == "rslab 0.1"
    assert d["ccx_solvers"] == {"modal": "SPOOLES", "heat": "PASTIX"}
    del info["ccx_solvers"]
    assert "ccx_solvers" not in results_summary(sess)                                          # nothing invented for a pure direct-solver result


def test_report_reads_the_solver_of_the_procedure(monkeypatch):
    from fea_core.report.word import _ccx_solver
    _pretend(monkeypatch, "ccx_static.exe")
    st = _study()
    res = NS(info={"ccx_solvers": {"mortar": "SPOOLES", "heat": "PASTIX"}, "ccx_solver": "SPOOLES (mortar), PASTIX (heat)"})
    assert _ccx_solver(st, res) == "SPOOLES" and _ccx_solver(st, res, "heat") == "PASTIX"     # the mechanics = the mortar procedure
    old = NS(info={"ccx_solver": "PASTIX"})                                                   # results saved before the per-procedure record
    assert _ccx_solver(st, old) == "PASTIX"
    assert _ccx_solver(st, None, "modal") == "SPOOLES" and _ccx_solver(st, None, "static") == "PASTIX"


# ================================================================= what the user is told

def test_mortar_substitution_is_recorded_and_warned_once(monkeypatch):
    """The readiness check warns, but it does not run when a finished mesh is taken over without the geometry → the nonlinear run says it too,
    with the same text (listed once), and the results record both formulations."""
    from fea_core.contacts.check import check
    from fea_core.contacts.model import ContactPair, MORTAR_SUBSTITUTED_WARNING, effective_method
    from fea_core.contacts.run import record_method
    from fea_core.mcp.f8_common import method_info
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictionless")]
    st.contact_settings.method = "mortar"
    monkeypatch.setenv("FEA_MORTAR", "0")
    assert effective_method(st) == "surface_to_surface"
    warnings: list = []
    info: dict = {"nonlinear": {"mode": "branches"}}
    record_method(st, info, warnings)
    record_method(st, info, warnings)
    assert warnings == [MORTAR_SUBSTITUTED_WARNING]
    assert info["nonlinear"]["contact_method"] == {"chosen": "mortar", "effective": "surface_to_surface", "substituted": True}
    assert info["nonlinear"]["mode"] == "branches"
    _, w = check(st, ["A", "B"])
    assert MORTAR_SUBSTITUTED_WARNING in w                                                    # the same text as the readiness check → no duplicate
    mi = method_info(st)                                                                      # MCP: the label of the choice AND the effective one
    assert mi["method_label"] == "mortar" and mi["effective_method"] == "surface_to_surface" and mi["substituted"] is True
    assert mi["effective_method_label"].startswith("surface-to-surface") and mi["note"] == MORTAR_SUBSTITUTED_WARNING
    monkeypatch.setenv("FEA_MORTAR", "1")
    warnings, info = [], {}
    record_method(st, info, warnings)
    assert warnings == [] and info["nonlinear"]["contact_method"]["substituted"] is False
    assert method_info(st)["substituted"] is False and "note" not in method_info(st)
    st.contacts = []                                                                          # no active contact: nothing to say
    warnings, info = [], {}
    record_method(st, info, warnings)
    assert warnings == [] and info == {}


def test_report_shows_the_formulation_that_was_computed(monkeypatch):
    from fea_core.contacts.model import ContactPair
    from fea_core.report.f8 import _effective_method
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictionless")]
    st.contact_settings.method = "mortar"
    monkeypatch.setenv("FEA_MORTAR", "1")
    assert _effective_method(st, None) == "mortar"
    done = NS(info={"nonlinear": {"contact_method": {"chosen": "mortar", "effective": "surface_to_surface", "substituted": True}}})
    assert _effective_method(st, done) == "surface_to_surface"                                 # what the results ran with wins over this installation


def test_explicit_pastix_warnings(monkeypatch):
    """An explicit PaStiX where the automatic choice would use SPOOLES: multi-case shells (the job dies / is slower) and mortar where it is solved."""
    from fea_core.contacts.model import ContactPair
    from fea_core.solvers import solver_warnings
    _pretend(monkeypatch, "ccx_static.exe")
    st = _study()
    st.solver.solver = "PASTIX"
    assert solver_warnings(st) == []                                       # a bar with one load case: PaStiX is the right choice
    st.shell.enabled = True
    assert solver_warnings(st) == []                                       # one load case
    st.load_cases.append(LoadCase("L2", [Pressure("z10", 0.1)]))
    w = solver_warnings(st)
    assert len(w) == 1 and "shell" in w[0]
    st.solver.solver = config.AUTO_SOLVER                                  # the automatic choice (stored word) never warns
    assert solver_warnings(st) == []
    st.solver.solver = "PASTIX"
    st.shell.enabled = False
    st.contacts = [ContactPair("p", "A", "B", kind="frictionless")]
    st.contact_settings.method = "mortar"
    monkeypatch.setenv("FEA_MORTAR", "0")                                   # Windows ccx_static: substituted by surface-to-surface – no problem
    assert solver_warnings(st) == []
    monkeypatch.setenv("FEA_MORTAR", "1")                                   # Linux: mortar is solved – with PaStiX it ends with NaN
    w = solver_warnings(st)
    assert len(w) == 1 and "mortar" in w[0]
    _pretend(monkeypatch, "ccx_spooles", win=False)                         # PaStiX is not in this ccx: the run replaces it and says so
    assert solver_warnings(st) == []
    from fea_core.solve import check_study
    _pretend(monkeypatch, "ccx_static.exe")
    assert any("mortar" in x and "PaStiX" in x for x in check_study(st, ["Tyc"])[1])


def test_nonlinear_mortar_engine_label_names_the_solver_of_the_deck(monkeypatch, tmp_path):
    """info["solver"]["engine"] / info["ccx_solver"] of a nonlinear run asked for the solver of the static procedure although the mortar deck
    (Linux) is written with SPOOLES."""
    from fea_core import solve
    from fea_core.contacts.model import ContactPair
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictionless")]
    st.contact_settings.method = "mortar"
    monkeypatch.setenv("FEA_MORTAR", "1")
    _pretend(monkeypatch, "ccx_pastix", win=False)
    monkeypatch.setattr("fea_core.contacts.run.run_nonlinear", lambda *a, **k: NS(info={}))
    res = solve._run_mechanics(st, bar_mesh(), tmp_path, [], lambda s: None, None, None, [], None, None, ("ccx", "test"))
    assert res.info["ccx_solver"] == "SPOOLES" and res.info["solver"]["engine"] == "CalculiX SPOOLES"
    monkeypatch.setenv("FEA_MORTAR", "0")                                  # substituted by surface-to-surface: PaStiX
    res = solve._run_mechanics(st, bar_mesh(), tmp_path, [], lambda s: None, None, None, [], None, None, ("ccx", "test"))
    assert res.info["ccx_solver"] == "PASTIX"
