"""FEA_THREADS as the cap of ALL solver threads: thermal pool, direct-solver / hypre thread counts, the environment of a serial ccx, the parsing of the variable,
and the in-order creep (progress). Without FEA_THREADS everything is unchanged."""
import numpy as np
import pytest

from fea_core import config
from fea_core.ccx import parallel, runner


@pytest.fixture
def cap4(monkeypatch):
    monkeypatch.setenv("FEA_THREADS", "4")
    monkeypatch.setattr(config, "NUM_THREADS", 4)
    monkeypatch.delenv("FEA_CCX_PARALLEL", raising=False)


def _threads_of(pool, cap):
    w, t = pool
    return w * (t if t is not None else cap)


def test_thermal_pool_inside_a_share_stays_in_the_cap(monkeypatch):
    """Print variants × thermal states: every variant's thermal states ran with workers × threads of the WHOLE machine (16 threads for the cap 4)."""
    from fea_core.thermal.run import _pool
    monkeypatch.delenv("FEA_CCX_PARALLEL", raising=False)
    monkeypatch.setenv("FEA_THREADS", "16")
    monkeypatch.setattr(config, "NUM_THREADS", 16)
    for share in ((1, 1), (1, 2), (1, 4), (2, 2), (4, 1), (3, 2), (1, 16)):
        with parallel.budget(*share):
            cap = share[0] * share[1]
            for n_jobs in (1, 2, 3, 5, 9):
                assert _threads_of(_pool(n_jobs), cap) <= cap, (share, n_jobs)
    with parallel.budget(2, 2):
        assert _pool(5) == (2, 2)                       # 4 threads of the share: 2 jobs × 2
    with parallel.budget(1, 2):
        assert _pool(5) == (1, None)                    # the job takes the share (run_ccx reads budget_threads)


def test_thermal_pool_without_a_share_follows_num_threads(monkeypatch):
    from fea_core.thermal.run import _pool
    monkeypatch.delenv("FEA_CCX_PARALLEL", raising=False)
    monkeypatch.delenv("FEA_THREADS", raising=False)
    monkeypatch.setattr(config, "NUM_THREADS", 8)
    assert _pool(5) == (4, 2) and _pool(2) == (2, 4) and _pool(1) == (1, None)      # the formula of 1.5.0, unchanged
    monkeypatch.setenv("FEA_THREADS", "4")
    monkeypatch.setattr(config, "NUM_THREADS", 4)
    assert _pool(5) == (2, 2)


def test_thermal_states_follow_fea_ccx_parallel_0(monkeypatch):
    from fea_core.thermal.run import _pool
    monkeypatch.setattr(config, "NUM_THREADS", 8)
    monkeypatch.setenv("FEA_CCX_PARALLEL", "0")
    assert _pool(5) == (1, None)
    monkeypatch.setenv("FEA_THREADS", "8")
    with parallel.budget(2, 4):
        assert _pool(5) == (1, None)


def test_cap_threads(monkeypatch):
    monkeypatch.setattr(config, "NUM_THREADS", 8)
    monkeypatch.delenv("FEA_THREADS", raising=False)
    assert parallel.cap_threads(0) is None and parallel.cap_threads(None) is None
    assert parallel.cap_threads(16) == 16                 # no FEA_THREADS: the request stands
    monkeypatch.setenv("FEA_THREADS", "8")
    assert parallel.cap_threads(16) == 8 and parallel.cap_threads(3) == 3 and parallel.cap_threads(0) is None
    with parallel.budget(1, 2):
        assert parallel.cap_threads(16) == 2 and parallel.cap_threads(1) == 1
    with parallel.budget(2, 8):                           # a share is never above NUM_THREADS (budget_threads)
        assert parallel.cap_threads(16) == 8


def test_direct_solver_threads_follow_the_cap(monkeypatch, tmp_path):
    """`study.solver.nproc` = 8 under FEA_THREADS=4 (and beside ccx side jobs: a share of 2) gave 8 threads to the direct solver."""
    from fea_core import direct, solve_large as SL
    got = {}

    class Stop(Exception):
        pass

    def run_prepared(job, threads=None, cancel=None):
        got["run"] = threads
        raise Stop

    monkeypatch.setattr(SL, "_assemble", lambda job, kind, log, cancel, threads, hom: got.update(asm=threads) or {"ndof": 30})
    monkeypatch.setattr(direct, "estimate_gb", lambda n: 0.1)
    monkeypatch.setattr(direct, "run_prepared", run_prepared)
    study = type("S", (), {"solver": type("V", (), {"nproc": 8})(), "load_cases": [1]})()

    def threads(share=None):
        got.clear()
        with (parallel.budget(*share) if share else _null()):
            with pytest.raises(Stop):
                SL._assemble_and_solve_direct(study, tmp_path, "30", print, None)
        return got["asm"], got["run"]

    monkeypatch.delenv("FEA_THREADS", raising=False)
    assert threads() == (8, 8)                            # default: the study's request, unchanged
    monkeypatch.setenv("FEA_THREADS", "4")
    monkeypatch.setattr(config, "NUM_THREADS", 4)
    assert threads() == (4, 4)
    assert threads((1, 2)) == (2, 2)                      # beside_direct: the direct solver has a share of 2
    study.solver.nproc = 0
    assert threads((1, 2)) == (2, 2) and threads() == (None, None)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def test_hypre_processes_follow_the_cap(monkeypatch, tmp_path):
    from fea_core import hypre
    seen = {}

    class Stop(Exception):
        pass

    def run_tree(cmd, *a, **k):
        seen["n"] = int(cmd[cmd.index("-n") + 1])
        raise Stop

    monkeypatch.setattr(hypre, "available", lambda: True)
    monkeypatch.setattr(hypre, "mpiexec", lambda: "mpiexec")
    import fea_core.procs as procs
    monkeypatch.setattr(procs, "run_tree", run_tree)
    np.save(tmp_path / "rhs.npy", np.zeros(3 * 50_000))
    monkeypatch.setattr(config, "NUM_THREADS", 4)
    monkeypatch.delenv("FEA_THREADS", raising=False)
    with pytest.raises(Stop):
        hypre.run_prepared(tmp_path, nproc=16)
    assert seen["n"] == 16                                 # unchanged without FEA_THREADS
    monkeypatch.setenv("FEA_THREADS", "4")
    with pytest.raises(Stop):
        hypre.run_prepared(tmp_path, nproc=16)
    assert seen["n"] == 4
    with pytest.raises(Stop):
        hypre.run_prepared(tmp_path)
    assert seen["n"] == 4


def test_preset_omp_num_threads_does_not_raise_a_serial_ccx(monkeypatch):
    monkeypatch.setattr(config, "NUM_THREADS", 4)
    for k in ("OMP_NUM_THREADS", "CCX_NPROC_STIFFNESS", "CCX_NPROC_RESULTS"):
        monkeypatch.setenv(k, "16")
    monkeypatch.delenv("FEA_THREADS", raising=False)
    env = runner._env(None)
    assert env["OMP_NUM_THREADS"] == "16"                  # without FEA_THREADS the user's own variable stays (as before)
    monkeypatch.setenv("FEA_THREADS", "4")
    env = runner._env(None)
    assert env["OMP_NUM_THREADS"] == env["CCX_NPROC_STIFFNESS"] == env["CCX_NPROC_RESULTS"] == "4"
    env = runner._env(2)
    assert env["OMP_NUM_THREADS"] == env["CCX_NPROC_STIFFNESS"] == env["CCX_NPROC_RESULTS"] == "2"
    assert config.ccx_env()["OMP_NUM_THREADS"] == "4"


def test_fea_threads_parsing(monkeypatch):
    for val, ok in (("4", True), (" 4 ", True), ("\t12\n", True), ("²", False), ("٣", False), ("４", False), ("0", False), ("-1", False), ("abc", False), ("", False), ("4.0", False)):
        monkeypatch.setenv("FEA_THREADS", val)
        assert config.threads_explicit() is ok, val
        n = config._physical_cores()                       # never raises, junk falls back to the machine's cores
        assert n >= 1
        if ok:
            assert n == int(val.strip())
    monkeypatch.delenv("FEA_THREADS")
    assert not config.threads_explicit()


def test_creep_in_order_uses_the_plain_log_and_its_own_phase(monkeypatch, tmp_path):
    """The nested run_study of the creep states computed in order got the progress-wrapped log: its messages moved the bar of the main calculation back."""
    from fea_core import solve
    from fea_core.printing import creep
    from fea_core.progress import Progress
    seen = {}

    class Rc:
        steps: list = []
        warnings: list = []

    def fake_run_study(st, wd, on_log=None, **k):
        seen["log"] = on_log
        return Rc()

    cs = type("Cs", (), {"hours": 1.0, "temperature": 20.0, "enabled": True})()
    study = type("S", (), {"printing": type("P", (), {"creep": cs})(), "load_cases": []})()
    st_c = type("C", (), {"load_cases": []})()
    monkeypatch.setattr(creep, "creep_study", lambda s: (st_c, {}))
    monkeypatch.setattr(solve, "run_study", fake_run_study)
    res = type("R", (), {"steps": [], "warnings": [], "info": {}, "thermal": []})()
    plain = []
    prog = Progress(None, ["check", "mech", "post", "background"])
    wrapped = prog.log(plain.append)
    creep.add_creep_steps(study, None, res, tmp_path, wrapped, raw_log=plain.append)
    assert seen["log"] == plain.append
    seen.clear()
    creep.add_creep_steps(study, None, res, tmp_path, wrapped)       # without raw_log: the old behaviour
    assert seen["log"] is wrapped
    prog.phase("post", "x")
    solve._creep_phase(study, prog)                                  # in order: booked under "background"
    solve._creep_phase(study, None)
    assert "background" in prog.timings()
    cs.enabled = False
    prog2 = Progress(None, ["check", "post", "background"])
    solve._creep_phase(study, prog2)
    assert "background" not in prog2.timings()


def test_mystran_cross_check_jobs_follow_the_cap(monkeypatch):
    """verification.mystran_check.run: at most FEA_THREADS MYSTRAN processes at once (the default cpu_count // 2 stays without the variable)."""
    from verification import mystran_check
    seen = {}

    class Pool:
        def __init__(self, max_workers):
            seen["jobs"] = max_workers
            raise RuntimeError("stop")

    monkeypatch.setattr(mystran_check, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(mystran_check, "run_study", lambda *a, **k: None)
    monkeypatch.setattr(config, "NUM_THREADS", 2)
    monkeypatch.setenv("FEA_THREADS", "2")
    with pytest.raises(RuntimeError):
        mystran_check.run([], 2.0, 8)
    assert seen["jobs"] == 2
    monkeypatch.delenv("FEA_THREADS")
    with pytest.raises(RuntimeError):
        mystran_check.run([], 2.0, 8)
    assert seen["jobs"] == 8
