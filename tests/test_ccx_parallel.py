"""Parallel steps of ccx (fea_core.ccx.parallel): splitting of the .inp and the plan of the jobs (without running ccx)."""
from fea_core.ccx.parallel import ccx_memory_gb, plan, split_steps

INP = """*NODE
1, 0, 0, 0
*ELEMENT, TYPE=C3D10, ELSET=EALL
*STEP
*STATIC
*CLOAD
1, 1, 1.
*END STEP
*STEP
*STATIC
*CLOAD, OP=NEW
1, 2, 1.
*END STEP
"""


def test_split_steps():
    model, steps = split_steps(INP)
    assert model.startswith("*NODE") and "*STEP" not in model
    assert len(steps) == 2
    assert steps[0].startswith("*STEP") and "1, 1, 1." in steps[0] and "OP=NEW" not in steps[0]
    assert steps[1].startswith("*STEP") and "OP=NEW" in steps[1]
    assert model + "".join(steps) == INP


def test_plan_respects_memory_and_cores(monkeypatch):
    """The shipped ccx solves on ONE thread (SPOOLES): one job per state, up to the logical processors and the memory, the threads of a job do not matter
    (measured 20. 11. 2026, see `plan`); up to 5 states the plan stays 5 × 3 (the one that verified example 2)."""
    from fea_core.ccx import parallel
    monkeypatch.setattr(parallel, "_single_thread_solver", lambda: True)
    assert plan(1, 200_000, 20, 8) == (1, 8)                 # jeden krok = postaru
    assert plan(5, 210_000, 20, 8) == (5, 3)                 # unchanged: 5 jobs × 3 threads
    assert plan(2, 66_000, 30, 8) == (2, 4) and plan(3, 66_000, 30, 8) == (3, 3)
    n, t = plan(5, 700_000, 10, 8)                           # a large model, little memory → one job
    assert n == 1
    assert plan(8, 50_000, 30, 8) == (8, 1)                  # more states than the former cap of 5: all at once (V1, 10 states: 11.1 s against 18.6 s)
    assert plan(10, 66_000, 30, 8) == (10, 1) and plan(16, 66_000, 30, 8) == (16, 1)
    assert plan(30, 66_000, 100, 8) == (16, 1)               # the logical processors of 8 cores with SMT
    assert plan(16, 210_000, 12, 8)[0] == 7                  # the memory limit holds: 0.8 × 12 GB / 1.37 GB per job
    n, t = plan(16, 210_000, 12, 8)
    assert n * ccx_memory_gb(210_000) <= 0.8 * 12 and t == 2
    assert plan(5, 210_000, 20, 4) == (5, 1)                 # few cores: still one job per state (the solver is one thread anyway)
    assert plan(5, 210_000, 20, 1) == (1, 1)


def test_plan_of_a_pastix_build_keeps_the_former_rule(monkeypatch):
    """A developer's ccx_pastix / FEA_CCX=ccx_static solves on several threads: at least 3 threads per job, at most (2 × cores) // 3 jobs."""
    from fea_core.ccx import parallel
    monkeypatch.setattr(parallel, "_single_thread_solver", lambda: False)
    assert plan(5, 210_000, 20, 8) == (5, 3)
    n, t = plan(8, 50_000, 30, 8)
    assert 2 <= n <= 5 and t >= 3
    assert plan(16, 66_000, 100, 8)[0] == 5
    assert plan(5, 210_000, 20, 4) == (1, 4)                 # few cores


def test_memory_estimate_grows_with_the_model():
    assert ccx_memory_gb(214_000) < ccx_memory_gb(706_000)


def _capped(monkeypatch, n):
    """FEA_THREADS = n set by the user: config.NUM_THREADS = n and config.threads_explicit() is true."""
    from fea_core import config
    monkeypatch.setenv("FEA_THREADS", str(n))
    monkeypatch.setattr(config, "NUM_THREADS", n)
    assert config.threads_explicit()


def test_explicit_fea_threads_caps_jobs_times_threads(monkeypatch):
    """FEA_THREADS=4 gave '7 CalculiX jobs × 1 threads' on 8 logical processors: with FEA_THREADS set jobs × threads <= FEA_THREADS for every
    number of states and every memory, for the SPOOLES (one-thread solve) plan and for the PaStiX one."""
    from fea_core.ccx import parallel
    for single in (True, False):
        monkeypatch.setattr(parallel, "_single_thread_solver", lambda s=single: s)
        for cap in (1, 2, 3, 4, 6, 8, 12):
            _capped(monkeypatch, cap)
            for n_steps in (1, 2, 3, 5, 7, 10, 16, 40):
                for ndof, free in ((50_000, 100.0), (210_000, 12.0), (700_000, 10.0)):
                    n, t = plan(n_steps, ndof, free)
                    assert n >= 1 and t >= 1 and n * t <= max(cap, 1) and n <= n_steps, (single, cap, n_steps, ndof, n, t)
    monkeypatch.setattr(parallel, "_single_thread_solver", lambda: True)
    _capped(monkeypatch, 4)
    assert plan(7, 50_000, 100.0) == (4, 1)                    # the reported case: 7 states, FEA_THREADS=4 (was 7 jobs × 1 thread)
    assert plan(2, 50_000, 100.0) == (2, 2) and plan(3, 50_000, 100.0) == (3, 1) and plan(1, 50_000, 100.0) == (1, 4)
    assert plan(7, 210_000, 4.0)[0] == 2                       # the memory limit still holds (0.8 × 4 GB / 1.37 GB)
    _capped(monkeypatch, 8)
    assert plan(5, 50_000, 100.0) == (5, 1) and plan(20, 50_000, 100.0) == (8, 1)      # no SMT oversubscription (the default plan gives 16 jobs)
    _capped(monkeypatch, 1)
    assert plan(5, 50_000, 100.0) == (1, 1)
    monkeypatch.setattr(parallel, "_single_thread_solver", lambda: False)
    _capped(monkeypatch, 9)
    assert plan(5, 50_000, 100.0) == (3, 3)                    # PaStiX: at least 3 threads per job
    _capped(monkeypatch, 4)
    assert plan(5, 50_000, 100.0) == (1, 4)


def test_default_plan_without_fea_threads_is_unchanged(monkeypatch):
    """No FEA_THREADS: the logical processors still count (the measured plan of the docstring); a `cores` given by the caller is never capped by the environment."""
    from fea_core import config
    from fea_core.ccx import parallel
    monkeypatch.setattr(parallel, "_single_thread_solver", lambda: True)
    monkeypatch.delenv("FEA_THREADS", raising=False)
    monkeypatch.setattr(config, "NUM_THREADS", 4)
    assert not config.threads_explicit()
    assert plan(7, 50_000, 100.0, 4) == (7, 1)
    assert plan(5, 210_000, 20, 8) == (5, 3) and plan(30, 66_000, 100, 8) == (16, 1)
    monkeypatch.setenv("FEA_THREADS", "4")                      # the caller states the machine (tests, a plan for another machine): unchanged
    assert plan(30, 66_000, 100, 8) == (16, 1)
    monkeypatch.setenv("FEA_THREADS", "x")                      # junk / zero = not explicit
    assert not config.threads_explicit()
    monkeypatch.setenv("FEA_THREADS", "0")
    assert not config.threads_explicit()


def test_beside_direct_respects_explicit_fea_threads(monkeypatch):
    from fea_core import config
    from fea_core.side_jobs import beside_direct
    monkeypatch.setenv("FEA_THREADS", "8")
    monkeypatch.setattr(config, "NUM_THREADS", 8)
    n, t = beside_direct(200_000, 3, free_gb=64.0)
    assert (n, t) == (2, 4) and n * t <= 8                      # the default (cores = 8, no FEA_THREADS) gives (4, 3) = 12 threads
    assert beside_direct(200_000, 3, free_gb=64.0, cores=8) == (4, 3)
    monkeypatch.setenv("FEA_THREADS", "4")
    monkeypatch.setattr(config, "NUM_THREADS", 4)
    assert beside_direct(200_000, 3, free_gb=64.0) == (1, 4)


def test_background_thermal_and_creep_follow_the_mechanics_with_explicit_fea_threads(monkeypatch):
    """Thermal states and the creep run beside the mechanics (each with all threads) – with FEA_THREADS set everything is computed in order."""
    from fea_core import config, solve
    from fea_core.printing import creep

    class C:
        def __init__(self, name):
            self.name = name

    class Th:
        cases = [C("a"), C("b")]

    class St:
        load_cases = [C("lc")]
        thermal = Th()

        def thermal_loads(self, lc):
            return [type("T", (), {"case": "a"})()]
    monkeypatch.delenv("FEA_THREADS", raising=False)
    assert solve._thermal_split(St()) == (["a"], ["b"])
    monkeypatch.setenv("FEA_THREADS", "4")
    assert config.threads_explicit() and solve._thermal_split(St()) == (["a", "b"], [])
    monkeypatch.setattr(creep, "creep_study", lambda study: (object(), {}))
    assert creep.start_creep(St(), None, [], None, print) is None
