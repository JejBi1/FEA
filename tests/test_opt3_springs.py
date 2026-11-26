"""Elastic supports in the modal interface (release 1.5.1): the compact reactions of `springs.CompactReactions` are those of `springs.reactions` to the last bit,
at the nodes of the support only."""
import numpy as np
import pytest

from fea_core import springs
from fea_core.study import Displacement, ElasticSupport
from tests.test_remote import _study
from tests.test_thermal import bar_mesh


@pytest.mark.parametrize("direction", ["all", "normal"])
def test_compact_reactions_equal_the_full_reactions(direction):
    m = bar_mesh()
    st = _study([Displacement("x0", ux=0.0, uy=0.0, uz=0.0), ElasticSupport("x100", 5.0, direction)])
    spr = springs.CompactReactions(st, m)
    rng = np.random.default_rng(0)
    for _ in range(3):
        U = rng.standard_normal((len(m.nodes), 3))
        full = springs.reactions(st, m, U)
        comp = spr(U)
        assert list(full) == list(comp) == list(spr.idx)
        for sel, R in full.items():
            idx = spr.idx[sel]
            assert np.array_equal(R[idx], comp[sel])                       # the same values to the last bit
            rest = np.ones(len(m.nodes), bool)
            rest[idx] = False
            assert not R[rest].any()                                       # and nothing is lost outside the support nodes
    assert len(spr.idx["x100"]) < len(m.nodes)


def test_two_supports_on_one_selection_accumulate_like_the_full_reactions():
    m = bar_mesh()
    st = _study([Displacement("x0", ux=0.0, uy=0.0, uz=0.0), ElasticSupport("x100", 5.0, "all"), ElasticSupport("x100", 3.0, "normal")])
    spr = springs.CompactReactions(st, m)
    U = np.random.default_rng(1).standard_normal((len(m.nodes), 3))
    full = springs.reactions(st, m, U)
    comp = spr(U)
    assert list(comp) == ["x100"] and np.array_equal(full["x100"][spr.idx["x100"]], comp["x100"])


def test_notch_regions_equal_the_loop_version():
    from fea_core.space.notching import _regions

    def loop(mask):
        out, i, n = [], 0, len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                out.append((i, j))
                i = j + 1
            else:
                i += 1
        return out
    rng = np.random.default_rng(3)
    cases = [np.zeros(0, bool), np.zeros(5, bool), np.ones(5, bool), np.array([True]), np.array([False, True]), np.array([True, False, True, True, False, True])]
    cases += [rng.random(n) < p for n in (7, 50, 1000) for p in (0.1, 0.5, 0.9)]
    for m in cases:
        got = _regions(m)
        assert got == loop(m) and all(type(a) is int and type(b) is int for a, b in got)
