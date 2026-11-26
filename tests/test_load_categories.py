"""EN 1990 load case categories (F10 C3): explicit LoadCase.category, fallback guess with a warning, persistence."""
import pytest

from fea_core.combinations import (Combination, case_categories, case_category_rows, generate_en1990_combinations,
                                   generate_for_study, guess_category, normalize_category)
from fea_core.study import Study, LoadCase, Force, Gravity


def _study(cats=None):
    st = Study(name="K", geometry="m.step")
    st.load_cases = [LoadCase("Tiha", [Gravity()]), LoadCase("Uzitne", [Force("f", [0, 0, -1.0])]),
                     LoadCase("Wing gust", [Force("f", [1.0, 0, 0])])]
    for lc in st.load_cases:
        lc.category = (cats or {}).get(lc.name)
    return st


def test_normalize_category():
    assert normalize_category("permanent") == "permanent"
    assert normalize_category("B") == "cat_B"
    assert normalize_category("cat_e") == "cat_E"
    assert normalize_category("G") == "permanent"                   # EN 1990 symbol G = permanent action
    assert normalize_category("g") == "permanent"
    assert normalize_category("cat_G") == "cat_G"                   # traffic category G only explicitly
    assert normalize_category("traffic_G") == "cat_G"
    assert normalize_category("snow") == "snow_cz_low"
    assert normalize_category("wind") == "wind"
    with pytest.raises(ValueError, match="allowed"):
        normalize_category("nonsense")


def test_guess_is_word_based():
    assert guess_category("Tiha") == "permanent"
    assert guess_category("Self weight") == "permanent"
    assert guess_category("Wing gust") == "cat_B"                  # contains "g", but it is not a word → not permanent
    assert guess_category("Snih") == "snow_cz_low"
    assert guess_category("Vitr X") == "wind"


def test_explicit_categories_win_and_guess_warns():
    st = _study({"Uzitne": "cat_C", "Wing gust": "wind"})
    cats, warns = case_categories(st)
    assert cats["Uzitne"] == {"category": "cat_C", "source": "explicit"}
    assert cats["Wing gust"]["category"] == "wind"
    assert cats["Tiha"] == {"category": "permanent", "source": "guessed from the name"}
    assert len(warns) == 1 and "Tiha" in warns[0]
    cats, warns = case_categories(st, overrides={"Tiha": "permanent"})
    assert cats["Tiha"]["source"] == "argument" and not warns
    with pytest.raises(ValueError):
        case_categories(st, overrides={"neni": "B"})


def test_generate_for_study_uses_categories():
    st = _study({"Tiha": "permanent", "Uzitne": "cat_C", "Wing gust": "wind"})
    combs, warns = generate_for_study(st, "6.10")
    assert not warns
    lead_u = next(c for c in combs if "Uzitne" in c.name)
    assert lead_u.factors == {"Tiha": 1.35, "Uzitne": 1.5, "Wing gust": pytest.approx(1.5 * 0.6)}
    lead_w = next(c for c in combs if "Wing gust" in c.name)
    assert lead_w.factors["Uzitne"] == pytest.approx(1.5 * 0.7)    # ψ0 of cat_C
    assert lead_w.case_categories["Uzitne"]["source"] == "explicit"
    # a guessed category travels with the combination as a warning (and persists)
    st2 = _study()
    combs2, warns2 = generate_for_study(st2, "6.10")
    assert warns2 and all(c.warnings == warns2 for c in combs2)
    back = Combination.from_dict(combs2[0].to_dict())
    assert back.warnings == warns2 and back.case_categories == combs2[0].case_categories
    # favourable permanent action
    combs3, _ = generate_for_study(st, "6.10", favorable=["Tiha"])
    assert all(c.factors["Tiha"] == 1.0 for c in combs3)


def test_case_without_kind_is_an_error():
    with pytest.raises(ValueError, match="action_type"):
        generate_en1990_combinations([{"name": "X"}], formula="6.10")
    # a category alone means a variable action
    c = generate_en1990_combinations([{"name": "G1", "action_type": "permanent"}, {"name": "Q", "category": "B"}])
    assert c[0].factors == {"G1": 1.35, "Q": 1.5}


def test_category_persisted_only_when_set():
    st = _study({"Uzitne": "cat_B"})
    d = st.to_dict()
    lcs = {lc["name"]: lc for lc in d["load_cases"]}
    assert lcs["Uzitne"]["category"] == "cat_B" and "category" not in lcs["Tiha"]
    assert "load_categories" in d["features"]
    st2 = Study.from_dict(d)
    assert [lc.category for lc in st2.load_cases] == [None, "cat_B", None]
    plain = _study().to_dict()
    assert "load_categories" not in plain["features"]
    assert all("category" not in lc for lc in plain["load_cases"])


def test_category_rows_for_report():
    rows = case_category_rows(_study({"Uzitne": "cat_A"}))
    r = {x["case"]: x for x in rows}
    assert r["Uzitne"]["psi_0"] == 0.7 and r["Uzitne"]["source"] == "explicit"
    assert r["Tiha"]["source"] == "guessed from the name" and "psi_0" not in r["Tiha"]


def test_g_means_permanent_in_both_apis():
    """Regression: category "G" was traffic cat_G in normalize_category but permanent as action_type "g"."""
    st = Study(name="K", geometry="m.step")
    st.load_cases = [LoadCase("Self weight", [Gravity()], category=normalize_category("G")),
                     LoadCase("Live", [Force("f", [0, 0, -1.0])], category="cat_B")]
    combs, _ = generate_for_study(st, "6.10")
    assert combs and all(c.case_categories["Self weight"]["category"] == "permanent" for c in combs)
    low_a = generate_en1990_combinations([{"name": "Self weight", "action_type": "g"},
                                          {"name": "Live", "category": "B"}], formula="6.10")
    low_b = generate_en1990_combinations([{"name": "Self weight", "category": "G"},
                                          {"name": "Live", "category": "B"}], formula="6.10")
    assert [c.factors for c in low_a] == [c.factors for c in low_b]
    assert [c.factors for c in combs] == [c.factors for c in low_a]
    traffic = generate_en1990_combinations([{"name": "Self weight", "action_type": "g"},
                                            {"name": "Live", "category": "traffic_G"}], formula="6.10")
    assert traffic and all(c.factors["Live"] > 0 for c in traffic)                 # cat_G = variable action


def test_report_category_table_only_for_categorised_combinations():
    """Regression: manual combinations (own factors, no categories) got the category table with GUESSED warnings."""
    from fea_core.report.word import uses_load_categories
    st = _study()
    st.combinations = [Combination(name="Manual", factors={"Tiha": 1.0, "Uzitne": 1.2})]
    assert not uses_load_categories(st)
    combs, _ = generate_for_study(st, "6.10")
    st.combinations = combs
    assert uses_load_categories(st)                     # generated → case_categories recorded
    st.combinations = []
    st.load_cases[1].category = "cat_B"
    assert uses_load_categories(st)
