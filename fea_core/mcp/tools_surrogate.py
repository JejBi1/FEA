"""Tools: surrogate model with calibrated error bars over a parametric study (fea_core.surrogate, F15-S4).

fea_param_study_run(mode="lhs", parameters=[{kind, bounds}…]) → fea_surrogate_fit → fea_surrogate_predict(params) answers a KPI ± band in
milliseconds and refuses where no converged analysis exists; fea_surrogate_info shows the validation (held-out points, coverage) and where
to analyse next. The bands cover interpolation and mesh noise only (not material scatter or manufacturing); the surrogate belongs to the
study it was fitted for (a changed study makes it stale).
"""
from __future__ import annotations

from typing import Any

from ..surrogate import LIMITATIONS, LIMITATIONS_SHORT
from .common import out
from .session import FeaError, Session


def _any(sess: Session) -> Any:
    """The session without the attribute types the checker infers from Session.reset (every attribute starts as None there)."""
    return sess


def _model(sess: Session):
    sm = _any(sess).surrogate
    if sm is None:
        raise FeaError("No surrogate model – run a parametric study (fea_param_study_run, mode lhs with bounds) and fea_surrogate_fit")
    return sm


def _fresh(sess: Session, sm) -> None:
    """Marks the surrogate stale when the study of the session is not the one it was fitted for (the answers then say so)."""
    S = _any(sess)
    sm.check_study(S.study, S.step_path)


def summary(sm) -> list[dict]:
    """One short row per (category, KPI): the data split, the accuracy on the held-out points, the band factor and the verdict."""
    rows = []
    for m in sm.describe()["members"]:
        rows.append({"kpi": m["kpi"], **({"category": m["category"]} if m["category"] else {}), "n_train": m["n_train"], "n_validation": m["n_validation"],
                     "r2": m["r2"], "median_error": m["median_rel_err"], "max_error": m["max_rel_err"], "band_factor_q": m["q"],
                     "coverage_cv": m["coverage_cv"], "guaranteed": m["guaranteed"], "reliable": m["reliable"],
                     **({"why": m["reasons"]} if m["reasons"] else {})})
    return rows


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_surrogate_fit(kpis: list[str] | None = None, validation_fraction: float = 0.2, seed: int = 0, backend: str = "auto") -> str:
        """Fits a surrogate model with 90 % error bars to the design points of the last parametric study (fea_param_study_run; mode lhs is made for it: 1-5 continuous parameters, at least ~20-30 points, more for more parameters). kpis: max_stress_clean | max_stress | max_disp | min_safety | mass | freq1 | print_safety (default all with data). A seeded share of the points (default 20 %, at least 9: fewer cannot guarantee 90 %) is held out and calibrates the band (split-conformal); a categorical parameter (material name) gets one model per category. Failed points are left out and reported; the model answers only inside the range and convex hull of the converged points, not in holes of the design, at the levels of a parameter with fewer than 6 values only (a coarse grid) and on the lines of a one-at-a-time design only. backend auto | gp (scikit-learn, module surrogate) | rbf (scipy). Refuses a training set that is too small or does not span the parameters, and a study that changed since the sweep. The bands cover interpolation and mesh noise, not material scatter; the coverage is an average over the analysed region."""
        from ..surrogate import SurrogateError, fit_surrogate
        with sess.use():
            S = _any(sess)
            res = S.param_study
            if res is None or not res.points:
                raise FeaError("No parametric study has been run – fea_param_study_run (mode lhs with bounds)")
            fp = getattr(res, "fingerprint", "")
            if fp and S.study is not None:
                from ..surrogate.stale import study_fingerprint
                if study_fingerprint(S.study, list(res.keys), S.step_path) != fp:
                    raise FeaError("The study was changed after the parametric study ran – its points describe another model: run "
                                   "fea_param_study_run again, then fit")
            try:
                sm = fit_surrogate(res.points, keys=list(res.keys), kpis=kpis, validation_fraction=validation_fraction, seed=seed,
                                   backend=backend, bounds=getattr(res, "bounds", None) or None, fingerprint=fp,
                                   sweep=getattr(res, "sweep", None) or None)
            except (SurrogateError, ValueError) as e:
                raise FeaError(str(e)) from None
            S.surrogate = sm
            d = sm.describe()
            return out({"backend": sm.backend, "n_points": d["n_points"], "n_failed": d["n_failed"], "level": sm.guaranteed_level,
                        "reliable": sm.reliable, "models": summary(sm),
                        "parameters": [{"key": p["key"], **({"range": [p["lo"], p["hi"]]} if p["kind"] == "continuous" else {"categories": p["categories"]})}
                                       for p in d["parameters"]],
                        "warnings": d["warnings"][:8], "note": LIMITATIONS_SHORT,
                        "next": "fea_surrogate_predict(params={key: value}) answers value ± band; fea_surrogate_info shows the validation"})

    @mcp.tool(structured_output=False)
    def fea_surrogate_predict(params: dict, kpis: list[str] | None = None, extrapolate: bool = False) -> str:
        """Predicts the KPI of the surrogate model (fea_surrogate_fit) at params = {parameter key: value} as value with a band (lower, upper; level = the coverage the calibration guarantees, 90 % with at least 9 held-out points). Refuses (error) a point where no converged analysis exists - outside the range or convex hull of the analysed points, in a hole of the design, between the few levels of a gridded parameter, off the lines of a one-at-a-time design, next to a failed analysis - unless extrapolate=true, which predicts anyway with a warning and NO guarantee; marks the answer stale when the study changed since the fit; warns when the model is not reliable; a point within 5 % of the edge of a parameter's analysed range is answered with a warning and reliable=false (the coverage is lower there). The band covers interpolation and mesh noise only, not material scatter or manufacturing."""
        from ..surrogate import SurrogateError
        with sess.use():
            sm = _model(sess)
            _fresh(sess, sm)
            try:
                r = sm.predict(dict(params), kpis=kpis, extrapolate=bool(extrapolate))
            except SurrogateError as e:
                raise FeaError(str(e)) from None
            r.pop("limitations", None)
            r["note"] = LIMITATIONS_SHORT
            return out(r)

    @mcp.tool(structured_output=False)
    def fea_surrogate_info(suggest: int = 0, kpi: str | None = None) -> str:
        """Shows the surrogate model: parameters and ranges of the analysed points, per KPI the accuracy on the held-out points (R², median error, coverage of the band, reliable or not), the held-out validation table (analysed value vs predicted value ± band), whether the study changed since the fit (stale) and, with suggest = n, the n points inside the analysed region where the model is least informed (analyse them with fea_param_study_run and fit again)."""
        from ..surrogate import SurrogateError
        with sess.use():
            sm = _model(sess)
            _fresh(sess, sm)
            d = sm.describe()
            try:
                table = sm.validation_table(kpi or sm.kpis[0])
                nxt = sm.suggest(int(suggest)) if suggest else []
            except SurrogateError as e:
                raise FeaError(str(e)) from None
            return out({"backend": sm.backend, "n_points": d["n_points"], "n_failed": d["n_failed"], "level": sm.guaranteed_level,
                        "seed": sm.seed, "reliable": sm.reliable, **({"stale": sm.stale} if sm.stale else {}),
                        "parameters": d["parameters"], "models": summary(sm), "skipped": d["skipped"],
                        "validation": {"kpi": kpi or sm.kpis[0], "points": [{"params": r["params"], "analysed": r["analysed"], "predicted": r["predicted"],
                                                                         "lower": r["lower"], "upper": r["upper"], "inside": r["inside"]} for r in table[:60]]},
                        **({"suggested_next": nxt} if nxt else {}), "warnings": d["warnings"][:8], "note": LIMITATIONS})
