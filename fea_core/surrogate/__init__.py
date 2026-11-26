"""Surrogate model with calibrated error bars over a parametric study (F15-S4).

    from fea_core.surrogate import fit_surrogate, SurrogateModel
    sm = fit_surrogate(result.points, keys=result.keys, kpis=["max_stress_clean", "max_disp"])   # sweep.ParametricStudyResult
    sm.predict({"sketch:W": 80.0, "sketch:d": 12.0})      # value ± 90 % band per KPI, refuses where no analysis exists

What the error bars are: a split-conformal band. A Gaussian process (hyperparameters by scikit-learn, optional module `surrogate`) or,
without it, a radial-basis interpolation (scipy) predicts the KPI and a local scale σ(x); the held-out calibration points give the
factor q (the 90 % quantile of |error| / σ), so the band μ ± q·σ holds for new points of the same design in at least 90 % of the cases
ON AVERAGE over the analysed region (exchangeable sampling; the calibration set is never used for the fit) – near the faces of the
range the coverage can be lower, near the centre higher. The bars cover interpolation and mesh noise only, NOT the scatter of
materials, manufacturing or the modelling assumptions (`LIMITATIONS`). With fewer than ⌈level / (1 − level)⌉ = 9 held-out points the
band guarantees only n / (n + 1) and says so. The model answers only where analyses exist: the range of the converged points, their
convex hull, no gaps and no failed regions (support.py); a changed study makes it stale (stale.py).

Modules: design (Latin hypercube / Sobol sampling, hold-out split), model (backends), calibrate (conformal quantile, reliability),
support (where the model may answer), stale (study fingerprint), fit (SurrogateModel, fit / predict), store (the file, read as
untrusted input). The imports are lazy: `import fea_core.surrogate` loads neither scipy nor scikit-learn.
"""
LIMITATIONS = ("The error bars cover the interpolation between the analysed design points and the mesh noise only. They do NOT "
               "cover scatter of the material, manufacturing tolerances, loads that differ from the study or modelling assumptions. "
               "The stated coverage is an AVERAGE over the analysed region: it can be lower near the edges of the parameter ranges "
               "and higher in the middle. The model answers only where analyses exist (inside the range and the convex hull of the "
               "converged design points, not in gaps or next to failed analyses) and only for the study it was fitted to.")

LIMITATIONS_SHORT = ("The band covers interpolation and mesh noise only, not material scatter or manufacturing; its coverage is an average over "
                     "the analysed region (lower near the edges); valid only where analyses exist and for the study it was fitted to.")

_LAZY = {"fit_surrogate": "fit", "SurrogateModel": "fit", "SurrogateError": "fit", "ParamSpec": "fit", "training_set": "fit",
         "load_surrogate": "fit", "lhs_design": "design", "split_indices": "design", "backends_available": "model",
         "conformal_quantile": "calibrate", "required_calibration": "calibrate", "SURROGATE_KPIS": "fit", "MIN_TRAIN_PER_DIM": "fit",
         "study_fingerprint": "stale"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
