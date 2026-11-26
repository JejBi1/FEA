"""Basic sketcher for 1D (beam systems) and 2D (plane stress, plane strain, axisymmetric) models without CAD (F15-S5).

    from fea_core.sketch import new_sketch, build_study
    sk = new_sketch("twod", "plane_stress", thickness=5)
    sk.set_param("W", 100); sk.set_param("d", 20)
    sk.add_entity("rect", [[0, 0], ["W", "W"]])
    sk.add_entity("circle", center=["W/2", "W/2"], radius="d/2")
    study, build = build_study(sk, "work")          # Study with sketch.step, 2D settings and a selection per entity

Typed coordinates and named parameters (expressions), NO geometric constraint solver. Modules: expr (restricted evaluator), model
(Sketch, entities, numeric curves), regions (2D: closed loops → regions → solids, OCP), frame (1D: nodes and members), build (sketch
→ study, rebuild with the settings kept by the CAD-link remap), dxf (import of lines / arcs, optional module ezdxf), render
(PNG of the sketch, matplotlib). The imports are lazy: `import fea_core.sketch` loads neither OCP nor shapely.
"""
_LAZY = {"Sketch": "model", "Entity": "model", "new_sketch": "model", "Plane3": "model", "make_plane": "model",
         "SketchError": "expr", "evaluate": "expr", "build_study": "build", "rebuild_study": "build",
         "build_geometry": "build", "SketchBuild": "build", "sketch_parameters": "build", "plan_rebuild": "build",
         "apply_rebuild": "build", "extrude_to_3d": "build", "import_dxf": "dxf", "render_png": "render"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
