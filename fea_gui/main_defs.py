"""Constants, small widgets and helpers of the main window: the step names and phases, colours and tips, the tool shelf (SHELF), the
quantity lists, the file filters and a few helper functions and classes. Shared by `main.py`, the window mixins `main_*.py`,
`chrome.py`, `ipc_bridge.py` and the panels; `fea_gui.main` re-exports every name defined here.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QMainWindow, QMessageBox, QSizePolicy, QWidget

from fea_core.i18n import decimal, tr

from .contact_panel import ContactPanel
from .dynamics_panel import DynamicsPanel
from .motion_panel import MotionPanel
from .frame_panel import FramePanel
from .eigen_gui import MODE_QUANTITIES
from .print_panel import PrintPanel


APP_NAME = "FEA Studio"
# F8: "Contacts" after Joints, "Motion" (mechanism kinematics, fea_core.motion) after Constraints, "Dynamics" after Loads
STAGES = ["Model", "Materials", "3D printing", "Joints", "Contacts", "Members", "Constraints", "Motion", "Heat", "Loads",
          "Dynamics", "Mesh", "Solve", "Results", "Fatigue"]
STAGE = {name: i for i, name in enumerate(STAGES)}      # step index by name (do not hard-code numbers)
# F11: steps grouped into phases (Progress bar); STAGES / STAGE and the panel indices are NOT changed by this
PHASES = [("Geometry", ["Model", "Materials", "3D printing"]),
          ("Assembly", ["Joints", "Contacts", "Members"]),
          ("Conditions", ["Constraints", "Motion", "Heat", "Loads", "Dynamics"]),
          ("Calculation", ["Mesh", "Solve"]),
          ("Evaluation", ["Results", "Fatigue"])]
PHASE_OF = {s: p for p, ss in PHASES for s in ss}
assert [s for _, ss in PHASES for s in ss] == STAGES
# optional steps: on the bar only when the study uses them, the user pinned them (Analyses menu) or opened them
OPTIONAL_STAGES = ("3D printing", "Contacts", "Members", "Motion", "Heat", "Dynamics", "Fatigue")
PHASE_TIPS = {
    "Geometry": "model, materials, 3D printing",
    "Assembly": "bolts and welds, contacts, members",
    "Conditions": "supports, loads, heat, motion, dynamics",
    "Calculation": "mesh and running the analysis",
    "Evaluation": "results and fatigue",
}
# names of optional steps in the Analyses menu
ANALYSIS_LABELS = {
    "3D printing": "3D printing – print settings of plastic parts",
    "Contacts": "Contacts – part contact, friction",
    "Members": "Members – beam structure",
    "Motion": "Motion – mechanisms",
    "Heat": "Heat – heat conduction",
    "Dynamics": "Dynamics – vibration",
    "Fatigue": "Fatigue – life",
}
# phase names on the header tabs and in the panel headings (proposal: Model, Assembly, Conditions, Calculation, Results)
PHASE_LABEL = {"Geometry": "Model", "Evaluation": "Results"}


class ElidedLabel(QLabel):
    """Label that is shortened with "…" in a narrow place (the full text is in the tooltip)."""

    def __init__(self, text=""):
        super().__init__()
        self._full = ""
        self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text):
        text = tr(text)                       # the subclass bypasses the translating wrapper of QLabel.setText
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(self.width(), 40)))


# header buttons per phase: (text, target as in SHELF, short label, long description)
HEADER_ACTIONS = {
    "Geometry": [("Replace geometry…", "replace_geometry_dialog",
                   "Replace the model with modified geometry and keep the study setup",
                   "Loads a modified STEP / .ipt / .f3d and transfers materials, selections, supports and loads to it (faces are matched by position and shape; whatever is not found is shown in the remapping window).")],
    "Assembly": [("Detect joints…", "joints_panel.detect", "Find bolts, nuts, washers and welds in the model",
                 "Detects bolts, nuts, washers and welds by name, shape, color or layer; you confirm uncertain suggestions in a window.")],
    "Calculation": [("Mesh convergence…", "open_convergence_dialog", "Convergence study: analysis on refined meshes",
                 "Solves the study on several refined meshes (globally or in a sphere around a point), Richardson extrapolation and the GCI index.")],
    "Evaluation": [("Export CSV / VTU", "open_export_dialog", "Export results to CSV (Excel) and VTU (ParaView)",
                     "Nodal fields, reactions and elements to CSV with solver node numbers; VTU + PVD for ParaView."),
                    ("Generate report", "export_report", "Technical report of the analysis in Word (.docx)  [Ctrl+P]",
                     "Word report: input, mesh, results by load case with maps, joint and fatigue assessments, conclusion.")],
}


def element_quality_nodes(mesh) -> np.ndarray:
    """Element quality for the mesh colour: q = 6√2·V / l_rms³ of the corner tetrahedron (1 = regular, 0 = degenerate),
    at a node the worst of the neighbouring elements. Cached on the mesh.
    """
    def calc():
        c = mesh.nodes[mesh.tets[:, :4]]
        e = [c[:, j] - c[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
        vol = np.abs(np.einsum("ij,ij->i", e[0], np.cross(e[1], e[2]))) / 6.0
        l_rms = np.sqrt(sum(np.einsum("ij,ij->i", v, v) for v in e) / 6.0)
        q = np.clip(6.0 * np.sqrt(2.0) * vol / np.maximum(l_rms, 1e-30) ** 3, 0.0, 1.0)
        node = np.ones(len(mesh.nodes))
        np.minimum.at(node, mesh.tets.ravel(), np.repeat(q, mesh.tets.shape[1]))
        return node
    return mesh.cached("gui_quality", ("nodes", "tets"), calc)


def phase_label(phase: str) -> str:
    return PHASE_LABEL.get(phase, phase)


# material colours (part colour by material) and a part without material
MATERIAL_COLORS = [(169, 188, 203), (232, 196, 120), (150, 196, 150), (214, 160, 160), (180, 160, 214),
                   (120, 190, 200), (220, 180, 130), (190, 200, 120)]
NO_MATERIAL_COLOR = (201, 203, 204)
VIEW_DIRECTIONS = (("iso", "Isometric"), ("front", "Front"), ("top", "Top"), ("right", "Right"))


# step labels of the progress bar: short (level 1), long STAGE_LONG (level 2: What is this? / F1)
STAGE_TIPS = {
    "Model": "load a STEP file or a .fea project, overview of parts",
    "Materials": "material of the parts – library, designation converter",
    "3D printing": "printing of plastic parts – perimeters, infill, build direction",
    "Joints": "bolts and welds – detection, preload, assessment",
    "Contacts": "part contact, friction, nonlinear steps",
    "Members": "beam structure – nodes, members, sections, EN 1993",
    "Constraints": "model supports – fixed, displacement, cylindrical",
    "Motion": "mechanism – bodies, joints, drives",
    "Heat": "heat conduction – steady-state and transient cases",
    "Loads": "load cases – force, pressure, moment, gravity, temperature",
    "Dynamics": "vibration – harmonic, transient, random, shock",
    "Mesh": "element size, refinement, mesh generation",
    "Solve": "readiness check, solver, running the analysis",
    "Results": "stress and displacement maps, safety factor, probe, maxima",
    "Fatigue": "load history, S-N, damage and life",
}
STAGE_LONG = {
    "Model": "loading STEP geometry or a .fea project, parts overview",
    "Materials": "assigning materials to parts (library, grade converter, custom material)",
    "3D printing": "print settings for plastic parts (perimeters, infill, build direction), variants, creep",
    "Joints": "bolts and welds (detection from the model, preload, VDI 2230 / EN 1993-1-8 assessment)",
    "Contacts": "contact of parts – frictionless, frictional, interference, compression only; friction and nonlinear steps (optional)",
    "Members": "frame – nodes, members, cross-sections, supports, EN 1993-1-1 check (optional)",
    "Constraints": "model restraints – fixed support, prescribed displacement, cylindrical and remote support",
    "Motion": "mechanism – bodies, joints, drivers; animation, joint reactions, transfer to FEA (optional)",
    "Heat": "heat conduction – steady-state / transient cases, boundary conditions (optional)",
    "Loads": "load cases and loads – force, pressure, moment, gravity, rotation, temperature",
    "Dynamics": "vibration – damping, harmonic / transient response, random vibration, shock, added mass (optional)",
    "Mesh": "element size, local refinement, mesh generation (optional – the analysis generates it)",
    "Solve": "readiness check, solver selection and running the analysis",
    "Results": "color maps of stress, displacement and safety factor, probe, maximum locations, reactions",
    "Fatigue": "load history, S-N curves, damage and life (optional)",
}
STAGE_COLORS = "Dot: green = done, orange = warning / optional, red = error, gray = nothing yet"
GREEN, ORANGE, RED, GREY = "#2e9e4f", "#e69a1a", "#d23b3b", "#9aa1ab"
C_CONSTRAINT = (40, 110, 230)
C_FORCE = (220, 40, 40)
C_PRESSURE = (190, 60, 200)
C_REFINE = (30, 170, 170)
BEARING = "Bearing load (hole / pin)"
THERMAL_LOAD = "Temperature (thermal stress)"
ROTATION = "Rotation (centrifugal force)"
FORCE = "Force (components)"
FRAME_NODE = "Member – nodal force / moment"
FRAME_MEMBER = "Member – distributed load"
FRAME_ONLY_OFF = ("3D printing", "Joints", "Contacts", "Constraints", "Heat", "Mesh")   # steps without meaning for beams without CAD
# left tool shelf: step commands (label, icon from theme.ICONS, target, description); target = window attribute
# (button of the panel – click, or a method) or (menu, text) = pick a menu item – see MainWindow._shelf_run
# tool shelf (Studio × Tables design): ("#", "NÁZEV") = group heading, otherwise (text, icon, target, label); target =
# window attribute (button → click, method → call; with a dot panel.button), "@Text" = panel button by its text,
# "focus:field" / "popup:menu" / "raise:dock" / "menu:button", (menu, text) = pick a menu item
SHELF = {
    "Model": [("#", "GEOMETRY"),
              ("Load", "open", "open_dialog", "Open STEP or .fea project"),
              ("Replace", "detect", "replace_geometry_dialog", "Replace geometry and keep the setup"),
              ("Check", "check", "show_geometry_check", "Geometry check: table of warnings from loading"),
              ("#", "PARTS"),
              ("Show", "show", "btn_show_all_parts", "Show all parts"),
              ("Isolate", "body", "isolate_selected", "Show only the selected parts"),
              ("Hide", "hide", "btn_toggle_hide_part", "Hide / show selected parts in 3D"),
              ("Exclude", "exclude", "btn_toggle_ignore_part", "Exclude selected parts from analysis / include again")],
    "Materials": [("#", "MATERIAL"),
                  ("Assign", "assign", "btn_assign_sel", "Assign the material from the form to selected parts"),
                  ("All", "assign", "@Assign to all", "Assign the material from the form to all parts"),
                  ("Library", "report", "popup:mat_combo", "Choose a material from the library"),
                  ("Converter", "combine", "focus:mat_find", "Steel and alloy grade converter (ČSN, EN, AISI…)")],
    "Joints": [("#", "JOINTS"),
              ("Detect", "detect", "joints_panel.detect", "Find bolts, nuts, washers and welds in the model"),
              ("From hole", "cylinder", "@Bolt into selected hole…", "Bolt into the hole selected in 3D"),
              ("Weld", "line", "@Mark the part of the selected face as a weld", "Mark the part of the selected face as a weld"),
              ("#", "CONNECTION"),
              ("Contact", "contact", "stage:Contacts", "Contact pairs (contact, friction) instead of a shared mesh"),
              ("#", "CHECK"),
              ("Exclude", "exclude", "@Exclude the part of the selected face", "Exclude the part of the selected face from the analysis")],
    "Constraints": [("#", "SUPPORTS"),
              ("Fixed support", "fixed", "btn_fixed", "Fix the selected faces rigidly"),
              ("Cylindrical", "cylinder", "add_cylindrical", "Cylindrical support of a hole / pin from selection"),
              ("Remote", "point", "add_remote_support", "Remote support: face connected to a point"),
              ("Displacement", "line", "@Displacement from selection (checked components)", "Prescribed displacement of the checked components")],
    "Loads": [("#", "LOADS"),
                 ("Force", "force", ("ld_type", FORCE), "Load type: force given by components"),
                 ("Pressure", "pressure", ("ld_type", "Pressure"), "Load type: pressure normal to the face"),
                 ("Moment", "moment", ("ld_type", "Moment (remote point)"), "Load type: moment via a remote point"),
                 ("Bearing", "cylinder", ("ld_type", BEARING), "Load type: bearing force in a hole"),
                 ("Gravity", "mass", ("ld_type", "Gravity"), "Load type: self-weight"),
                 ("Add", "add", "btn_add_load", "Add a load to the faces selected in 3D"),
                 ("#", "STATES"),
                 ("New case", "add", "add_lc", "New load case"),
                 ("Combination", "combine", "open_combinations_dialog", "Load case combinations to EN 1990")],
    "3D printing": [("#", "PRINTING"),
                ("Assign", "assign", "print_panel.assign_material", "Assign the printed material to the selected parts"),
                ("Shell", "section", "print_panel.show_shell", "Mesh section: print shell and core"),
                ("Refine", "mesh", "print_panel.btn_refine", "Refine mesh at critical locations and recompute")],
    "Contacts": [("#", "CONTACTS"),
                 ("Find", "detect", "contact_panel.btn_find", "Find touching parts and suggest contact pairs"),
                 ("Add pair", "contact", "contact_panel.add_pair", "Add a contact pair from the form")],
    "Members": [("#", "MEMBERS"),
              ("Node", "point", "frame_panel.add_node", "Add a frame node"),
              ("Member", "line", "frame_panel.add_member", "Add a member between nodes"),
              ("Cross-section", "profile", "frame_panel.add_section", "Add a cross-section from the library"),
              ("From wires", "open", "frame_panel.from_wires", "Members from a STEP wireframe")],
    "Motion": [("#", "MECHANISM"),
              ("Body", "body", "motion_panel.add_body", "Add a mechanism body"),
              ("Joint", "joint", "motion_panel.add_joint", "Add a joint between bodies"),
              ("Driver", "moment", "motion_panel.add_driver", "Add a joint driver"),
              ("Play", "run", "motion_panel.a_play", "Play the mechanism motion")],
    "Heat": [("#", "CONDITIONS"),
              ("Temperature", "temperature", ("thermal_panel.bc_type", "Prescribed temperature"),
               "Boundary condition: prescribed surface temperature"),
              ("Flux", "heat", ("thermal_panel.bc_type", "Heat flux [W/m²]"), "Boundary condition: heat flux through a surface"),
              ("Convection", "convection", ("thermal_panel.bc_type", "Heat transfer – convection"),
               "Boundary condition: heat transfer to the surroundings"),
              ("Radiation", "radiation", ("thermal_panel.bc_type", "Radiation to surroundings"), "Boundary condition: radiation to the surroundings"),
              ("Add", "add", "thermal_panel.add_bc", "Add the boundary condition to the faces selected in 3D")],
    "Dynamics": [("#", "DYNAMICS"),
                 ("Mass", "mass", "dynamics_panel.add_mass", "Add a point or surface mass"),
                 ("Probe", "point", "dynamics_panel.btn_pr_pt", "Response probe at the selected point")],
    "Mesh": [("#", "MESH"),
            ("Create", "mesh", "btn_mesh", "Create the finite element mesh"),
            ("Size", "line", "focus:ms_size", "Global element size [mm]"),
            ("Refine", "add", "@Refine selection", "Refine the mesh on the faces selected in 3D"),
            ("Cancel", "exclude", "btn_mesh_cancel", "Cancel meshing")],
    "Solve": [("#", "SOLVER"),
               ("Run", "run", "btn_run", "Run analysis"),
               ("Stop", "exclude", "btn_cancel", "Stop the analysis"),
               ("#", "OUTPUT"),
               ("Log", "report", "raise:dock_log", "Solver log (Gmsh and CalculiX output)"),
               ("Messages", "check", "raise:dock_messages", "Messages: errors, warnings and information")],
    "Results": [("#", "EXPLORE"),
                 ("Section", "section", "menu:sec_button", "Section the model by a plane"),
                 ("Along line", "line", "btn_path", "Quantity along a line segment, linearization"),
                 ("Isosurface", "isosurface", "btn_iso", "Isosurface of a chosen value"),
                 ("Vectors", "force", "@Stress vectors...", "Principal stress vectors"),
                 ("Section force", "cut_force", "btn_sec_force", "Resultant internal forces in a cut"),
                 ("#", "VIEW"),
                 ("Animation", "animate", "btn_anim", "Deformation animation"),
                 ("Export", "export", "open_export_dialog", "Export results to VTU / CSV")],
    "Fatigue": [("#", "FATIGUE"),
              ("PragTic", "detect", "fat_panel.sn_from_db", "S-N curve from the PragTic database"),
              ("Calculate", "run", "fat_panel.btn", "Run fatigue analysis")],
}
SHELF_LABELS = {"Results": "Explore"}
P_MODES = ["constant", "linear", "hydrostatic (liquid)"]
# labels of the load types (field Type in the Loads step) and their vector
LOAD_TIPS = {
    FORCE: "Force given by components, distributed uniformly over the selected faces (by area), edges (Edge filter,\nby length) or to points (Point filter, each F/n). The total force is entered, not the density.",
    "Force normal to face": "Total force normal to the selected planar faces (positive pushes into the material).\nUse Pressure for curved faces.",
    "Pressure": "Pressure on the selected faces [MPa], positive acts into the material (against the outward normal).\nProfile: constant, linear (gradient) or hydrostatic (liquid below the surface).",
    "Moment (remote point)": "Moment [N·mm] at a remote point attached to the selected faces (rigid or distributing\ncoupling) – e.g. torque on a shaft. Point = face center or user-specified.",
    "Force at remote point": "Force [N] at a point outside the model attached to the selected faces – a moment F × arm is also produced\n(e.g. a weight on an arm, a force at the end of an unmodeled part).",
    BEARING: "Bearing load: cosine pressure distribution on the half of the hole / pin facing the force (like a pin in an eye),\naxial component as uniform shear. Select a cylindrical face (Hole filter H).",
    "Gravity": "Self-weight of all parts from material density: acceleration g [mm/s²], default (0; 0; −9810).\nNeeds no face selection; one per load case.",
    ROTATION: "Centrifugal force of rotating parts (ρ·ω²·r) and optionally angular acceleration α.\nEnter the axis (point + direction or “Axis from selection”) and the speed.",
    THERMAL_LOAD: "Thermal stress: temperature from a thermal case (Heat step) or a uniform part temperature.\nStress arises from ΔT = T − reference temperature and restrained expansion (requires material α).",
    FRAME_NODE: "Point force [N] and moment [N·mm] at a frame node (global axes). Select the node in the list\nor by clicking it in 3D.",
    FRAME_MEMBER: "Distributed member load [N/mm] along its length: uniform, or linearly varying up to the value at the end;\nin global axes (snow, wind) or in the member local axes (x, y, z). Select the member in the list\nor by clicking in 3D. Member self-weight is added by the Gravity load.",
}
VEC_TIPS = {"Moment (remote point)": "Moment components [N·mm] about the point (right-handed about the axes)",
            "Gravity": "Gravitational acceleration [mm/s²] – downward direction; usually (0; 0; −9810)",
            "Force at remote point": "Force components [N] acting at the remote point",
            BEARING: "Components of the bearing load resultant [N]; the radial part becomes pressure on the hole, the axial part becomes shear"}
# short labels of the load types (field Type); the long LOAD_TIPS go to the detailed help (F1 / ⓘ / What is this?)
LOAD_SHORT = {
    FORCE: "Force spread over faces, edges or points (total force, not density)",
    "Force normal to face": "Total force normal to the selected planar faces (positive pushes into the material)",
    "Pressure": "Pressure on the selected faces [MPa] – constant, linear or hydrostatic",
    "Moment (remote point)": "Moment [N·mm] at a remote point attached to the selected faces",
    "Force at remote point": "Force [N] at a point outside the model attached to the faces (also produces a moment)",
    BEARING: "Bearing load: cosine pressure on the half of the hole / pin facing the force",
    "Gravity": "Self-weight of all parts from the material density (one per load case)",
    ROTATION: "Centrifugal force of rotating parts and optionally angular acceleration",
    THERMAL_LOAD: "Thermal stress from a thermal case or from a uniform part temperature",
    FRAME_NODE: "Point force and moment at a node of the beam structure",
    FRAME_MEMBER: "Distributed member load along its length (uniform or linear)",
}
VEC_SHORT = {"Moment (remote point)": "{} component of the moment [N·mm] about the point",
             "Gravity": "{} component of the gravity acceleration [mm/s²]",
             "Force at remote point": "{} component of the force [N] at the remote point",
             BEARING: "{} component of the bearing load resultant [N]"}
G_STD = 9810.0                               # g [mm/s²] for hydrostatic pressure
C_EDGE = (200, 30, 120)
C_ELASTIC = (20, 150, 120)                   # elastic support (springs)
STRESS_ERROR_TIP = ("Stress error estimate [%] (CalculiX, ccx solver only): difference in principal stress between adjacent\nelements at a node (gradient estimate). Below ~5 % is good, above ~10 % refine the mesh there. At singularities\n(sharp corners, fixed support edges) it stays high even after refinement.")

STORED_SOLVER = " (stored in the study)"        # suffix of the temporary item of the equation-solver menu (a stored value that the menu does not offer)

BACKENDS = [("auto", "Automatic (recommended – direct / iterative by model)"),
            ("ccx", "Direct sparse solver – CalculiX"),
            ("ngs_direct", "Direct, one factorization for all cases – rslab"),
            ("ngs_hypre", "Iterative PCG + AMG (large models) – hypre")]

QUANTITIES = [
    ("von Mises", "MPa"), ("Total displacement", "mm"), ("Displacement X", "mm"), ("Displacement Y", "mm"), ("Displacement Z", "mm"),
    ("1st principal stress S1", "MPa"), ("3rd principal stress S3", "MPa"),
    ("Stress σx", "MPa"), ("Stress σy", "MPa"), ("Stress σz", "MPa"),
    ("Stress τxy", "MPa"), ("Stress τyz", "MPa"), ("Stress τzx", "MPa"),
    ("Safety factor to Re", "–"),
    ("3D printing – safety factor (Tsai-Wu/Hill)", "–"), ("3D printing – failure index", "–"),
    ("Temperature", "°C"),                       # only a state with a thermal load (StepResult.fields["T"])
    ("Stress error estimate", "%"),             # ccx gradient estimate (StepResult.fields["stress_error"], ccx backend only)
    # contact (F8, StepResult.fields CPRESS, STATUS, OPENING, CSLIP, CSHEAR) – at the end, old indices stay valid
    ("Contact – pressure", "MPa"), ("Contact – contact state", "–"), ("Contact – opening", "mm"),
    ("Contact – slip", "mm"), ("Contact – frictional stress", "MPa"),
    # Stress Tool (F9.1)
    ("Safety factor – Tresca", "–"),
    ("Safety factor – Mohr-Coulomb", "–"),
    ("Safety factor (Stress Tool)", "–"),
    # topology optimisation (BESO, StepResult.fields["topo_density"]) – at the end, old indices stay valid
    ("Topology – density", "–"), ("Topology – final shape", "–"),
    # plasticity (StepResult.fields["peeq"], nonlinear CalculiX) – at the end, old indices stay valid
    ("Equivalent plastic strain PEEQ", "–"),
]
PEEQ_Q = "Equivalent plastic strain PEEQ"
TOPO_Q = ("Topology – density", "Topology – final shape")
TOPO_SHAPE_Q = "Topology – final shape"
TOPO_MISSING = ("The result has no topology optimization – run it with the Topology (BESO)… button in the results panel.")
CONTACT_Q = {"Contact – pressure": "CPRESS", "Contact – contact state": "STATUS", "Contact – opening": "OPENING",
             "Contact – slip": "CSLIP", "Contact – frictional stress": "CSHEAR"}
STATUS_NOTE = "Contact status: −1 not a contact node, 0 open, 1 sliding, 2 sticking"
# quantities of the thermal state (Results.thermal)
THERMAL_QUANTITIES = [("Temperature", "°C"), ("Heat flux |q|", "W/m²"), ("Heat flux qx", "W/m²"),
                      ("Heat flux qy", "W/m²"), ("Heat flux qz", "W/m²")]
SAFETY_QTY = ("Safety factor to Re", "3D printing – safety factor (Tsai-Wu/Hill)", "Safety factor – Tresca", "Safety factor – Mohr-Coulomb", "Safety factor (Stress Tool)")
NO_MIN_QTY = ("von Mises", "Total displacement", "3D printing – failure index", "Heat flux |q|", "Stress error estimate",
              MODE_QUANTITIES[0][0], "Contact – pressure", "Contact – contact state", "Contact – opening",
              "Contact – slip", "Contact – frictional stress", "Safety factor – Tresca", "Safety factor – Mohr-Coulomb", "Safety factor (Stress Tool)",
              "Topology – density", "Topology – final shape", "Equivalent plastic strain PEEQ")   # without a minimum marker
# maximum places with singularity classification (fea_core.mcp.diagnostics): GUI quantity → diagnostics field;
# other mechanical quantities (displacement, safety, print) → von Mises
HS_STRESS = {"von Mises": "mises", "1st principal stress S1": "s1", "3rd principal stress S3": "s3", "Stress σx": "sxx",
             "Stress σy": "syy", "Stress σz": "szz", "Stress τxy": "sxy", "Stress τyz": "syz", "Stress τzx": "szx"}
HS_THERMAL = {"Temperature": "temperature", "Heat flux |q|": "heat_flux", "Heat flux qx": "hfx",
              "Heat flux qy": "hfy", "Heat flux qz": "hfz"}
HS_TOP = 8                                   # number of places in the table
F8_SOURCE = "nonlinear analysis and dynamics"  # message source (Messages dock) – partial result, advice, dynamics checks
CAD_SOURCE = "CAD import"        # message source: metadata from the CAD next to the STEP (fea_core.cad_import)
DROP_SUFFIXES = (".step", ".stp", ".fea", ".ipt", ".iam", ".f3d", ".f3z")   # files to drop onto the window (as Open);
# .ipt / .f3d / .f3z = native Autodesk files (module native, fea_core.native_cad)
NATIVE_FILTER = "Inventor / Fusion (*.ipt *.iam *.f3d *.f3z)"
# selection filter in 3D (fea_gui.view.PICK_FILTERS): key, text, shortcut, help
PICK_FILTER_ACTIONS = [
    ("face", "Face", "P", "Selection filter Face: a click adds / removes one face (click on empty space = clear selection)"),
    ("body", "Body", "B", "Selection filter Body: a click adds / removes all faces of a part.\nIn steps without face selection (Model, Materials…) a click always selects whole parts."),
    ("hole", "Hole", "H", "Selection filter Hole: a click on a cylindrical face selects the whole hole / pin\n(all coaxial cylindrical faces of the same radius). Switching also expands the already selected cylinders."),
    ("edge", "Edge", "E", "Selection filter Edge: a click adds / removes an edge (edge load in the Loads step) and shows\nthe length and type of the edge, for a circle the radius and center (“Use center” in the status bar = point\nof a remote load / support). The face selection is not changed."),
    ("point", "Point", "V", "Selection filter Point: a click on a vertex or face adds / removes a point (point load in the Loads\nstep). A point on a face is projected exactly onto the CAD face; a click on empty space clears the point selection."),
]


# short labels of the selection filters (tool shelf); the long text from PICK_FILTER_ACTIONS goes to whatsThis (F1 / What is this?)
PICK_FILTER_SHORT = {
    "face": "Face selection: a click adds / removes one face (click on empty space = clear)",
    "body": "Body selection: a click adds / removes all faces of the part",
    "hole": "Hole selection: a click on a cylindrical face selects the whole hole / pin",
    "edge": "Edge selection: a click adds / removes an edge and shows its length, radius and center",
    "point": "Point selection: a click on a vertex or face adds / removes a point",
}


def fmt(v, dec=4):
    """Number with a decimal comma (decimal point in English)."""
    if v is None:
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", " ")
    s = f"{v:.{dec}g}"
    if "e+" in s and abs(v) < 1e9:           # 20 000 instead of 2e+04 (heat fluxes in W/m²)
        return f"{float(s):,.0f}".replace(",", " ")
    return decimal(s)


def nice_floor(v):
    """Rounding down to 1 / 2 / 2.5 / 5 × 10^n (default element size)."""
    if v <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(v))
    return max(m * e for m in (1.0, 2.0, 2.5, 5.0) if m * e <= v * (1 + 1e-9))


def dot_icon(color):
    pm = QPixmap(12, 12)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, 10, 10)
    p.end()
    return QIcon(pm)


def msg(parent, title, text, icon=QMessageBox.Warning):
    b = QMessageBox(icon, title, str(text), QMessageBox.Ok, parent)
    b.setTextFormat(Qt.PlainText)
    return b.exec()


def msg_via_main(*args, **kwargs):
    """`msg` as the window mixins (main_*.py) call it: looked up in the module `fea_gui.main` at call time, because the GUI smoke tests
    replace `fea_gui.main.msg` (a modal message box would block them)."""
    main_module = sys.modules.get("fea_gui.main")
    return getattr(main_module, msg.__name__, msg)(*args, **kwargs)


def _prewarm_imports():
    """Optional (FEA_PREWARM=1): import of libraries that the GUI needs only for results / plots (scipy.spatial ~1.2 s,
    matplotlib ~0.6 s), in a background thread – only the import of modules, no widgets (those may be created only in the
    main thread). Off by default: the two libraries add ~45 MB to the idle program, the first chart then waits ~1–2 s.
    """
    import threading

    def run():
        try:
            import matplotlib.figure  # noqa: F401
            import scipy.sparse  # noqa: F401
            import scipy.spatial  # noqa: F401
        except Exception:  # noqa: BLE001 – only a speed-up, the error shows up when the feature is really used
            pass
    threading.Thread(target=run, daemon=True, name="fea-prewarm").start()


def _remember_ui() -> bool:
    """Remember the state of the bar / sections between runs (only a real start of the program, not the tests)."""
    try:
        from .widgets import remember_ui
        return remember_ui()
    except ImportError:
        import os
        return os.environ.get("FEA_UI_REMEMBER") == "1"


class _InputBlocker(QObject):
    """Application event filter of MainWindow.run_busy: while a worker runs, user input goes only to the progress
    window (clicks, keys, wheel and closing of other windows are dropped – the progress window appears only after 0.3 s).
    Windows opened after the blocker was installed (a message box from a signal handler running in the nested loop,
    e.g. on_solve_failed) keep their input – otherwise they could not be dismissed; the main window and the windows
    that already existed (the dialog that started the operation) stay blocked."""
    BLOCK = {QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.KeyPress,
             QEvent.KeyRelease, QEvent.Wheel, QEvent.Close, QEvent.Drop, QEvent.ShortcutOverride, QEvent.Shortcut}

    def __init__(self, allowed):
        super().__init__(allowed)
        self.allowed = allowed
        self._before = list(QApplication.topLevelWidgets())     # references keep the ids unique
        self._before_ids = {id(w) for w in self._before}

    def _new_window(self, w) -> bool:
        top = w.window()
        return (top is not None and not isinstance(top, QMainWindow) and id(top) not in self._before_ids
                and top.isWindow())

    def eventFilter(self, obj, ev):
        if ev.type() not in self.BLOCK:
            return False
        if ev.type() == QEvent.Close and not isinstance(obj, QMainWindow):
            return False                           # programmatic closing of popups / dialogs goes on
        if not isinstance(obj, QWidget):           # QWindow: Qt passes the event on to the widget (filtered there)
            return False
        if self._new_window(obj):
            return False
        w = obj
        while w is not None:
            if w is self.allowed:
                return False
            w = w.parentWidget()
        if ev.type() == QEvent.Close:
            ev.ignore()
        return True


def _busy_progress(title: str, cancellable: bool, parent):
    """Progress window of run_busy: Escape / the window's close button = Cancel when the operation can be cancelled,
    otherwise they are ignored (QProgressDialog would only hide itself – the program then looks frozen)."""
    from PySide6.QtWidgets import QProgressDialog

    class _BusyProgress(QProgressDialog):
        def reject(self):
            if cancellable:
                self.canceled.emit()           # same as the Cancel button (run_busy then shows "Cancelling…")

        def closeEvent(self, ev):
            ev.ignore()
            if cancellable:
                self.canceled.emit()

    dlg = _BusyProgress(title, "Cancel" if cancellable else "", 0, 0, parent)
    if not cancellable:
        dlg.setCancelButton(None)              # also removes the Escape shortcut of QProgressDialog (= canceled)
    return dlg


def _batched(fn):
    """Method of the main window with batched redraw of the 3D view (one UpdateCurrentViewer at the end)."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        view = getattr(self, "view", None)
        if view is None or not hasattr(view, "batch"):
            return fn(self, *a, **k)
        with view.batch():
            return fn(self, *a, **k)
    return wrapper

def form_hide_empty_rows(form: QFormLayout) -> None:
    """Form rows whose widgets are all hidden are removed from the layout (QFormLayout.setRowVisible) –
    a hidden row with a layout of boxes (QHBoxLayout) would otherwise leave a gap. The caller sets the widget visibility;
    a row with at least one visible widget is shown whole (the widgets of a row are switched together).
    """
    def widgets(item):
        if item is None:
            return []
        w = item.widget()
        if w is not None:
            return [w]
        lay = item.layout()
        return [x for i in range(lay.count()) for x in widgets(lay.itemAt(i))] if lay is not None else []

    for row in range(form.rowCount()):
        ws = []
        for role in (QFormLayout.LabelRole, QFormLayout.FieldRole, QFormLayout.SpanningRole):
            ws += widgets(form.itemAt(row, role))
        form.setRowVisible(row, any(not w.isHidden() for w in ws))


class NearestPoints:
    """Nearest displayed point for the result probe – the part of cKDTree the GUI uses (`data`, `query(x)` → (distance, index)).
    A click asks for one point: a numpy pass over the points (~1 ms for 100 k) instead of building a tree on every redraw (animation
    frames, section moves) and importing scipy.spatial only for that. Several points at once → cKDTree."""

    def __init__(self, pts):
        self.data = np.asarray(pts, float)

    def query(self, x, *args, **kwargs):
        x = np.asarray(x, float)
        if x.ndim == 1 and not args and not kwargs:
            d2 = np.einsum("ij,ij->i", self.data - x, self.data - x)
            j = int(np.argmin(d2))
            return float(np.sqrt(d2[j])), j
        from scipy.spatial import cKDTree
        return cKDTree(self.data).query(x, *args, **kwargs)


class _LazyPanel:
    """Panel of an optional step built at the first access (start-up of the window: these panels are ~40 % of the widgets).
    Non-data descriptor – after the build the instance attribute takes over."""

    def __init__(self, attr: str):
        self.attr = attr

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        return obj._build_lazy_panel(self.attr)


# attribute → (step, class); the stack holds a placeholder until the panel is needed
LAZY_PANELS = {"print_panel": ("3D printing", PrintPanel), "frame_panel": ("Members", FramePanel),
               "contact_panel": ("Contacts", ContactPanel), "dynamics_panel": ("Dynamics", DynamicsPanel),
               "motion_panel": ("Motion", MotionPanel)}
# the step colour of a panel that is not built yet: color() reads only the study through `win` → the same functions on a light object
_SHELL_ATTRS = ("color", "study", "frame", "mech", "part_names", "errors", "body_names")
