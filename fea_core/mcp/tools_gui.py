"""Tools for live control of the FEA Studio GUI (Dynamic IPC Bridge).

When the application is open, these tools let the LLM assistant interact
directly with the user on the screen: rotate the camera, highlight elements, switch
steps of the workflow or show messages in the status bar.
If the GUI is not running, they return information about the headless state without raising an error.
"""
from __future__ import annotations


from .common import out
from .session import FeaError, Session

# workflow steps of the GUI (fea_gui.main.STAGES – English names) + Czech aliases (older clients, Czech users)
STAGE_NAMES = ("Model", "Materials", "3D printing", "Joints", "Contacts", "Members", "Constraints", "Motion", "Heat",
               "Loads", "Dynamics", "Mesh", "Solve", "Results", "Fatigue")
STAGE_ALIASES = {
    "materiály": "Materials", "3d tisk": "3D printing", "tisk": "3D printing", "spoje": "Joints", "kontakty": "Contacts",
    "pruty": "Members", "vazby": "Constraints", "pohyb": "Motion", "teplo": "Heat", "zatížení": "Loads",
    "dynamika": "Dynamics", "síť": "Mesh", "řešení": "Solve", "výpočet": "Solve", "výsledky": "Results", "únava": "Fatigue",
    "material": "Materials", "printing": "3D printing", "contact": "Contacts", "beams": "Members", "bars": "Members",
    "supports": "Constraints", "thermal": "Heat", "load": "Loads", "solver": "Solve", "results": "Results",
}


def _fold(s: str) -> str:
    import unicodedata
    s = " ".join(str(s).strip().lower().split())
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_stage(stage: str) -> str:
    """Step name of the GUI (English) from an English / Czech name, case and diacritics ignored; unknown = FeaError."""
    key = _fold(stage)
    for n in STAGE_NAMES:
        if _fold(n) == key:
            return n
    for alias, n in STAGE_ALIASES.items():
        if _fold(alias) == key:
            return n
    raise FeaError(f"Unknown step '{stage}'. Allowed: {', '.join(STAGE_NAMES)}")


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_gui_status() -> str:
        """GUI state (window open, active step, study); headless: gui_running false."""
        return out({
            "gui_running": False,
            "message": "The FEA Studio application runs in headless mode (no window is open).",
            "study": sess.study.name if sess.study is not None else None,
        })

    @mcp.tool(structured_output=False)
    def fea_gui_camera_focus(selection: str | None = None, part: str | None = None) -> str:
        """Zooms the open window's 3D camera onto a selection (name or face id 'Díl#0') or part ('*' = fit all). No arguments = Fit All."""
        return out({
            "gui_running": False,
            "focused": False,
            "message": "The FEA Studio application is not open on the desktop – the camera command was skipped.",
            "target": selection or part or "all",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_highlight(entities: list[str] | None = None) -> str:
        """Highlights faces/bodies in the open window: entities = selection names or face ids 'Díl#i'; empty list clears."""
        return out({
            "gui_running": False,
            "highlighted": entities or [],
            "message": "The FEA Studio application is not open on the desktop – the highlight was skipped.",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_show_mode(mode_number: int = 1) -> str:
        """Shows mode shape (vibration/buckling) number mode_number (from 1) in the 3D window."""
        return out({
            "gui_running": False,
            "mode_shown": mode_number,
            "message": "The FEA Studio application is not open on the desktop – the mode switch was skipped.",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_toast(message: str, timeout_ms: int = 4000) -> str:
        """Shows `message` in the status bar of the open window for timeout_ms."""
        return out({
            "gui_running": False,
            "toast": message,
            "message": f"The GUI is not open – the message '{message}' was not shown on the screen.",
        })

    @mcp.tool(structured_output=False)
    def fea_gui_stage_switch(stage: str) -> str:
        """Switches the workflow step in the window; stage: Model, Materials, 3D printing, Joints, Contacts, Members, Constraints, Motion, Heat, Loads, Dynamics, Mesh, Solve, Results, Fatigue (case-insensitive; Czech names accepted)."""
        name = normalize_stage(stage)
        return out({
            "gui_running": False,
            "stage": name,
            "message": f"The GUI is not open – the step '{name}' was not switched.",
        })

    # the GUI knows only the English step names → the server translates the argument before forwarding (server._Registrar)
    fea_gui_stage_switch.ipc_args = lambda k: {**k, "stage": normalize_stage(k.get("stage", ""))}
