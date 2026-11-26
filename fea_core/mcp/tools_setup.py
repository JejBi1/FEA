"""Tool `fea_setup_review` and the confirmation of the setup before an AI-started calculation (F15-S1).

The agent shows the user one numbered picture and a table of what will be computed (`fea_setup_review`); when the session
requires confirmation (`Session.require_confirmation`, variable FEA_MCP_CONFIRM, `--confirm`, the GUI setting "Ask before an AI
calculation") `fea_solve_run` and `fea_job_start(kind='solve')` run only for a setup that the user confirmed. The best available
way of asking the user, in this order:

1. The FEA Studio window is open: the review opens a dialog there (`fea_gui.setup_confirm`); only a click on Confirm in the
   window counts. The agent cannot make this decision – the window records it, the digest the agent passes back is only compared.
2. The MCP client supports elicitation: the server asks the user a yes / no question with the table when the calculation is
   started (the picture came in the result of `fea_setup_review` before).
3. Otherwise the agent passes the digest back after asking the user in the conversation. This is recorded honestly as
   `confirmed_by: "agent"` (it proves that the agent knew the setup, not that the user saw it).

The decision is stored in `Results.info["setup_confirmation"]` and printed in the report.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from mcp.server.mcpserver import Image
from mcp.server.mcpserver.context import Context
from pydantic import BaseModel, Field

from . import setup_proof as sp
from .common import out
from .session import FeaError, Session

log = logging.getLogger("fea_core.mcp")

MAX_ELICIT_CHARS = 3000


def strip_ctx(k: dict) -> dict:
    """`ipc_args` of the tools that take the MCP context: the context cannot be sent to the window."""
    return {n: v for n, v in k.items() if n != "ctx"}


def without_ctx(fn):
    """Decorator (under `@mcp.tool`): the window gets no MCP context (its own confirmation applies there)."""
    fn.ipc_args = strip_ctx
    return fn


# ===================================================================== review

def _status(sess: Session, digest: str) -> dict:
    if not sp.confirmation_required(sess):
        return {"required": False, "state": "not required in this session"}
    rec = sp.current_record(sess, digest)
    if rec is None:
        return {"required": True, "state": "waiting for the user's decision"}
    return {"required": True, "state": rec["decision"], "by": rec["by"], "channel": rec["channel"], "at": rec["at"]}


def _next_step(sess: Session, digest: str) -> str:
    short = digest[:sp.MIN_PREFIX]
    if not sp.confirmation_required(sess):
        return "Confirmation is switched off in this session – the calculation may be started; the digest identifies this setup in the report."
    if sess.confirm_ui is not None:
        return (f"The numbered picture and the table are open in the FEA Studio window: ask the user to check them there and click Confirm "
                f"(or Reject). Calling this tool again shows the decision. After the user has confirmed, call fea_solve_run / fea_job_start "
                f"with confirmed='{short}'. Do not repeat the table instead of the window – only the user's click counts.")
    return (f"Show the picture and the table to the user and ask them to confirm the setup. Only after they agree, pass confirmed='{short}' "
            f"to fea_solve_run / fea_job_start (a changed setup has another digest - also a changed mesh size: run fea_mesh_generate "
            f"BEFORE the review). Do not pass it without the user's agreement.")


def build_review(sess: Session, output_path: str | None = None, width: int = 1280, height: int = 720) -> tuple[dict, Path | None]:
    """The review of the setup: (result dict, path of the numbered picture). Opens the dialog when the session has a window."""
    st = sess.require_study()
    scene = None
    path: Path | None = None
    notes: list[str] = []
    if not sess.frame_only:
        _ = sess.parts                                      # the geometry (an error if the STEP file is missing)
        sess.sync_geometry_caches()
    from .render import render
    from .scenes import model_scene
    from .tools_results import _png_path
    scene = model_scene(sess, None, None, False, None, None, numbered=True)
    path = _png_path(sess, output_path, "setup_review")
    render(scene, path, "multi", width, height)
    if sess.frame_only:
        notes.append("Member-only model: the picture shows the members, the numbers of the table are not drawn in it.")
    rows = sp.setup_table(st, sess, scene.numbers)
    digest = sp.session_digest(sess)
    summary = sp.summary_lines(st, sess)
    text = sp.table_text(rows)
    d: dict = {"digest": digest, "digest_short": digest[:sp.MIN_PREFIX], "study": st.name, "summary": summary, "table": text,
               "confirmation": _status(sess, digest), "next": _next_step(sess, digest)}
    if notes:
        d["notes"] = notes
    if sess.geometry_warnings:
        d["geometry_warnings"] = sess.geometry_warnings[:5]
    if sess.confirm_ui is not None and sp.confirmation_required(sess):
        sess.confirm_ui.open_review({"digest": digest, "study": st.name, "rows": rows, "summary": summary, "image": str(path)})
        d["picture_shown_to_user"] = "in the FEA Studio window (the picture file: " + str(path) + ")"
    return d, path


# ===================================================================== the gate of the calculation

class _Form(BaseModel):
    confirm: bool = Field(description="Run the calculation with exactly this setup (numbers as in the picture)")


def ask_user(ctx: Context | None, message: str) -> str:
    """Asks the user through the MCP client: "accept" | "decline" | "cancel" | "unsupported" (no context, or the client has not
    declared the elicitation capability). Called from the worker thread of a tool. A client that DECLARED the capability and then
    fails the request (or a missing worker thread) is an error: the calculation is not started and the question is not replaced by
    the agent's own word."""
    if ctx is None:
        return "unsupported"
    try:
        from mcp import types
        caps = types.ClientCapabilities(elicitation=types.ElicitationCapability(form=types.FormElicitationCapability()))
        supported = bool(ctx.session.check_client_capability(caps))
    except Exception as e:  # noqa: BLE001 – the capability cannot be read: the client is treated as one without elicitation
        log.info("elicitation capability of the client not readable: %s", e)
        return "unsupported"
    if not supported:
        return "unsupported"
    try:
        import anyio.from_thread
        r = anyio.from_thread.run(ctx.elicit, message[:MAX_ELICIT_CHARS], _Form)
    except Exception as e:  # noqa: BLE001 – the client claims elicitation and the question failed
        log.warning("elicitation of the setup confirmation failed: %s", e)
        raise FeaError("The question to the user (setup confirmation) could not be asked through the MCP client: "
                       f"{type(e).__name__}: {e}. The calculation was NOT started. Show the picture and the table to the user in the "
                       "conversation, or ask them to confirm in the FEA Studio window, and repeat the call.") from e
    if r.action == "accept":
        return "accept" if r.data.confirm else "decline"
    return r.action


def gate(sess: Session, confirmed: str | None = None, ctx: Context | None = None) -> dict | None:
    """Called before a calculation is started by `fea_solve_run` / `fea_job_start`. Returns None when the session does not
    require confirmation, otherwise the entry for `Results.info["setup_confirmation"]`; raises FeaError (with the instruction
    for the agent) when the setup is not confirmed."""
    if not sp.confirmation_required(sess):
        return None
    with sess.use(wait=5):
        sess.require_study()
        digest = sp.session_digest(sess)
    rec = sp.current_record(sess, digest)
    given = sp.digest_matches(digest, confirmed)
    if sess.confirm_ui is not None:                           # the window: only the user's click counts, never the passed digest
        if rec and rec["decision"] == "confirmed" and rec["by"] == "user":
            return sp.result_info(rec)
        if rec and rec["decision"] == "rejected":
            raise FeaError("The user REJECTED this setup in the FEA Studio window. Ask what to change, change it, and call "
                           "fea_setup_review again.")
        if confirmed and not given:
            raise FeaError("The setup changed since the review (the digest does not match). Call fea_setup_review again so that "
                           "the user can check and confirm the current setup in the window.")
        raise FeaError("This session requires the user's confirmation of the setup before a calculation, and the user has not "
                       "confirmed it in the FEA Studio window. Call fea_setup_review (it opens the numbered picture and the table "
                       "there), ask the user to click Confirm, then repeat this call with confirmed=<digest_short>.")
    if rec and rec["decision"] == "confirmed" and rec["by"] == "user":      # already asked through the client for this very setup
        return sp.result_info(rec)
    if not confirmed:
        raise FeaError("This session requires the user's confirmation of the setup before a calculation. Call fea_setup_review, "
                       "show the picture and the table to the user, and pass the digest as confirmed=<digest_short> after they agree.")
    if not given:
        raise FeaError("The setup changed since the review (the digest does not match). Call fea_setup_review again, show it to the "
                       "user, and pass the new digest after they agree.")
    rows = sp.setup_table(sess.require_study(), sess)
    answer = ask_user(ctx, f"Run the calculation of study “{sess.require_study().name}” with this setup?\n\n{sp.table_text(rows)}\n\n"
                           f"Setup digest {digest[:sp.MIN_PREFIX]}")
    if answer in ("decline", "cancel"):
        sp.record_decision(sess, digest, "rejected", "user", "elicitation")
        raise FeaError("The user declined the calculation with this setup. Ask what to change, then call fea_setup_review again.")
    if answer == "accept":
        with sess.use(wait=5):
            now = sp.session_digest(sess)
        if now != digest:
            raise FeaError("The setup changed while the user was deciding. Call fea_setup_review again.")
        return sp.result_info(sp.record_decision(sess, digest, "confirmed", "user", "elicitation"))
    return sp.result_info(sp.record_decision(sess, digest, "confirmed", "agent", "agent"))


def verify(sess: Session, confirmation: dict | None) -> None:
    """With the session lock held, immediately before the calculation: the setup is still the confirmed one."""
    if confirmation and sp.session_digest(sess) != confirmation["digest"]:
        raise FeaError("The setup changed after it was confirmed (the digest no longer matches). Call fea_setup_review again.")


def stamp(results, confirmation: dict | None) -> None:
    """Writes who confirmed the setup into the results (printed in the report)."""
    if confirmation:
        results.info["setup_confirmation"] = {**confirmation, "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# ===================================================================== registration

def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_setup_review(output_path: str | None = None, width: int = 1280, height: int = 720, return_image: bool = True):
        """Numbered picture, table and digest of everything that will be computed (supports, loads, thermal conditions, values, faces, cases) - SHOW IT TO THE USER before a calculation. Where confirmation is required the result says what to do next. Read-only."""
        with sess.use(op="setup review"):
            d, p = build_review(sess, output_path, width, height)
        txt = out(d)
        if return_image and p is not None and sess.confirm_ui is None:
            return [txt, Image(path=p)]
        return txt

    # (with a window the result is text only: the picture object does not pass through the pipe to the server, and the user
    # sees the picture in the dialog of the window anyway)
