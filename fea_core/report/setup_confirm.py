"""Report – "Setup confirmation" (F15-S1): who confirmed the setup before the calculation and when, the digest of the setup,
the numbered picture and the table of the supports and loads as they were shown to the user. Printed only when the calculation was
started after a confirmation (`Results.info["setup_confirmation"]` only; a stamp read from a saved file is marked as such)."""
from __future__ import annotations

from pathlib import Path

from ..mcp import setup_proof as sp

WHO = {
    ("user", "gui"): "the user – clicked Confirm in the FEA Studio window",
    ("user", "elicitation"): "the user – answered the confirmation request of the MCP client",
    ("agent", "agent"): "the AI agent – it passed the digest back; this is NOT a proof that the user saw the setup",
}


def record_of(sess, res) -> dict | None:
    """The confirmation of this calculation: only the stamp in the results (`Results.info["setup_confirmation"]`). The decision
    recorded in the session is NOT used – it belongs to the last setup that was asked about, not to the setup that was calculated."""
    info = (res.info or {}).get("setup_confirmation") if res is not None else None
    return dict(info) if info else None


def has_confirmation(sess, res) -> bool:
    return record_of(sess, res) is not None


def section(D, sess, o, img_dir: Path, images, warnings, label: str) -> None:
    from .images import numbered, render
    study = sess.study
    rec = record_of(sess, sess.results) or {}
    D.h(f"{label} Setup confirmation", 2)
    D.p("Before the calculation the setup (supports, loads and the other inputs) was shown as a numbered picture and a table "
        "and confirmed. The digest identifies exactly that setup: any change of a value, a face, a material, the mesh settings or "
        "the geometry gives another digest.", size=9)
    try:
        now = sp.session_digest(sess)
    except Exception as e:  # noqa: BLE001 – without the geometry the digest cannot be recomputed; the report goes on
        now = None
        warnings.append(f"Setup digest: {e}")
    digest = str(rec.get("digest", ""))
    if now is None:
        same = "not checked (the geometry is not available)"
    else:
        same = ("identical to the confirmed one" if now == digest
                else "DIFFERS from the confirmed one – the setup was changed after the confirmation")
    by, channel = str(rec.get("confirmed_by", "?")), str(rec.get("channel", "?"))
    who = WHO.get((by, channel), f"{by} ({channel})")
    if rec.get("read_from_file"):
        who += " (the record was read from a saved results file; it was not made in this session)"
    D.kv([("Confirmed by", who), ("Time of the confirmation (UTC)", str(rec.get("at", ""))),
          ("Setup digest (SHA-256)", digest), ("Setup of this report", same)])
    scene = None
    try:
        from ..mcp.scenes import model_scene
        scene = model_scene(sess, None, None, False, None, None, numbered=True)
        if o.images:
            img_dir.mkdir(parents=True, exist_ok=True)
            p = numbered(img_dir, "setup_")
            render(scene, p, "multi", 1600, 1000)
            images.append(str(p))
            D.image(p, "Numbered setup picture – the numbers are those of the table (supports blue, loads red / orange, heat by kind)")
    except Exception as e:  # noqa: BLE001 – the table is the proof; the report continues without the picture
        warnings.append(f"Setup figure: {e}")
    rows = sp.setup_table(study, sess, scene.numbers if scene is not None else None)
    if rows:
        D.table(["No.", "Kind", "Selection", "Faces", "Value", "Cases"],
                [[r["no"], r["kind"], r.get("selection") or "–", sp.faces_cell(r), r["value"], ", ".join(r["cases"])] for r in rows],
                caption="Setup confirmed before the calculation", widths=(0.06, 0.16, 0.14, 0.16, 0.34, 0.14), font=8)
    for line in sp.summary_lines(study, sess):
        D.bullet(line)
