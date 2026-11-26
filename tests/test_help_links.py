"""The help pages and the manuals built from them: every chapter of the contents has its page in both languages (four chapters were missing in the
program and in every manual: help.py asked for contacts / friction / members / constraints, the files are kontakty / treni / pruty / vazby), every
local link resolves, and neither the HTML nor the PDF manual (nor the Testing Guide, a subset of the chapters) has a dead anchor. No window, no
WebEngine: the manuals are only assembled.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from fea_core.i18n import language, set_language, tr
from fea_gui import help as helpmod
from fea_gui import help_export, help_pdf, help_web

ROOT = Path(__file__).resolve().parents[1]
HELP = ROOT / "fea_gui" / "help"
LANGS = {"en": HELP, "cs": HELP / "cs"}
SLUGS = [s for s, _ in helpmod.PAGES]

HREF = re.compile(r'<a\s[^>]*?href="([^"]*)"')
ANCHOR = re.compile(r'\s(?:id|name)="([^"]+)"')


def test_every_chapter_has_a_page_in_both_languages_and_every_page_a_chapter():
    for lang, folder in LANGS.items():
        missing = [s for s in SLUGS + list(helpmod.STAGE_PAGES.values()) if not (folder / f"{s}.html").is_file()]
        assert not missing, (lang, missing)
        unreachable = [p.stem for p in folder.glob("*.html") if p.stem not in SLUGS]
        assert not unreachable, (lang, unreachable)
    assert set(helpmod.STAGE_PAGES.values()) <= set(SLUGS)
    assert set(help_pdf.TESTING_PAGES) <= set(SLUGS)


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_every_local_link_of_the_help_resolves(lang):
    folder = LANGS[lang]
    pages = {p.stem: p.read_text(encoding="utf-8") for p in folder.glob("*.html")}
    anchors = {s: set(ANCHOR.findall(t)) for s, t in pages.items()}
    bad = []
    for slug, text in pages.items():
        for href in HREF.findall(text):
            if href.startswith(("http://", "https://", "mailto:", "fea:")):
                continue
            name, _, anchor = href.partition("#")
            target = Path(name).stem if name else slug
            if name and not name.endswith(".html"):
                bad.append((slug, href, "not a help page"))
            elif target not in pages:
                bad.append((slug, href, "no such page"))
            elif anchor and anchor not in anchors[target]:
                bad.append((slug, href, "no such anchor"))
    assert not bad, bad


def test_the_contents_are_translated():
    old = language()
    try:
        set_language("cs")
        untranslated = [t for s, t in helpmod.PAGES if s in ("kontakty", "treni", "pruty", "vazby") and tr(t) == t]
        assert not untranslated
    finally:
        set_language(old)


def _dead_anchors(html: str) -> list[str]:
    ids = set(ANCHOR.findall(html))
    return sorted({h for h in HREF.findall(html) if h.startswith("#") and h[1:] and h[1:] not in ids}
                  | {h for h in HREF.findall(html) if not h.startswith(("#", "http://", "https://", "mailto:"))})


@pytest.mark.parametrize("only", [None, help_pdf.TESTING_PAGES], ids=["manual", "testing"])
@pytest.mark.parametrize("lang", sorted(LANGS))
def test_the_html_manual_has_no_dead_anchor(tmp_path, lang, only):
    html = help_export.export_html(tmp_path / "m.html", only, lang=lang).read_text(encoding="utf-8")
    assert not _dead_anchors(html), _dead_anchors(html)[:20]
    assert "The help page is not available yet." not in html
    for slug in (only or SLUGS):
        assert f'<section class="chapter" id="{slug}"' in html, slug


@pytest.mark.parametrize("kind", ["manual", "testing"])
@pytest.mark.parametrize("lang", sorted(LANGS))
def test_the_pdf_manual_has_no_dead_anchor(lang, kind):
    html = help_pdf.build_html(lang, kind)
    assert not _dead_anchors(html), _dead_anchors(html)[:20]
    assert help_pdf.TEXTS[lang]["missing"] not in html


def test_the_rewriter_keeps_anchors_of_headings_and_drops_links_out_of_the_document():
    src = ('<h2 id="a">Head</h2><p><a href="#a">here</a> <a href="other.html#b">there</a> <a href="gone.html">away <b>x</b></a> '
           '<a href="https://example.org/">web</a></p><table id="t"><tr><td>1</td></tr></table><a name="n"></a>')
    out = help_web.link_chapter(src, "page", {"page", "other"})
    assert '<h2><a class="anchor" id="page--a"></a>Head</h2>' in out
    assert 'href="#page--a"' in out and 'href="#other--b"' in out and "away <b>x</b>" in out and "gone" not in out
    assert 'href="https://example.org/"' in out
    assert '<a class="anchor" id="page--t"></a><table>' in out and 'id="page--n"' in out
