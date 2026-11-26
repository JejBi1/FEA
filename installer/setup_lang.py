"""Language of the installer's output (Czech / English) – shared by fea_setup.py, upgrade.py, migrations.py and slim.py.
Only the standard library. The switch --lang of the command line (setup.exe passes the language of the wizard) decides; without it Czech
if FEA_LANG / the national environment / LANG is Czech, otherwise English.
"""
from __future__ import annotations

import locale
import os
import sys


def argv_language(argv: list[str]) -> str:
    """The value of --lang X / --lang=X ("cs" | "en"), "" = not given. Read before argparse: many texts are made when the modules are loaded."""
    for i, a in enumerate(argv):
        value = a[len("--lang="):] if a.startswith("--lang=") else (argv[i + 1] if a == "--lang" and i + 1 < len(argv) else None)
        if value in ("cs", "en"):
            return value
    return ""


def _cs() -> bool:
    lang = argv_language(sys.argv[1:]) or os.environ.get("FEA_LANG") or (locale.getlocale()[0] or "") or os.environ.get("LANG", "")
    return lang.lower().startswith(("cs", "czech"))


CS = _cs()


def T(cs: str, en: str) -> str:
    return cs if CS else en
