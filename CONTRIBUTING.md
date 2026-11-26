# Contributing to FEA Studio

Thank you for your interest. FEA Studio is maintained by a single volunteer, so reviews may take a while; small, focused
pull requests with tests are the easiest to accept.

- Bugs and feature requests: open an issue (templates: bug report / feature request). For a crash, attach the crash log –
  **Help → About FEA Studio → Open log folder** (the `crash-<time>.log` files).
- Security problems: **do not** open an issue – see [`SECURITY.md`](SECURITY.md).

## Licence of contributions

FEA Studio is licensed under the **GNU GPL version 3 or later** (`GPL-3.0-or-later`, [`LICENSE`](LICENSE)) with an
additional permission under section 7 ([`LICENSE-EXCEPTIONS.md`](LICENSE-EXCEPTIONS.md): linking with Microsoft MPI and the CAD / Abaqus scripting runtimes). By submitting a contribution you agree that it is licensed under the
GPL-3.0-or-later **together with that additional permission**. Only submit code you wrote yourself or that is available under
a compatible licence – name its source and licence in the pull request; third-party components are listed in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Do not add personal data, company names, real customer models or data without a clear licence (the PragTic fatigue database
is deliberately not distributed).

## Development setup

```bash
python installer/fea_setup.py install verify --profile standard --no-shortcuts --no-mcp --no-fusion   # .venv + CalculiX
.venv/bin/python -m pytest -m "not slow"          # Windows: .venv\Scripts\python
.venv/bin/python -m verification.run_all V1 S1    # verification cases against analytical references
```

Optional modules (GUI, Numba acceleration, MCP / REST, NGSolve, …) are listed in `fea_core/modules.json`; tests of modules
that are not installed are skipped. The same steps run in CI (`.github/workflows/tests.yml`) on Linux and Windows.

## Code rules

- **English** in code, comments, messages, docstrings and documentation. Czech appears only through the translation catalogue
  (`fea_core/i18n/cs.json`) and the Czech help pages (`fea_gui/help/cs/`).
- Units mm, N, MPa, t, s everywhere in the core.
- Keep the core (`fea_core`) free of GUI imports; optional dependencies are imported through `fea_core.modules.need()`.
- User text that reaches a solver input deck goes through `fea_core.ccx.deck_text()`.
- Names from files (parts, load cases, selections) are untrusted: show them in Qt as plain text or escaped.
- Lint: `ruff check .` (rule set in `pyproject.toml` – real errors only, no formatter). Do not reformat unrelated code.
- Types: `basedpyright` (basic mode, `fea_core` + `fea_gui`, CI job `typecheck`) fails on every error that is not in the baseline
  `.basedpyright/baseline.json`, so new code must be clean; see "Type checking" below. A suppression names its rule
  (`# type: ignore[attr-defined]  # reason`), a bare `# type: ignore` is rejected by `tests/test_code_health.py`.
- Structure: `tests/test_code_health.py` keeps `fea_gui/main.py` and the `fea_gui/main_*.py` modules small and every other module
  under 2,500 lines: a new feature gets its own module (dialog, panel, mixin) instead of growing an existing one.
- Every new computational feature needs an **independent reference** (analytical solution, literature or a full model) as a
  verification case in `verification/` with real tolerances, plus fast unit tests in `tests/`. Classify the case in
  `verification/catalogue.py` (reference type, citation, published or internal); only cases with a cited independent reference
  are published on `docs/verification/` (`python -m verification.run_public`, guarded by `tests/test_verification_public.py`).
- Every interactive GUI element gets a short tooltip (≤ 80 characters, `widgets.help_tip`); UI strings use only characters
  present in Segoe UI (`tests/test_ui_glyphs.py`).

## Type checking

The checker is `basedpyright` (the pure-Python distribution of pyright, no Node.js to install), configured under
`[tool.basedpyright]` in `pyproject.toml`: `basic` mode, Python 3.11, Windows platform (fixed, so the result does not depend on
your machine), `fea_core` and `fea_gui`. The findings that existed when it was introduced are recorded in
`.basedpyright/baseline.json` and hidden; everything else is an error. The result depends on the installed packages, so the
baseline is made in an environment of exactly the pinned packages of `pyproject.toml`. The list is derived, not copied:
`.github/scripts/typecheck_requirements.py` prints `dependencies` + the extras `gui`, `plots`, `report`, `beams`, `mcp` + the
checker's pin from the `dev` extra, and the CI job `typecheck` builds its environment from the same script:

```bash
uv venv .typecheck-venv --python 3.11        # any folder outside the checkout works as well
python .github/scripts/typecheck_requirements.py > typecheck-requirements.txt
uv pip install --python .typecheck-venv/bin/python -r typecheck-requirements.txt
.typecheck-venv/bin/python -m basedpyright --pythonpath .typecheck-venv/bin/python        # Windows: .typecheck-venv\Scripts\python
```

A full run takes a few minutes (about 1 GB of memory; the 13 main-window files alone peak near 800 MB; `basedpyright <files>` checks only those files). Optional modules that
are not installed (numba, rslab, pychrono ...) only give warnings.

- Fix a finding rather than suppress it. When a suppression is unavoidable (dynamic OCP / PySide6 attributes, optional imports), name the
  rule and say why: `# type: ignore[attr-defined]  # OCP dynamic cast`.
- basedpyright rewrites `.basedpyright/baseline.json` by itself whenever a run (also `basedpyright <files>`) finds fewer of the baselined
  findings: check `git diff .basedpyright` after a run and commit the shrink only together with the fix.
- When you have fixed old findings, shrink the baseline: `python -m basedpyright --pythonpath <env python> --writebaseline` and
  commit `.basedpyright/baseline.json` (the diff must only remove entries).
- Qt enums: write `Qt.AlignmentFlag.AlignLeft`, `QEvent.Type.KeyPress` in new code; the short forms (`Qt.AlignLeft`) work at run time
  but the PySide6 stubs report them (about 700 of the baselined findings are of this kind).
- The main window is put together from mixins (`fea_gui/chrome.py`, `ipc_bridge.py`, `main_*.py`) that derive, for the checker only,
  from `MainWindowHost` (`fea_gui/main_host.py`). A mixin that starts to use another member of the window or of a sibling mixin through
  `self` declares it there (otherwise: `Cannot access attribute ... for class "<Mixin>*"`); a new mixin copies the `_Base` pattern of the others.
- Annotate the parameters and return values of functions that other modules call: that is what makes `reportCallIssue` and
  `reportArgumentType` effective on the callers.

## Translations (i18n workflow)

The source language is English; the Czech text is looked up at display time.

1. Write new user-visible texts in English in the code. Do not translate persisted identifiers (JSON / `.fea` keys, enum
   values, MCP tool and argument names, solver cards).
2. `python -m fea_core.i18n.extract --todo todo.json --where` lists the texts without a Czech translation and where they are.
3. Translate them in `todo.json` (`""` = not a text for the user; terminology in `fea_core/i18n/glossary.md`).
4. `python -m fea_core.i18n.extract --merge todo.json` adds them to `fea_core/i18n/cs.json`; `tests/test_i18n.py` checks
   that nothing is missing.
5. Help: the English page `fea_gui/help/<slug>.html` is the source; update the Czech page `fea_gui/help/cs/<slug>.html` and
   run `python -m fea_gui.help_pdf --mark-translated <slug>`.

## Pull requests

- One topic per pull request; describe what changed and how it was verified (tests, verification cases, GUI check).
- Run the fast tests and the related verification cases before submitting.
- Add a line to `installer/CHANGELOG.md` under "Unreleased" for user-visible changes.
