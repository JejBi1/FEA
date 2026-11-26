"""Summary of the public verification page from its bundled table (docs/verification/results.csv).

The table is written by `verification.public_report` from a real run of the published verification cases; the About dialog
shows one line of it. Only the standard library is used, and a missing or damaged table gives None (nothing is shown),
never an error.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

CSV_RELATIVE = Path("docs") / "verification" / "results.csv"
_MAX_BYTES = 8 * 1024 * 1024             # the table is ~0.3 MB; anything much larger is not ours


@dataclass(frozen=True)
class Summary:
    cases: int            # published cases
    passed: int           # of them run and within tolerance
    failed: int           # run and outside tolerance
    not_run: int          # not run on the platform of the snapshot (module, solver feature or time)
    date: str             # date of the run (yyyy-mm-dd), "" when unknown
    version: str          # program version of the run, "" when unknown

    @property
    def all_ok(self) -> bool:
        return self.failed == 0 and self.passed > 0


def default_path(root: Path) -> Path:
    return Path(root) / CSV_RELATIVE


def read_summary(path: Path) -> Summary | None:
    """Summary of a results.csv, or None when the file is missing, too large or not a verification table."""
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    status: dict[str, str] = {}
    runs: dict[str, tuple[str, str]] = {}          # case -> (date, program version) of its run
    for r in rows:
        case, st = (r.get("case") or "").strip(), (r.get("case_status") or "").strip()
        # "incomplete" = within tolerance, only the cross-check against the separately downloaded MYSTRAN did not run
        if not case or st not in ("pass", "fail", "incomplete", "not run"):
            return None
        status[case] = st
        d, v = (r.get("run_date") or "").strip(), (r.get("program_version") or "").strip()
        if d and case not in runs:
            runs[case] = (d, v)
    if not status:
        return None
    # a merged run holds cases of several runs: the line names the OLDEST one (the date and version it states hold for every case)
    date, version = min(runs.values()) if runs else ("", "")
    n_pass = sum(1 for v in status.values() if v in ("pass", "incomplete"))
    n_fail = sum(1 for v in status.values() if v == "fail")
    return Summary(len(status), n_pass, n_fail, len(status) - n_pass - n_fail, date, version)
