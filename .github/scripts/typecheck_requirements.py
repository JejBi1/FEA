"""Requirements of the environment the type checker (basedpyright) runs in: the pinned packages of pyproject.toml
(`dependencies` + the extras `gui`, `plots`, `report`, `beams`, `mcp`) and the checker itself (its pin in the `dev` extra).

One derivation for the CI job `typecheck` (.github/workflows/tests.yml) and for a local environment (CONTRIBUTING.md,
"Type checking"), so the pinned list is written down only in pyproject.toml.

usage: python .github/scripts/typecheck_requirements.py > typecheck-requirements.txt      (Python 3.11 or newer: tomllib)
"""
import tomllib
from pathlib import Path

EXTRAS = ("gui", "plots", "report", "beams", "mcp")
CHECKER = "basedpyright"


def requirements(pyproject: Path) -> list[str]:
    with open(pyproject, "rb") as f:
        project = tomllib.load(f)["project"]
    extras = project["optional-dependencies"]
    reqs = list(project["dependencies"])
    for name in EXTRAS:
        reqs += extras[name]
    reqs += [r for r in extras["dev"] if r.replace(" ", "").lower().startswith(CHECKER + "==")]
    return reqs


if __name__ == "__main__":
    print(*requirements(Path(__file__).resolve().parents[2] / "pyproject.toml"), sep="\n")
