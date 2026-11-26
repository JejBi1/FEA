"""installer: the menu entry fea-studio.desktop of a Linux installation – the Exec key is quoted when the installation folder has a space (or another reserved
character) in its name (Desktop Entry Specification, "The Exec key")."""
import importlib
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))
fs = importlib.import_module("fea_setup")

BS = "\\"                    # one backslash
PAIR = "\x01"                # stands for a backslash that the first level of unescaping produced


def unquote(arg: str) -> str:
    """What a desktop environment makes of an Exec argument (the two steps of the specification: the string escapes of the file, then the quoting)."""
    arg = arg.replace(BS * 2, PAIR)                      # level 1: the doubled backslash of the file format
    if arg.startswith('"') and arg.endswith('"'):
        out, i, body = "", 0, arg[1:-1]
        while i < len(body):
            if body[i] == PAIR and i + 1 < len(body) and body[i + 1] in '"`$' + PAIR:     # level 2: a backslash that escapes " ` $ or a backslash
                out += body[i + 1]
                i += 2
            else:
                out += body[i]
                i += 1
        arg = out
    return arg.replace(PAIR, BS).replace("%%", "%")


@pytest.mark.parametrize("path,quoted", [
    ("/opt/fea-studio/fea-studio", False),
    ("/home/me/my programs/fea studio/fea-studio", True),
    ("/home/me/fea (new)/fea-studio", True),
    ("/home/me/$HOME dir/fea-studio", True),
    ('/home/me/say "hi"/fea-studio', True),
    ("/home/me/a`b/fea-studio", True),
    ("/home/me/100% sure/fea-studio", True),
    ("/home/me/a b" + BS + "c/fea-studio", True),
])
def test_exec_argument_is_quoted_when_needed_and_round_trips(path, quoted):
    arg = fs.desktop_exec_arg(PurePosixPath(path))
    assert arg.startswith('"') is quoted
    assert unquote(arg) == path


def test_a_space_in_the_path_stays_one_argument():
    arg = fs.desktop_exec_arg(PurePosixPath("/home/me/my programs/fea studio/fea-studio"))
    assert arg == '"/home/me/my programs/fea studio/fea-studio"'
    text = fs.desktop_entry(PurePosixPath("/home/me/my programs/fea studio"), "fea-studio")
    assert 'Exec="/home/me/my programs/fea studio/fea-studio" %f\n' in text
    assert "Path=/home/me/my programs/fea studio\n" in text            # Path is a plain value, not a command line
    assert text.startswith("[Desktop Entry]\n") and text.endswith("model/step;\n")


def test_dollar_and_backslash_follow_the_specification_examples():
    """The specification: a literal dollar sign in a quoted argument is written with two backslashes, a literal backslash with four."""
    assert fs.desktop_exec_arg("/a b/$x/f") == '"/a b/' + BS * 2 + '$x/f"'
    assert fs.desktop_exec_arg("/a b" + BS + "c") == '"/a b' + BS * 4 + 'c"'
