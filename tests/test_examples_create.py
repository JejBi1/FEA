"""`python -m examples.create` (used by `make_package.py --examples-no-results`, the documented `--only N`): the example number is read from the file name."""
import pytest


@pytest.mark.parametrize("name,k", [("example1_bracket", 1), ("example5_heat_sink", 5), ("example9_base_contact", 9),
                                    ("example10_bracket_topology", 10)])
def test_example_number_from_the_file_name(name, k):
    from examples.create import example_number
    assert example_number(name) == k


def test_create_only_one_example(tmp_path):
    """The command line runs: `--only 3` makes exactly the flange study (a rename of the files once broke the whole command with int('e1'))."""
    from examples import create
    assert create.main(["--only", "3", "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["example3_flange.fea", "example3_flange.step"]
