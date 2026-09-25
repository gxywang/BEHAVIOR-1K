"""Step zero (tiptop/docs/skills/SPEC.md, D25): a harness refuses to run code from another checkout (no Isaac Sim).

A worktree runs the main tree's code, or b1k-submission's, whenever its PYTHONPATH misses it, and three branches
were once verified that way without testing anything. run.check_imports logs where b1k and each planner's tiptop
and cutamp came from and raises unless all are inside the checkout the harness runs from.
"""

from pathlib import Path

import pytest

import b1k
from omnigibson.tiptop import bench
from omnigibson.tiptop.run import check_imports

ROOT = Path(__file__).resolve().parents[2]  # the checkout these tests (and run.py) belong to


def planner(root: Path = ROOT) -> dict:
    modules = {"tiptop": root / "tiptop/tiptop/__init__.py", "cutamp": root / "tiptop/cutamp/cutamp/__init__.py"}
    return {"modules": {name: str(path) for name, path in modules.items()}}


def test_this_checkout_passes_and_is_one_line_naming_every_module():
    """The gate line. It fails here too when the test itself runs with PYTHONPATH off the worktree."""
    line = check_imports(planner(), None)
    assert line == (
        f"imports: b1k={b1k.__file__}, planner0.tiptop={ROOT}/tiptop/tiptop/__init__.py, "
        f"planner0.cutamp={ROOT}/tiptop/cutamp/cutamp/__init__.py"
    )
    assert Path(b1k.__file__).resolve().is_relative_to(ROOT)


def test_a_planner_running_another_checkout_fails():
    other = planner(Path("/home/wding8/projects/BEHAVIOR-1K"))
    with pytest.raises(RuntimeError, match=r"^planner0\.tiptop, planner0\.cutamp not imported from"):
        check_imports(other)
    with pytest.raises(RuntimeError, match=r"^planner1\.cutamp not imported"):  # the press planner is checked too
        check_imports(planner(), {"modules": {**planner()["modules"], "cutamp": other["modules"]["cutamp"]}})


def test_a_planner_that_does_not_say_fails():
    """An older server has no ``modules``: what cannot be checked is refused, not assumed."""
    with pytest.raises(RuntimeError, match=r"^planner0\.tiptop, planner0\.cutamp not imported"):
        check_imports({"embodiment": {}})


def test_b1k_from_another_checkout_fails(tmp_path):
    with pytest.raises(RuntimeError, match=r"^b1k not imported"):
        check_imports(planner(tmp_path), root=tmp_path)


def test_bench_checks_before_the_simulator_starts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bench, "connect_planners", lambda args: (None, {"embodiment": {}}, None, None))
    monkeypatch.setattr(bench, "build_r1pro_sim", lambda *a, **k: pytest.fail("the simulator was started"))
    with pytest.raises(SystemExit) as end:
        bench.main(["--task-name", "turning_on_radio", "--instances", "0", "--out-dir", str(tmp_path)])
    assert end.value.code == 1
    assert "planner0.tiptop, planner0.cutamp not imported from" in capsys.readouterr().out
