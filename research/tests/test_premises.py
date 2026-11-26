"""The premises are fixed: the document, its lock, the main-line plans and the runner agree."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab"))
import runner  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_document_matches_its_lock():
    lock = json.loads((ROOT / "docs" / "PREMISES.sha256").read_text())
    assert _sha(ROOT / "docs" / "PREMISES.md") == lock["sha256"], \
        "docs/PREMISES.md was edited. The premises change only on the researcher's explicit instruction (tools/lock_premises.py)."
    assert lock["history"][-1]["sha256"] == lock["sha256"] and lock["history"][-1]["researcher_instruction"]


def test_every_main_line_plan_carries_the_locked_hash():
    lock = json.loads((ROOT / "docs" / "PREMISES.sha256").read_text())["sha256"]
    plans = sorted((ROOT / "colab" / "plans").glob("mq-*.json"))
    assert len(plans) >= 5
    for f in plans:
        plan = json.loads(f.read_text())
        assert plan.get("premises_sha") == lock, f.name
        assert runner.check_premises(plan, code=ROOT) == lock


def test_runner_refuses_other_premises(tmp_path):
    lock = json.loads((ROOT / "docs" / "PREMISES.sha256").read_text())["sha256"]
    plan = {"name": "mq-x", "premises_sha": lock, "runs": []}
    with pytest.raises(RuntimeError):                                   # a plan stamped for other premises
        runner.check_premises({**plan, "premises_sha": "0" * 64}, code=ROOT)
    with pytest.raises(RuntimeError):                                   # a main-line plan with no stamp
        runner.check_premises({"name": "mq-x", "runs": []}, code=ROOT)
    assert runner.check_premises({"name": "checks12", "runs": []}, code=ROOT) is None    # earlier plans: not checked
    (tmp_path / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "PREMISES.sha256", tmp_path / "docs")
    (tmp_path / "docs" / "PREMISES.md").write_text((ROOT / "docs" / "PREMISES.md").read_text() + "\n- 追加の前提\n")
    with pytest.raises(RuntimeError):                                   # the document was edited
        runner.check_premises(plan, code=tmp_path)
    with pytest.raises(RuntimeError):                                   # nothing to check against
        runner.check_premises(plan, code=tmp_path / "nowhere")
