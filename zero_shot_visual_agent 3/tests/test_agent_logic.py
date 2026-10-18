"""Logic tests using a fake model hub (no GPU / downloads needed).  Run: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import VisualAgent                      # noqa: E402
from agent.text_utils import count_targets, extract_terms, singularize   # noqa: E402
from agent.utils import annotate, box_gap, load_runs, save_run             # noqa: E402
from agent import config as C                      # noqa: E402


class FakeHub:
    """Deterministic stand-in for ModelHub."""
    def __init__(self, vqa_answer="cars and trees", dets=None, worksite=False, broken=()):
        self.vqa_answer, self.worksite, self.broken = vqa_answer, worksite, set(broken)
        self.dets = dets if dets is not None else [
            {"label": "bus", "score": 0.57, "box": [10, 10, 200, 120]},
            {"label": "person", "score": 0.45, "box": [150, 40, 190, 140]},
            {"label": "car", "score": 0.30, "box": [300, 60, 380, 120]}]
        self.detect_calls = []

    def ensure(self, name): return name not in self.broken
    def error(self, name): return "boom" if name in self.broken else None

    def vqa(self, image, question): return self.vqa_answer

    def detect(self, image, labels, threshold=0.12, **kw):
        self.detect_calls.append(list(labels))
        return [d for d in self.dets if d["label"] in labels and d["score"] >= threshold]

    def clip_logits(self, images, texts):
        out = np.zeros((len(images), len(texts)))
        for j, t in enumerate(texts):
            if t == C.WORKSITE_CHECK[0]: out[:, j] = 5 if self.worksite else -5
            if "bare head" in t or "ordinary clothes" in t: out[:, j] = 4      # PPE missing
        return out

    def cnn_topk(self, image, k=5): return [{"class": "school bus", "score": 0.9}]


IMG = Image.new("RGB", (400, 300), "gray")


def test_text_helpers():
    assert singularize("buses") == "bus" and singularize("lights") == "light"
    assert count_targets("How many people are in the picture?", C.KNOWN_PHRASES) == ["person"]
    assert count_targets("How many traffic lights are visible?", C.KNOWN_PHRASES) == ["traffic light"]
    assert extract_terms("Is there a dog in the picture?") == ["dog"]


def test_plan_general_vs_safety():
    agent = VisualAgent(FakeHub())
    r = agent.run(IMG, "What objects are visible?")
    assert r["plan"] == ["vqa", "detection"]
    r = agent.run(IMG, "Any safety hazards here?")
    assert r["plan"] == ["vqa", "detection", "safety"]
    r = agent.run(IMG, "What kind of scene is this?")
    assert "clip" in r["plan"] and "cnn" in r["plan"]


def test_verification_loop_runs_once_and_flags_unsupported_claim():
    hub = FakeHub(vqa_answer="cars and trees")
    r = VisualAgent(hub).run(IMG, "What is here?")
    steps = [t["step"] for t in r["trace"]]
    assert steps.count("verify") == 1 and steps[-1] == "compose"
    status = {c["term"]: c["status"] for c in r["claims"]}
    assert status["car"] == "supported" and status["tree"] == "not_corroborated"
    assert r["agreement"] == "mixed"


def test_count_question_uses_detector_and_flags_mismatch():
    r = VisualAgent(FakeHub(vqa_answer="5")).run(IMG, "How many people are there?")
    assert "1 x person" in r["summary"]
    assert any("counted 1" in u for u in r["uncertainty"])


def test_yes_no_conflict():
    r = VisualAgent(FakeHub(vqa_answer="no")).run(IMG, "Is there a bus?")
    assert any(c["status"] == "conflict" for c in r["claims"])


def test_safety_street_scene_does_not_apply_ppe_and_flags_proximity():
    r = VisualAgent(FakeHub(worksite=False)).run(IMG, "Any safety hazards?", {"mode": "safety"})
    s = r["tool_results"]["safety"]
    types = {f["type"] for f in s["findings"]}
    assert "ppe" not in types and "context" in types and "proximity" in types
    assert "Not a compliance" not in s["warning"] and "not a compliance" in s["warning"].lower()


def test_safety_worksite_flags_missing_ppe():
    hub = FakeHub(worksite=True)
    r = VisualAgent(hub).run(IMG, "Check PPE", {"mode": "safety"})
    assert any(f["type"] == "ppe" for f in r["tool_results"]["safety"]["findings"])
    # worksite-only labels are queried only when safety is planned
    assert "hard hat" in hub.detect_calls[0]


def test_general_mode_does_not_query_ppe_labels():
    hub = FakeHub()
    VisualAgent(hub).run(IMG, "What objects are visible?")
    assert "hard hat" not in hub.detect_calls[0]


def test_failed_model_is_reported_not_crashing():
    r = VisualAgent(FakeHub(broken={"vqa"})).run(IMG, "What objects are visible?")
    assert any("BLIP VQA failed" in u for u in r["uncertainty"])
    assert r["summary"]


def test_forced_tools_and_custom_labels():
    hub = FakeHub()
    r = VisualAgent(hub).run(IMG, "hello", {"forced_tools": ["clip", "cnn"], "custom_labels": ["bus", "dog"]})
    assert r["plan"] == ["clip", "cnn"]
    assert r["tool_results"]["clip"]["label_set"] == "custom labels"


def test_utils_and_logging(tmp_path):
    assert box_gap([0, 0, 10, 10], [5, 5, 20, 20]) == 0
    assert box_gap([0, 0, 10, 10], [13, 0, 20, 10]) == 3
    out = annotate(IMG, [{"label": "bus", "score": 0.5, "box": [5, 5, 50, 50]}])
    assert out.size == IMG.size
    r = VisualAgent(FakeHub()).run(IMG, "What objects are visible?")
    p = save_run("x.png", "q", r, path=tmp_path / "runs.jsonl")
    assert load_runs(p)[0]["question"] == "q"
