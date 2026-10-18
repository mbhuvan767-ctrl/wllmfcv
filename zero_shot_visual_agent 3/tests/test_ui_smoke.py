"""Headless UI smoke test with a fake model layer.  Run: python -m pytest -q tests/test_ui_smoke.py"""
import io
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from PIL import Image                                     # noqa: E402
from streamlit.testing.v1 import AppTest                  # noqa: E402

from test_agent_logic import FakeHub                      # noqa: E402


class UIHub(FakeHub):
    device = "cpu"
    def __init__(self): super().__init__(worksite=False)
    def status(self): return {"vqa": "loaded", "clip": "not loaded", "owl": "loaded", "cnn": "failed"}
    def preload(self, *a, **k): pass


def test_app_renders_and_runs(tmp_path, monkeypatch):
    fake = types.ModuleType("agent.models")
    fake.ModelHub = UIHub
    fake.MODEL_LABELS = {"vqa": "BLIP VQA", "clip": "CLIP", "owl": "OWL-ViT", "cnn": "ResNet-50"}
    monkeypatch.setitem(sys.modules, "agent.models", fake)
    from agent import config as C
    monkeypatch.setattr(C, "RUNS_FILE", tmp_path / "runs.jsonl")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    # No image yet -> analyze button disabled
    assert [b for b in at.button if "Analyze" in b.label][0].disabled
    assert "Zero-Shot" in at.markdown[0].value


def test_result_view_renders_all_tabs(tmp_path, monkeypatch):
    from agent import VisualAgent
    fake = types.ModuleType("agent.models")
    fake.ModelHub = UIHub
    fake.MODEL_LABELS = {"vqa": "BLIP VQA", "clip": "CLIP", "owl": "OWL-ViT", "cnn": "ResNet-50"}
    monkeypatch.setitem(sys.modules, "agent.models", fake)
    from agent import config as C
    monkeypatch.setattr(C, "RUNS_FILE", tmp_path / "runs.jsonl")

    img = Image.new("RGB", (400, 300), "gray")
    for worksite in (False, True):
        hub = FakeHub(worksite=worksite)
        res = VisualAgent(hub).run(img, "What kind of scene is this, and any safety hazards? How many people?",
                                   {"mode": "safety", "custom_labels": ["bus", "dog"]})
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.session_state["last"] = {"result": res, "image": img, "name": "x.png", "question": "q"}
        at.run()
        assert not at.exception, at.exception
        assert at.subheader and any("Result" in s.value for s in at.subheader)
        assert len(at.tabs) >= 9
