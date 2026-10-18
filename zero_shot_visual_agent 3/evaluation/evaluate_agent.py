"""Run the full agent over a human-labelled CSV and compute metrics.

Fill evaluation/evaluation_template.csv (copy it, keep the header) with at least:
    image_path, question, reference_answer          -> VQA answer accuracy
    gt_objects_json  e.g. ["bus","person"]          -> object precision / recall
    hazard_reference_0_or_1                         -> hazard precision / recall / F1
Then:  python evaluation/evaluate_agent.py --csv evaluation/my_labels.csv
Predictions are written to outputs/evaluation_results.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import VisualAgent  # noqa: E402
from agent import config as C  # noqa: E402
from agent.models import ModelHub  # noqa: E402
from agent.utils import prepare_image  # noqa: E402


def norm(s):
    return " ".join(str(s).lower().replace(".", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "evaluation" / "evaluation_template.csv"))
    ap.add_argument("--safety", action="store_true", help="run in Safety mode (needed for hazard metrics)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        sys.exit("The CSV has no rows yet. Add human-labelled examples first (see the docstring).")
    agent = VisualAgent(ModelHub())
    preds, objs, hazards, auto = [], [], [], []
    for _, row in df.iterrows():
        img = prepare_image(row["image_path"])
        res = agent.run(img, str(row["question"]), {"mode": "safety" if args.safety else "general"})
        ans = (res["tool_results"].get("vqa", {}) or {}).get("answer", "")
        preds.append(ans)
        det = res["tool_results"].get("detection", {}) or {}
        objs.append(json.dumps(sorted(set(d["label"] for d in det.get("detections", [])))))
        s = res["tool_results"].get("safety")
        hazards.append(int(any(f["severity"] in ("medium", "low") for f in s["findings"])) if s else None)
        ref = norm(row.get("reference_answer", ""))
        auto.append(int(bool(ref) and (ref in norm(ans) or norm(ans) in ref)))
        print(f"{Path(str(row['image_path'])).name}: {ans!r}")
    df["predicted_answer"], df["pred_objects_json"], df["hazard_prediction_0_or_1"] = preds, objs, hazards
    df["auto_match_0_or_1"] = auto
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    out = C.OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(out, index=False)

    print("\n=== Metrics ===")
    if df["reference_answer"].notna().any():
        print(f"VQA auto-match accuracy (substring match, check manually): {df['auto_match_0_or_1'].mean():.3f}")
    if "answer_correct_0_or_1" in df and df["answer_correct_0_or_1"].notna().any():
        print(f"VQA human-labelled accuracy: {pd.to_numeric(df['answer_correct_0_or_1'], errors='coerce').mean():.3f}")
    tp = fp = fn = 0
    for _, r in df.dropna(subset=["gt_objects_json"]).iterrows():
        gt, pr = set(json.loads(r["gt_objects_json"])), set(json.loads(r["pred_objects_json"]))
        tp += len(gt & pr); fp += len(pr - gt); fn += len(gt - pr)
    if tp + fp + fn:
        print(f"Object precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f} (image-level label sets)")
    haz = df.dropna(subset=["hazard_reference_0_or_1", "hazard_prediction_0_or_1"])
    if len(haz):
        from sklearn.metrics import classification_report, confusion_matrix
        yt, yp = haz["hazard_reference_0_or_1"].astype(int), haz["hazard_prediction_0_or_1"].astype(int)
        print(classification_report(yt, yp, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(yt, yp))
    print("Saved:", out)


if __name__ == "__main__":
    main()
