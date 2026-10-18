# Zero-Shot Visual Agent

Ask any question about an image. A **LangGraph agent** decides which vision tools to run, cross-checks
its own answer, and reports what it is unsure about. Everything runs zero-shot (no training).

| Tool | Model | Used for |
|---|---|---|
| VQA | BLIP `Salesforce/blip-vqa-base` | answering the question |
| Detection | OWL-ViT `google/owlvit-base-patch32` | open-vocabulary boxes, **counting**, claim verification |
| Zero-shot classification | CLIP `openai/clip-vit-base-patch32` | scene tags / custom labels |
| CNN baseline | ResNet-50 (ImageNet) | supervised comparison |
| Safety screening | CLIP paired prompts + detector geometry | PPE, person-vehicle proximity, obstruction / fall hazards |

Agent graph: `plan → tools → reconcile ⇄ verify → compose`

## Run it (3 steps)

**Requirements:** Python 3.10 – 3.12 (3.13 works if PyTorch wheels exist for it), ~6 GB free disk,
Internet on the first run (≈3 GB of model weights are downloaded once and cached).

1. **Unzip** the folder anywhere.
2. **Start it**
   - **Windows:** double-click `run.bat`
   - **macOS / Linux:** `chmod +x run.sh && ./run.sh`

   The script creates a virtual environment, installs `requirements.txt` (first time only) and launches the app.

   *Manual alternative:*
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. **Open** <http://localhost:8501> (opens automatically). Upload an image, type a question, click **Analyze image**.

The first analysis is slow because models download and load; later runs are fast. Models load lazily, so a
simple question only loads what it needs. Use **Preload all models** in the sidebar to download everything up front.

## Using the app
* **Mode:** *General image Q&A* or *Safety analysis* (adds person / vehicle / PPE screening).
* **Custom zero-shot labels:** e.g. `zebra, bicycle lane` → CLIP classifies among them and the detector looks for them.
* **Detection threshold**, **cross-check toggle**, and **manual tool selection** are in the sidebar.
* Result tabs: Evidence · Detections (boxes + counts) · Classification · Safety · Agent trace · Export (JSON / annotated PNG).
* **History** tab shows every run (`outputs/agent_runs.jsonl`).

Try: *"How many people are there?"* (answered by the detector, since VQA models count badly),
*"Is there a dog?"* (yes/no is cross-checked by the detector), *"Are there any safety hazards?"*.

## What changed vs. the original notebook
* The Streamlit app now actually **uses the LangGraph agent** (before it ran a separate fixed pipeline).
* **Verification loop:** the VQA answer is turned into claims (`"cars and trees"` → `car`, `tree`) and checked with the detector; disagreements are reported.
* **Counting** and **yes/no** questions are verified with detections instead of trusting BLIP.
* **NMS** removes duplicate OWL-ViT boxes; noisy PPE labels get higher minimum scores and are only queried in safety mode.
* **Safety rewritten:** the old softmax over unrelated labels always summed to 1 (so "0.947" meant nothing). Now each check
  compares a hazard prompt with its safe opposite, PPE is checked per person crop and **only on work-site scenes**, and
  proximity uses box geometry.
* Lazy model loading, error isolation per tool (one failing model no longer crashes a run), EXIF-aware image loading.
* **Evaluation:** `evaluation/evaluate_cifar10.py` (CLIP zero-shot vs ResNet-50) and `evaluation/evaluate_agent.py`.
* Portable paths (no hard-coded Kaggle paths), tests, one-click launchers.

## Evaluation
```bash
python evaluation/evaluate_cifar10.py --n 1000        # downloads CIFAR-10; writes outputs/cifar10_*
python evaluation/evaluate_cifar10.py --kaggle-dir /kaggle/input/competitions/cifar-10   # Kaggle layout
```
For your own images: copy `evaluation/evaluation_template.csv`, add human-labelled rows
(`image_path, question, reference_answer, gt_objects_json, hazard_reference_0_or_1`), then
`python evaluation/evaluate_agent.py --csv evaluation/my_labels.csv --safety`.
Do not report hazard accuracy without human-reviewed labels.

## Tests (no GPU / downloads needed)
```bash
pip install pytest
python -m pytest -q tests
```

## Project layout
```
app.py                    Streamlit frontend
agent/graph.py            LangGraph agent (plan / tools / reconcile / verify / compose)
agent/models.py           lazy model loading + inference (BLIP, CLIP, OWL-ViT, ResNet-50)
agent/safety.py           safety screening
agent/config.py           model IDs, vocabularies, thresholds
evaluation/               CIFAR-10 benchmark + agent evaluation
zero_shot_visual_agent.ipynb   cleaned notebook using the same package
tests/                    logic + UI smoke tests
```

## Troubleshooting
* **Slow / "Running…" for minutes on first run:** models are downloading. Check the terminal for progress.
* **`pip install torch` fails:** use Python 3.10–3.12, or follow <https://pytorch.org/get-started/locally/> then re-run.
* **Out of memory on CPU:** analyse smaller images, or use *Advanced → choose tools manually* to load fewer models.
* **No GPU?** It works on CPU; expect several seconds per image.
* **Port busy:** `streamlit run app.py --server.port 8502`.

> Safety output is a preliminary visual screening, **not** a certified inspection or compliance determination.
