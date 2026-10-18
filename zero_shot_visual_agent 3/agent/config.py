"""Central configuration: model IDs, vocabularies and thresholds."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"

MODEL_IDS = {
    "vqa": "Salesforce/blip-vqa-base",
    "clip": "openai/clip-vit-base-patch32",
    "owl": "google/owlvit-base-patch32",
}

# ---------------------------------------------------------------- detection
CORE_DETECTION_LABELS = [
    "person", "car", "bus", "truck", "bicycle", "motorcycle", "traffic light",
    "dog", "cat", "bird", "backpack", "chair", "table", "bottle", "laptop",
]
# Only added when the safety tool is part of the plan (avoids false "hard hat"
# detections on ordinary street photos).
WORKSITE_DETECTION_LABELS = [
    "hard hat", "safety vest", "forklift", "excavator", "crane", "machinery",
    "ladder", "fire extinguisher", "traffic cone",
]
# Extra per-label minimum scores (OWL-ViT is noisy on PPE-like labels).
LABEL_MIN_SCORE = {"hard hat": 0.30, "safety vest": 0.30, "fire extinguisher": 0.25,
                   "machinery": 0.25}
MAX_DETECTION_LABELS = 40

VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "forklift", "excavator"}

# Multi-word phrases the text parser should keep together.
KNOWN_PHRASES = [l for l in CORE_DETECTION_LABELS + WORKSITE_DETECTION_LABELS if " " in l] + ["stop sign"]

# --------------------------------------------------------------------- CLIP
GENERAL_SCENE_LABELS = [
    "a street scene", "a busy city crosswalk", "a highway", "an indoor room", "an office",
    "a kitchen", "a living room", "a construction site", "a factory or warehouse",
    "a natural landscape", "a beach", "a forest", "a mountain", "a park",
    "a building exterior", "food on a plate", "a close-up of a person", "a group of people",
    "an animal", "a vehicle",
]

# ------------------------------------------------------------------- safety
# Each check compares a "hazard" prompt against its "safe" opposite, so the score is
# P(hazard | hazard-or-safe) and not a softmax over unrelated labels.
WORKSITE_CHECK = ("a photo of a construction site or industrial workplace",
                  "a photo of an ordinary street, home or public place")
PPE_CHECKS = {
    "hard_hat": ("a person wearing a hard hat", "a person with a bare head and no helmet"),
    "safety_vest": ("a person wearing a high visibility safety vest",
                    "a person wearing ordinary clothes without a safety vest"),
}
# (name, hazard prompt, safe prompt, needs_person, needs_vehicle)
SCENE_HAZARD_CHECKS = [
    ("Obstructed walkway", "a photo of a cluttered, obstructed walkway",
     "a photo of a clear, unobstructed walkway", False, False),
    ("Fall hazard / unprotected edge",
     "a photo of an unprotected edge or open drop where someone could fall",
     "a photo of a safe area with guardrails and solid floor", False, False),
    ("Pedestrians close to moving vehicles",
     "a photo of pedestrians very close to moving vehicles",
     "a photo of pedestrians safely separated from vehicles", True, True),
    ("Person close to exposed machinery",
     "a photo of a person close to exposed moving machinery",
     "a photo of a safe workspace with guarded machinery", True, False),
]
SCENE_LOW, SCENE_MEDIUM = 0.70, 0.85      # P(hazard) thresholds
PPE_MISSING_THRESHOLD = 0.70              # P(missing PPE) to flag on a worksite
WORKSITE_THRESHOLD = 0.50
PROXIMITY_LOW, PROXIMITY_MEDIUM = 0.35, 0.15   # gap / person-height

SAFETY_WARNING = (
    "Preliminary visual screening only. CLIP probabilities compare two prompts and are not "
    "calibrated hazard probabilities; 2D box proximity ignores depth. This is not a compliance "
    "determination - a qualified person must verify any potential hazard."
)

# -------------------------------------------------------------- agent rules
SAFETY_TERMS = ["safety", "hazard", "danger", "ppe", "helmet", "hard hat", "safety vest",
                "unsafe", "workplace", "protective equipment", "risk"]
OBJECT_KEYWORDS = ["object", "detect", "where", "locate", "box", "visible", "identify",
                   "what is in", "what's in", "how many", "count", "find", "people",
                   "person", "vehicle", "car", "bus", "truck"]
CLASSIFY_KEYWORDS = ["classify", "category", "scene", "what kind", "image type",
                     "type of image", "what is this", "genre"]
