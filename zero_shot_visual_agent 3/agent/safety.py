"""Preliminary safety screening built on OWL-ViT detections + CLIP paired prompts.

Improvements over a plain softmax over unrelated labels:
  * every check compares a hazard prompt with its *safe opposite* (pairwise probability)
  * PPE is checked per detected person (crop-level) and only flagged on a work-site scene
  * person/vehicle proximity uses box geometry (explainable) instead of a CLIP guess
"""

from . import config as C
from .utils import box_gap, crop_box, softmax


def _pair_prob(pos_logit: float, neg_logit: float) -> float:
    return float(softmax([pos_logit, neg_logit])[0])


def analyze_safety(hub, image, detections=None, threshold: float = 0.12) -> dict:
    # ------------------------------------------------------------ detections
    if detections is None:
        labels = (C.CORE_DETECTION_LABELS + C.WORKSITE_DETECTION_LABELS)
        detections = hub.detect(image, labels, threshold=threshold)

    person_min = max(threshold, 0.18)
    persons = sorted((d for d in detections if d["label"] == "person" and d["score"] >= person_min),
                     key=lambda d: -d["score"])[:6]
    vehicles = [d for d in detections if d["label"] in C.VEHICLE_LABELS and d["score"] >= person_min]

    findings = []

    # ----------------------------------------------- scene-level CLIP checks
    texts = list(C.WORKSITE_CHECK)
    for _, hazard, safe, *_ in C.SCENE_HAZARD_CHECKS:
        texts += [hazard, safe]
    logits = hub.clip_logits([image], texts)[0]
    worksite_p = _pair_prob(logits[0], logits[1])

    scene_checks = []
    for i, (name, _, _, needs_person, needs_vehicle) in enumerate(C.SCENE_HAZARD_CHECKS):
        p = _pair_prob(logits[2 + 2 * i], logits[3 + 2 * i])
        applicable = (not needs_person or persons) and (not needs_vehicle or vehicles)
        scene_checks.append({"check": name, "hazard_probability": round(p, 3),
                             "applicable": bool(applicable)})
        if applicable and p >= C.SCENE_LOW:
            findings.append({
                "type": "scene", "severity": "medium" if p >= C.SCENE_MEDIUM else "low",
                "detail": f"{name}: CLIP prefers the hazard description (p={p:.2f}).",
                "score": round(p, 3)})

    # ------------------------------------------------ per-person PPE (crops)
    ppe_checks, crops, crop_idx = [], [], []
    for i, d in enumerate(persons):
        x1, y1, x2, y2 = d["box"]
        if (x2 - x1) >= 24 and (y2 - y1) >= 48:
            crops.append(crop_box(image, d["box"]))
            crop_idx.append(i)
    is_worksite = worksite_p >= C.WORKSITE_THRESHOLD
    if crops:
        hat, vest = C.PPE_CHECKS["hard_hat"], C.PPE_CHECKS["safety_vest"]
        ppe_logits = hub.clip_logits(crops, [hat[0], hat[1], vest[0], vest[1]])
        for row, i in zip(ppe_logits, crop_idx):
            hat_p, vest_p = _pair_prob(row[0], row[1]), _pair_prob(row[2], row[3])
            ppe_checks.append({"person": i + 1, "box": persons[i]["box"],
                               "hard_hat_probability": round(hat_p, 3),
                               "safety_vest_probability": round(vest_p, 3)})
            if is_worksite:
                if 1 - hat_p >= C.PPE_MISSING_THRESHOLD:
                    findings.append({"type": "ppe", "severity": "medium", "score": round(1 - hat_p, 3),
                                     "detail": f"Person #{i + 1}: no hard hat visible (p_missing={1 - hat_p:.2f})."})
                if 1 - vest_p >= C.PPE_MISSING_THRESHOLD:
                    findings.append({"type": "ppe", "severity": "medium", "score": round(1 - vest_p, 3),
                                     "detail": f"Person #{i + 1}: no high-visibility vest visible (p_missing={1 - vest_p:.2f})."})
    if persons and not is_worksite:
        findings.append({"type": "context", "severity": "info", "score": round(worksite_p, 3),
                         "detail": f"PPE rules not applied: the scene looks like a non-work-site "
                                   f"(work-site probability {worksite_p:.2f})."})

    # ------------------------------------------------------ 2D proximity
    proximity = []
    for pi, p in enumerate(persons):
        ph = max(1.0, p["box"][3] - p["box"][1])
        for v in vehicles:
            rel = box_gap(p["box"], v["box"]) / ph
            if rel <= C.PROXIMITY_LOW:
                proximity.append({"person": pi + 1, "vehicle": v["label"], "gap_ratio": round(rel, 2),
                                  "person_box": p["box"],
                                  "severity": "medium" if rel <= C.PROXIMITY_MEDIUM else "low"})
    proximity.sort(key=lambda r: r["gap_ratio"])
    for r in proximity[:5]:
        findings.append({"type": "proximity", "severity": r["severity"], "score": r["gap_ratio"],
                         "detail": f"Person #{r['person']} is close to a {r['vehicle']} in the image "
                                   f"(gap {r['gap_ratio']:.2f} x person height; depth unknown)."})

    order = {"medium": 0, "low": 1, "info": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    flagged = [f for f in findings if f["severity"] in ("medium", "low")]
    return {
        "overall": "potential indicators flagged" if flagged else "no obvious indicators flagged",
        "worksite_probability": round(worksite_p, 3),
        "findings": findings,
        "scene_checks": scene_checks,
        "ppe_checks": ppe_checks,
        "proximity": proximity,
        "persons_considered": len(persons),
        "vehicles_considered": len(vehicles),
        "warning": C.SAFETY_WARNING,
    }
