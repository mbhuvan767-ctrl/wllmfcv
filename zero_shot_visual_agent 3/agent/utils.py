"""Image helpers, geometry and run logging. No torch dependency."""
import json
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import config as C


def prepare_image(src, max_side: int = 1600) -> Image.Image:
    """Open an image, fix EXIF rotation, convert to RGB and cap the longest side."""
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def softmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def box_gap(a, b) -> float:
    """Pixel gap between two [x1,y1,x2,y2] boxes (0 when they overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float((dx ** 2 + dy ** 2) ** 0.5)


def crop_box(image: Image.Image, box, pad: float = 0.1) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(image.width, x2 + pad * w), min(image.height, y2 + pad * h)
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080",
            "#9a6324", "#800000", "#808000", "#000075"]


def _color(label: str) -> str:
    return _PALETTE[zlib.crc32(label.encode()) % len(_PALETTE)]


def annotate(image: Image.Image, detections, highlight_boxes=None) -> Image.Image:
    """Draw detection boxes (colour per label). `highlight_boxes` get a thick yellow outline."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    size = max(12, canvas.width // 55)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:               # Pillow < 10.1
        font = ImageFont.load_default()
    width = max(2, canvas.width // 400)
    for box in highlight_boxes or []:
        draw.rectangle(box, outline="#ffd400", width=width * 3)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        col = _color(d["label"])
        draw.rectangle((x1, y1, x2, y2), outline=col, width=width)
        text = f'{d["label"]} {d["score"]:.2f}'
        tb = draw.textbbox((x1, y1), text, font=font)
        th = tb[3] - tb[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle((x1, ty, x1 + (tb[2] - tb[0]) + 6, ty + th + 4), fill=col)
        draw.text((x1 + 3, ty), text, fill="white", font=font)
    return canvas


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_run(image_name, question, result, options=None, path=None) -> Path:
    path = Path(path or C.RUNS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_name": str(image_name),
        "question": question,
        "options": options or {},
        **{k: result.get(k) for k in ("answer", "summary", "agreement", "plan", "tool_results",
                                      "evidence", "uncertainty", "elapsed_seconds")},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record)) + "\n")
    return path


def load_runs(path=None, limit: int = 200):
    path = Path(path or C.RUNS_FILE)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:][::-1]
