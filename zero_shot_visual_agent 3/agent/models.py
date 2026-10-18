"""ModelHub: lazy, thread-safe loading + inference wrappers for BLIP, CLIP, OWL-ViT, ResNet-50.

All torch / transformers code lives here so the rest of the agent stays lightweight and testable.
A model is only downloaded/loaded the first time a tool needs it.
"""
import threading

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

MODEL_LABELS = {
    "vqa": "BLIP VQA",
    "clip": "CLIP",
    "owl": "OWL-ViT",
    "cnn": "ResNet-50",
}


class ModelHub:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._objs: dict = {}
        self.errors: dict = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ loading
    def ensure(self, name: str) -> bool:
        """Load model `name` if needed. Returns True when it is ready to use."""
        with self._lock:
            if name in self._objs:
                return True
            if name in self.errors:
                return False
            try:
                self._objs[name] = getattr(self, f"_load_{name}")()
                return True
            except Exception as e:  # network down, out of memory, version mismatch...
                self.errors[name] = f"{type(e).__name__}: {e}"
                return False

    def error(self, name: str):
        return self.errors.get(name)

    def status(self) -> dict:
        return {name: ("loaded" if name in self._objs else
                       "failed" if name in self.errors else "not loaded")
                for name in MODEL_LABELS}

    def preload(self, names=None, on_progress=None):
        for name in names or MODEL_LABELS:
            if on_progress:
                on_progress(f"Loading {MODEL_LABELS[name]} ...")
            self.ensure(name)

    def _load_vqa(self):
        from transformers import BlipForQuestionAnswering, BlipProcessor
        proc = BlipProcessor.from_pretrained(C.MODEL_IDS["vqa"])
        model = BlipForQuestionAnswering.from_pretrained(C.MODEL_IDS["vqa"]).to(self.device).eval()
        return proc, model

    def _load_clip(self):
        from transformers import CLIPModel, CLIPProcessor
        proc = CLIPProcessor.from_pretrained(C.MODEL_IDS["clip"])
        model = CLIPModel.from_pretrained(C.MODEL_IDS["clip"]).to(self.device).eval()
        return proc, model

    def _load_owl(self):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        proc = OwlViTProcessor.from_pretrained(C.MODEL_IDS["owl"])
        model = OwlViTForObjectDetection.from_pretrained(C.MODEL_IDS["owl"]).to(self.device).eval()
        return proc, model

    def _load_cnn(self):
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights).to(self.device).eval()
        return weights, model

    # ---------------------------------------------------------------- inference
    def vqa(self, image, question: str) -> str:
        proc, model = self._objs["vqa"]
        inputs = proc(images=image.convert("RGB"), text=question.strip(), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=50)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    def clip_logits(self, images, texts, batch_size: int = 16) -> np.ndarray:
        """Raw CLIP image-text logits, shape (n_images, n_texts)."""
        proc, model = self._objs["clip"]
        images = [im.convert("RGB") for im in images]
        chunks = []
        for i in range(0, len(images), batch_size):
            inputs = proc(text=list(texts), images=images[i:i + batch_size],
                          return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                chunks.append(model(**inputs).logits_per_image.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def detect(self, image, labels, threshold: float = 0.12, nms_iou: float = 0.5,
               max_det: int = 50) -> list:
        """Open-vocabulary detection with OWL-ViT + class-aware NMS."""
        from torchvision.ops import batched_nms
        proc, model = self._objs["owl"]
        labels = list(dict.fromkeys(l for l in labels if l and l.strip()))
        if not labels:
            return []
        image = image.convert("RGB")
        inputs = proc(text=[labels], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        # Transformers versions differ in the post-processing signature; support both.
        try:
            result = proc.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes,
                threshold=threshold, text_labels=[labels])[0]
            names = [str(x) for x in result["text_labels"]]
        except (TypeError, AttributeError, KeyError):
            result = proc.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold)[0]
            names = [labels[i] if i < len(labels) else str(i) for i in result["labels"].tolist()]
        scores, boxes = result["scores"].float().cpu(), result["boxes"].float().cpu()
        if len(scores) == 0:
            return []
        label_ids = torch.tensor([labels.index(n) if n in labels else -1 for n in names])
        keep = batched_nms(boxes, scores, label_ids, nms_iou).tolist()[:max_det]
        return [{"label": names[i], "score": round(float(scores[i]), 4),
                 "box": [round(float(v), 1) for v in boxes[i].tolist()]} for i in keep]

    def cnn_probs(self, image) -> np.ndarray:
        weights, model = self._objs["cnn"]
        x = weights.transforms()(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return F.softmax(model(x), dim=1)[0].cpu().numpy()

    def cnn_topk(self, image, k: int = 5) -> list:
        weights, _ = self._objs["cnn"]
        probs = self.cnn_probs(image)
        cats = weights.meta["categories"]
        idx = np.argsort(-probs)[:k]
        return [{"class": cats[i], "score": float(probs[i])} for i in idx]
