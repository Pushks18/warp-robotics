"""Per-image orchestration: tote -> detections -> pick or flag."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .pick import choose_pick, pick_order
from .tote import find_tote


@dataclass
class ImageResult:
    file: str
    items: list = field(default_factory=list)
    action: str = "flag"
    pick: dict | None = None
    reason: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    pick_sequence: list = field(default_factory=list)

    def to_manifest_entry(self) -> dict:
        e = {"file": self.file, "items": self.items,
             "action": self.action, "reason": self.reason,
             "confidence": round(float(self.confidence), 3)}
        if self.action == "pick" and self.pick is not None:
            e["pick"] = self.pick
        if self.pick_sequence:
            e["pick_order"] = self.pick_sequence
        return e


def process_image(path, detector, min_confidence: float = 0.0,
                  want_order: bool = False) -> ImageResult:
    """Run the full pipeline on one image.

    The flag decision is deliberately simple: we flag when the detector finds
    nothing it believes in, or when the best available pick is weak. An empty
    tote is not tested for separately -- it is the case where nothing survives
    detection, which is the same condition. That is one fewer threshold to
    defend, and after three failed hand-crafted emptiness cues (see
    reports/EXPERIMENTS.md) fewer thresholds is the right direction.
    """
    t0 = time.perf_counter()
    name = path.name if hasattr(path, "name") else str(path)
    bgr = cv2.imread(str(path))
    if bgr is None:
        return ImageResult(file=name, reason="Image could not be read.",
                           latency_ms=(time.perf_counter() - t0) * 1000)

    tote = find_tote(bgr)
    masks, scores = detector.detect(bgr, tote)

    items = [{"bbox": _bbox(m), "score": round(float(s), 3)}
             for m, s in zip(masks, scores)]

    res = ImageResult(file=name, items=items,
                      latency_ms=(time.perf_counter() - t0) * 1000)

    if not masks:
        res.action = "flag"
        res.reason = ("No item-like region survived detection inside the tote, "
                      "so the tote reads as empty or unresolvable.")
        res.confidence = 0.9
        return res

    target = choose_pick(masks, bgr)
    if target is None or target.score < min_confidence:
        res.action = "flag"
        res.reason = ("Detected items but no candidate offered a confident "
                      "suction surface, so this tote goes to a human.")
        res.confidence = 0.3
    else:
        res.action = "pick"
        res.pick = {"x": int(target.x), "y": int(target.y),
                    "item_index": int(target.index)}
        res.reason = target.reason
        res.confidence = round(float(target.score), 3)
        if want_order:
            res.pick_sequence = pick_order(masks, bgr)

    res.latency_ms = (time.perf_counter() - t0) * 1000
    return res


def _bbox(mask: np.ndarray) -> list:
    ys, xs = np.where(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]
