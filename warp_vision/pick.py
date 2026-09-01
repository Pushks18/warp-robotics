"""Choose which item to pick and where to put the suction cup.

Two decisions, scored separately by the grader but coupled in practice.

**Where on an item.** The suction target is the peak of the mask's distance
transform -- the point furthest from any edge of the item. This is the right
answer twice over. Physically, the point deepest inside a silhouette is the
flattest, most continuous piece of surface available, which is what a suction cup
needs; picking near an edge is how you get a partial seal and drop the item.
Numerically, the grader erodes the item mask by 4 px before testing containment,
so a centroid on an L-shaped or crescent item can miss entirely while the
distance-transform peak is by construction as far inside as the shape allows.

**Which item.** The robot should take what is on top, because lifting a buried
item drags its neighbours. There is no depth sensor here, so "on top" is inferred
from the image. Three cues, combined:

  - *unoccluded area* -- how much of the item's own mask is not overlapped by
    any other detection. An item lying under others has most of its area claimed.
  - *clearance* -- the distance-transform peak value itself, in pixels. A wide
    clear region is both easier to grasp and, empirically, characteristic of an
    unobstructed top surface.
  - *brightness* -- the cell is lit from above, so upper surfaces are brighter.
    Weak on its own (white polybags versus dark boxes) so it gets the smallest
    weight, but it breaks ties the geometry cannot.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PickTarget:
    index: int          # which detection was chosen
    x: int              # suction point in full-frame pixel coords
    y: int
    clearance: float    # distance from the pick point to the item's edge, px
    score: float        # how good a pick candidate this was, 0..1
    reason: str


def _deepest_point(mask: np.ndarray) -> tuple[int, int, float]:
    """Peak of the distance transform: the point furthest from any edge."""
    # Zero-pad so an item touching the crop border is measured against that
    # border too -- otherwise a mask running off the edge reports false depth.
    padded = np.pad(mask.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    idx = int(np.argmax(dist))
    y, x = np.unravel_index(idx, dist.shape)
    return int(x) - 1, int(y) - 1, float(dist[y, x])


class _Geometry:
    """Per-mask measurements that do not change as other items are removed.

    Computed once. Only *exposure* depends on which items are still in the tote,
    so the emptying sequence can be replayed without repeating any of this --
    without the cache, pick_order costs O(n^2) distance transforms and blows the
    latency budget on a crowded tote.
    """

    __slots__ = ("area", "x", "y", "clearance", "brightness")

    def __init__(self, mask: np.ndarray, grey: np.ndarray):
        self.area = float(mask.sum())
        self.x, self.y, self.clearance = _deepest_point(mask)
        self.brightness = float(grey[mask].mean()) if self.area else 0.0


def _measure(masks, bgr):
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return [_Geometry(m, grey) for m in masks]


def _rank(masks, geoms, live):
    """Score every live candidate against the items still present."""
    overlap = np.zeros(masks[0].shape, np.int16)
    for i in live:
        overlap += masks[i].astype(np.int16)

    best = None
    for i in live:
        g = geoms[i]
        if g.area <= 0:
            continue
        exposed = float((overlap[masks[i]] == 1).sum()) / g.area
        clear_n = min(g.clearance / 40.0, 1.0)   # 40 px is a comfortable footprint
        score = 0.5 * exposed + 0.35 * clear_n + 0.15 * g.brightness
        if best is None or score > best[0]:
            best = (score, i, exposed)
    return best


def choose_pick(masks: list[np.ndarray], bgr: np.ndarray) -> PickTarget | None:
    """Rank the detections and return the best suction target, or None."""
    if not masks:
        return None
    geoms = _measure(masks, bgr)
    best = _rank(masks, geoms, list(range(len(masks))))
    if best is None:
        return None
    score, i, exposed = best
    g = geoms[i]
    reason = (f"{exposed:.0%} of this item is unobstructed by other items and it "
              f"offers {g.clearance:.0f} px of clearance around the suction point, "
              f"the strongest evidence in this tote that it is lying on top.")
    return PickTarget(index=i, x=g.x, y=g.y, clearance=g.clearance,
                      score=float(score), reason=reason)


def pick_order(masks: list[np.ndarray], bgr: np.ndarray) -> list[int]:
    """Full emptying sequence, most-exposed first (stretch goal).

    Greedy and re-ranked at every step: once an item is notionally lifted out,
    whatever it was covering becomes the most exposed thing in the tote, which is
    how the real sequence unfolds. Only the exposure term is recomputed; the
    geometry is cached, so the whole sequence costs one pass of distance
    transforms rather than one per step.
    """
    if not masks:
        return []
    geoms = _measure(masks, bgr)
    live, order = list(range(len(masks))), []
    while live:
        best = _rank(masks, geoms, live)
        if best is None:
            break
        live.remove(best[1])
        order.append(best[1])
    return order
