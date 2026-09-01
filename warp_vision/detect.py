"""Segment the tote's contents and reduce the result to one box per item.
 
Backend: **FastSAM-x**, chosen on measurement, in two steps.

MobileSAM's automatic mode needs 20.6 s per image on this laptop (M4 Air).
FastSAM produces a comparable candidate set in a fraction of a second -- an 80x
difference that decides both the sub-second latency goal and whether a full
120-image run is a 1-minute loop or a 40-minute one.

Within FastSAM, the larger backbone is worth its cost here:

    FastSAM-s   F1 0.494   579 ms p95   22 MB weights
    FastSAM-x   F1 0.614   726 ms p95  138 MB weights

+0.12 F1 for +137 ms, still inside the 1 s budget, so -x ships. If the latency
budget tightened -- a faster pack line, or a cell running several cameras off one
machine -- -s is a drop-in swap for about 0.12 F1.

Neither model is trained on warehouse totes and neither knows what an "item" is.
Both are class-agnostic region proposers, which is the right tool -- the task
never asks what an item *is*, only where one ends and the next begins -- but it
means FastSAM returns roughly 97 regions per image of which about 6% correspond
to a real item at IoU >= 0.5. Everything interesting happens in the filter.

Measured over 25 dev images, good masks versus junk:

    feature        good p10/p50/p90      junk p10/p50/p90
    area_frac      0.006 0.033 0.107     0.000 0.001 0.012   <- strongest cue
    containment    1.000 1.000 1.000     0.000 1.000 1.000
    confidence     0.276 0.644 0.897     0.233 0.419 0.804
    border contact 0.000 0.000 0.000     0.000 0.000 0.003   <- useless

The recall ceiling of the raw candidate set is 0.841, so that is the most any
filter can deliver and the number the pipeline is measured against.
"""
from __future__ import annotations

import cv2
import numpy as np

# Area bounds as a fraction of the frame. The lower bound sits below the p10 of
# real items (0.006) to keep small items, while still cutting the fragment noise
# that dominates junk (p90 0.012).
#
# The upper bound exists to reject the bin floor, and it must sit between the
# largest real item and the smallest floor region. It was 0.45, which was simply
# wrong: 22 of the 707 ground-truth items (3.1%) are larger than that, up to
# 0.75, so the filter was excluding real items by construction. 0.60 covers the
# p99 of real items and still sits below a bin floor, which fills 0.5-0.7 of the
# frame when the tote is empty.
MIN_AREA_FRAC = 0.004
MAX_AREA_FRAC = 0.60

MIN_CONTAINMENT = 0.90   # a real item lies inside the tote footprint
MIN_CONF = 0.25          # FastSAM's own objectness
NMS_IOU = 0.60           # (legacy pairwise suppression, kept for comparison)
CONTAINMENT_NMS = 0.90   # (legacy pairwise suppression, kept for comparison)

# A proposal is accepted only if this fraction of it is not already explained by
# the masks chosen before it. See ItemDetector._select.
MIN_NEW_FRACTION = 0.50

# --- Is this region the bin, or something in it? ---
#
# Three per-pixel appearance models failed at this (reports/EXPERIMENTS.md
# E2.1-E2.3), all beaten by shadow inside the bin. The same cues applied to a
# whole mask work, because a median over tens of thousands of pixels is stable
# where a single pixel is not.
#
# Three cues, each measured on the mask, each roughly independent of the others
# (balanced accuracy over 604 labelled masks):
#
#   chroma distance from the tote's own colour   0.859   items are not tote-coloured
#   chromatic spread within the mask             0.870   a tote mask spans lit rim,
#                                                        shadowed floor and colour
#                                                        bounce; an item is one
#                                                        consistent object
#   edge density inside the mask                 0.840   print and creases versus
#                                                        smooth moulded plastic
#
# A mask is kept if at least TWO of the three call it an item. The vote beats any
# single cue (0.908) and beats every tuned AND/OR chain tried, and it degrades
# gracefully: no single threshold being wrong on an unfamiliar tote can sink the
# decision. Solidity was also tested and rejected (0.590) -- the intuition that
# the floor is concave because items are punched out of it does not survive
# contact with FastSAM, which returns floor fragments rather than one region.
TOTE_CHROMA_DIST = 22.0    # further than this from tote colour -> item
TOTE_AB_STD = 22.0         # chromatic spread below this -> item
TOTE_EDGE_DENSITY = 0.03   # edge density above this -> item
MIN_ITEM_VOTES = 2


class ItemDetector:
    def __init__(self, backend: str = "fastsam", imgsz=1024,
                 min_area_frac: float = MIN_AREA_FRAC,
                 max_area_frac: float = MAX_AREA_FRAC,
                 min_conf: float = MIN_CONF, device: str = "mps",
                 weights: str = "FastSAM-x.pt"):
        self.backend = backend
        self.weights = weights
        self.imgsz = imgsz
        self.device = device
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.min_conf = min_conf
        self._model = None

    def _load(self):
        if self._model is None:
            if self.backend == "fastsam":
                from ultralytics import FastSAM
                self._model = FastSAM(self.weights)
            else:
                from ultralytics import SAM
                self._model = SAM("mobile_sam.pt")
        return self._model

    def _select(self, cands):
        """Reduce overlapping proposals to one per item.

        Greedy **set cover**: repeatedly take the candidate contributing the most
        pixels not already explained by the masks chosen so far, and stop
        accepting one once most of it is already covered.

        This replaced pairwise suppression (sort by area, drop anything mostly
        inside a kept mask). Pairwise testing has a blind spot: a proposal 45%
        inside one kept mask and 45% inside another passes both tests
        individually while being 90% redundant overall, and cluttered totes are
        full of exactly that geometry. Measuring against the union of everything
        chosen so far cannot be fooled that way.

        Ordering by marginal contribution rather than by raw area also stops a
        large group-of-items proposal from winning simply for being big: once the
        items it spans are individually explained, it contributes nothing new.
        """
        cands = sorted(cands, key=lambda t: -t[0])
        kept, covered = [], None
        for a, c, m in cands:
            if covered is None:
                covered = np.zeros(m.shape, bool)
            new = float(np.logical_and(m, ~covered).sum())
            if new / a < MIN_NEW_FRACTION:
                continue
            kept.append((a, c, m))
            covered |= m
        return kept

    def detect(self, bgr: np.ndarray, tote) -> tuple[list[np.ndarray], list[float]]:
        """Return full-frame boolean masks and scores, one per detected item."""
        cands = self._candidates(bgr, tote)
        if not cands:
            return [], []
        x0, y0, x1, y1 = tote.roi
        H, W = bgr.shape[:2]
        masks, scores = [], []
        for a, c, m in self._select(cands):
            full = np.zeros((H, W), bool)
            full[y0:y1, x0:x1] = m
            masks.append(full)
            scores.append(c)
        return masks, scores

    def _candidates(self, bgr: np.ndarray, tote):
        """Proposals surviving the per-mask filters, in crop coordinates."""
        model = self._load()
        x0, y0, x1, y1 = tote.roi
        crop = bgr[y0:y1, x0:x1]

        # One or several input scales. retina_masks returns masks at crop
        # resolution regardless of input size, so proposals from different scales
        # are directly comparable and can share one candidate pool; the
        # suppression step below already resolves the duplicates that creates.
        scales = (self.imgsz,) if isinstance(self.imgsz, int) else tuple(self.imgsz)
        md_all, conf_all = [], []
        for sz in scales:
            res = model.predict(crop, verbose=False, imgsz=sz, device=self.device,
                                retina_masks=True, conf=0.2, iou=0.7)[0]
            if res.masks is None or len(res.masks.data) == 0:
                continue
            md_all.append(res.masks.data.cpu().numpy().astype(bool))
            conf_all.append(res.boxes.conf.cpu().numpy())
        if not md_all:
            return []
        md = np.concatenate(md_all, axis=0)
        conf = np.concatenate(conf_all, axis=0)
        H, W = bgr.shape[:2]
        frame_area = float(H * W)

        # --- per-mask filtering, done on the crop to keep memory small ---
        fp_crop = tote.footprint[y0:y1, x0:x1]
        lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        edges = cv2.Canny(cv2.GaussianBlur(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120) > 0
        cands = []
        for m, c in zip(md, conf):
            a = float(m.sum())
            if not (self.min_area_frac <= a / frame_area <= self.max_area_frac):
                continue
            if c < self.min_conf:
                continue
            if np.logical_and(m, fp_crop).sum() / a < MIN_CONTAINMENT:
                continue
            # Two of three cues must agree that this is an item, not the bin.
            ab = lab_crop[..., 1:][m]
            # int() on each: numpy treats bool + bool as logical OR, not a sum.
            votes = (int(np.linalg.norm(np.median(ab, axis=0) - tote.colour[1:]) > TOTE_CHROMA_DIST)
                     + int(ab.std() < TOTE_AB_STD)
                     + int(edges[m].mean() > TOTE_EDGE_DENSITY))
            if votes < MIN_ITEM_VOTES:
                continue
            cands.append((a, float(c), m))

        return cands
