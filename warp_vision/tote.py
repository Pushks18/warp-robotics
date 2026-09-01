"""Locate the tote, crop to it, and model its colour.

Everything downstream rests on this stage. It gives the segmenter a region of
interest so it never wastes effort on the warehouse machinery around the bin, and
it gives us a per-image model of what the tote plastic looks like -- which is how
we later tell "that mask is an item" from "that mask is the empty tote floor".

Design note. The obvious cue is tote colour: the bins are vivid yellow or blue.
Measured over the dev split:

    tote pixels   S p10 >= 175,  V median  71..177
    background    S p10  27..48, V median  18..34   (blue-lit machinery)

Hue is useless here -- the machinery behind the bin is lit blue and lands on
almost the same hue as a blue tote -- but saturation separates them well. Keying
on saturation alone still failed on 16/120 images, in two distinct ways:

  1. Some totes are beige and dimly lit (tote_0054, tote_0073). Low saturation,
     so the whole bin was rejected.
  2. Big shiny polybags hang over the rim (tote_0035, tote_0047). They break the
     ring of tote plastic, so hole-filling cannot recover them.

Both say the same thing: tote colour is the wrong primitive. What is true of
every image is that the bin and its contents are lit and everything around them
is dark. Taking (bright OR saturated) covers each cue's blind spot -- the beige
tote is bright but not saturated, the shadowed yellow rim is saturated but not
bright -- and lifts item-pixel recall from 0.878 to 0.996. Cropping to the
bounding rectangle of that region rather than its exact outline takes it to
1.000 on all 120 dev images while still discarding ~18% of the frame.
"""
from __future__ import annotations

from dataclasses import dataclass
 
import cv2
import numpy as np

S_MIN = 140        # saturation floor for "vivid tote plastic"
V_MIN = 40         # brightness floor, keeps deeply shadowed walls out
ROI_MARGIN = 0.02  # slack on the crop; dev needs none, held-out might
RIM_BAND = 0.06    # inward band used to sample tote colour, as a frac of frame

_K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


@dataclass
class Tote:
    """Everything the later stages need to know about the bin in one image."""
    footprint: np.ndarray   # bool mask: tote plastic + whatever sits in it
    roi: tuple              # (x0, y0, x1, y1) crop rectangle
    colour: np.ndarray      # median Lab colour of the tote plastic
    spread: np.ndarray      # per-channel robust spread of that colour

    def is_tote_colour(self, bgr_patch: np.ndarray, k: float = 3.0) -> np.ndarray:
        """Which pixels of a patch look like this tote's plastic."""
        lab = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = np.abs(lab - self.colour) / np.maximum(self.spread, 4.0)
        return d.max(axis=-1) <= k


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill regions enclosed by the mask. Padded so the flood always starts
    outside, even when the tote runs to the image border."""
    h, w = binary.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = binary
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return (binary.astype(bool) | ~ff[1:-1, 1:-1].astype(bool))


def _central_component(m: np.ndarray) -> np.ndarray:
    """Largest connected component that reaches the middle of the frame.

    The bin is always roughly centred under the camera; the bright things we want
    to reject (machinery, lit panels) are always at the border. Requiring central
    overlap is what stops a bright wall from being mistaken for the tote.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    h, w = m.shape
    centre = lab[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
    ids = [i for i in range(1, n) if np.any(centre == i)]
    if not ids:
        ids = list(range(1, n))
    return lab == max(ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])


def find_tote(bgr: np.ndarray) -> Tote:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    saturated = (S >= S_MIN) & (V >= V_MIN)
    thresh, _ = cv2.threshold(cv2.GaussianBlur(V, (5, 5), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lit = V >= thresh

    m = (lit | saturated).astype(np.uint8)
    # Close before component analysis: glare and the moulded drain holes
    # fragment the shell, and a fragmented shell loses the largest-component race.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K9)
    footprint = _fill_holes(_central_component(m))

    h, w = footprint.shape
    ys, xs = np.where(footprint)
    if len(ys) == 0:                       # nothing found; fall back to the frame
        return Tote(np.ones((h, w), bool), (0, 0, w, h),
                    np.array([128., 128., 128.], np.float32),
                    np.array([10., 10., 10.], np.float32))
    my, mx = int(ROI_MARGIN * h), int(ROI_MARGIN * w)
    roi = (max(0, xs.min() - mx), max(0, ys.min() - my),
           min(w, xs.max() + 1 + mx), min(h, ys.max() + 1 + my))

    colour, spread = _sample_tote_colour(bgr, footprint)
    return Tote(footprint, roi, colour, spread)


def _sample_tote_colour(bgr: np.ndarray, footprint: np.ndarray):
    """Estimate the tote's own colour from a band just inside its outline.

    Learned per image rather than hardcoded, so a tote colour we have never seen
    -- the held-out split may well have one -- costs us nothing. The band is the
    rim and upper walls, which is tote plastic in almost every frame; the median
    absorbs the occasional item that hangs over the edge.
    """
    h, w = footprint.shape
    r = int(RIM_BAND * min(h, w))
    inner = cv2.erode(footprint.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    band = footprint & ~inner.astype(bool)
    if band.sum() < 500:
        band = footprint
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[band]
    colour = np.median(lab, axis=0)
    spread = 1.4826 * np.median(np.abs(lab - colour), axis=0)   # MAD -> sigma
    return colour.astype(np.float32), spread.astype(np.float32)
