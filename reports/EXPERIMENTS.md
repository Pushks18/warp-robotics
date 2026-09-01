# Experiment log

A running record of what was tried, what the numbers said, and why the design
moved where it did. Every measurement here is reproducible from the scripts in
`experiments/`. Ground-truth masks are used in this file **only to validate
design decisions** -- no stage of the shipped pipeline reads them.

---

## Stage 1 -- Locating the tote

### Why this stage exists

Two things are needed before any segmentation happens:

1. **A region of interest.** Roughly 20-35% of every frame is warehouse
   machinery, conveyor and wall. Segmenting that is wasted compute and a source
   of false positives.
2. **A model of the tote itself.** The grader scores a pick as correct only if it
   lands on an *item*. A pick on the tote floor scores 0. So the pipeline has to
   be able to say "that region is the bin, not something in it."

The measure that matters for (1) is **item-pixel recall**: of all ground-truth
item pixels, what fraction survives the crop. Anything lost here can never be
recovered downstream, so the target is 1.0, and area saved is secondary.

### E1.1 -- Colour separability probe

Percentiles of HSV over ground-truth regions, 8 sample images:

| region | H p10-p90 | S p10 | V median |
|---|---|---|---|
| tote | 27-32 (yellow) or 106-112 (blue) | >= 175 | 71-177 |
| background | 30-126, mostly 115-118 | 27-48 | 18-34 |

**Finding.** Hue is not usable. The machinery behind the bin is lit blue and
lands on almost the same hue as a blue tote. Saturation and brightness both
separate cleanly. Chose `S >= 140 and V >= 40`, placed in the gap between the two
distributions and biased toward the background side so shadowed tote walls
(V p10 as low as 41) survive.

### E1.2 -- Saturation-keyed footprint: 0.966 recall, 6 failures

Largest saturated connected component, holes filled (items sit inside the tote,
so they punch holes in an otherwise solid blob).

    item-pixel recall  mean 0.9655   min 0.000   n<0.99: 6/120

Aggregate looks fine; the tail does not. Inspecting the 6 failures showed **two
distinct causes**, not one:

| failure | images | cause |
|---|---|---|
| **Tote colour outside the assumed palette** | tote_0054 (recall 0.00), tote_0073 (0.00) | These bins are **beige** and dimly lit, not yellow or blue. Saturation falls under the threshold, the entire tote is rejected, and the largest "saturated" component becomes a sliver of wall. |
| **Item overflows the tote outline** | tote_0035 (0.03), tote_0047 (0.48) | Large shiny polybags hang over the rim or spill out of frame. They break the ring of tote plastic, so hole-filling cannot enclose them. |

**This is the finding that shaped the stage.** A 0.966 mean would have shipped a
pipeline that returns *nothing* on any tote colour Amazon did not happen to use
in my sample -- exactly the kind of thing the held-out split exists to catch.
Tote colour is the wrong primitive.

### E1.3 -- Brightness, and the union: 0.996 recall

What is true of every image, beige bins included, is that **the bin and its
contents are lit and everything around them is dark** (foreground V median
88-170, background 19-31). Brightness alone is noisy -- the background has bright
outliers at the frame edge (V p95 up to 233) -- so it is combined with the
saturation cue and constrained to the connected component reaching the centre of
the frame.

| ROI rule | recall mean | min | n<0.99 | area kept |
|---|---|---|---|---|
| saturated only | 0.878 | 0.000 | 16 | 0.647 |
| bright only (Otsu on V) | 0.846 | 0.040 | 43 | 0.545 |
| **bright OR saturated** | **0.996** | 0.506 | **1** | 0.723 |

The union works because each cue covers the other's blind spot: the beige tote is
bright but unsaturated, the shadowed yellow rim is saturated but dark. Requiring
the component to reach the centre of the frame is what stops a lit wall panel
from winning the largest-component race.

### E1.4 -- Rectangle vs exact outline

The one remaining failure (tote_0047, 0.51) is a bag spilling out of the top of
the frame. Since the ROI's only job is to bound the crop, using the bounding
rectangle rather than the exact outline recovers it for free:

| ROI | recall mean | min | area kept |
|---|---|---|---|
| exact footprint | 0.996 | 0.506 | 0.723 |
| bounding rect, 0% margin | **1.000** | **1.000** | 0.820 |
| bounding rect, 2% margin | **1.000** | **1.000** | 0.863 |

**Shipped: bounding rect + 2% margin.** Dev needs no margin at all; the 2% costs
4% more area and buys slack if the held-out split seats the bin slightly
differently under the camera. Cheap insurance against the thing I cannot measure.

**Stage 1 result: item-pixel recall 1.0000 on all 120 dev images, ~57 ms/image,
18% of each frame discarded.**

---

## Stage 2 -- Telling the tote from its contents (two failed attempts)

Both the empty-tote decision and the "is this detection an item or the bin floor"
filter need a way to recognise tote plastic. Two hypotheses were tested and
**both failed**. Recorded here because the failures constrain the design.

### E2.1 -- FAILED: parametric Lab colour model

Learn the tote's colour per image from a band just inside its outline (the rim,
which is tote plastic in nearly every frame; the median absorbs the occasional
overhanging item), then classify pixels by normalised per-channel Lab distance.

Learning the colour per image rather than hardcoding it was the right instinct --
it costs nothing when the held-out split has an unseen bin colour. The *distance
metric* was wrong:

    GT tote pixels correctly kept   mean 0.946   min 0.510
    GT item pixels wrongly kept     mean 0.510   max 0.998    <-- unusable

Half of all item pixels pass as tote plastic. Cause: normalising by a MAD-derived
spread across all three channels, **lightness included**. A tote spans a huge
brightness range (lit rim to shadowed wall), so the learned spread is wide, and a
wide spread admits everything.

### E2.2 -- FAILED: chroma-only (drop the lightness channel)

Standard fix for illumination variation: compare only the a,b chromaticity
channels. It made things worse in the other direction.

| tolerance | tote px kept | item px wrongly kept |
|---|---|---|
| k=6 | 0.217 | 0.022 |
| k=10 | 0.387 | 0.049 |
| k=14 | 0.538 | 0.075 |

Rejecting 78% of the tote is no more usable than admitting half the items. Cause:
shadowed plastic **desaturates toward neutral grey**, so its chromaticity drifts
far from the reference sampled on the brightly lit rim. Chromaticity is only
illumination-invariant while the surface is actually illuminated.

### E2.3 -- FAILED: emptiness by edge density

Different cue entirely: an empty tote should be a smooth surface, items should
introduce edges. Canny edge density inside the eroded interior:

    empty totes (n=8)     0.0086 .. 0.0193
    full totes, lowest    0.0038 (tote_0054), 0.0100, 0.0108, 0.0119 ...

**The signal is inverted.** Empty totes score *higher* than several full ones.
Empty interiors are covered in scratches, drain holes and moulding seams, while
tote_0054 holds one small flat item in a large bin and is genuinely almost
featureless. No threshold separates these.

### What these three failures imply

Every hand-crafted appearance model tried so far has been beaten by the same two
things: **shadow inside the bin**, and **the bin's own surface texture**. Rather
than keep tuning a fourth one blind, the next step is to run the segmenter and
design the tote-vs-item filter against its real output, using cues that do not
depend on absolute appearance:

- **area** -- GT items span 29.6k px (p5) to 505k px (p95) against a 1.25M px
  frame; the tote floor is larger than any single item
- **containment** -- items sit inside the footprint, walls run along its border
- **the segmenter's own stability/IoU scores**

Emptiness is deferred to fall out of the same filter: a tote where nothing
survives it is a tote to flag. That removes a threshold instead of adding one,
which given three failed thresholds is the right direction.

**Open risk:** tote_0054 (one small flat item in a large bin) is hard for *any*
appearance-based method and is a likely failure on the held-out split too.

---

## Stage 3 -- Segmentation backend

### E3.1 -- MobileSAM is 80x too slow

Both models are **class-agnostic** region proposers, which is the right family of
tool: the task never asks what an item *is*, only where one ends and the next
begins, and the ground truth carries no category labels at all. A COCO-trained
detector would confidently box the one bottle it recognises and miss six
anonymous polybags beside it.

| backend | masks | time / image |
|---|---|---|
| MobileSAM, automatic mask generation | 86 | **20.6 s** |
| FastSAM-s, single forward pass | 133 | **0.25 s** |

MobileSAM's automatic mode grids the image with point prompts and decodes each
one. On an M4 Air that is 41 minutes for one pass over the dev split, which ends
any hope of the sub-second stretch goal and, more importantly, makes the
build-measure loop unusable. FastSAM ships.

*(An early version of this benchmark also exhausted memory by upsampling a
256-prompt batch to full resolution in one allocation, ~1.2 GB, while holding two
other models in the same process. 16 GB of unified memory is shared with the GPU;
one model per process, modest batches.)*

### E3.2 -- Which FastSAM

| backend | F1 | p95 latency | weights |
|---|---|---|---|
| FastSAM-s | 0.494 | 579 ms | 22 MB |
| **FastSAM-x** | **0.614** | 726 ms | 138 MB |

**Shipped: FastSAM-x.** +0.12 F1 for +137 ms, still inside the 1 s budget. If
that budget tightened -- a faster line, or one machine driving several cells --
`-s` is a one-line swap costing about 0.12 F1.

### E3.3 -- The recall ceiling

Of 138 ground-truth items across 25 images, **116 (0.841)** are representable by
*some* mask in FastSAM's raw output at IoU >= 0.5. That is the hard upper bound
on detection F1 for this architecture, and the number worth comparing against --
not 1.0.

The raw output is ~97 masks per image of which **about 6% are real items**. The
filter has to discard 94% of what the model returns.

---

## Stage 4 -- Telling the bin from its contents, properly

### E4.1 -- The same cue, at mask level instead of pixel level

Stage 2 failed three times trying to classify *pixels*. Applying the identical
chromaticity comparison to a whole *mask* -- median colour of ~40,000 pixels
versus the tote colour learned from the rim -- works:

    balanced accuracy 0.859      items kept 0.919      tote rejected 0.918

The cue was never wrong; the granularity was. A median over tens of thousands of
pixels is stable where a single pixel is destroyed by shadow.

**Effect on the shipped pipeline, from this one filter:**

| | detection F1 | pick score | flags |
|---|---|---|---|
| before | 0.237 | 0.583 | 0 |
| after | **0.445** | **0.850** | 7 |

It also fixed the empty-tote case for free (7 of 8 flagged, from 0), exactly as
intended: an empty bin is one where nothing survives the filter, so no separate
emptiness threshold is needed.

### E4.2 -- A cautionary measurement

Re-running E4.1 on 40 images instead of 30 dropped the same rule from **0.919 to
0.859**. The extra accuracy was the threshold fitting a small sample. Recorded
because it is the whole argument for the held-out split, and a reason every
threshold here is a round number chosen between two distributions rather than the
arg-max of a sweep.

### E4.3 -- Two more cues, and a majority vote

Chroma alone was at its limit; a better *reference* colour did not help (the
footprint chromaticity mode scored 0.568, far worse than the rim-band median).
Two further mask-level cues, largely independent of chroma:

| cue | balanced acc | direction |
|---|---|---|
| chroma distance from tote colour | 0.859 | items are not tote-coloured |
| chromatic spread *within* the mask | 0.870 | **tote masks vary more** (31.2 vs 11.2) |
| edge density inside the mask | 0.840 | items carry print and creases (0.110 vs 0.015) |
| solidity | 0.590 | **rejected** |

Two results here were the opposite of what I predicted, which is why they were
measured rather than assumed:

- **Chromatic spread runs backwards.** I expected moulded plastic to be uniform
  and printed packaging varied. In fact a single tote mask spans lit rim,
  shadowed floor and colour bounce, while an item is one consistent object.
- **Solidity is useless.** The intuition that the floor is concave because items
  are punched out of it does not survive contact with FastSAM, which returns
  floor *fragments*, not one wrapping region.

| rule | balanced acc |
|---|---|
| chroma only | 0.859 |
| chroma AND edge density | 0.891 |
| **2 of 3 cues agree** | **0.908** |

**Shipped: 2-of-3 majority vote.** It beats every tuned AND/OR chain tried, and
it degrades gracefully -- on an unfamiliar tote, one cue being wrong cannot sink
the decision. Preferred over a fitted classifier precisely because there is a
held-out split.

*(Implementation note: the first version of this vote flagged all 120 images.
NumPy treats `bool + bool` as logical OR, so the vote was always `True` and
`True < 2` rejected every mask. Each term now casts to `int`.)*

---

## Stage 5 -- Suppression, and how many boxes to emit

FastSAM proposes an item, its parts, and the group it belongs to simultaneously.
Which survives is architecture, not tuning. Masks computed once per image, every
config scored on identical candidates:

| config | F1 | boxes/img (gt 5.5) |
|---|---|---|
| area sort, containment 0.80 | 0.439 | 4.8 |
| area sort, containment 0.90 | 0.458 | 5.1 |
| area sort, containment **off** | 0.398 | 9.1 |
| confidence sort, containment 0.80 | 0.437 | 7.2 |
| **area sort, containment 0.90, conf 0.25** | **0.473** | 5.0 |

**Hypothesis corrected.** I expected containment suppression to be destroying
recall, since a genuinely separate item in a cluttered tote is often 80% buried
under its neighbour. Turning it off is clearly *worse* (0.398, 9.1 boxes against
5.5 real). Sorting by area and resolving nesting toward whole objects is right;
the suppression threshold just needed loosening from 0.80 to 0.90.

The gains here are small, which is itself the finding: suppression is not the
bottleneck. Mask granularity is.

---

## Stage 6 -- The flag decision, and why the pipeline flags rarely

The grader pays 1.0 for a correct pick, 0.0 for a wrong one, and **0.25 for
flagging a tote that has items**. So a confidence gate is only worth having if it
is right about failure more than **3 times in 4**: catching a wrong pick gains
0.25, but flagging a pick that would have succeeded loses 0.75.

Measured over the 113 picks in the final run (109 correct, 4 wrong):

    confidence of correct picks   p10 0.888   median 0.925
    confidence of the 4 wrong     0.931, 0.935, 0.938, 0.963

**The wrong picks are more confident than the median correct pick.** The signal
is not weak, it is absent. Flagging the N least-confident picks never catches a
failure at any N and only destroys points:

| flag lowest N | wrong caught | correct lost | net points |
|---|---|---|---|
| 0 | 0 | 0 | **109.00** |
| 5 | 0 | 5 | 105.25 |
| 20 | 0 | 20 | 94.00 |

**Shipped: flag only when nothing survives detection** (an empty or unreadable
bin). At 96% pick accuracy the 0.25 credit is a trap, and the honest reading is
that this pipeline's confidence is a *pick-quality* score, not a calibrated
probability of being right. Making it one -- fitting a failure predictor against
held-out labels -- is the first thing I would do with more time, and it is the
only route to spending flags profitably.

For scale: flagging every tote scores 0.31. This pipeline scores 0.958.

---

## Stage 7 -- Failure taxonomy

### Detection degrades monotonically with clutter

| items in tote | images | mean F1 | boxes predicted / actual |
|---|---|---|---|
| 0 (empty) | 8 | 0.875 | 0.25 |
| 1-3 | 39 | 0.717 | **1.38** |
| 4-6 | 29 | 0.566 | 0.96 |
| 7-10 | 22 | 0.491 | 0.98 |
| 11+ | 22 | **0.366** | 0.88 |

The ratio column shows **two opposite failure modes**: sparse totes are
over-segmented (1.38 boxes per real item), crowded ones under-segmented (0.88).

### Mode A -- "item" means pickable unit, not visible object (dominant)

`tote_0108`: ground truth draws **one** box around a polybag holding ~15 loose
bottles. FastSAM finds the bottles and returns **19** boxes. F1 0.10.

This is not a segmentation error. FastSAM segments by visual coherence; the task
defines an item as **one pickable unit** -- one bag, one SKU, one suction event --
regardless of how many distinct objects are visible through the plastic. A
class-agnostic segmenter has no way to know that a clear polybag is a container
rather than a window.

**This is the ceiling on detection F1 and no threshold reaches it.** It is also
why the pick score (0.958) is so far ahead of detection (0.585): a pick needs
only to land on *a* real item, not to partition the tote correctly.

**Fix: better code, but not better tuning.** Fine-tune the segmenter on
ARMBench's own instance labels so it learns the pickable-unit concept, or add a
merge step that detects polybag boundaries (specular sheen, seams) and unions the
objects inside one. Both are real work, neither is a knob.

### Mode B -- occlusion in crowded totes

`tote_0053` (18 items, F1 0.18): heavily overlapping white polybags. Ground truth
boxes overlap enormously; a mostly-buried item has too little visible surface for
any single-view method to recover its true extent, and IoU 0.5 against its full
box is unreachable.

**Fix: a better camera, not better code.** This is the honest answer. Depth from
a stereo or ToF sensor would separate touching surfaces that share colour and
texture. Failing that, the operational fix is what a real cell does anyway --
pick the top item, re-image, repeat, which is why pick order matters more than
a perfect single-shot manifest.

### Mode C -- adjacent lookalikes merge

`tote_0082` (5 items, 5 boxes, F1 0.00): the right *number* of boxes, none
matching. Two identical cylindrical products lying parallel are merged into one
mask while another item is split. Same root cause as Mode A, without the polybag.

### Mode D -- residual tote-as-item (4 images)

The 2-of-3 vote is 0.908, not 1.0. In `tote_0030` (an empty bin) two floor
fragments still pass, and the pipeline picks in an empty tote -- the worst
outcome the grader defines. **Fix: better code.** The cue that would settle it is
geometric rather than appearance-based: the bin floor is a plane at a known
height, so any depth signal separates it trivially.

### Known weak case carried from Stage 1

`tote_0054` -- one small flat item in a large beige bin -- is hard for every
appearance-based method tried here and is a likely failure on the held-out split.

---

## Stage 8 -- How much of this score is real?

### E8.1 -- Bootstrap over the dev split

The held-out split is unseen, so its score cannot be computed. Part of the gap is
plain sampling noise: 120 images is a small sample and a fresh draw from the same
distribution lands elsewhere. Resampling the 120 per-image scores with
replacement, 20,000 draws:

    dev split          F1 0.585         pick 0.958
    bootstrap 90% CI   F1 0.540-0.631   pick 0.942-0.992
    standard error     F1 0.028         pick 0.016

**So F1 0.54-0.63 from sampling alone**, before any distribution shift.

*(The bootstrap script recomputes pick score without the grader's 4 px erosion
and therefore reads 0.967 rather than 0.958. `grade.py` is authoritative; the
interval is centred on its number.)*

### E8.2 -- What the bootstrap does NOT capture

Resampling the same 120 images cannot model a tote colour, item mix, lighting or
camera pose that does not occur in them. Every such shift can only push the score
down. The one instance actually observed -- two beige totes among 118 yellow and
blue ones (E1.2) -- took item-pixel recall from 1.000 to 0.000 on those images
until the design changed. That is the size of effect a single unseen condition
can have, and it is entirely invisible to a bootstrap.

**Honest expectation for the held-out run: F1 in the low 0.5s, pick in the low
0.9s.** Stated here so the estimate is on record before the number is known.

### E8.3 -- Design choices made specifically to survive the shift

Recorded because each cost something measurable on the dev split:

| choice | cost on dev | what it buys |
|---|---|---|
| ROI = bright OR saturated, not tote colour | ~14% more area kept | works on a bin colour never seen |
| Tote colour learned per image from the rim | none | no palette is hardcoded anywhere |
| 2-of-3 vote instead of a fitted classifier | ~0.01 balanced acc | no single wrong threshold can sink an image |
| Round-number thresholds, not sweep optima | up to 0.06 balanced acc | E4.2 showed sweep optima were fitting noise |
| 2% ROI margin | 4% more area | slack if the bin sits differently under the camera |
| Emptiness inferred, not thresholded | none | one fewer thing to be wrong about |

None of these were tuned against the dev score, and no threshold was chosen as
the arg-max of a sweep.

---

## Decision register

Every design decision, what it was chosen over, and the measurement behind it.

| # | decision | alternatives rejected | evidence |
|---|---|---|---|
| D1 | ROI keyed on bright OR saturated | saturation only (0.878 recall); brightness only (0.846) | E1.3 |
| D2 | ROI as bounding rectangle + 2% margin | exact footprint outline (0.996 recall, fails on overhanging bags) | E1.4 |
| D3 | Tote colour learned per image from the rim band | hardcoded yellow/blue palette; footprint chromaticity mode (0.568) | E1.2, E4.3 |
| D4 | FastSAM over MobileSAM | MobileSAM auto-mode at 20.6 s/image | E3.1 |
| D5 | FastSAM-**x** over FastSAM-s | -s at F1 0.494 vs -x at 0.614, +137 ms | E3.2 |
| D6 | Tote/item decided at **mask** level | three per-pixel models, all failed | E2.1-E2.3, E4.1 |
| D7 | 2-of-3 majority vote over three cues | chroma alone (0.859); tuned AND chains (0.891); solidity (0.590, dropped) | E4.3 |
| D8 | Area-sorted greedy suppression, containment 0.90 | containment off (0.398); confidence sort (0.437) | E5 |
| D9 | Emptiness inferred from an empty detection set | edge density threshold (signal inverted) | E2.3 |
| D10 | Flag only when nothing is detected | confidence gate (catches 0 of 4 failures at any N) | E6 |
| D11 | Pick point = distance-transform peak | mask centroid (fails on L-shaped and crescent items) | pick.py rationale |
| D12 | Pick ranking = exposure + clearance + brightness | area or brightness alone (ranks the bin floor first) | E6 diagnosis |

## Open hypotheses, not yet tested

1. **The parent mask may already exist and be getting filtered out.** In
   `tote_0108` FastSAM returns 19 bottles where GT wants one polybag. If FastSAM
   *also* proposes the whole-bag region and it is being killed by `MAX_AREA_FRAC`,
   the confidence floor or the 2-of-3 vote, recovering it is cheap. If it is
   never proposed, this needs a merge step or a fine-tune. **Untested, and the
   single highest-value thing to check.**
2. **Near-miss boxes.** If unmatched GT items are being covered at IoU 0.35-0.49
   rather than missed outright, mask boundary refinement pays; if they sit near
   0, the proposal set is the problem. Decides refinement versus proposal work.
3. **Multi-scale proposals.** Running FastSAM at two input sizes and merging
   should raise the 0.841 ceiling, at roughly double the segmentation cost. There
   is ~390 ms of latency headroom for it.
4. **Box-prompted mask refinement.** MobileSAM's image encoder runs once (~50-100
   ms) and each box prompt decodes in single-digit ms, so FastSAM proposals could
   be refined by SAM cheaply. Better boundaries, therefore better IoU.
5. **Monocular depth.** Would settle floor-as-item (Mode D) geometrically rather
   than by appearance, and give a true occlusion order. Cost likely 200-500 ms.

---

## Stage 9 -- Chasing the remaining F1

### E9.1 -- Where the loss actually is

Over 40 images / 218 ground-truth items:

    matched (IoU >= 0.5)                      120  (0.550)
    near-miss (IoU 0.35-0.49, median 0.445)    42
    genuine miss (IoU < 0.35)                  56

Of the 98 misses, **32 had a raw FastSAM proposal that would have matched at
IoU >= 0.5 and were removed by this pipeline's own filters**:

| stage that removed it | count |
|---|---|
| 2-of-3 tote/item vote | 12 |
| suppression / NMS | 8 |
| max-area cap | 5 |
| confidence floor | 5 |
| containment | 2 |

### E9.2 -- The max-area cap was a bug

The cap existed to reject the bin floor, at 0.45 of the frame. Measured against
all 707 ground-truth items:

    p50 0.098   p95 0.402   p99 0.560   max 0.750
    items above 0.45: 22 (3.1%)

**The filter was excluding 3.1% of real items by construction.** Raised to 0.60,
which covers the p99 of real items and still sits below an empty bin floor
(0.5-0.7 of frame).

**Result: no measurable change on dev (F1 0.585, pick 0.958 both before and
after).** The cap was not binding in practice. Kept regardless -- a bound that
provably excludes real items is wrong whether or not this particular split
notices, and the held-out split may contain the large items that dev does not.

### E9.3 -- FAILED: box-prompted SAM refinement

42 items sit at IoU 0.35-0.49, just under the bar. If those were boundary errors,
re-segmenting FastSAM's boxes with MobileSAM (encoder once per image, cheap
per-box decode) should convert them.

| | F1 | latency |
|---|---|---|
| baseline | 0.614 | 602 ms |
| + SAM refinement | **0.612** | 857 ms (+196) |

**No gain, at +196 ms**, despite SAM replacing 98% of masks. Rejected.

The value is in what it rules out: the near-misses are **not** boundary errors.
SAM draws a sharper outline around the same region, and the same region is the
problem.

### E9.4 -- What the near-misses actually are

Geometry of the 188 near-miss pairs across the full split:

    predicted box area / GT box area:  p25 0.44   median 0.56   p75 1.14
    predicted smaller than GT: 66%     larger: 26%     similar: 9%

Two thirds of near-misses cover **roughly half** the item's true extent.

**Hypothesis tested and rejected:** that GT items are disconnected (visible on
both sides of an occluder) while FastSAM returns one contiguous blob. Measured:
only 6% of GT items break into more than one visible piece, and when they do the
largest piece holds 98% of the pixels. Not the explanation.

**Remaining explanation: granularity.** FastSAM locks onto a dominant sub-region
of a large item -- one face of a box, one panel of a bag -- rather than the whole
object, and never proposes the whole, so suppression has nothing better to keep.
This is the same failure as the polybag case (Stage 7, Mode A) running in the
opposite direction: there, many objects should have been one; here, one object is
returned as one of its parts.

**Consequence for what to do next.** Both directions are a *definition* problem --
the segmenter's notion of a coherent region is not the task's notion of a
pickable unit -- and neither is reachable by tuning a threshold or sharpening a
boundary. The fix is to teach the model the right notion: fine-tune on ARMBench's
own instance labels. That is the single highest-value remaining piece of work and
it is not a four-hour job.

---

## Stage 10 -- Is the pipeline actually colour-agnostic?

The design claims colour-independence in two places: the crop is keyed on
brightness OR saturation rather than hue (D1), and the reference tote colour is
learned per image from the rim (D3). Both claims were **untested** -- the dev
split contains only yellow, blue and beige bins, so "it would work on a green
tote" was an assertion about code, not a measurement.

### E10.1 -- Method

Using the GT masks (validation only, never at inference), the tote's own pixels
are recoloured in HSV and every item pixel is left untouched. Ground-truth boxes
and masks are unchanged, so `grade.py` applies as-is. Four variants, none of
which occur anywhere in the dev split:

| variant | transform |
|---|---|
| green | hue +60 deg |
| magenta | hue +150 deg |
| grey | saturation x 0.10 |
| white | saturation x 0.05, value x 1.35 |

Reproduce with `python experiments/recolour.py`.

### E10.2 -- Result

| tote colour | detection F1 | pick score | vs original |
|---|---|---|---|
| original (yellow / blue / beige) | 0.589 | 0.958 | -- |
| **green** | 0.590 | 0.950 | +0.001 / -0.008 |
| **magenta** | 0.584 | 0.958 | -0.005 / 0.000 |
| **grey** | 0.474 | 0.904 | **-0.115 / -0.054** |
| **white** | 0.480 | 0.867 | **-0.109 / -0.091** |

*(Re-run against the final pipeline, after the Stage 12 selection change, so these
numbers and the headline scores come from one build.)*

**Hue is irrelevant, as designed.** Green and magenta land within noise of the
original (the bootstrap standard error is 0.028 F1, so both differences are
inside one standard error). A bin colour appearing nowhere in the data costs
nothing, and this is now measured rather than asserted.

**Saturation is the real dependency.** An achromatic bin costs ~0.11 F1 and
0.05-0.09 pick. This is the limitation predicted before the test was run: the
strongest of the three tote/item cues is chromatic distance from the tote colour,
and a grey bin holding grey polybags offers no chroma to measure.

**The vote is what turns a collapse into a slope.** Losing one of three cues
outright costs about 0.11 F1, not the pipeline. A single-cue design keyed on
chroma -- which is what the first working version was -- would have failed
outright on these images. This is the clearest evidence for D7 in the whole log,
and it comes from a condition the dev split never contains.

Even the worst variant (pick 0.867) remains nearly 3x the flag-everything
baseline of 0.31.

### E10.3 -- What this does not cover

The items are untouched, so this measures robustness to bin appearance only. It
does not model a different camera, different lighting, a different item mix, or a
bin at a different distance or angle. A pipeline that survives a hue rotation is
not thereby proven to survive a new fulfilment centre.

---

## Stage 11 -- FAILED: multi-scale proposals, and what it revealed

### E11.1 -- The test

Case 2 of the taxonomy is FastSAM returning one *part* of a large item -- the
printed face of a box rather than the box. A coarser input scale sees less
texture detail and might group the whole object instead. `retina_masks` returns
masks at crop resolution whatever the input size, so proposals from several
scales share one candidate pool and the existing suppression resolves duplicates.

| config | F1 | item recall | boxes/img | latency |
|---|---|---|---|---|
| **1024 (current)** | **0.614** | 0.550 | 5.7 | 600 ms |
| 1280 | 0.554 | 0.509 | 5.1 | 742 ms |
| 1024 + 1280 | 0.568 | 0.518 | 5.1 | 1232 ms |
| 768 + 1024 + 1280 | 0.566 | 0.532 | 5.2 | 1541 ms |

**Rejected.** Every multi-scale configuration is *worse* than the single scale
and every one breaks the 1 s budget. Multi-scale support is left in
`ItemDetector` (`imgsz` accepts a tuple) because it costs ten lines and makes the
experiment reproducible, but it defaults to a single scale.

### E11.2 -- The finding underneath: selection is the bottleneck, not proposals

    proposal ceiling (best any filter could reach)   0.841
    recall actually achieved                         0.550
    -------------------------------------------------------
    lost in selection                                0.291

**The right mask is already in the pool 84% of the time and gets chosen 55% of
the time.** Nearly thirty points of recall are lost after segmentation, in the
filter-and-suppress stage.

That reframes every result in Stages 9-11. Three separate attempts to buy F1
returned nothing --

| attempt | aimed at | result |
|---|---|---|
| E9.2 relaxing the area cap | the filter | no change |
| E9.3 SAM boundary refinement | mask quality | -0.002, +196 ms |
| E11.1 multi-scale proposals | the proposal pool | -0.046, +632 ms |

-- and adding proposals made things actively worse, because a greedy
largest-first selection over a larger, more redundant pool has *more* ways to
choose the wrong region, not fewer.

**What would actually pay, in order:**

1. **A better selection rule.** The current one is greedy area-descending with
   containment suppression: a single ordering heuristic, no notion of which
   *combination* of masks best explains the tote. Choosing a coherent set --
   maximise coverage of the tote interior, penalise overlap, prefer proposals the
   model is confident in -- is a set-selection problem, and it is where the 0.29
   sits. This is the highest-value remaining work and it needs no new model.
2. **Fine-tuning on ARMBench labels**, for the definition mismatch (Cases 1 and
   3) that no selection rule can reach.
3. Nothing else measured here moves the number.

---

## Stage 12 -- Set-cover selection

E11.2 located 0.29 of recall in the selection stage. This attacks it directly.

**Pairwise suppression has a geometric blind spot.** Sorting by area and dropping
any proposal mostly inside a *single* kept mask cannot see cumulative redundancy:
a proposal 45% inside one kept mask and 45% inside another passes both tests
individually while being 90% explained overall. Cluttered totes are full of
exactly that geometry.

**Greedy set cover** measures each candidate against the union of everything
already chosen: take the proposal contributing the most pixels not yet explained,
and reject it once most of it is already covered.

All configs run on identical cached candidates -- the segmenter is not re-run --
so every difference is selection alone.

| selection rule | F1 | item recall | boxes/img |
|---|---|---|---|
| pairwise, area order (previous) | 0.614 | 0.550 | 5.7 |
| set cover, area order, new > 0.30 | 0.618 | 0.550 | 5.6 |
| **set cover, area order, new > 0.50** | **0.621** | 0.546 | 5.6 |
| set cover, area order, new > 0.70 | 0.621 | 0.546 | 5.6 |
| set cover, confidence order, new > 0.50 | 0.560 | 0.564 | 7.1 |
| set cover, area x confidence order | 0.617 | 0.555 | 5.8 |

Full dev split: **F1 0.585 -> 0.589**, pick unchanged at 0.958.

### The gain is not real, and it ships anyway

+0.004 sits well inside one bootstrap standard error (0.028). It is not an
improvement and is not claimed as one.

It ships because of the **threshold sensitivity**, which is a robustness property
rather than a score:

    set cover     new > 0.30 / 0.50 / 0.70   ->   0.618 / 0.621 / 0.621
    pairwise      containment 0.80 / 0.90    ->   0.439 / 0.458

Set cover barely responds to where its knob is set; the rule it replaces swings
0.019 between two reasonable settings. Given a held-out split and E4.2's
demonstration that tuned optima here are partly fitting noise, the rule that does
not depend on its threshold is the safer one to ship. That is the whole argument.

### And the gap is still not closed

Recall did not move (0.550 -> 0.546). Set cover fixed cumulative redundancy and
recall was unaffected, so **redundancy was not what the 0.29 was made of**.

What remains is granularity: among several overlapping proposals for the same
region -- the whole box, its printed face, the box plus its neighbour -- neither
area order nor confidence order knows which one is a *pickable unit*. No ordering
heuristic over these candidates can know, because the information is not in area
or in FastSAM's objectness.

This is the same conclusion reached from Case 1 (polybags), Case 2 (part of an
item), Case 3 (merged lookalikes) and now from selection: **the segmenter's
notion of a coherent region is not the task's notion of a pickable unit.** Four
independent lines of evidence, one cause. Fine-tuning on ARMBench's own instance
labels is the fix, and it is not an afternoon's work.

### Cumulative record of attempts to raise F1 beyond 0.585

| attempt | aimed at | result |
|---|---|---|
| E9.2 max-area cap | filter bounds | no change (real bug, not binding) |
| E9.3 SAM refinement | mask boundaries | -0.002, +196 ms -- rejected |
| E11.1 multi-scale | proposal pool | -0.046, +632 ms -- rejected |
| E12 set cover | selection rule | +0.004 -- shipped for robustness, not score |

Four targeted attempts, each aimed at a different stage, none moving the number.
That is the strongest available evidence that the remaining error is not in any
stage of this pipeline but in the model's definition of an object.
