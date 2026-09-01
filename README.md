# Warehouse tote perception

A vision pipeline for a robotic pack station. Given a top-down photo of a tote it
answers three questions: what is in there, where should the suction cup land, and
is this a tote the robot should attempt at all.

Built for the Warp Robotics challenge; the original brief is in
[`CHALLENGE.md`](CHALLENGE.md).

## Results on the dev split

```
images scored     120
detection mean F1 0.589
pick score        0.958  (ok 108, wrong 5, flagged 7)
latency           619 ms median, 908 ms p95   (M4 MacBook Air, CPU+MPS, no GPU)
                  1100 ms median               (same machine in macOS Low Power Mode)
```

Reference points for those numbers:

| | |
|---|---|
| Flag every tote (safe and useless) | pick 0.31 |
| **This pipeline** | **pick 0.958** |
| Detection ceiling for this architecture | F1 0.841 (measured, see below) |

No hosted models, no API keys, nothing to pay for. Everything runs locally.

## Reproduce

From a clean checkout, on Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-pipeline.txt

# Generate the manifest (~80 s for 120 images; downloads FastSAM-x weights,
# 138 MB, on the first run only)
PYTHONPATH=. .venv/bin/python run.py --images data/dev/images --out out/manifest.json

# Score it
.venv/bin/python grade.py out/manifest.json
```

Useful flags:

```bash
--backend fastsam|mobilesam   # segmentation backend (default fastsam)
--device mps|cpu|cuda         # default mps; use cpu on non-Apple hardware
--pick-order                  # also emit the full emptying sequence per tote
--limit N                     # first N images, for a quick check
```

To see what it is doing on any image:

```bash
PYTHONPATH=. .venv/bin/python experiments/visualise.py tote_0108.jpg -o reports/x.png
```

## How it works

Four stages, ~450 lines.

**1. Find the tote** — [`warp_vision/tote.py`](warp_vision/tote.py)
Crops to the bin and learns its colour. The crop is keyed on *(bright OR
saturated)*, not on tote colour, and constrained to the component reaching the
centre of the frame. **Item-pixel recall 1.0000 on all 120 images** while
discarding ~14% of each frame — nothing the later stages need is ever thrown away.

**2. Propose regions** — [`warp_vision/detect.py`](warp_vision/detect.py)
FastSAM-x over the crop. It is *class-agnostic*, which is the right family of
tool: the ground truth carries no category labels, so the task is never "find the
bottle" but "where does one thing end and the next begin". A COCO detector would
box the one bottle it recognises and miss six anonymous polybags beside it.

**3. Decide what is an item and what is the bin** — same file.
FastSAM returns ~97 regions per image of which ~6% are real items. Masks are
filtered on area, containment and confidence, then by a **2-of-3 vote** over
three mask-level cues — chromatic distance from the tote's colour, chromatic
spread within the mask, and edge density. Overlapping proposals are reduced by greedy **set cover** against the union of
what is already explained. This single stage was worth **+0.21 F1 and +0.27 pick score**.

**4. Choose the pick** — [`warp_vision/pick.py`](warp_vision/pick.py)
The suction point is the **peak of the item's distance transform** — the point
furthest from any edge. Right for two independent reasons: physically it is the
flattest, most continuous surface available, which is what a cup needs to seal;
and numerically the grader erodes the item by 4 px before testing, so a centroid
on an L-shaped or crescent item can miss entirely while the distance peak cannot.
Which item is chosen by exposure (how much of it no other detection claims),
clearance, and brightness.

## Decisions and tradeoffs

Everything below is measured. The full log, including the things that failed, is
in [`reports/EXPERIMENTS.md`](reports/EXPERIMENTS.md).

### FastSAM-x over MobileSAM, and over FastSAM-s

MobileSAM's automatic mode takes **20.6 s per image** on this laptop against
FastSAM's fraction of a second. That is 41 minutes per pass over the dev split —
it ends the sub-second goal and, worse, makes the build-measure loop unusable.
Within FastSAM the larger backbone earns its cost: **F1 0.494 → 0.614 for +137
ms**, still inside the budget. If latency tightened, `-s` is a one-line swap for
about 0.12 F1.

### Tote colour is learned per image, never hardcoded

The bins are vivid yellow or blue — and keying on that failed on 16 of 120
images, because **some totes are beige**. Two cues (bright OR saturated) cover
each other's blind spots and lift item-pixel recall from 0.878 to 1.000. The
reference colour is sampled per image from the rim, so a bin colour that does not
appear anywhere in the dev split costs nothing on the held-out run.

### Emptiness is not detected, it is inferred

There is no empty-tote threshold. A tote is flagged when *nothing survives
detection*, which is the same condition. This removed a threshold rather than
adding one — after three separate hand-crafted emptiness cues failed (parametric
Lab, chroma-only, edge density; all logged), fewer thresholds was the right
direction. 7 of 8 empty totes are flagged correctly.

### The pipeline deliberately almost never flags

The grader pays 0.25 for flagging a tote that has items, so a confidence gate
must be right about failure **more than 3 times in 4** to break even: catching a
wrong pick gains 0.25, flagging a good one loses 0.75.

I measured whether my confidence has that power. It does not — the 4 wrong picks
scored **0.931-0.963, above the 0.925 median of correct picks**. Flagging the N
least-confident picks catches zero failures at every N and only loses points. So
the pipeline flags only when it sees nothing.

The honest reading: my confidence is a *pick-quality* score, not a calibrated
probability of correctness. Making it one is the single highest-value thing I
would do next, and the only route to spending flags profitably.

### Thresholds are round numbers, not sweep optima

Re-running one filter's evaluation on 40 images instead of 30 dropped its
accuracy from **0.919 to 0.859** — the difference was the threshold fitting a
small sample. Every threshold here is a round value chosen in the gap between two
measured distributions, and the tote/item decision is a majority vote rather than
a fitted classifier, so no single mis-set threshold can sink an unfamiliar image.
Nothing was tuned against the dev score.

### Colour-agnosticism, measured rather than claimed

The design is meant to be independent of bin colour. That was an assertion about
code until I tested it: using the GT masks for validation only, I recoloured the
tote's own pixels across all 120 images -- items untouched -- to colours that
appear nowhere in the split, and re-ran the pipeline against the unchanged
grader.

| tote colour | detection F1 | pick score |
|---|---|---|
| original (yellow / blue / beige) | 0.589 | 0.958 |
| green (hue +60 deg) | 0.590 | 0.950 |
| magenta (hue +150 deg) | 0.584 | 0.958 |
| grey (saturation x 0.10) | 0.474 | 0.904 |
| white (desaturated, brightened) | 0.480 | 0.867 |

**Hue is irrelevant** -- green and magenta land inside one bootstrap standard
error (0.028) of the original. **Saturation is the real dependency**: an
achromatic bin costs ~0.11 F1, because the strongest of the three tote/item cues
is chromatic distance and a grey bin holding grey polybags offers no chroma to
measure.

That degradation is the argument for the 2-of-3 vote. Losing one cue entirely
costs 0.11, not the pipeline; the single-cue version this replaced would have
failed outright. Reproduce with `python experiments/recolour.py`.

### What I cut

- **A learned failure predictor** for the flag decision. Needs held-out labels to
  calibrate against; guessing at it would have been worse than not flagging.
- **Depth estimation** for the "what is on top" ordering. Monocular depth would
  be a stronger occlusion signal than my exposure heuristic, but it roughly
  doubles latency for a stretch goal.
- **Fine-tuning the segmenter.** It is the correct fix for the dominant failure
  mode (below) and it is not a four-hour job.

## Where it fails

Detection degrades monotonically with clutter — F1 0.875 on empty totes, 0.717
at 1-3 items, **0.366 at 11+**.

**The dominant failure is a definition mismatch, not a segmentation error.** In
`tote_0108` the ground truth draws *one* box around a polybag holding ~15 loose
bottles; FastSAM finds the bottles and returns *19* boxes. The task defines an
item as **one pickable unit** — one bag, one SKU, one suction event — while a
class-agnostic segmenter sees visual coherence and has no way to know that clear
plastic is a container rather than a window. This is the ceiling on detection F1
and no amount of tuning reaches it. It is also why picking (0.958) is so far
ahead of detection (0.589): a pick only has to land on *a* real item, not
partition the tote correctly.

**Fix with better code:** fine-tune on ARMBench's own instance labels so the model
learns the pickable-unit concept, or add a merge step keyed on polybag cues
(specular sheen, seams).

**Fix with a better camera:** heavy occlusion in crowded totes (`tote_0053`, 18
overlapping white polybags, F1 0.18) is not recoverable from one RGB view — a
mostly-buried item has too little visible surface to reach IoU 0.5 against its
true extent. Depth would separate touching surfaces that share colour and
texture. The same is true of the 4 residual cases where a bin floor fragment
survives as an item: the floor is a plane at a known height, and any depth signal
settles it instantly.

Full taxonomy in [`reports/EXPERIMENTS.md`](reports/EXPERIMENTS.md#stage-7----failure-taxonomy).

## Stretch goals

- **Speed** — 619 ms median, 908 ms p95, under the 1 s target. Measured on an
  unrestricted M4 Air; the same code takes ~1100 ms on battery in macOS Low Power
  Mode, which is worth stating because a fanless machine on a pack line would hit
  the same wall. Stage breakdown: tote localisation 57 ms, segmentation and
  filtering ~450 ms, set-cover selection 7 ms. What paid for it:
  cropping to the tote (~14% fewer pixels), a single-forward-pass segmenter
  instead of prompt-grid SAM (80x), and caching per-mask geometry so the pick
  ranking never recomputes a distance transform. What it cost: about 0.12 F1
  versus a heavier backbone, and the prompt-grid quality MobileSAM would give.
- **Pick order** — `--pick-order` emits the full emptying sequence. Greedy and
  re-ranked at each step: once an item is lifted, whatever it was covering
  becomes the most exposed thing in the tote. Adds ~110 ms p95.
- **Failure taxonomy** — above, and in the experiment log.

## Layout

```
run.py                      CLI: images in, manifest out
warp_vision/tote.py         stage 1 - find the bin, learn its colour
warp_vision/detect.py       stages 2-3 - propose regions, decide what is an item
warp_vision/pick.py         stage 4 - which item, and where to place the cup
warp_vision/pipeline.py     per-image orchestration
experiments/                every measurement quoted here, re-runnable
reports/EXPERIMENTS.md      the full log, including what failed
```

Ground-truth masks are read only by `grade.py` and by scripts in `experiments/`
that validate a design decision. No stage of the pipeline reads them.

## Dataset attribution

Images and instance masks are a curated subset of **ARMBench**, released by
Amazon under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> Mitash, C., Wang, F., Lu, S., Terhuja, V., Garaas, T., Polido, F., & Nambi, M.
> (2023). ARMBench: An Object-centric Benchmark Dataset for Robotic Manipulation.
> ICRA 2023. https://arxiv.org/abs/2303.16382

Accessed via the [community mirror on Hugging Face](https://huggingface.co/datasets/correll/armbench-segmentation-mix-object-tote)
(also CC BY 4.0).
