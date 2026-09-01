# Warp Robotics - Warehouse Vision Challenge

Warp runs freight and warehouses. We are automating the pack station: a tote full of
mixed items arrives in front of a robot arm, and the arm has to decide what to pick up
first. Before any motor moves, a camera has to answer three questions. What is in the
tote? Where should the suction cup land? And is this tote one the robot should even
attempt, or should a human take it?

That perception step is this challenge. You build the vision pipeline. Real robot
optional (and not expected).

Budget about **half a day (~4 hours)** and submit within **2 days**. We care far more
about the decisions you make than about how many features you finish. A smaller thing
done well beats a broad thing half-working.

## The data

`data/dev/` contains **120 real photos of totes** from a working Amazon fulfillment
center, taken from above by the camera on a robotic picking cell. They come from
[ARMBench](https://www.amazon.science/code-and-datasets/amazon-robotic-manipulation-benchmark-armbench),
a public dataset Amazon released for exactly this kind of research (license: CC BY 4.0,
full attribution at the bottom). We curated the subset, cleaned the labels, and resized
everything to 1224x1024.

For every image you get ground truth:

- `data/dev/images/tote_XXXX.jpg` - the photo
- `data/dev/masks/tote_XXXX.png` - a label image the same size as the photo. Each pixel
  holds an integer id: 0 is background, one id is the tote itself, and every other id is
  one item. The `tote_id` for each image is in the ground truth file.
- `data/dev/ground_truth.json` - per image: the item ids, their bounding boxes
  `[x, y, w, h]`, and whether the tote is empty.

A few totes are empty on purpose. Some items are in clear polybags, some are reflective,
some overlap each other. That mess is real warehouse life and it is the point of the
exercise.

## What you build

A pipeline (script, notebook, small service, your call) that takes the images in
`data/dev/images/` and writes **one manifest JSON** describing, for every image:

1. **What is in the tote.** A bounding box per detected item.
2. **Where to pick.** One suction target `(x, y)` in pixel coordinates, on the item you
   would pick first, and one sentence on why that item.
3. **Or when not to pick.** If the tote is empty, or your pipeline is not confident
   enough for a safe pick, output `"action": "flag"` instead and say why. A robot that
   knows when to call a human is worth more than one that guesses.

The exact output format is in [`manifest.schema.json`](manifest.schema.json), with a
filled-in example in [`example/manifest.example.json`](example/manifest.example.json).

Any approach is fair game: classical OpenCV, a pretrained detector or segmenter (YOLO,
SAM, Detectron2), a vision language model, or a mix. You do not need a GPU and you do
not need to train anything, though you may. Using pretrained models is not cheating
here, it is how this work is actually done. What we grade is whether your pipeline's
judgment holds up on messy input.

## How grading works

Score yourself locally, as often as you like:

```bash
pip install -r requirements.txt
python grade.py out/manifest.json
```

The grader prints a per-image table and two numbers:

- **Detection mean F1.** Your boxes are matched one-to-one against ground truth boxes at
  IoU >= 0.5. F1 is computed per image and averaged. On an empty tote, predicting no
  items scores 1.0.
- **Pick score.** Per image: a pick point that lands on an item earns 1.0, flagging a
  tote that has pickable items earns 0.25, a pick that lands on the tote floor, a wall,
  or in an empty tote earns 0. Flagging an empty tote earns 1.0. Scores are averaged
  over all images.

The 0.25 is deliberate. Flagging everything is safe but useless, picking blindly is
worse than useless, because in production a bad suction attempt jams the cell. Your job
is to find the trade-off and defend it in your writeup.

**The held-out set.** We keep a second split of images, curated the same way from the
same dataset, that you never see. After you submit, we run your pipeline on it with this
same grader. That score is the one that counts, so tuning thresholds to squeeze the dev
split is wasted effort. Build something that generalizes.

## Rules

- **Every number in your manifest must come from your pipeline.** Hand-tuning a
  threshold is engineering. Hand-editing boxes or pick points in the JSON is
  disqualifying, and it will show immediately on the held-out run.
- Your pipeline must run from a clean checkout with the steps in your README. If we
  cannot reproduce your manifest, we cannot grade you.
- Keep any API keys server-side and out of the repo. (None are required. If you use a
  hosted vision model, note roughly what the run cost.)
- Do not retrain on the dev split and call it generalization. If you fine-tune anything,
  say so and say on what.

## Stretch goals (only after the must-haves work)

- **Pick order.** Output the full sequence to empty the tote, top-most and least
  occluded first. Explain how you decided what is on top.
- **Speed.** Measure your per-image latency and get it under 1 second on your laptop.
  Say what you traded to get there.
- **Failure taxonomy.** Cluster your worst dev images and name the failure modes. Which
  would you fix with better code, and which need a better camera?

## Using AI - encouraged

We build with AI here and you should too. Use Claude Code, Cursor, Copilot, whatever you
are fastest in. We expect fluency with these tools and want to see it. The one thing
that matters: the result has to be yours. You understand every line, the design calls
were deliberate, and you can explain and extend it live on a follow-up call.
AI-generated code you cannot defend is what sinks a submission, not the AI.

## How to submit

See [`SUBMISSION.md`](SUBMISSION.md). In short: click **"Use this template"** to make
your own private repo, build, invite `rahulharikumarr`, and send us the link, your
README, and a short screen recording.

## What we value

Judgment over coverage. How you handle the ugly images, how honestly you report what
does not work, and how clearly you explain the calls you made. A submission with a
mediocre F1 and an excellent failure analysis beats a submission with a good F1 and no
idea why.

---

## Dataset attribution

Images and instance masks are a curated subset of **ARMBench**, released by Amazon
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> Mitash, C., Wang, F., Lu, S., Terhuja, V., Garaas, T., Polido, F., & Nambi, M. (2023).
> ARMBench: An Object-centric Benchmark Dataset for Robotic Manipulation. ICRA 2023.
> https://arxiv.org/abs/2303.16382

We accessed the mix-object-tote subset via the
[community mirror on Hugging Face](https://huggingface.co/datasets/correll/armbench-segmentation-mix-object-tote)
(also CC BY 4.0). Changes made: selected 120 images, denoised the compressed label
masks, derived bounding boxes, resized to 1224x1024. No image content was altered.
