# How to submit

This is a take-home. Budget about **half a day (~4 hours)** of work, and submit within
**2 days** of receiving it.

## 1. Read the brief

Read [`README.md`](README.md) fully before writing code. Meet the three must-haves
first (detect items, choose a pick point, flag totes you should not pick from). Reach
for stretch goals only after those work end to end. A small pipeline that handles the
ugly images honestly beats a big one that only works on the easy ones.

## 2. Set up

- Python users: `pip install -r requirements.txt` gets you the grader's dependencies
  (numpy, pillow). Your own pipeline can use any language and any libraries.
- The data is already in the repo under `data/dev/`. There is nothing to download and
  no API key to obtain.
- If you use a hosted model (a vision API, a VLM), keep the key in an environment
  variable, out of the repo, and note the approximate cost of one full run in your
  README.

## 3. Build and check your score

Your pipeline reads `data/dev/images/` and writes one manifest JSON matching
[`manifest.schema.json`](manifest.schema.json). Score it whenever you like:

```bash
python grade.py out/manifest.json
```

The grader is the same one we run on the held-out split after you submit, so make sure
it runs clean on your manifest before you send anything.

## 4. Submit

1. Click **"Use this template" -> Create a new repository** (top of this repo's GitHub
   page). Set your new repo to **Private**. Please use the template rather than
   forking, so your work stays your own.
2. Build your project inside your new repo. Commit as you go; we like seeing how the
   work developed.
3. When you are done, invite **`rahulharikumarr`** as a collaborator so we can see it.
4. Send us:
   - the **repo link**
   - your **README**, which must include:
     - exact steps to reproduce your manifest from a clean checkout
     - your final dev-split scores from `grade.py`, reported honestly
     - a short **"Decisions and tradeoffs"** section: what you chose, what you cut,
       where your pipeline fails, and what you would do next with more time
   - a **2 to 3 minute screen recording** (Loom or similar) of a full run: pipeline in,
     manifest out, grader output. Include at least one hard tote (empty, reflective, or
     cluttered) and show how your pipeline handled it.

## 5. What happens after

We run your pipeline on a held-out split of images you have not seen, using the same
grader. Then we get on a call, walk through your code together, and ask you to extend
it live. Build something that is genuinely yours: you understand every line and can
defend every threshold.

## Ground rules

- Every value in your manifest must be produced by your pipeline. Hand-editing outputs
  is disqualifying and shows up immediately on the held-out run.
- Using AI coding tools is encouraged. Claude Code, Cursor, Copilot, whatever you are
  fastest in. Understanding what they wrote for you is mandatory.
- Do not commit the data to any other public place, and keep the attribution notice in
  your README if you publish anything derived from it. The images are CC BY 4.0 from
  Amazon's ARMBench dataset (details in the README).
