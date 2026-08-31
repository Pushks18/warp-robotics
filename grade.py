#!/usr/bin/env python3
"""Score a manifest against the dev split ground truth.

Usage:
    python grade.py path/to/manifest.json [--data data/dev]

Needs numpy and pillow (pip install -r requirements.txt).
Grading is deterministic. The held-out split is scored with this exact
script, so if it runs clean here it will run clean there.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image

IOU_MATCH = 0.5
FLAG_CREDIT = 0.25
ERODE_PX = 4


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_boxes(preds, gts):
    """Greedy one-to-one matching by descending IoU. Returns tp count."""
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            v = iou(p, g)
            if v >= IOU_MATCH:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp


def erode(binary, iterations):
    """8-neighbour binary erosion without scipy."""
    a = binary
    for _ in range(iterations):
        shrunk = a.copy()
        shrunk[1:, :] &= a[:-1, :]
        shrunk[:-1, :] &= a[1:, :]
        shrunk[:, 1:] &= a[:, :-1]
        shrunk[:, :-1] &= a[:, 1:]
        shrunk[1:, 1:] &= a[:-1, :-1]
        shrunk[:-1, :-1] &= a[1:, 1:]
        shrunk[1:, :-1] &= a[:-1, 1:]
        shrunk[:-1, 1:] &= a[1:, :-1]
        a = shrunk
    return a


def fail(msg):
    print(f"MANIFEST ERROR: {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--data", default="data/dev")
    args = ap.parse_args()

    gt_path = os.path.join(args.data, "ground_truth.json")
    with open(gt_path) as f:
        gt = {r["file"]: r for r in json.load(f)["images"]}
    with open(args.manifest) as f:
        manifest = json.load(f)

    if "results" not in manifest or not isinstance(manifest["results"], list):
        fail('top level must be {"results": [...]}')
    results = {}
    for r in manifest["results"]:
        if "file" not in r:
            fail("every result needs a \"file\" field")
        results[r["file"]] = r

    missing = sorted(set(gt) - set(results))
    if missing:
        fail(f"{len(missing)} images missing from manifest, first: {missing[0]}")

    det_f1s, pick_scores = [], []
    n_pick_ok = n_pick_bad = n_flag = 0
    rows = []
    for fname, g in sorted(gt.items()):
        r = results[fname]
        gt_boxes = [it["bbox"] for it in g["items"]]
        pred_boxes = []
        for it in r.get("items", []):
            b = it.get("bbox")
            if (not isinstance(b, list) or len(b) != 4
                    or not all(isinstance(v, (int, float)) for v in b)
                    or b[2] <= 0 or b[3] <= 0):
                fail(f"{fname}: bad bbox {b}, expected [x, y, w, h]")
            pred_boxes.append(b)

        # detection F1
        if not gt_boxes and not pred_boxes:
            f1 = 1.0
        elif not gt_boxes or not pred_boxes:
            f1 = 0.0
        else:
            tp = match_boxes(pred_boxes, gt_boxes)
            prec = tp / len(pred_boxes)
            rec = tp / len(gt_boxes)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        det_f1s.append(f1)

        # pick scoring
        action = r.get("action")
        if action not in ("pick", "flag"):
            fail(f'{fname}: "action" must be "pick" or "flag"')
        if action == "flag":
            n_flag += 1
            score = 1.0 if g["empty"] else FLAG_CREDIT
            note = "flag (empty tote, correct)" if g["empty"] else "flag"
        else:
            p = r.get("pick")
            if (not isinstance(p, dict) or "x" not in p or "y" not in p):
                fail(f'{fname}: action "pick" needs "pick": {{"x": .., "y": ..}}')
            if g["empty"]:
                score, note = 0.0, "picked in an empty tote"
                n_pick_bad += 1
            else:
                mask = np.array(Image.open(
                    os.path.join(args.data, "masks",
                                 fname.replace(".jpg", ".png"))))
                item_ids = [it["id"] for it in g["items"]]
                inside = np.isin(mask, item_ids)
                inside = erode(inside, ERODE_PX)
                x, y = int(round(p["x"])), int(round(p["y"]))
                if 0 <= y < inside.shape[0] and 0 <= x < inside.shape[1] and inside[y, x]:
                    score, note = 1.0, "pick ok"
                    n_pick_ok += 1
                else:
                    score, note = 0.0, "pick point not on an item"
                    n_pick_bad += 1
        pick_scores.append(score)
        rows.append((fname, len(gt_boxes), len(pred_boxes), f1, score, note))

    print(f"{'image':<16}{'gt':>4}{'pred':>6}{'detF1':>8}{'pick':>6}  note")
    for fname, ng, np_, f1, s, note in rows:
        print(f"{fname:<16}{ng:>4}{np_:>6}{f1:>8.2f}{s:>6.2f}  {note}")

    n = len(rows)
    print()
    print(f"images scored     {n}")
    print(f"detection mean F1 {sum(det_f1s) / n:.3f}")
    print(f"pick score        {sum(pick_scores) / n:.3f}"
          f"  (ok {n_pick_ok}, wrong {n_pick_bad}, flagged {n_flag})")
    print()
    print("Score meaning: a correct pick earns 1.0, flagging a tote that has")
    print(f"items earns {FLAG_CREDIT}, a wrong pick earns 0. Flagging an empty tote")
    print("earns 1.0, picking in one earns 0.")


if __name__ == "__main__":
    main()
