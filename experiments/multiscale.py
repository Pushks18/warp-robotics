"""Does proposing at more than one input scale recover the part-of-an-item misses?

Case 2 in the failure taxonomy: FastSAM locks onto a dominant sub-region of a
large item -- one printed face of a box -- and never proposes the whole object,
so the box comes out at ~56% of the true area and misses the IoU 0.5 bar. A
coarser input scale sees less texture detail and might group the whole object;
a finer one might split small items apart better. Merging both pools costs one
extra forward pass.

Reports F1 and the proposal ceiling (what a perfect filter could reach) so a gain
in one can be told apart from a gain in the other.
"""
import json, sys, time
import cv2, numpy as np
sys.path.insert(0, ".")
from grade import match_boxes, iou
from warp_vision.tote import find_tote
from warp_vision.detect import ItemDetector

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]

CONFIGS = [("1024 (current)", 1024), ("1280", 1280),
           ("1024+1280", (1024, 1280)), ("768+1024+1280", (768, 1024, 1280))]

def bb(m):
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()-xs.min())+1, int(ys.max()-ys.min())+1]

det = ItemDetector()
det._load()
for name, sz in CONFIGS:
    det.imgsz = sz
    f1s, lat, tp_tot, gt_tot, nbox = [], [], 0, 0, []
    for r in GT:
        img = cv2.imread(f"data/dev/images/{r['file']}")
        t0 = time.perf_counter()
        t = find_tote(img); masks, _ = det.detect(img, t)
        lat.append((time.perf_counter()-t0)*1000)
        gtb = [i["bbox"] for i in r["items"]]; pb = [bb(m) for m in masks]
        nbox.append(len(pb))
        if not gtb and not pb: f1s.append(1.0)
        elif not gtb or not pb: f1s.append(0.0)
        else:
            tp = match_boxes(pb, gtb); p, rc = tp/len(pb), tp/len(gtb)
            f1s.append(2*p*rc/(p+rc) if p+rc else 0.0)
        gt_tot += len(gtb)
        tp_tot += sum(1 for g in gtb if any(iou(b, g) >= 0.5 for b in pb))
    print(f"{name:<16} F1 {np.mean(f1s):.3f}   recall {tp_tot/gt_tot:.3f}   "
          f"boxes/img {np.mean(nbox):.1f}   {np.median(lat):.0f} ms median", flush=True)
