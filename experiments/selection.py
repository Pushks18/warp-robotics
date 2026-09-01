"""Greedy set cover versus pairwise suppression, and how strict to be.

Measured gap: the proposal pool contains a matching mask for 84.1% of items but
only 55.0% are selected. Everything below operates on identical candidates -- the
segmenter is not re-run between configs -- so any difference is selection alone.
"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, ".")
from grade import match_boxes, iou
from warp_vision.tote import find_tote
from warp_vision.detect import ItemDetector, CONTAINMENT_NMS, NMS_IOU

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
det = ItemDetector()

def pairwise(cands, cont=CONTAINMENT_NMS, nms=NMS_IOU):
    cands = sorted(cands, key=lambda t: -t[0]); kept = []
    for a, c, m in cands:
        drop = False
        for ka, kc, km in kept:
            inter = float(np.logical_and(m, km).sum())
            if inter and (inter/a >= cont or inter/(a+ka-inter) >= nms):
                drop = True; break
        if not drop: kept.append((a, c, m))
    return kept

def setcover(cands, thr, key):
    cands = sorted(cands, key=key); kept, cov = [], None
    for a, c, m in cands:
        if cov is None: cov = np.zeros(m.shape, bool)
        if float(np.logical_and(m, ~cov).sum())/a < thr: continue
        kept.append((a, c, m)); cov |= m
    return kept

CONFIGS = [("pairwise (previous)", lambda cs: pairwise(cs))]
for thr in (0.30, 0.50, 0.70):
    CONFIGS.append((f"set cover, area order, new>{thr:.2f}",
                    lambda cs, t=thr: setcover(cs, t, lambda x: -x[0])))
CONFIGS.append(("set cover, conf order, new>0.50",
                lambda cs: setcover(cs, 0.50, lambda x: -x[1])))
CONFIGS.append(("set cover, area*conf order, new>0.50",
                lambda cs: setcover(cs, 0.50, lambda x: -x[0]*x[1])))

def bb(m, ox, oy):
    ys, xs = np.where(m)
    return [int(xs.min())+ox, int(ys.min())+oy, int(xs.max()-xs.min())+1, int(ys.max()-ys.min())+1]

cache = []
for r in GT:
    img = cv2.imread(f"data/dev/images/{r['file']}")
    t = find_tote(img)
    cache.append((r, t.roi, det._candidates(img, t)))
print(f"cached candidates for {len(cache)} images\n", flush=True)

print(f"{'config':<38}{'F1':>8}{'recall':>9}{'boxes':>8}")
for name, fn in CONFIGS:
    f1s, tp_tot, gt_tot, nb = [], 0, 0, []
    for r, roi, cands in cache:
        x0, y0, _, _ = roi
        kept = fn(cands)
        pb = [bb(m, x0, y0) for _, _, m in kept]
        gtb = [i["bbox"] for i in r["items"]]; nb.append(len(pb))
        if not gtb and not pb: f1s.append(1.0)
        elif not gtb or not pb: f1s.append(0.0)
        else:
            tp = match_boxes(pb, gtb); p, rc = tp/len(pb), tp/len(gtb)
            f1s.append(2*p*rc/(p+rc) if p+rc else 0.0)
        gt_tot += len(gtb); tp_tot += sum(1 for g in gtb if any(iou(b, g) >= 0.5 for b in pb))
    print(f"{name:<38}{np.mean(f1s):>8.3f}{tp_tot/gt_tot:>9.3f}{np.mean(nb):>8.1f}", flush=True)
