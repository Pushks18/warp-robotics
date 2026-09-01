"""Are missed items near-misses or genuine misses -- and does the parent mask exist?

Decides where effort should go:
  * unmatched GT items covered at IoU 0.35-0.49 -> boundary refinement pays
  * unmatched GT items near IoU 0               -> the proposal set is the problem
Also checks, for every GT item, whether ANY raw FastSAM proposal matched it, and
if so which filter stage removed it.
"""
import json, sys
import cv2, numpy as np
from ultralytics import FastSAM
sys.path.insert(0, ".")
from grade import iou
from warp_vision.tote import find_tote
from warp_vision.detect import (ItemDetector, MIN_AREA_FRAC, MAX_AREA_FRAC,
                                MIN_CONTAINMENT, MIN_CONF, TOTE_CHROMA_DIST,
                                TOTE_AB_STD, TOTE_EDGE_DENSITY, MIN_ITEM_VOTES)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
det = ItemDetector()
model = det._load()

def bb(m, ox=0, oy=0):
    ys, xs = np.where(m)
    return [int(xs.min())+ox, int(ys.min())+oy, int(xs.max()-xs.min())+1, int(ys.max()-ys.min())+1]

near, gone, matched = [], [], 0
killed = {"area_min": 0, "area_max": 0, "conf": 0, "containment": 0, "vote": 0,
          "survived_filter_lost_to_nms": 0}
total_gt = 0
for r in GT:
    f = r["file"]; img = cv2.imread(f"data/dev/images/{f}")
    t = find_tote(img); x0, y0, x1, y1 = t.roi; crop = img[y0:y1, x0:x1]
    out_masks, _ = det.detect(img, t)
    out_boxes = [bb(m) for m in out_masks]
    res = model.predict(crop, verbose=False, imgsz=1024, device="mps",
                        retina_masks=True, conf=0.2, iou=0.7)[0]
    md = res.masks.data.cpu().numpy().astype(bool) if res.masks is not None else []
    cf = res.boxes.conf.cpu().numpy() if res.masks is not None else []
    fp = t.footprint[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ed = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5,5),0),40,120) > 0
    fa = float(img.shape[0]*img.shape[1])

    for it in r["items"]:
        total_gt += 1
        g = it["bbox"]
        best_out = max((iou(b, g) for b in out_boxes), default=0.0)
        if best_out >= 0.5:
            matched += 1; continue
        # which raw proposal, if any, would have matched?
        best_raw, why = 0.0, None
        for m, c in zip(md, cf):
            b = bb(m, x0, y0); v = iou(b, g)
            if v <= best_raw: continue
            a = float(m.sum())
            if a/fa < MIN_AREA_FRAC: reason = "area_min"
            elif a/fa > MAX_AREA_FRAC: reason = "area_max"
            elif c < MIN_CONF: reason = "conf"
            elif np.logical_and(m, fp).sum()/a < MIN_CONTAINMENT: reason = "containment"
            else:
                ab = lab[..., 1:][m]
                votes = (int(np.linalg.norm(np.median(ab,axis=0)-t.colour[1:]) > TOTE_CHROMA_DIST)
                         + int(ab.std() < TOTE_AB_STD) + int(ed[m].mean() > TOTE_EDGE_DENSITY))
                reason = "vote" if votes < MIN_ITEM_VOTES else "survived_filter_lost_to_nms"
            best_raw, why = v, reason
        (near if best_out >= 0.35 else gone).append(best_out)
        if best_raw >= 0.5 and why: killed[why] += 1

print(f"\n{N} images, {total_gt} GT items: {matched} matched ({matched/total_gt:.3f})")
miss = total_gt - matched
print(f"missed {miss}:  near-miss (IoU 0.35-0.49) {len(near)}   genuine miss (<0.35) {len(gone)}")
if near: print(f"  near-miss IoU distribution: {np.round(np.percentile(near,[25,50,75]),3).tolist()}")
print(f"\nOf the missed items, a RAW proposal would have matched {sum(killed.values())} of them.")
print("Which stage removed it:")
for k, v in sorted(killed.items(), key=lambda kv: -kv[1]):
    if v: print(f"   {k:<32}{v:>4}   ({v/max(miss,1):.1%} of all misses)")
