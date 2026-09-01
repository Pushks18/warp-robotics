"""Does box-prompted SAM refinement convert the near-misses?

42 of 218 ground-truth items are covered at IoU 0.35-0.49, just under the 0.50
bar. Those are boundary errors, not missed objects. MobileSAM's image encoder
runs once per image and each box prompt decodes cheaply, so FastSAM's proposals
can be re-segmented with SAM's sharper boundaries for a fixed cost.

Guarded: a refined mask replaces the original only if it still looks like the
same object (IoU with the original above a floor) and stays inside the area
bounds. Otherwise SAM is free to latch onto a neighbour.
"""
import json, sys, time
import cv2, numpy as np
from ultralytics import SAM
sys.path.insert(0, ".")
from grade import match_boxes, iou
from warp_vision.tote import find_tote
from warp_vision.detect import ItemDetector, MIN_AREA_FRAC, MAX_AREA_FRAC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
det = ItemDetector()
sam = SAM("mobile_sam.pt")
KEEP_IOU = 0.30      # refined mask must still be the same object

def bb(m):
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()-xs.min())+1, int(ys.max()-ys.min())+1]

def f1_of(pb, gtb):
    if not gtb and not pb: return 1.0
    if not gtb or not pb: return 0.0
    tp = match_boxes(pb, gtb); p, r = tp/len(pb), tp/len(gtb)
    return 2*p*r/(p+r) if p+r else 0.0

base_f1, ref_f1, t_base, t_ref, swapped, total = [], [], [], [], 0, 0
for r in GT:
    f = r["file"]; img = cv2.imread(f"data/dev/images/{f}")
    gtb = [i["bbox"] for i in r["items"]]
    t0 = time.perf_counter()
    t = find_tote(img); masks, sc = det.detect(img, t)
    t_base.append(time.perf_counter()-t0)
    base_f1.append(f1_of([bb(m) for m in masks], gtb))

    x0, y0, x1, y1 = t.roi; crop = img[y0:y1, x0:x1]
    fa = float(img.shape[0]*img.shape[1])
    t1 = time.perf_counter()
    out = []
    if masks:
        boxes = []
        for m in masks:
            sub = m[y0:y1, x0:x1]
            ys, xs = np.where(sub)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        res = sam.predict(crop, bboxes=boxes, verbose=False, device="mps")[0]
        rm = res.masks.data.cpu().numpy().astype(bool) if res.masks is not None else []
        for i, m in enumerate(masks):
            orig = m[y0:y1, x0:x1]
            new = rm[i] if i < len(rm) else None
            use = orig
            if new is not None and new.sum() > 0:
                inter = np.logical_and(orig, new).sum()
                ov = inter/(orig.sum()+new.sum()-inter)
                if ov >= KEEP_IOU and MIN_AREA_FRAC <= new.sum()/fa <= MAX_AREA_FRAC:
                    use = new; swapped += 1
            total += 1
            full = np.zeros(img.shape[:2], bool); full[y0:y1, x0:x1] = use
            out.append(full)
    t_ref.append(time.perf_counter()-t1)
    ref_f1.append(f1_of([bb(m) for m in out], gtb))

print(f"\n{N} images, {total} masks, {swapped} replaced by SAM ({swapped/max(total,1):.0%})")
print(f"baseline  F1 {np.mean(base_f1):.3f}   {np.median(t_base)*1000:.0f} ms/img")
print(f"refined   F1 {np.mean(ref_f1):.3f}   +{np.median(t_ref)*1000:.0f} ms/img "
      f"-> {np.median(np.array(t_base)+np.array(t_ref))*1000:.0f} ms total")
