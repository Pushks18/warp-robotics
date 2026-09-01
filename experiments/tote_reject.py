"""Can a MASK be classified as tote-plastic, where a PIXEL could not?

Three per-pixel appearance models failed (see reports/EXPERIMENTS.md E2.1-E2.3),
all beaten by shadow inside the bin. A mask gives thousands of pixels to take a
robust statistic over, which is a much easier question. This measures whether
that holds.
"""
import json, sys
import cv2, numpy as np
from ultralytics import FastSAM
from warp_vision.tote import find_tote

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
model = FastSAM("FastSAM-s.pt")

item_rows, tote_rows = [], []
for r in GT:
    f = r["file"]
    img = cv2.imread(f"data/dev/images/{f}")
    gtm = cv2.imread("data/dev/masks/" + f.replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
    t = find_tote(img); x0, y0, x1, y1 = t.roi
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[y0:y1, x0:x1]
    res = model.predict(img[y0:y1, x0:x1], verbose=False, imgsz=1024,
                        retina_masks=True, conf=0.2, iou=0.7)[0]
    if res.masks is None: continue
    md = res.masks.data.cpu().numpy().astype(bool)
    gt_crop = gtm[y0:y1, x0:x1]
    item_ids = [i["id"] for i in r["items"]]

    for m in md:
        a = m.sum()
        if a < 3000: continue
        # what is this mask, according to GT? majority label
        vals, cnt = np.unique(gt_crop[m], return_counts=True)
        label = vals[np.argmax(cnt)]
        purity = cnt.max() / a
        if purity < 0.7: continue                     # ambiguous, skip
        med = np.median(lab[m], axis=0)
        d_full = float(np.linalg.norm(med - t.colour))
        d_ab   = float(np.linalg.norm(med[1:] - t.colour[1:]))
        d_L    = float(abs(med[0] - t.colour[0]))
        row = (d_full, d_ab, d_L, a / img[..., 0].size)
        if label == r["tote_id"]: tote_rows.append(row)
        elif label in item_ids:   item_rows.append(row)

I = np.array(item_rows); T = np.array(tote_rows)
print(f"\nmasks profiled: {len(I)} item, {len(T)} tote")
names = ["dist_full_Lab", "dist_ab_only", "dist_L_only", "area_frac"]
print(f"{'feature':<16}{'ITEM p10':>10}{'p50':>9}{'p90':>9} | {'TOTE p10':>10}{'p50':>9}{'p90':>9}")
for i, n in enumerate(names):
    print(f"{n:<16}" + "".join(f"{np.percentile(I[:,i],p):>10.1f}" if p==10 else f"{np.percentile(I[:,i],p):>9.1f}" for p in (10,50,90))
          + " | " + "".join(f"{np.percentile(T[:,i],p):>10.1f}" if p==10 else f"{np.percentile(T[:,i],p):>9.1f}" for p in (10,50,90)))

# how well does a single threshold on each feature separate them?
print("\nbest single-threshold separation (keep item, reject tote):")
for i, n in enumerate(names[:3]):
    best = max(((((I[:,i] > th).mean() + (T[:,i] <= th).mean())/2), th) for th in np.arange(2, 80, 1.0))
    acc, th = best
    print(f"  {n:<16} threshold {th:>5.1f} -> balanced acc {acc:.3f}  "
          f"(items kept {(I[:,i] > th).mean():.3f}, tote rejected {(T[:,i] <= th).mean():.3f})")
