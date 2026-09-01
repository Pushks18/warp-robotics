"""What is in FastSAM's raw output, and what separates a real item mask from junk?

Establishes the recall ceiling (best achievable F1 with a perfect filter) and
profiles candidate filter features against ground truth.
"""
import json, sys, time
import cv2, numpy as np
from ultralytics import FastSAM
from warp_vision.tote import find_tote

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
model = FastSAM("FastSAM-s.pt")

rows, ceiling, latency = [], [], []
for r in GT:
    f = r["file"]
    img = cv2.imread(f"data/dev/images/{f}")
    gtm = cv2.imread("data/dev/masks/" + f.replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
    t0 = time.time()
    t = find_tote(img); x0, y0, x1, y1 = t.roi
    res = model.predict(img[y0:y1, x0:x1], verbose=False, imgsz=1024,
                        retina_masks=True, conf=0.2, iou=0.7)[0]
    latency.append(time.time() - t0)
    if res.masks is None:
        continue
    md = res.masks.data.numpy().astype(bool)
    conf = res.boxes.conf.numpy()
    H, W = img.shape[:2]

    gt_items = {it["id"]: (gtm == it["id"]) for it in r["items"]}
    tote_px = (gtm == r["tote_id"])
    footprint = t.footprint

    matched = set()
    for k, m in enumerate(md):
        full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = m
        a = full.sum()
        if a == 0: continue
        # best IoU against any GT item
        best_iou, best_id = 0.0, None
        for iid, g in gt_items.items():
            inter = np.logical_and(full, g).sum()
            if inter == 0: continue
            iou = inter / (a + g.sum() - inter)
            if iou > best_iou: best_iou, best_id = iou, iid
        if best_iou >= 0.5: matched.add(best_id)
        # candidate filter features
        border = (m[0].sum() + m[-1].sum() + m[:,0].sum() + m[:,-1].sum()) / (2*(m.shape[0]+m.shape[1]))
        rows.append(dict(area_frac=a/(H*W),
                         contain=np.logical_and(full, footprint).sum()/a,
                         tote_overlap=np.logical_and(full, tote_px).sum()/a,
                         border=border, conf=float(conf[k]),
                         iou=best_iou, good=best_iou >= 0.5))
    ceiling.append((len(matched), len(r["items"])))
    del md

R = {k: np.array([d[k] for d in rows]) for k in rows[0]}
good = R["good"].astype(bool)
tp = sum(m for m, _ in ceiling); ngt = sum(n for _, n in ceiling)
print(f"\n=== {len(GT)} images, {len(rows)} raw masks, {latency and np.mean(latency)*1000:.0f} ms/img ===")
print(f"CEILING: {tp}/{ngt} GT items representable at IoU>=0.5  (recall {tp/ngt:.3f})")
print(f"raw masks per image: {len(rows)/len(GT):.0f}   of which good: {good.sum()} ({good.mean():.1%})")
print(f"\n{'feature':<14}{'good p10':>10}{'good p50':>10}{'good p90':>10} | {'junk p10':>10}{'junk p50':>10}{'junk p90':>10}")
for k in ("area_frac", "contain", "tote_overlap", "border", "conf"):
    g, b = R[k][good], R[k][~good]
    print(f"{k:<14}" + "".join(f"{np.percentile(g,p):>10.3f}" for p in (10,50,90))
          + " | " + "".join(f"{np.percentile(b,p):>10.3f}" for p in (10,50,90)))
np.savez("reports/mask_features.npz", **R)
