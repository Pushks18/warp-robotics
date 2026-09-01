"""Which reference colour best represents 'this tote'?

The rim-band median fails on shadowed floors: the rim is lit, the floor is not,
so the floor reads as chromatically distant and survives as an 'item'. Compares
three references by how well a mask-level chroma test separates tote from item.
"""
import json, sys
import cv2, numpy as np
from ultralytics import FastSAM
from warp_vision.tote import find_tote

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
model = FastSAM("FastSAM-s.pt")

def ab_mode(lab, sel, bins=64):
    """Peak of the 2D chromaticity histogram over a region: the single colour the
    region is most made of. Robust to shadow, which moves L but not the mode."""
    a = lab[..., 1][sel]; b = lab[..., 2][sel]
    H, ae, be = np.histogram2d(a, b, bins=bins, range=[[0, 255], [0, 255]])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    return np.array([(ae[i]+ae[i+1])/2, (be[j]+be[j+1])/2], np.float32)

rows = []
for r in GT:
    f = r["file"]
    img = cv2.imread(f"data/dev/images/{f}")
    gtm = cv2.imread("data/dev/masks/" + f.replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
    t = find_tote(img); x0, y0, x1, y1 = t.roi
    lab_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_rim = t.colour[1:]
    ref_mode = ab_mode(lab_full, t.footprint)
    lab = lab_full[y0:y1, x0:x1]
    res = model.predict(img[y0:y1, x0:x1], verbose=False, imgsz=1024, device="mps",
                        retina_masks=True, conf=0.2, iou=0.7)[0]
    if res.masks is None: continue
    gt_crop = gtm[y0:y1, x0:x1]; item_ids = [i["id"] for i in r["items"]]
    for m in res.masks.data.cpu().numpy().astype(bool):
        a = m.sum()
        if a < 3000: continue
        vals, cnt = np.unique(gt_crop[m], return_counts=True)
        label = vals[np.argmax(cnt)]
        if cnt.max()/a < 0.7: continue
        med = np.median(lab[m], axis=0)[1:]
        is_tote = label == r["tote_id"]
        is_item = label in item_ids
        if not (is_tote or is_item): continue
        rows.append((float(np.linalg.norm(med-ref_rim)),
                     float(np.linalg.norm(med-ref_mode)),
                     float(min(np.linalg.norm(med-ref_rim), np.linalg.norm(med-ref_mode))),
                     is_tote))

A = np.array(rows); tote = A[:, 3].astype(bool); item = ~tote
print(f"\n{len(GT)} images | {item.sum()} item masks, {tote.sum()} tote masks")
for i, name in enumerate(["rim-band median", "footprint ab-mode", "min of both"]):
    d = A[:, i]
    best = max((((d[item] > th).mean() + (d[tote] <= th).mean())/2, th) for th in np.arange(2, 90, 1.0))
    acc, th = best
    print(f"{name:<20} th={th:>4.0f}  balanced acc {acc:.3f}   "
          f"items kept {(d[item] > th).mean():.3f}   tote rejected {(d[tote] <= th).mean():.3f}")
