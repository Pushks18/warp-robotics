"""Mask-level chroma tops out at 0.919 balanced accuracy. What else separates
the tote floor from an item?

Candidates, all cheap and all independent of absolute appearance:
  solidity   - the floor has items punched out of it, so it is concave; items
               are compact blobs. area / convex-hull area.
  ab_std     - moulded plastic is chromatically uniform; printed packaging is not
  edge_dens  - Canny density inside the mask; print and creases versus smooth plastic
  fp_border  - fraction of the mask's outline that runs along the tote outline
               (walls do, items do not)
"""
import json, sys
import cv2, numpy as np
from ultralytics import FastSAM
from warp_vision.tote import find_tote

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
model = FastSAM("FastSAM-s.pt")
rows = []
for r in GT:
    f = r["file"]
    img = cv2.imread(f"data/dev/images/{f}")
    gtm = cv2.imread("data/dev/masks/" + f.replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
    t = find_tote(img); x0, y0, x1, y1 = t.roi
    crop = img[y0:y1, x0:x1]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[y0:y1, x0:x1]
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5,5), 0), 40, 120) > 0
    fp_edge = cv2.morphologyEx(t.footprint[y0:y1, x0:x1].astype(np.uint8), cv2.MORPH_GRADIENT,
                               np.ones((9,9), np.uint8)) > 0
    res = model.predict(crop, verbose=False, imgsz=1024, device="mps",
                        retina_masks=True, conf=0.2, iou=0.7)[0]
    if res.masks is None: continue
    gt_crop = gtm[y0:y1, x0:x1]; item_ids = [i["id"] for i in r["items"]]
    for m in res.masks.data.cpu().numpy().astype(bool):
        a = int(m.sum())
        if a < 3000: continue
        vals, cnt = np.unique(gt_crop[m], return_counts=True)
        label = vals[np.argmax(cnt)]
        if cnt.max()/a < 0.7: continue
        is_tote = label == r["tote_id"]; is_item = label in item_ids
        if not (is_tote or is_item): continue
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = max(cv2.contourArea(cv2.convexHull(c)) for c in cs) if cs else 0
        outline = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)) > 0
        rows.append((float(np.linalg.norm(np.median(lab[m],axis=0)[1:]-t.colour[1:])),
                     a/hull if hull > 0 else 0.0,
                     float(lab[...,1:][m].std()),
                     float(edges[m].mean()),
                     float((outline & fp_edge).sum()/max(outline.sum(),1)),
                     is_tote))
A = np.array(rows); np.savez("reports/tote_features.npz", A=A)
tote = A[:,5].astype(bool); item = ~tote
names = ["chroma_dist","solidity","ab_std","edge_dens","fp_border"]
print(f"\n{item.sum()} item masks, {tote.sum()} tote masks")
print(f"{'feature':<13}{'ITEM p10':>10}{'p50':>9}{'p90':>9} | {'TOTE p10':>10}{'p50':>9}{'p90':>9}{'  bal.acc':>10}")
for i,n in enumerate(names):
    d=A[:,i]
    lo=max((((d[item]>th).mean()+(d[tote]<=th).mean())/2, th) for th in np.linspace(d.min(),d.max(),120))
    hi=max((((d[item]<th).mean()+(d[tote]>=th).mean())/2, th) for th in np.linspace(d.min(),d.max(),120))
    acc=max(lo[0],hi[0])
    print(f"{n:<13}"+"".join(f"{np.percentile(d[item],p):>10.3f}" if p==10 else f"{np.percentile(d[item],p):>9.3f}" for p in (10,50,90))
          +" | "+"".join(f"{np.percentile(d[tote],p):>10.3f}" if p==10 else f"{np.percentile(d[tote],p):>9.3f}" for p in (10,50,90))+f"{acc:>10.3f}")
# combined rule: chroma OR shape
ch, so = A[:,0], A[:,1]
best=max(((( (ch[item]>c)|(so[item]>s) ).mean() + ( (ch[tote]<=c)&(so[tote]<=s) ).mean())/2, c, s)
         for c in np.arange(8,40,2.0) for s in np.arange(0.3,0.95,0.05))
print(f"\ncombined  keep if chroma>{best[1]:.0f} OR solidity>{best[2]:.2f}  -> balanced acc {best[0]:.3f}")
best2=max(((( (ch[item]>c)&(so[item]>s) ).mean() + ( (ch[tote]<=c)|(so[tote]<=s) ).mean())/2, c, s)
         for c in np.arange(8,40,2.0) for s in np.arange(0.3,0.95,0.05))
print(f"combined  keep if chroma>{best2[1]:.0f} AND solidity>{best2[2]:.2f} -> balanced acc {best2[0]:.3f}")
