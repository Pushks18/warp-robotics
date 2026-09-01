"""How should overlapping FastSAM proposals be reduced to one box per item?

FastSAM proposes an item, its parts, and the group it belongs to simultaneously.
Which survives is an architectural choice, not a threshold: sort by area and you
resolve toward whole objects and groups; sort by confidence and you resolve
toward whatever the model is surest of. Containment suppression is the risky
knob -- in a cluttered tote a genuinely separate item is often mostly buried
under its neighbour, and deleting it costs recall that cannot be recovered.

Masks are computed once per image and every config scored on the same candidates.
"""
import json, sys
import cv2, numpy as np
from ultralytics import FastSAM
sys.path.insert(0, ".")
from grade import match_boxes
from warp_vision.tote import find_tote
from warp_vision.detect import (MIN_CONTAINMENT, TOTE_CHROMA_DIST, TOTE_AB_STD,
                                TOTE_EDGE_DENSITY, MIN_ITEM_VOTES,
                                MIN_AREA_FRAC, MAX_AREA_FRAC)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]
model = FastSAM("FastSAM-s.pt")

CONFIGS = [
    ("area sort, contain 0.80 (current)", dict(sort="area", cont=0.80, iou=0.60, conf=0.30)),
    ("area sort, contain 0.90",           dict(sort="area", cont=0.90, iou=0.60, conf=0.30)),
    ("area sort, contain off",            dict(sort="area", cont=2.00, iou=0.60, conf=0.30)),
    ("conf sort, contain 0.80",           dict(sort="conf", cont=0.80, iou=0.60, conf=0.30)),
    ("conf sort, contain 0.90",           dict(sort="conf", cont=0.90, iou=0.60, conf=0.30)),
    ("conf sort, contain off",            dict(sort="conf", cont=2.00, iou=0.60, conf=0.30)),
    ("area sort, contain 0.90, iou 0.75", dict(sort="area", cont=0.90, iou=0.75, conf=0.30)),
    ("area sort, contain 0.90, conf 0.25",dict(sort="area", cont=0.90, iou=0.60, conf=0.25)),
]
acc = {k: [] for k, _ in CONFIGS}
npred = {k: [] for k, _ in CONFIGS}

def bbox(m):
    ys, xs = np.where(m); return [int(xs.min()), int(ys.min()),
                                  int(xs.max()-xs.min())+1, int(ys.max()-ys.min())+1]

for r in GT:
    f = r["file"]; img = cv2.imread(f"data/dev/images/{f}")
    t = find_tote(img); x0, y0, x1, y1 = t.roi; crop = img[y0:y1, x0:x1]
    res = model.predict(crop, verbose=False, imgsz=1024, device="mps",
                        retina_masks=True, conf=0.2, iou=0.7)[0]
    gtb = [i["bbox"] for i in r["items"]]
    cands = []
    if res.masks is not None:
        md = res.masks.data.cpu().numpy().astype(bool); cf = res.boxes.conf.cpu().numpy()
        fp = t.footprint[y0:y1, x0:x1]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        ed = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5,5), 0), 40, 120) > 0
        fa = float(img.shape[0]*img.shape[1])
        for m, c in zip(md, cf):
            a = float(m.sum())
            if not (MIN_AREA_FRAC <= a/fa <= MAX_AREA_FRAC): continue
            if np.logical_and(m, fp).sum()/a < MIN_CONTAINMENT: continue
            ab = lab[..., 1:][m]
            v = (int(np.linalg.norm(np.median(ab,axis=0)-t.colour[1:]) > TOTE_CHROMA_DIST)
                 + int(ab.std() < TOTE_AB_STD) + int(ed[m].mean() > TOTE_EDGE_DENSITY))
            if v < MIN_ITEM_VOTES: continue
            cands.append((a, float(c), m))

    for name, cfg in CONFIGS:
        cs = [c for c in cands if c[1] >= cfg["conf"]]
        cs.sort(key=(lambda t_: -t_[0]) if cfg["sort"] == "area" else (lambda t_: -t_[1]))
        kept = []
        for a, c, m in cs:
            drop = False
            for ka, kc, km in kept:
                inter = float(np.logical_and(m, km).sum())
                if inter == 0: continue
                if inter/a >= cfg["cont"] or inter/(a+ka-inter) >= cfg["iou"]:
                    drop = True; break
            if not drop: kept.append((a, c, m))
        pb = [bbox(m) for _, _, m in kept]
        npred[name].append(len(pb))
        if not gtb and not pb: acc[name].append(1.0)
        elif not gtb or not pb: acc[name].append(0.0)
        else:
            tp = match_boxes(pb, gtb)
            p, rc = tp/len(pb), tp/len(gtb)
            acc[name].append(2*p*rc/(p+rc) if p+rc else 0.0)

print(f"\n{N} images, gt {np.mean([len(r['items']) for r in GT]):.1f} boxes/img\n")
print(f"{'config':<36}{'mean F1':>9}{'pred/img':>10}")
for name, _ in CONFIGS:
    print(f"{name:<36}{np.mean(acc[name]):>9.3f}{np.mean(npred[name]):>10.1f}")
