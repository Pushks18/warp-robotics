"""FastSAM-s versus FastSAM-x: is a bigger backbone worth the latency?

Measures the recall ceiling of the raw proposals (the most any filter could
achieve) alongside the F1 the shipped filter actually delivers, plus latency.
"""
import json, sys, time
import cv2, numpy as np
from ultralytics import FastSAM
sys.path.insert(0, ".")
from grade import match_boxes, iou
from warp_vision.tote import find_tote
from warp_vision.detect import ItemDetector

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
GT = json.load(open("data/dev/ground_truth.json"))["images"][:N]

for weights in ("FastSAM-s.pt", "FastSAM-x.pt"):
    det = ItemDetector(); det._model = FastSAM(weights)
    f1s, lat, ceil_tp, ceil_n = [], [], 0, 0
    for r in GT:
        f = r["file"]; img = cv2.imread(f"data/dev/images/{f}")
        t0 = time.perf_counter()
        t = find_tote(img); masks, sc = det.detect(img, t)
        lat.append((time.perf_counter()-t0)*1000)
        gtb = [i["bbox"] for i in r["items"]]
        pb = []
        for m in masks:
            ys, xs = np.where(m)
            pb.append([int(xs.min()), int(ys.min()), int(xs.max()-xs.min())+1, int(ys.max()-ys.min())+1])
        if not gtb and not pb: f1s.append(1.0)
        elif not gtb or not pb: f1s.append(0.0)
        else:
            tp = match_boxes(pb, gtb); p, rc = tp/len(pb), tp/len(gtb)
            f1s.append(2*p*rc/(p+rc) if p+rc else 0.0)
        ceil_n += len(gtb)
        ceil_tp += sum(1 for g in gtb if any(iou(b, g) >= 0.5 for b in pb))
    print(f"{weights:<14} F1 {np.mean(f1s):.3f}   matched {ceil_tp}/{ceil_n} "
          f"({ceil_tp/ceil_n:.3f})   latency median {np.median(lat):.0f} ms  p95 {np.percentile(lat,95):.0f} ms")
    del det
