"""Validate the tote footprint against GT masks. GT is used here to check a
design decision, never at inference time."""
import json, time
import cv2, numpy as np
from warp_vision.tote import tote_footprint, tote_colour_mask

GT = json.load(open("data/dev/ground_truth.json"))["images"]
rec_list, area_list, t0 = [], [], time.time()
worst = []
for r in GT:
    img = cv2.imread(f"data/dev/images/{r['file']}")
    m = cv2.imread(f"data/dev/masks/{r['file'].replace('.jpg','.png')}", cv2.IMREAD_UNCHANGED)
    fp = tote_footprint(img)
    area_list.append(fp.mean())
    if r["items"]:
        items = np.isin(m, [it["id"] for it in r["items"]])
        rec = (items & fp).sum() / items.sum()
        rec_list.append(rec)
        worst.append((rec, r["file"]))
rec_list = np.array(rec_list); area_list = np.array(area_list)
print(f"images {len(GT)}   {time.time()-t0:.1f}s total, {(time.time()-t0)/len(GT)*1000:.0f} ms/img")
print(f"item-pixel recall inside footprint: mean {rec_list.mean():.4f}  min {rec_list.min():.4f}  "
      f"n<0.99 {(rec_list<0.99).sum()}  n<0.95 {(rec_list<0.95).sum()}")
print(f"footprint area fraction of frame:   mean {area_list.mean():.3f}  min {area_list.min():.3f}  max {area_list.max():.3f}")
print("worst 5:", [(f, round(float(v),3)) for v, f in sorted(worst)[:5]])
