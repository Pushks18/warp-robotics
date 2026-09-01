"""Probe: what does the tote look like in color space, and can we isolate its interior?

Uses GT masks ONLY to validate the color heuristic, never at inference.
"""
import json, sys
import numpy as np, cv2

GT = json.load(open("data/dev/ground_truth.json"))["images"]

def stats(rec):
    img = cv2.imread(f"data/dev/images/{rec['file']}")
    m = cv2.imread(f"data/dev/masks/{rec['file'].replace('.jpg','.png')}", cv2.IMREAD_UNCHANGED)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tote = (m == rec["tote_id"])
    items = np.isin(m, [it["id"] for it in rec["items"]]) if rec["items"] else np.zeros_like(tote)
    bg = (m == 0)
    out = {}
    for name, sel in (("tote", tote), ("items", items), ("bg", bg)):
        if sel.sum() == 0:
            continue
        h, s, v = (hsv[..., i][sel] for i in range(3))
        out[name] = dict(frac=round(float(sel.mean()), 3),
                         H=[int(np.percentile(h, p)) for p in (10, 50, 90)],
                         S=[int(np.percentile(s, p)) for p in (10, 50, 90)],
                         V=[int(np.percentile(v, p)) for p in (10, 50, 90)])
    return out

for rec in GT[:6] + [r for r in GT if r["empty"]][:2]:
    print(rec["file"], "empty" if rec["empty"] else f"{len(rec['items'])} items")
    for k, v in stats(rec).items():
        print(f"   {k:6} frac={v['frac']:<6} H={v['H']} S={v['S']} V={v['V']}")
