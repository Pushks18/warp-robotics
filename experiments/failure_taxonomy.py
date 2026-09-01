"""Cluster the worst images and characterise how the pipeline fails."""
import json, sys
import numpy as np, cv2
sys.path.insert(0, ".")
from grade import match_boxes

GT = {r["file"]: r for r in json.load(open("data/dev/ground_truth.json"))["images"]}
M = {r["file"]: r for r in json.load(open("out/manifest.json"))["results"]}

rows = []
for f, g in sorted(GT.items()):
    r = M[f]; gb = [i["bbox"] for i in g["items"]]; pb = [i["bbox"] for i in r.get("items", [])]
    if not gb and not pb: f1 = 1.0
    elif not gb or not pb: f1 = 0.0
    else:
        tp = match_boxes(pb, gb); p, rc = tp/len(pb), tp/len(gb)
        f1 = 2*p*rc/(p+rc) if p+rc else 0.0
    pick_ok = None
    if not g["empty"] and r["action"] == "pick":
        m = cv2.imread("data/dev/masks/"+f.replace(".jpg",".png"), cv2.IMREAD_UNCHANGED)
        x, y = int(round(r["pick"]["x"])), int(round(r["pick"]["y"]))
        pick_ok = bool(m[y, x] in [i["id"] for i in g["items"]])
    elif g["empty"]:
        pick_ok = r["action"] == "flag"
    rows.append(dict(f=f, f1=f1, ngt=len(gb), npred=len(pb), empty=g["empty"],
                     action=r["action"], pick_ok=pick_ok))

rows.sort(key=lambda d: d["f1"])
print(f"{'image':<16}{'gt':>4}{'pred':>6}{'F1':>7}{'pick':>7}")
for d in rows[:15]:
    print(f"{d['f']:<16}{d['ngt']:>4}{d['npred']:>6}{d['f1']:>7.2f}"
          f"{('ok' if d['pick_ok'] else 'FAIL'):>7}")

print("\n--- F1 as a function of how crowded the tote is ---")
for lo, hi in [(0,0),(1,3),(4,6),(7,10),(11,30)]:
    sel = [d for d in rows if lo <= d["ngt"] <= hi]
    if not sel: continue
    print(f"  {lo:>2}-{hi:<3} items ({len(sel):>3} images): mean F1 {np.mean([d['f1'] for d in sel]):.3f}   "
          f"pred/gt ratio {np.mean([d['npred']/max(d['ngt'],1) for d in sel]):.2f}")

bad_pick = [d['f'] for d in rows if d['pick_ok'] is False]
print(f"\npick failures: {bad_pick}")
