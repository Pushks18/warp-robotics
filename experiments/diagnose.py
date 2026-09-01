"""Split detection F1 into precision and recall, and profile the pick failures."""
import json, sys
import numpy as np, cv2

sys.path.insert(0, ".")
from grade import iou, match_boxes

GT = {r["file"]: r for r in json.load(open("data/dev/ground_truth.json"))["images"]}
M = {r["file"]: r for r in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/manifest.json"))["results"]}

P, R, npred, ngt = [], [], [], []
empty_bad, pick_bad, pick_ok = [], [], []
for f, g in sorted(GT.items()):
    r = M[f]
    gb = [i["bbox"] for i in g["items"]]
    pb = [i["bbox"] for i in r.get("items", [])]
    npred.append(len(pb)); ngt.append(len(gb))
    if gb and pb:
        tp = match_boxes(pb, gb)
        P.append(tp/len(pb)); R.append(tp/len(gb))
    if g["empty"]:
        if r["action"] == "pick": empty_bad.append((f, len(pb)))
    elif r["action"] == "pick":
        m = cv2.imread("data/dev/masks/"+f.replace(".jpg",".png"), cv2.IMREAD_UNCHANGED)
        ids = [i["id"] for i in g["items"]]
        x, y = int(round(r["pick"]["x"])), int(round(r["pick"]["y"]))
        hit = m[y, x]
        (pick_ok if hit in ids else pick_bad).append((f, "tote" if hit == g["tote_id"] else ("bg" if hit == 0 else "edge")))

print(f"detection  precision {np.mean(P):.3f}   recall {np.mean(R):.3f}")
print(f"boxes      predicted {np.mean(npred):.1f}/img   gt {np.mean(ngt):.1f}/img")
print(f"\nempty totes picked in: {len(empty_bad)}  -> {empty_bad}")
from collections import Counter
print(f"\npick failures ({len(pick_bad)}): {Counter(x[1] for x in pick_bad)}")
print("  examples:", [f for f,_ in pick_bad[:8]])
