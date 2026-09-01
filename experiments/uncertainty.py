"""How much would these scores move on a different 120 images?

The held-out split is unseen, so its score cannot be computed. But part of the
gap is just sampling noise -- 120 images is a small sample, and a fresh draw from
the same distribution would land somewhere else. Bootstrapping the per-image
scores estimates that part honestly. It does NOT capture distribution shift
(different tote colours, different item mixes), which can only push it further.
"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, ".")
from grade import match_boxes

GT = {r["file"]: r for r in json.load(open("data/dev/ground_truth.json"))["images"]}
M = {r["file"]: r for r in json.load(open("out/manifest.json"))["results"]}

f1s, picks = [], []
for f, g in sorted(GT.items()):
    r = M[f]
    gb = [i["bbox"] for i in g["items"]]; pb = [i["bbox"] for i in r.get("items", [])]
    if not gb and not pb: f1 = 1.0
    elif not gb or not pb: f1 = 0.0
    else:
        tp = match_boxes(pb, gb); p, rc = tp/len(pb), tp/len(gb)
        f1 = 2*p*rc/(p+rc) if p+rc else 0.0
    f1s.append(f1)
    if r["action"] == "flag":
        picks.append(1.0 if g["empty"] else 0.25)
    elif g["empty"]:
        picks.append(0.0)
    else:
        m = cv2.imread("data/dev/masks/"+f.replace(".jpg",".png"), cv2.IMREAD_UNCHANGED)
        x, y = int(round(r["pick"]["x"])), int(round(r["pick"]["y"]))
        picks.append(1.0 if m[y, x] in [i["id"] for i in g["items"]] else 0.0)

f1s, picks = np.array(f1s), np.array(picks)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(f1s), size=(20000, len(f1s)))
bf1, bp = f1s[idx].mean(1), picks[idx].mean(1)
print(f"dev split          F1 {f1s.mean():.3f}   pick {picks.mean():.3f}")
print(f"bootstrap 90% CI   F1 {np.percentile(bf1,5):.3f}-{np.percentile(bf1,95):.3f}"
      f"   pick {np.percentile(bp,5):.3f}-{np.percentile(bp,95):.3f}")
print(f"std error          F1 {bf1.std():.3f}          pick {bp.std():.3f}")
