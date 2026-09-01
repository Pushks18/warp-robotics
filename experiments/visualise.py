"""Draw predictions against ground truth. Also the tool to demo in a recording.

    python experiments/visualise.py tote_0108.jpg tote_0082.jpg -o reports/x.png
"""
import argparse, json
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("-o", "--out", default="reports/vis.png")
ap.add_argument("-m", "--manifest", default="out/manifest.json")
ap.add_argument("--width", type=int, default=560)
a = ap.parse_args()

GT = {r["file"]: r for r in json.load(open("data/dev/ground_truth.json"))["images"]}
M = {r["file"]: r for r in json.load(open(a.manifest))["results"]}

tiles = []
for f in a.files:
    img = cv2.imread(f"data/dev/images/{f}")
    g, r = GT[f], M[f]
    gt_img, pr_img = img.copy(), img.copy()
    for it in g["items"]:
        x, y, w, h = [int(v) for v in it["bbox"]]
        cv2.rectangle(gt_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    for it in r.get("items", []):
        x, y, w, h = [int(v) for v in it["bbox"]]
        cv2.rectangle(pr_img, (x, y), (x+w, y+h), (0, 165, 255), 3)
    if r["action"] == "pick":
        px, py = int(r["pick"]["x"]), int(r["pick"]["y"])
        cv2.circle(pr_img, (px, py), 22, (255, 255, 255), 4)
        cv2.circle(pr_img, (px, py), 8, (0, 0, 255), -1)
    for im, txt in ((gt_img, f"GT  {f}  {len(g['items'])} items"),
                    (pr_img, f"PRED  {len(r.get('items',[]))} boxes  {r['action']}")):
        cv2.rectangle(im, (0, 0), (im.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(im, txt, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    pair = np.hstack([gt_img, pr_img])
    s = (a.width * 2) / pair.shape[1]
    tiles.append(cv2.resize(pair, (int(pair.shape[1]*s), int(pair.shape[0]*s))))

cv2.imwrite(a.out, np.vstack(tiles))
print(f"wrote {a.out}")
