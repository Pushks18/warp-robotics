"""Should the pipeline flag more often than 'nothing detected'?

The grader pays 0.25 for flagging a tote that has items, against 1.0 for a
correct pick and 0.0 for a wrong one. So a flag is only worth casting when the
pick would have failed: converting a wrong pick to a flag gains 0.25, but
converting a correct pick to a flag loses 0.75. A confidence gate must therefore
be right about failure more than 3 times in 4 to pay for itself.

This measures whether the pipeline's confidence carries that much signal.
"""
import json
import numpy as np, cv2

GT = {r["file"]: r for r in json.load(open("data/dev/ground_truth.json"))["images"]}
M = {r["file"]: r for r in json.load(open("out/manifest.json"))["results"]}

rows = []
for f, g in sorted(GT.items()):
    r = M[f]
    if r["action"] != "pick":
        continue
    if g["empty"]:
        ok = False
    else:
        m = cv2.imread("data/dev/masks/" + f.replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
        ids = [i["id"] for i in g["items"]]
        x, y = int(round(r["pick"]["x"])), int(round(r["pick"]["y"]))
        ok = bool(m[y, x] in ids)
    rows.append((r.get("confidence", 0.0), ok, f))

rows.sort()
conf = np.array([c for c, _, _ in rows]); ok = np.array([o for _, o, _ in rows])
print(f"{len(rows)} picks: {ok.sum()} correct, {(~ok).sum()} wrong")
print(f"confidence of correct picks: p10 {np.percentile(conf[ok],10):.3f} median {np.median(conf[ok]):.3f}")
print(f"confidence of wrong picks  : {sorted(np.round(conf[~ok],3).tolist())}")

base = ok.sum() * 1.0
print(f"\nbaseline pick points from these {len(rows)} picks: {base:.2f}")
print(f"{'flag lowest N':>14}{'wrong caught':>14}{'correct lost':>14}{'net points':>12}")
for n in (0, 2, 5, 10, 15, 20, 30):
    sel = np.zeros(len(rows), bool); sel[:n] = True
    caught = int((sel & ~ok).sum()); lost = int((sel & ok).sum())
    net = base - lost * 1.0 + lost * 0.25 + caught * 0.25
    print(f"{n:>14}{caught:>14}{lost:>14}{net:>12.2f}")
