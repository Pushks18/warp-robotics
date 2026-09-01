import json, time, cv2, numpy as np

GT = json.load(open("data/dev/ground_truth.json"))["images"]
K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def fill_holes(binary):
    h, w = binary.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)      # pad so flood always starts outside
    ff[1:-1, 1:-1] = binary
    pad = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    return (binary | (1 - ff[1:-1, 1:-1])).astype(bool)

def central_component(m):
    """Largest component weighted to the middle of the frame: the bin is always
    roughly centred, bright machinery is always at the border."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1: return np.zeros(m.shape, bool)
    h, w = m.shape
    cy, cx = h // 2, w // 2
    # a generous central box; require the component to overlap it
    box = lab[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
    ids = [i for i in range(1, n) if (box == i).sum() > 0]
    if not ids: ids = list(range(1, n))
    best = max(ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return lab == best

def roi(img, mode):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    sat = (S >= 140) & (V >= 40)
    if mode == "sat":
        m = sat
    else:
        t, _ = cv2.threshold(cv2.GaussianBlur(V, (5, 5), 0), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = V >= t
        m = bright if mode == "bright" else (bright | sat)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, K9)
    return fill_holes(central_component(m).astype(np.uint8))

for mode in ("sat", "bright", "both"):
    recs, areas, t0 = [], [], time.time()
    for r in GT:
        img = cv2.imread(f"data/dev/images/{r['file']}")
        fp = roi(img, mode); areas.append(fp.mean())
        if r["items"]:
            m = cv2.imread("data/dev/masks/" + r["file"].replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
            items = np.isin(m, [it["id"] for it in r["items"]])
            recs.append(((items & fp).sum() / items.sum(), r["file"]))
    v = np.array([x[0] for x in recs])
    print(f"{mode:7} recall mean {v.mean():.4f} min {v.min():.3f} "
          f"n<0.99 {(v<0.99).sum():2d} n<0.95 {(v<0.95).sum():2d} | "
          f"area mean {np.mean(areas):.3f} max {np.max(areas):.3f} | {(time.time()-t0)/len(GT)*1000:.0f} ms/img")
    print("        worst:", [(f, round(float(s),2)) for s, f in sorted(recs)[:4]])

print("\n--- rectangular ROI (bounding rect of footprint + margin) ---")
for margin in (0.0, 0.02, 0.04):
    recs, areas = [], []
    for r in GT:
        img = cv2.imread(f"data/dev/images/{r['file']}")
        fp = roi(img, "both")
        ys, xs = np.where(fp)
        h, w = fp.shape
        my, mx = int(margin * h), int(margin * w)
        y0, y1 = max(0, ys.min()-my), min(h, ys.max()+1+my)
        x0, x1 = max(0, xs.min()-mx), min(w, xs.max()+1+mx)
        rect = np.zeros_like(fp); rect[y0:y1, x0:x1] = True
        areas.append(rect.mean())
        if r["items"]:
            m = cv2.imread("data/dev/masks/" + r["file"].replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
            items = np.isin(m, [it["id"] for it in r["items"]])
            recs.append(((items & rect).sum() / items.sum(), r["file"]))
    v = np.array([x[0] for x in recs])
    print(f"margin {margin:.2f}  recall mean {v.mean():.4f} min {v.min():.3f} n<0.99 {(v<0.99).sum():2d} | area mean {np.mean(areas):.3f}")
    print("          worst:", [(f, round(float(s),3)) for s, f in sorted(recs)[:3]])
