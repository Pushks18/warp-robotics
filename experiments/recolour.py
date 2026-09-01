"""Is the pipeline actually colour-agnostic, or does it just happen to work on
yellow and blue bins?

The design claims colour-independence: the crop is keyed on brightness OR
saturation rather than hue, and the reference tote colour is learned per image
from the rim. That claim is untested -- the dev split contains only yellow, blue
and beige bins.

This tests it directly. Using the GT masks (validation only, never at inference)
the tote's own pixels are recoloured to hues and saturations that appear nowhere
in the data, leaving every item untouched. Ground truth boxes and masks are
unchanged, so the original grader applies as-is. If the score holds, the pipeline
is colour-agnostic in fact and not just by intention.
"""
import json, os, shutil, subprocess, sys
import cv2, numpy as np

VARIANTS = {
    "green":   dict(hue=60,  sat=1.0, val=1.0),
    "magenta": dict(hue=150, sat=1.0, val=1.0),
    "grey":    dict(hue=0,   sat=0.10, val=1.0),   # no chroma at all
    "white":   dict(hue=0,   sat=0.05, val=1.35),  # bright and colourless
}
GT = json.load(open("data/dev/ground_truth.json"))["images"]
root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/recolour"

for name, v in VARIANTS.items():
    d = f"{root}/{name}/images"
    os.makedirs(d, exist_ok=True)
    for r in GT:
        f = r["file"]
        img = cv2.imread(f"data/dev/images/{f}")
        m = cv2.imread("data/dev/masks/" + f.replace(".jpg", ".png"), cv2.IMREAD_UNCHANGED)
        tote = (m == r["tote_id"])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h[tote] = (h[tote] + v["hue"]) % 180
        s[tote] = np.clip(s[tote] * v["sat"], 0, 255)
        val[tote] = np.clip(val[tote] * v["val"], 0, 255)
        out = cv2.cvtColor(np.stack([h, s, val], -1).astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"{d}/{f}", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[{name}] wrote {len(GT)} recoloured images", flush=True)

    mf = f"{root}/{name}/manifest.json"
    subprocess.run([".venv/bin/python", "run.py", "--images", d, "--out", mf],
                   env={**os.environ, "PYTHONPATH": "."},
                   stdout=subprocess.DEVNULL, check=True)
    print(f"[{name}] ", end="", flush=True)
    out = subprocess.run([".venv/bin/python", "grade.py", mf], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "detection mean F1" in line or "pick score" in line:
            print(line.strip(), end="   ", flush=True)
    print(flush=True)
