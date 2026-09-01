#!/usr/bin/env python3
"""Run the warehouse vision pipeline over a directory of tote images.

    python run.py --images data/dev/images --out out/manifest.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

from warp_vision.detect import ItemDetector
from warp_vision.pipeline import process_image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/dev/images")
    ap.add_argument("--out", default="out/manifest.json")
    ap.add_argument("--backend", default="fastsam", choices=["fastsam", "mobilesam"])
    ap.add_argument("--device", default="mps",
                    help="torch device: mps (Apple), cpu, or cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="below this pick score, flag instead of picking")
    ap.add_argument("--pick-order", action="store_true",
                    help="also emit the full emptying sequence per tote")
    args = ap.parse_args()

    paths = sorted(Path(args.images).glob("*.jpg"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no .jpg images under {args.images}", file=sys.stderr)
        return 1

    detector = ItemDetector(backend=args.backend, device=args.device)
    results, t0 = [], time.perf_counter()
    for i, p in enumerate(paths, 1):
        r = process_image(p, detector, min_confidence=args.min_confidence,
                          want_order=args.pick_order)
        results.append(r)
        print(f"[{i:>3}/{len(paths)}] {r.file:<16} {len(r.items):>3} items  "
              f"{r.action:<4} {r.latency_ms:>6.0f} ms", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"results": [r.to_manifest_entry() for r in results]}, indent=2))

    lat = sorted(r.latency_ms for r in results)
    total = time.perf_counter() - t0
    print(f"\nwrote {out}  ({len(results)} images, {total:.1f}s total)")
    print(f"latency  median {lat[len(lat)//2]:.0f} ms   p95 {lat[int(len(lat)*0.95)]:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
