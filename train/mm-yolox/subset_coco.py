#!/usr/bin/env python3
import json, random, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source instances.json")
    ap.add_argument("--dst", required=True, help="dest instances.json")
    ap.add_argument("--k", type=int, required=True, help="# images to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_faces", type=int, default=1, help="skip images with < min_faces")
    args = ap.parse_args()

    data = json.loads(Path(args.src).read_text())
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # Filter out images with too few faces (optional)
    candid = [im for im in data["images"] if len(anns_by_img.get(im["id"], [])) >= args.min_faces]
    if len(candid) < args.k:
        raise SystemExit(f"Requested {args.k} images but only {len(candid)} meet min_faces={args.min_faces}")

    random.Random(args.seed).shuffle(candid)
    keep_images = candid[:args.k]
    keep_ids = {im["id"] for im in keep_images}

    keep_anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]

    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": keep_images,
        "annotations": keep_anns,
        "categories": data["categories"],  # unchanged
    }
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dst).write_text(json.dumps(out))
    print(f"Wrote {args.dst}: {len(keep_images)} images, {len(keep_anns)} anns")

if __name__ == "__main__":
    main()

