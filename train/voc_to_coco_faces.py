#!/usr/bin/env python3
import argparse, json, random, xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from PIL import Image
import csv

# ---------------- Config via CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Merge VOC 'Human face'/'Human head' into single-class COCO + attribution.")
    p.add_argument("--root", required=True,
                   help="Root of the downloaded dataset (has 'Human face' and 'Human head' subdirs).")
    p.add_argument("--meta_dir", required=True,
                   help="Directory containing Open Images metadata CSVs (the same --meta_dir you used).")
    p.add_argument("--out", required=True,
                   help="Output directory for COCO json + attribution csv.")
    p.add_argument("--include_head", action="store_true",
                   help="If set, includes 'Human head' boxes as positives (mapped to face).")
    p.add_argument("--min_box_pixels", type=int, default=16*16,
                   help="Discard boxes smaller than this area in pixels (default: 256).")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="Validation fraction (default: 0.1).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    return p.parse_args()

# ---------------- Helpers ----------------
def read_voc_annotation(xml_path: Path) -> Tuple[Path, int, int, List[Dict]]:
    """Returns (image_path, width, height, boxes). Boxes are dicts with xmin,ymin,xmax,ymax."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # filename may be present, but the image lives next to sibling /images/ folder
    filename = root.findtext("filename")
    size = root.find("size")
    if size is not None:
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
    else:
        width = height = None

    objs = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        objs.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return filename, width, height, objs

def ensure_size(image_path: Path, w: int, h: int) -> Tuple[int, int]:
    if w and h:
        return w, h
    with Image.open(image_path) as im:
        return im.size  # (w, h)

def image_id_from_filename(fname: str) -> str:
    # openimages tool typically saves images as <OpenImagesID>.<ext>
    return Path(fname).stem

def load_meta_tables(meta_dir: Path) -> Dict[str, Dict]:
    """
    Build a dict: image_id -> {OriginalURL, OriginalLandingURL, AuthorProfileURL, License}
    We’ll try to read all splits; some filenames may appear in train/val/test.
    """
    meta = {}
    # Common file names in various versions
    candidates = [
        "oidv6-train-images-with-labels-with-rotation.csv",
        "oidv6-validation-images-with-labels-with-rotation.csv",
        "oidv6-test-images-with-labels-with-rotation.csv",
        "train-images-boxable-with-rotation.csv",
        "validation-images-with-rotation.csv",
        "test-images-with-rotation.csv",
    ]
    cols_norm = {
        "imageid": "ImageID",
        "originalurl": "OriginalURL",
        "originallandingurl": "OriginalLandingURL",
        "authorprofileurl": "AuthorProfileURL",
        "license": "License",
    }
    for fname in candidates:
        fpath = meta_dir / fname
        if not fpath.exists():
            continue
        with fpath.open("r", newline="") as f:
            reader = csv.DictReader(f)
            # normalize header keys to lowercase
            for row in reader:
                norm = {k.lower(): v for k, v in row.items()}
                rec = {cols_norm[k]: norm.get(k, "") for k in cols_norm}
                imgid = rec["ImageID"]
                if imgid and imgid not in meta:
                    meta[imgid] = rec
    return meta

# ---------------- Main conversion ----------------
def main():
    args = parse_args()
    random.seed(args.seed)

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input folders
    face_dir = root / "humanface"
    head_dir = root / "humanhead"
    use_classes = [("Human face", face_dir)]
    if args.include_head:
        use_classes.append(("Human head", head_dir))

    # Gather all VOC xmls and map to image paths
    records = []  # list of dicts with keys: img_path, ann_path, width, height, boxes, oi_id
    for cls_name, cls_dir in use_classes:
        voc_dir = cls_dir / "pascal"
        img_dir = cls_dir / "images"
        xmls = sorted(voc_dir.glob("*.xml"))
        for xp in xmls:
            fname, w, h, boxes = read_voc_annotation(xp)
            # fallback: if filename missing in XML, infer from xml stem
            if not fname:
                # many VOC writers name the xml the same as the image id
                # try any image extension
                candidates = list(img_dir.glob(f"{xp.stem}.*"))
                if candidates:
                    img_path = candidates[0]
                else:
                    continue
            else:
                img_path = img_dir / fname
                if not img_path.exists():
                    # try any extension with the same stem
                    candidates = list(img_dir.glob(f"{Path(fname).stem}.*"))
                    if candidates:
                        img_path = candidates[0]
                    else:
                        continue

            if not img_path.exists():
                continue

            W, H = ensure_size(img_path, w, h)
            oi_id = image_id_from_filename(img_path.name)
            # Filter tiny boxes now that we know W,H
            final_boxes = []
            for b in boxes:
                bw = max(0.0, b["xmax"] - b["xmin"])
                bh = max(0.0, b["ymax"] - b["ymin"])
                if bw * bh >= args.min_box_pixels and bw > 0 and bh > 0:
                    final_boxes.append(b)
            if not final_boxes:
                continue

            records.append({
                "img_path": img_path,
                "ann_path": xp,
                "width": W, "height": H,
                "boxes": final_boxes,
                "oi_id": oi_id,
            })

    # Train/val split by image (not by box)
    all_imgs = sorted({r["img_path"] for r in records})
    random.shuffle(all_imgs)
    n_val = int(len(all_imgs) * args.val_frac)
    val_set = set(all_imgs[:n_val])
    train_set = set(all_imgs[n_val:])

    # Build COCO dicts
    def coco_skeleton(desc):
        return {
            "info": {"year": 2025, "version": "1.0", "description": desc},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "face", "supercategory": "person"}],
        }

    coco_train = coco_skeleton("Faces (Open Images → VOC → COCO), single-class = face")
    coco_val = coco_skeleton("Faces (Open Images → VOC → COCO), single-class = face (validation)")

    img_id_map = {}   # path -> coco image id (train)
    img_id_map_v = {} # path -> coco image id (val)
    ann_id = 1
    ann_id_v = 1

    def add_record(rec, coco_dict, img_id_map_local, ann_counter):
        path = rec["img_path"]
        if path not in img_id_map_local:
            new_id = len(img_id_map_local) + 1
            img_id_map_local[path] = new_id
            coco_dict["images"].append({
                "id": new_id,
                "file_name": str(path),  # keep absolute path; or make it relative if you prefer
                "width": rec["width"],
                "height": rec["height"],
            })
        img_id = img_id_map_local[path]
        for b in rec["boxes"]:
            xmin = max(0.0, b["xmin"])
            ymin = max(0.0, b["ymin"])
            bw = max(0.0, b["xmax"] - b["xmin"])
            bh = max(0.0, b["ymax"] - b["ymin"])
            coco_dict["annotations"].append({
                "id": ann_counter,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_counter += 1
        return ann_counter

    for rec in records:
        if rec["img_path"] in val_set:
            ann_id_v = add_record(rec, coco_val, img_id_map_v, ann_id_v)
        else:
            ann_id = add_record(rec, coco_train, img_id_map, ann_id)

    # Write COCO JSONs
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    with (out_dir / "annotations" / "instances_train.json").open("w") as f:
        json.dump(coco_train, f)
    with (out_dir / "annotations" / "instances_val.json").open("w") as f:
        json.dump(coco_val, f)

    # Attribution: join image IDs against OI metadata CSVs
    meta = load_meta_tables(Path(args.meta_dir))
    def attribution_rows(img_paths):
        rows = []
        for p in img_paths:
            oid = image_id_from_filename(Path(p).name)
            m = meta.get(oid, {})
            rows.append({
                "ImageID": oid,
                "File": str(p),
                "OriginalURL": m.get("OriginalURL", ""),
                "OriginalLandingURL": m.get("OriginalLandingURL", ""),
                "AuthorProfileURL": m.get("AuthorProfileURL", ""),
                "License": m.get("License", ""),
            })
        return rows

    train_rows = attribution_rows(img_id_map.keys())
    val_rows = attribution_rows(img_id_map_v.keys())

    # Write attribution CSVs
    with (out_dir / "ATTRIBUTION_train.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ImageID","File","OriginalURL","OriginalLandingURL","AuthorProfileURL","License"])
        writer.writeheader()
        writer.writerows(train_rows)
    with (out_dir / "ATTRIBUTION_val.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ImageID","File","OriginalURL","OriginalLandingURL","AuthorProfileURL","License"])
        writer.writeheader()
        writer.writerows(val_rows)

    print(f"Wrote COCO to: {out_dir/'annotations'}")
    print(f"Train images: {len(img_id_map)} | Val images: {len(img_id_map_v)}")
    print(f"Attribution CSVs: {out_dir/'ATTRIBUTION_train.csv'}, {out_dir/'ATTRIBUTION_val.csv'}")

if __name__ == "__main__":
    main()

