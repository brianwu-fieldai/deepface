# download_oi.py
from pathlib import Path
from openimages.download import download_dataset

DEST_DIR = "/home/ubuntu/brianwu/data/openimages/faces"   # images + annotations will be put under here
META_DIR = "/home/ubuntu/brianwu/data/openimages/meta"    # cache for Open Images CSVs (created if missing)
LABELS   = ["Human face", "Human head"]                   # Open Images class names (case-sensitive!)

Path(DEST_DIR).mkdir(parents=True, exist_ok=True)
Path(META_DIR).mkdir(parents=True, exist_ok=True)

result = download_dataset(
    dest_dir=DEST_DIR,
    class_labels=LABELS,
    annotation_format="pascal",  # or "darknet" for YOLO .txt
    exclusions_path=None,        # or path to a file of image IDs to skip
    limit=None                   # or an int while testing, e.g. 5000
)

print("Done. Wrote to:")
for cls, paths in result.items():
    print(f"  {cls}: images={paths['images_dir']}  annotations={paths['annotations_dir']}")

