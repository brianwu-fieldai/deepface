#!/usr/bin/env python3
# eval_face_retinanet.py
# Eval RetinaNet face detector on a COCO val set.
# Requires: torch, torchvision, numpy, tqdm, pillow, pycocotools

import os, json, math, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Dataset (val-only use)
# ----------------------------
class COCOSingleClassFaces(Dataset):
    """COCO reader for single 'face' class (category_id=1)."""
    def __init__(self, ann_file, image_base=None, transform=None, limit=None, seed=42):
        with open(ann_file, "r") as f:
            data = json.load(f)
        self.images = data["images"]
        self.annos = data["annotations"]
        self.transform = transform
        self.image_base = image_base

        # index GT by image for simple PR/F1 calc
        ann_by_img = {}
        for a in self.annos:
            if a.get("category_id") != 1:
                continue
            ann_by_img.setdefault(a["image_id"], []).append(a)
        self.ann_by_img = ann_by_img

        if isinstance(limit, int) and limit < len(self.images):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(self.images), size=limit, replace=False)
            self.images = [self.images[i] for i in idx]

    def __len__(self): return len(self.images)

    def _resolve(self, name):
        p = Path(name)
        if p.is_absolute(): return p
        return Path(self.image_base)/name if self.image_base else p

    def __getitem__(self, idx):
        rec = self.images[idx]
        img_path = self._resolve(rec["file_name"])
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        target = {
            "image_id": torch.tensor([rec["id"]], dtype=torch.int64),
            "path": str(img_path),
        }
        return img, target

def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

# ----------------------------
# Model (RetinaNet) — MUST match training anchors & scales
# ----------------------------
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead

def build_retinanet(device="cuda", num_classes=1, short_side=896, long_side=1536):
    # base model
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes=num_classes)

    # IMPORTANT: keep the same anchors-per-location across FPN levels as in training
    # (we used 1 anchor per level with square ratio)
    anchor_gen = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),   # P3..P7
        aspect_ratios=((1.0,),) * 5
    )
    model.anchor_generator = anchor_gen

    # Rebuild head to match anchor count
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]  # = 1
    out_channels = model.backbone.out_channels
    model.head = RetinaNetHead(out_channels, num_anchors, num_classes)

    # Input scaling (should match training)
    model.transform.min_size = (short_side,)
    model.transform.max_size = long_side
    return model.to(device)

# ----------------------------
# Post-processing helpers
# ----------------------------
from torchvision.ops import nms

def postprocess_single(boxes, scores, score_thr=0.75, nms_iou=0.4, min_sz=8):
    keep = scores >= score_thr
    boxes, scores = boxes[keep], scores[keep]
    if min_sz > 0 and boxes.numel() > 0:
        w = boxes[:,2]-boxes[:,0]; h = boxes[:,3]-boxes[:,1]
        k2 = (w >= min_sz) & (h >= min_sz)
        boxes, scores = boxes[k2], scores[k2]
    if boxes.numel() > 0:
        idx = nms(boxes, scores, nms_iou)
        boxes, scores = boxes[idx], scores[idx]
    return boxes, scores

def retinaface_like(image_path, boxes, scores):
    faces = {}
    for i, (b, s) in enumerate(zip(boxes.tolist(), scores.tolist()), 1):
        x1,y1,x2,y2 = b
        faces[f"face_{i}"] = {
            "score": float(s),
            "facial_area": [float(x1), float(y1), float(x2), float(y2)],
            "landmarks": None  # we didn't train landmarks
        }
    return {"image_path": image_path, "predictions": faces}

# ----------------------------
# Simple PR/F1 at IoU=0.5
# ----------------------------
def iou_np(a, b):  # a,b: xyxy arrays
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    xx1 = np.maximum(a[:,None,0], b[None,:,0])
    yy1 = np.maximum(a[:,None,1], b[None,:,1])
    xx2 = np.minimum(a[:,None,2], b[None,:,2])
    yy2 = np.minimum(a[:,None,3], b[None,:,3])
    inter = np.clip(xx2-xx1, 0, None) * np.clip(yy2-yy1, 0, None)
    area_a = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    union = area_a[:,None] + area_b[None,:] - inter
    return inter / np.clip(union, 1e-6, None)

def greedy_match_PR(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    ious = iou_np(pred_boxes, gt_boxes)
    order = np.argsort(-np.asarray(pred_scores))
    matched = set()
    TP = FP = 0
    for i in order:
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and j not in matched:
            TP += 1; matched.add(j)
        else:
            FP += 1
    FN = len(gt_boxes) - len(matched)
    return TP, FP, FN

# ----------------------------
# COCO eval
# ----------------------------
def run_coco_eval(val_json, coco_dets):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print("pycocotools not installed; skipping COCO eval.", e)
        return None
    coco_gt = COCO(val_json)
    coco_dt = coco_gt.loadRes(coco_dets)
    ce = COCOeval(coco_gt, coco_dt, iouType='bbox')
    ce.evaluate(); ce.accumulate(); ce.summarize()
    return {
        "AP": float(ce.stats[0]),
        "AP50": float(ce.stats[1]),
        "AP75": float(ce.stats[2]),
        "AP_small": float(ce.stats[3]),
        "AP_medium": float(ce.stats[4]),
        "AP_large": float(ce.stats[5]),
        "AR_1": float(ce.stats[6]),
        "AR_10": float(ce.stats[7]),
        "AR_100": float(ce.stats[8]),
        "AR_small": float(ce.stats[9]),
        "AR_medium": float(ce.stats[10]),
        "AR_large": float(ce.stats[11]),
    }

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pt file saved as {'model': state_dict}")
    ap.add_argument("--val_json", required=True, help="COCO instances_val.json")
    ap.add_argument("--image_base_dir", required=True, help="Root to resolve file_name if relative")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--short_side", type=int, default=896)
    ap.add_argument("--long_side", type=int, default=1536)
    ap.add_argument("--score_thr", type=float, default=0.75)
    ap.add_argument("--nms_iou", type=float, default=0.4)
    ap.add_argument("--min_size_px", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate on a subset of images")
    ap.add_argument("--save_coco_dets", default=None, help="Optional: path to save COCO detections JSON")
    ap.add_argument("--save_retinaface_jsonl", default=None, help="Optional: path to save RetinaFace-like JSONL")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # Dataset & Loader
    val_tf = transforms.Compose([transforms.ToTensor()])
    val_ds = COCOSingleClassFaces(args.val_json, args.image_base_dir, transform=val_tf, limit=args.limit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Build model (must match training anchors & transforms), then load weights
    model = build_retinanet(device=device, num_classes=1, short_side=args.short_side, long_side=args.long_side)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded checkpoint. Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

    model.eval()

    # Prepare GT index for simple PR/F1
    with open(args.val_json, "r") as f:
        vdata = json.load(f)
    gt_by_img = defaultdict(list)
    for ann in vdata["annotations"]:
        if ann.get("category_id") != 1: continue
        x,y,w,h = ann["bbox"]
        gt_by_img[ann["image_id"]].append([x, y, x+w, y+h])

    # Inference → build COCO dets + simple PR/F1
    coco_dets = []
    TP = FP = FN = 0
    rf_out = None
    if args.save_retinaface_jsonl:
        rf_out = open(args.save_retinaface_jsonl, "w")

    with torch.no_grad():
        pbar = tqdm(val_loader, total=len(val_loader), desc="Evaluating")
        for imgs, tgts in pbar:
            img = imgs[0].to(device)
            tgt = tgts[0]
            img_id = int(tgt["image_id"].item())
            path = tgt["path"]

            pred = model([img])[0]
            boxes, scores = postprocess_single(
                pred["boxes"].cpu(), pred["scores"].cpu(),
                score_thr=args.score_thr, nms_iou=args.nms_iou, min_sz=args.min_size_px
            )

            # COCO dets (xyxy -> xywh)
            for b, s in zip(boxes.tolist(), scores.tolist()):
                x1,y1,x2,y2 = b
                coco_dets.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "score": float(s),
                })

            # Simple PR/F1
            TP_i, FP_i, FN_i = greedy_match_PR(boxes.numpy(), scores.numpy(), gt_by_img.get(img_id, []), iou_thr=0.5)
            TP += TP_i; FP += FP_i; FN += FN_i

            # Optional RetinaFace-like JSONL
            if rf_out is not None:
                js = retinaface_like(path, boxes, scores)
                rf_out.write(json.dumps(js) + "\n")

    if rf_out is not None:
        rf_out.close()
        print("Saved RetinaFace-like JSONL to", args.save_retinaface_jsonl)

    # Save COCO detections if requested
    if args.save_coco_dets:
        with open(args.save_coco_dets, "w") as f:
            json.dump(coco_dets, f)
        print("Saved COCO detections to", args.save_coco_dets)

    # Simple metrics
    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    print("\n=== Simple PR/F1 (IoU=0.5, after postproc) ===")
    print(f"Precision: {P:.4f}  Recall: {R:.4f}  F1: {F1:.4f}")
    print(f"TP={TP}  FP={FP}  FN={FN}  Detections={len(coco_dets)}")

    # COCO eval
    coco_metrics = run_coco_eval(args.val_json, coco_dets)
    if coco_metrics is not None:
        print("\n=== COCO Metrics ===")
        for k in ["AP","AP50","AP75","AP_small","AP_medium","AP_large","AR_1","AR_10","AR_100","AR_small","AR_medium","AR_large"]:
            print(f"{k}: {coco_metrics[k]:.4f}")

    # Persist summary
    summary = {
        "postproc": {"score_thr": args.score_thr, "nms_iou": args.nms_iou, "min_size_px": args.min_size_px},
        "simple_PRF1": {"precision": P, "recall": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN, "detections": len(coco_dets)},
        "coco": coco_metrics
    }
    with open("eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote eval_summary.json")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()

