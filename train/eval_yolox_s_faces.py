#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mmengine import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.api_wrappers import COCO, COCOeval


def xyxy_to_xywh(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def draw_image(ax, img_path, preds_xyxy, pred_scores, gt_xyxy=None, title=None):
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img); ax.axis("off")
    # draw predictions (green)
    for (x1, y1, x2, y2), s in zip(preds_xyxy, pred_scores):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 4), f"{s:.2f}", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.4, pad=1), color="white")
    # draw GT (optional, dashed red)
    if gt_xyxy is not None:
        for (x1, y1, x2, y2) in gt_xyxy:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
            ax.add_patch(rect)
    if title:
        ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--score_thr", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="eval only first N images")
    ap.add_argument("--device", default="cuda:0")
    # grid options
    ap.add_argument("--num_grids", type=int, default=3, help="# figures to save")
    ap.add_argument("--grid_rows", type=int, default=2)
    ap.add_argument("--grid_cols", type=int, default=2)
    ap.add_argument("--draw_gt", action="store_true", help="overlay ground-truth boxes")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load config / dataset info (Option A layout) ---
    cfg = Config.fromfile(args.config)
    val_cfg = cfg.get("val_dataloader", cfg.get("test_dataloader"))
    ds = val_cfg["dataset"]
    data_root = ds["data_root"]
    ann_file = ds["ann_file"]
    img_prefix = ds["data_prefix"]["img"]

    ann_path = Path(data_root) / ann_file
    coco_gt = COCO(str(ann_path))
    cat_ids = coco_gt.getCatIds()
    assert len(cat_ids) == 1, f"Expected 1 class, found {cat_ids}"
    face_cat_id = int(cat_ids[0])

    # list of (image_id, path)
    img_id_to_path = []
    for img in coco_gt.dataset["images"]:
        img_path = Path(data_root) / img_prefix / img["file_name"]
        img_id_to_path.append((img["id"], str(img_path)))

    if args.limit and args.limit > 0:
        img_id_to_path = img_id_to_path[: args.limit]

    # --- Init model ---
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # --- Inference loop ---
    preds_for_coco = []
    # cache for visualization: only keep images with at least one prediction
    vis_pool = []  # list of dicts: {img_id, img_path, preds_xyxy, scores, (optional) gt_xyxy}
    print(f"Running inference on {len(img_id_to_path)} images...")
    for img_id, img_path in tqdm(img_id_to_path):
        result = inference_detector(model, img_path)
        inst = result.pred_instances
        bboxes = inst.bboxes.detach().cpu().numpy() if hasattr(inst, "bboxes") else np.zeros((0, 4))
        scores = inst.scores.detach().cpu().numpy() if hasattr(inst, "scores") else np.zeros((0,))
        labels = inst.labels.detach().cpu().numpy() if hasattr(inst, "labels") else np.zeros((0,), dtype=np.int64)

        keep = scores >= args.score_thr
        bboxes, scores = bboxes[keep], scores[keep]

        # COCO results
        for box, sc in zip(bboxes, scores):
            preds_for_coco.append({
                "image_id": int(img_id),
                "category_id": face_cat_id,
                "bbox": [float(box[0]), float(box[1]),
                         float(box[2]-box[0]), float(box[3]-box[1])],
                "score": float(sc),
            })

        # Save for visualization if there is at least 1 prediction
        if len(bboxes) > 0:
            item = dict(img_id=img_id, img_path=img_path,
                        preds_xyxy=bboxes, scores=scores)
            if args.draw_gt:
                # gather GT bboxes for this image
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[face_cat_id], iscrowd=None)
                anns = coco_gt.loadAnns(ann_ids)
                gt_xyxy = []
                for a in anns:
                    x, y, w, h = a["bbox"]
                    gt_xyxy.append([x, y, x + w, y + h])
                item["gt_xyxy"] = np.array(gt_xyxy, dtype=float) if gt_xyxy else None
            vis_pool.append(item)

    # --- Save detections for COCOeval ---
    det_path = out_dir / "detections.json"
    with det_path.open("w") as f:
        json.dump(preds_for_coco, f)
    print(f"Wrote detections to {det_path}")

    # --- COCO evaluation ---
    if len(preds_for_coco) > 0:
        coco_dt = coco_gt.loadRes(str(det_path))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
            "AR_1": coco_eval.stats[6],
            "AR_10": coco_eval.stats[7],
            "AR_100": coco_eval.stats[8],
            "AR_small": coco_eval.stats[9],
            "AR_medium": coco_eval.stats[10],
            "AR_large": coco_eval.stats[11],
        }
        with (out_dir / "metrics.json").open("w") as f:
            json.dump(stats, f, indent=2)
        print("Saved metrics to", out_dir / "metrics.json")
    else:
        print("No detections above threshold; skipping COCOeval.")

    # --- Visualization grids (3 figures × 4 images each) ---
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    total_slots = args.num_grids * args.grid_rows * args.grid_cols

    if len(vis_pool) == 0:
        print("No predicted faces above threshold to visualize.")
        return

    # sample without replacement (or cap to available)
    idxs = list(range(len(vis_pool)))
    random.shuffle(idxs)
    idxs = idxs[: total_slots]

    # pad if fewer than slots (repeat some images)
    while len(idxs) < total_slots:
        idxs.append(idxs[len(idxs) % len(idxs)])

    # build figures
    k = 0
    for g in range(args.num_grids):
        fig, axs = plt.subplots(args.grid_rows, args.grid_cols, figsize=(5*args.grid_cols, 5*args.grid_rows))
        axs = np.array(axs).reshape(args.grid_rows, args.grid_cols)

        for r in range(args.grid_rows):
            for c in range(args.grid_cols):
                item = vis_pool[idxs[k]]
                k += 1
                gt = item.get("gt_xyxy") if args.draw_gt else None
                title = Path(item["img_path"]).name
                draw_image(axs[r, c], item["img_path"], item["preds_xyxy"], item["scores"], gt_xyxy=gt, title=title)

        out_path = grids_dir / f"yolox_s_preds_grid_{g+1:02d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print("Saved", out_path)


if __name__ == "__main__":
    main()

