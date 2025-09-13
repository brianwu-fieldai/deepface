import os, json, math, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead
import matplotlib.pyplot as plt, matplotlib.patches as patches


class COCOSingleClassFaces(Dataset):
    """COCO reader for single 'face' class (category_id=1). RetinaNet expects 0-based labels."""
    def __init__(self, ann_file, image_base=None, transform=None, limit=None, seed=42):
        with open(ann_file, "r") as f:
            data = json.load(f)
        self.images = data["images"]
        self.annos = data["annotations"]
        self.transform = transform
        self.image_base = image_base

        ann_by_img = {}
        for a in self.annos:
            if a.get("category_id") != 1:
                continue
            ann_by_img.setdefault(a["image_id"], []).append(a)
        self.ann_by_img = ann_by_img

        if isinstance(limit, int) and limit < len(self.images):
            rng = random.Random(seed)
            rng.shuffle(self.images)
            self.images = self.images[:limit]

    def __len__(self):
        return len(self.images)

    def _resolve(self, name):
        p = Path(name)
        if p.is_absolute():
            return p
        return Path(self.image_base) / name if self.image_base else p

    def __getitem__(self, idx):
        rec = self.images[idx]
        img_path = self._resolve(rec["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.ann_by_img.get(rec["id"], [])
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(0)  # RetinaNet: single class -> label 0

        boxes_t  = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64).reshape(-1,)      if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,                      # int64, 1D, values in {0}
            "image_id": torch.tensor([rec["id"]], dtype=torch.int64),
            "path": str(img_path),
        }

        if self.transform:
            img = self.transform(img)
        return img, target
    
    
class RandomShortestSideResize:
    def __init__(self, min_short=640, max_short=1024): 
        self.min_short, self.max_short = min_short, max_short
    def __call__(self, img):
        short = random.randint(self.min_short, self.max_short)
        w,h = img.size
        scale = short/min(w,h)
        new_size = (int(h*scale), int(w*scale))
        return F.resize(img, new_size)
    
    
def make_warmup_cosine(optimizer, base_lr, total_epochs, warmup_epochs=1, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs: return (epoch+1)/warmup_epochs
        prog = (epoch-warmup_epochs)/(total_epochs-warmup_epochs)
        return (min_lr/base_lr)+0.5*(1-(min_lr/base_lr))*(1+math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay=decay
        self.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def apply_to(self, model): model.load_state_dict(self.shadow, strict=False)
    



def build_retinanet(device="cuda", num_classes=1, short_side=896, long_side=1536):
    # base model
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes=num_classes)

    # --- IMPORTANT: same anchors-per-location on all levels ---
    # Use ONE size per level (constant num_anchors=1)
    anchor_gen = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),   # P3..P7
        aspect_ratios=((1.0,),) * 5                    # faces ~square
    )
    model.anchor_generator = anchor_gen

    # Rebuild head to match new num_anchors
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]  # = 1
    out_channels = model.backbone.out_channels  # 256 for FPN
    model.head = RetinaNetHead(out_channels, num_anchors, num_classes)

    # input scaling
    model.transform.min_size = (short_side,)
    model.transform.max_size = long_side

    # classification prior bias (stabilize early training)
    import math
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("classification_head.cls_logits.bias"):
                pi = 0.01
                p.fill_(-math.log((1 - pi) / pi))

    return model.to(device)


def train_one_epoch(model, loader, optimizer, device="cuda", ema=None, print_every=50):
    model.train(); running=0
    for i,(imgs,tgts) in enumerate(loader):
        imgs=[im.to(device) for im in imgs]
        tgts=[{k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in t.items()} for t in tgts]
        loss_dict=model(imgs,tgts); loss=sum(loss_dict.values())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if ema: ema.update(model)
        running+=loss.item()
        if (i+1)%print_every==0:
            print(f"[{i+1}/{len(loader)}] loss={running/print_every:.4f}")
            running=0


# --- Checkpointing ---
def save_checkpoint(model, optimizer, scheduler, epoch, ema=None, path_prefix="checkpoint"):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if ema:
        state["ema"] = ema.shadow
    torch.save(state, f"{path_prefix}_ep{epoch}.pt")
    print(f"Checkpoint saved: {path_prefix}_ep{epoch}.pt")


# Paths
TRAIN_JSON="/home/ubuntu/brianwu/data/openimages/faces_coco/annotations/instances_train.json"
VAL_JSON="/home/ubuntu/brianwu/data/openimages/faces_coco/annotations/instances_val.json"
IMAGE_BASE="/home/ubuntu/brianwu/data/openimages/faces"
DEVICE="cuda"

# Datasets
train_tf=transforms.Compose([RandomShortestSideResize(640,1024), transforms.ToTensor()])
val_tf=transforms.Compose([transforms.ToTensor()])
train_ds=COCOSingleClassFaces(TRAIN_JSON, IMAGE_BASE, transform=train_tf)
val_ds=COCOSingleClassFaces(VAL_JSON, IMAGE_BASE, transform=val_tf)

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

print(f"Train={len(train_ds)} | Val={len(val_ds)}")

# Model/optim
model=build_retinanet(DEVICE)
optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=5e-4,momentum=0.9,weight_decay=1e-4)
lr_sch=make_warmup_cosine(optimizer,5e-4,total_epochs=12)
ema=EMA(model,decay=0.999)

def sanity_check(loader, num_classes=1, device=DEVICE):
    model.train()
    imgs, tgts = next(iter(loader))
    # label & box integrity
    for t in tgts:
        assert t["labels"].dtype == torch.int64
        assert t["labels"].ndim == 1
        if t["labels"].numel() > 0:
            assert int(t["labels"].min()) >= 0 and int(t["labels"].max()) < num_classes
        assert t["boxes"].ndim == 2 and t["boxes"].shape[1] == 4
    imgs = [im.to(device) for im in imgs]
    tgts = [{k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in t.items()} for t in tgts]
    with torch.autograd.set_detect_anomaly(True):
        losses = model(imgs, tgts)
    print({k: float(v) for k,v in losses.items()})

sanity_check(train_loader, num_classes=1, device=DEVICE)


EPOCHS=3
for ep in range(EPOCHS):
    print(f"Epoch {ep+1}/{EPOCHS}")
    train_one_epoch(model,train_loader,optimizer,device=DEVICE,ema=ema)
    lr_sch.step()
    save_checkpoint(model, optimizer, lr_sch, ep+1, ema=ema, path_prefix="retinanet_face")


ema.apply_to(model)  # use EMA weights
torch.save({"model":model.state_dict()},"retinanet_face.pt")
print("Saved to retinanet_face.pt")


def retinaface_like_json(image_path,pred,thr=0.7):
    keep=pred["scores"]>=thr
    boxes=pred["boxes"][keep].cpu().numpy(); scores=pred["scores"][keep].cpu().numpy()
    faces={}
    for i,(b,s) in enumerate(zip(boxes,scores),1):
        x1,y1,x2,y2=b
        faces[f"face_{i}"]={"score":float(s),"facial_area":[x1,y1,x2,y2],"landmarks":None}
    return {"image_path":image_path,"predictions":faces}

model.eval()
with torch.no_grad():
    for i in range(2):
        img,tgt=val_ds[i]
        pred=model([img.to(DEVICE)])[0]
        js=retinaface_like_json(tgt["path"],pred,thr=0.7)
        print(json.dumps(js,indent=2)[:600])
        

def visualize_predictions(model, dataset, num_images=3, thr=0.7):
    fig, axs = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    if num_images == 1: axs = [axs]
    model.eval()
    correct, failure = [], []
    with torch.no_grad():
        for i in range(num_images):
            img, tgt = dataset[i]
            out = model([img.to(DEVICE)])[0]
            keep = out["scores"] >= thr
            boxes = out["boxes"][keep].cpu().numpy()
            gt_boxes = tgt["boxes"].cpu().numpy()
            img_np = img.permute(1,2,0).numpy()
            ax = axs[i]; ax.imshow(img_np); ax.axis("off")
            # Draw predictions
            for b in boxes:
                x1, y1, x2, y2 = b
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)
            # Draw ground truth
            for b in gt_boxes:
                x1, y1, x2, y2 = b
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="red", facecolor="none", linestyle='dashed')
                ax.add_patch(rect)
            # Simple correct/failure logic: if any pred overlaps any gt (IoU > 0.5), correct; else, failure
            def iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                return iou
            found = False
            for pb in boxes:
                for gb in gt_boxes:
                    if iou(pb, gb) > 0.5:
                        found = True
                        break
                if found: break
            if found:
                correct.append(i)
                plt.imsave(f"correct_pred_{i}.png", img_np)
            else:
                failure.append(i)
                plt.imsave(f"failure_pred_{i}.png", img_np)
    plt.savefig("retinanet_face_predictions.png")
    print(f"Correct predictions saved: {[f'correct_pred_{i}.png' for i in correct]}")
    print(f"Failure predictions saved: {[f'failure_pred_{i}.png' for i in failure]}")

visualize_predictions(model, val_ds, num_images=3)
