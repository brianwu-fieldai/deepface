# configs/face/yolox_l_face.py
# Keep the official YOLOX-L base (with MultiImageMixDataset + Mosaic/MixUp)
_base_ = '/home/ubuntu/brianwu/data/openimages/mm-yolox/yolox_l/yolox_l_8x8_300e_coco.py'

# ----- single class -----
metainfo = dict(classes=('face',))

model = dict(
    bbox_head=dict(num_classes=1),
)

# ----- data -----
data_root = '/home/ubuntu/brianwu/data/openimages/'
train_ann = 'faces_coco/annotations/instances_train.json'
val_ann   = 'faces_coco/annotations/instances_val.json'

# IMPORTANT: For YOLOX we keep the base's MultiImageMixDataset wrapper.
# We only override the *inner* COCO dataset paths + metainfo.
train_dataloader = dict(
    batch_size=12,                 # adjust if you OOM on A10G
    num_workers=4,
    persistent_workers=True,
    dataset=dict(                  # <- this remains the MultiImageMixDataset from base
        dataset=dict(              # <- inner COCO dataset we override
            type='CocoDataset',
            data_root=data_root,
            ann_file=train_ann,
            data_prefix=dict(img='faces/'),  # resolves to /openimages/faces/<file_name>
            metainfo=metainfo,
            # keep base filter_cfg/pipeline unless you want to override them too
        )
    )
)

# Val/test are plain COCO datasets; just point to your JSON + prefix
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img='faces/'),
        metainfo=metainfo,
        test_mode=True,
    )
)

test_dataloader = val_dataloader

# ----- evaluators -----
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann,
    metric='bbox',
)
test_evaluator = val_evaluator

# ----- schedule tweaks -----
max_epochs = 120
train_cfg = dict(max_epochs=max_epochs)

# Auto-scale LR if total batch size differs from the base (64)
auto_scale_lr = dict(enable=True, base_batch_size=64)

# Keep EMA and switch off Mosaic in the last ~10% of epochs
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49
    ),
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=int(max_epochs * 0.1),
        priority=48
    ),
]

