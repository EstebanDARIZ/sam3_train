# Fine-tuning SAM3 on a Custom Fish Segmentation Dataset

This document describes the changes made to fine-tune SAM3 on a custom fish segmentation dataset
in YOLO polygon format, and how to reproduce the training setup.

---

## Dataset

- **Source**: 29 videos, ~10 frames per video (~215 train / 39 val images)
- **Classes** (7): Squid, Sardine, Ray, Sunfish, Pilot Fish, Shark, Jellyfish
- **Original format**: YOLO segmentation (polygon masks, normalized coordinates)
- **Location**: `/home/esteban-dreau-darizcuren/doctorat/dataset/SAM3/res_sam/finetune_dataset/`

SAM3 requires COCO JSON format. A conversion script was written and the converted dataset
is stored at:
```
/home/esteban-dreau-darizcuren/doctorat/dataset/SAM3/sam3_coco_finetuning/
├── train/
│   ├── images/          (symlinks to original images)
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

---

## New Files

### `scripts/yolo_seg_to_coco.py`

Converts a YOLO segmentation dataset (polygon format) to COCO JSON format compatible with SAM3.

**YOLO polygon format** (one line per object):
```
class_id x1 y1 x2 y2 x3 y3 ...   (normalized coordinates, 0-1)
```

**Usage:**
```bash
python scripts/yolo_seg_to_coco.py \
  --train-images /path/to/train/images \
  --train-labels /path/to/train/labels \
  --val-images   /path/to/val/images \
  --val-labels   /path/to/val/labels \
  --output-dir   /path/to/output \
  --class-names  Squid Sardine Ray Sunfish "Pilot Fish" Shark Jellyfish
```

The script:
- Reads each `.txt` label file alongside its image
- Converts normalized polygon coordinates to absolute pixel coordinates
- Computes bounding boxes and polygon areas (Shoelace formula)
- Writes `_annotations.coco.json` in each split folder (COCO 1-indexed categories)
- Creates symlinks to images so no data is duplicated

### `sam3/train/configs/fish_seg/fish_seg_finetuning.yaml`

Hydra training config for fine-tuning SAM3 on the fish segmentation dataset.

Key settings:
| Parameter | Value | Notes |
|---|---|---|
| `fish_train.num_images` | `null` | Use all images; set to e.g. `10` for few-shot |
| `scratch.enable_segmentation` | `true` | Enables polygon mask loss |
| `scratch.resolution` | `1008` | Native SAM3 resolution (RoPE is hardcoded for this) |
| `trainer.max_epochs` | `40` | |
| `trainer.val_epoch_freq` | `5` | |
| `model.freeze_vision_backbone` | `true` | Required to fit in <8 GB GPU |
| `model.freeze_language_backbone` | `true` | Recommended for small datasets |
| `checkpoint.save_freq` | `5` | Save every 5 epochs |

**Loss functions enabled:**
- `Boxes` (L1 + GIoU)
- `IABCEMdetr` (presence-aware focal classification)
- `Masks` (focal + Dice, weight 200/10)

**To run locally (single GPU, backbone frozen):**
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python sam3/train/train.py \
  -c configs/fish_seg/fish_seg_finetuning.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

**To run on SLURM (recommended, full fine-tuning):**

Set in the config:
```yaml
model:
  freeze_vision_backbone: false
  freeze_language_backbone: false  # or true to save memory

launcher:
  gpus_per_node: 2

submitit:
  use_cluster: True
  partition: <your_partition>
  account: <your_account>
  timeout_hour: 24
```

Then:
```bash
python sam3/train/train.py \
  -c configs/fish_seg/fish_seg_finetuning.yaml \
  --use-cluster 1
```

---

## Code Changes

### `sam3/model/vitdet.py` — `Mlp.forward`

**Problem:** The fused kernel `addmm_act` (in `sam3/perflib/fused.py`) is inference-only
and raises `ValueError("Expected grad to be disabled.")` when called with gradients enabled.
During training, `torch.utils.checkpoint.checkpoint` re-enables gradients inside the ViT
blocks, triggering this error.

**Fix:** Fall back to standard PyTorch ops when gradients are enabled.

```python
# Before
def forward(self, x):
    x = addmm_act(type(self.act), self.fc1, x)
    ...

# After
def forward(self, x):
    if torch.is_grad_enabled():
        # addmm_act is inference-only; use standard ops during training
        x = self.act(self.fc1(x))
    else:
        x = addmm_act(type(self.act), self.fc1, x)
    ...
```

Inference performance is unchanged (fused kernel still used in eval/no_grad).

---

### `sam3/model_builder.py` — `build_sam3_image_model`

**Problem:** SAM3's ViT-L backbone requires ~40+ GB GPU memory to train end-to-end.
Consumer/laptop GPUs (8 GB) cannot fit a full training pass.

**Fix:** Added `freeze_vision_backbone` and `freeze_language_backbone` parameters.
When set to `True`, the corresponding parameters are marked `requires_grad=False`
after checkpoint loading, so no gradients or optimizer states are allocated for them.

```python
# New parameters added to build_sam3_image_model()
def build_sam3_image_model(
    ...
    freeze_vision_backbone=False,
    freeze_language_backbone=False,
):
    ...
    if freeze_vision_backbone:
        for param in model.backbone.vision_backbone.parameters():
            param.requires_grad = False
    if freeze_language_backbone:
        for param in model.backbone.language_backbone.parameters():
            param.requires_grad = False
```

**What gets trained with both backbones frozen:**
- `transformer` encoder + decoder (~100M params)
- `segmentation_head` (pixel decoder + universal segmentation head)
- `input_geometry_encoder`

This is sufficient to adapt SAM3 to a new domain when using a small dataset.

---

## Monitoring

```bash
tensorboard --logdir /home/esteban-dreau-darizcuren/doctorat/experiments/sam3_fish_seg/tensorboard
```

Checkpoints are saved every 5 epochs to:
```
/home/esteban-dreau-darizcuren/doctorat/experiments/sam3_fish_seg/checkpoints/
```

---

## Inference After Fine-tuning

Use the exact category names from the dataset as text prompts:

```python
from sam3 import build_sam3_predictor

predictor = build_sam3_predictor(
    checkpoint="path/to/finetuned_checkpoint.pt",
    device="cuda"
)
predictor.set_image(image)
masks, scores, _ = predictor.predict(text_prompt="Sunfish")
```

Supported prompt names: `Squid`, `Sardine`, `Ray`, `Sunfish`, `Pilot Fish`, `Shark`, `Jellyfish`.
