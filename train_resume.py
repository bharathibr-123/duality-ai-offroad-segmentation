"""
train_resume.py — Deep Thinkers | Offroad Semantic Segmentation
Resume training from a saved checkpoint with:
  - Lower LR fine-tuning
  - Aggressive augmentation
  - Weighted Focal + Dice loss
  - Per-class IoU tracking
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
TRAIN_IMG_DIR  = r"C:\Users\Dell\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
TRAIN_MASK_DIR = r"C:\Users\Dell\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
VAL_IMG_DIR    = r"C:\Users\Dell\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images"
VAL_MASK_DIR   = r"C:\Users\Dell\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation"

MODEL_PATH  = r"C:\Users\Dell\Desktop\Hackathon\runs\best_model.pth"
SAVE_DIR    = r"C:\Users\Dell\Desktop\Hackathon\runs"

IMAGE_SIZE  = 384
BATCH_SIZE  = 2
NUM_EPOCHS  = 20           # Additional fine-tuning epochs
LR          = 5e-5         # Lower LR for fine-tuning
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = torch.cuda.is_available()

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}  |  AMP: {USE_AMP}")

# ─────────────────────────────────────────
#  CLASS MAPPING
# ─────────────────────────────────────────
CLASS_MAP = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}
NUM_CLASSES = 11

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers", "Logs",
    "Rocks", "Landscape", "Sky"
]

CLASS_WEIGHTS = torch.tensor([
    0.5, 1.5, 1.5, 1.2, 1.5, 2.5, 5.0, 5.0, 2.5, 1.0, 1.0
], dtype=torch.float32).to(DEVICE)


def map_mask(mask):
    out = np.zeros_like(mask, dtype=np.uint8)
    for raw_id, idx in CLASS_MAP.items():
        out[mask == raw_id] = idx
    return out


# ─────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────
class DesertDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = sorted([
            f for f in os.listdir(img_dir) if f.lower().endswith(".png")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        image     = cv2.imread(os.path.join(self.img_dir,  img_name))
        image     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask      = cv2.imread(os.path.join(self.mask_dir, img_name), cv2.IMREAD_UNCHANGED)
        mask      = map_mask(mask.astype(np.int32))
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]
        return image, mask.long()


# ─────────────────────────────────────────
#  AUGMENTATIONS — aggressive for fine-tuning
# ─────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.15),
    A.RandomRotate90(p=0.25),
    A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.15, rotate_limit=20, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(0.35, 0.35, p=1.0),
        A.HueSaturationValue(20, 40, 25, p=1.0),
        A.CLAHE(clip_limit=5.0, p=1.0),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 60), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
    ], p=0.35),
    A.CoarseDropout(max_holes=6, max_height=40, max_width=40, p=0.25),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────
train_dataset = DesertDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
val_dataset   = DesertDataset(VAL_IMG_DIR,   VAL_MASK_DIR,   val_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
model = smp.DeepLabV3Plus(
    encoder_name    = "efficientnet-b4",
    encoder_weights = None,
    in_channels     = 3,
    classes         = NUM_CLASSES,
    encoder_output_stride = 16,
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
start_epoch = 0
prev_best   = 0.0

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    prev_best   = checkpoint.get("best_iou", 0.0)
    print(f"Resumed from epoch {start_epoch}  |  Previous best IoU: {prev_best:.4f}")
else:
    model.load_state_dict(checkpoint)
    print("Loaded raw state dict")

model = model.to(DEVICE)


# ─────────────────────────────────────────
#  LOSS & OPTIMIZER
# ─────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, ignore_index=255, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        probs   = F.softmax(inputs, dim=1)
        valid   = (targets != 255)
        t       = targets.clone(); t[~valid] = 0
        one_hot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
        mask    = valid.unsqueeze(1).float()
        probs   = probs * mask; one_hot = one_hot * mask
        inter   = (probs * one_hot).sum(dim=(0, 2, 3))
        union   = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        return 1 - ((2 * inter + 1) / (union + 1)).mean()


focal     = FocalLoss(gamma=2.0, weight=CLASS_WEIGHTS)
dice      = DiceLoss()
criterion = lambda inp, tgt: 0.6 * focal(inp, tgt) + 0.4 * dice(inp, tgt)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)


# ─────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────
def compute_iou(preds, labels):
    iou_list = []
    preds  = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    for cls in range(NUM_CLASSES):
        inter = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or (preds == cls, labels == cls).sum()
        if union > 0:
            iou_list.append(inter / union)
    return np.mean(iou_list) if iou_list else 0.0


def compute_per_class_iou(preds, labels):
    preds  = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    result = {}
    for cls in range(NUM_CLASSES):
        inter = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or (preds == cls, labels == cls).sum()
        result[CLASS_NAMES[cls]] = round(inter / union, 4) if union > 0 else None
    return result


# ─────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────
train_losses, val_losses, val_ious = [], [], []
best_iou = prev_best

print(f"\n Resuming training from epoch {start_epoch + 1}  (best so far: {best_iou:.4f})\n")

for epoch in range(NUM_EPOCHS):
    ep_num = start_epoch + epoch + 1

    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {ep_num} [Train]"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(images)
            loss    = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss  = 0.0
    all_preds, all_masks = [], []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {ep_num} [Val]"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                loss    = criterion(outputs, masks)
            val_loss += loss.item()
            all_preds.append(torch.argmax(outputs, dim=1))
            all_masks.append(masks)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    iou       = compute_iou(all_preds, all_masks)
    val_ious.append(iou)
    scheduler.step()

    print(f"\nEpoch {ep_num}")
    print(f"  Train Loss : {avg_train_loss:.4f}")
    print(f"  Val   Loss : {avg_val_loss:.4f}")
    print(f"  Val   IoU  : {iou:.4f}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

    if iou > best_iou:
        best_iou = iou
        torch.save({
            'epoch': ep_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
        }, os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"  ✓ New best model saved! IoU = {best_iou:.4f}")

    if (epoch + 1) % 5 == 0:
        per_class = compute_per_class_iou(all_preds, all_masks)
        print("  Per-Class IoU:")
        for name, val in per_class.items():
            bar = "█" * int((val or 0) * 20)
            print(f"    {name:20s}: {val if val is not None else 'N/A':>6}  {bar}")


# ─────────────────────────────────────────
#  SAVE PER-CLASS IoU CHART
# ─────────────────────────────────────────
per_class = compute_per_class_iou(all_preds, all_masks)
classes   = [k for k, v in per_class.items() if v is not None]
values    = [v for v in per_class.values()   if v is not None]

plt.figure(figsize=(14, 5))
colors = ['#2ecc71' if v >= 0.5 else '#e67e22' if v >= 0.3 else '#e74c3c' for v in values]
bars   = plt.bar(classes, values, color=colors, edgecolor='white', linewidth=0.5)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.ylabel('IoU Score')
plt.title(f'Per-Class IoU — Team Deep Thinkers  (mIoU = {np.mean(values):.4f})')
plt.ylim(0, 1.0)
plt.axhline(y=np.mean(values), color='white', linestyle='--', alpha=0.6, label=f'mIoU={np.mean(values):.3f}')
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "per_class_iou.png"), dpi=150)
plt.close()

# ─────────────────────────────────────────
#  LOSS & IoU CURVES
# ─────────────────────────────────────────
epochs_range = range(start_epoch + 1, start_epoch + NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_range, train_losses, label="Train Loss", color="#00d4ff")
axes[0].plot(epochs_range, val_losses,   label="Val Loss",   color="#7c3aed")
axes[0].set_title("Loss Curve (Resume)"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(epochs_range, val_ious, label="Val mIoU", color="#10b981")
axes[1].axhline(y=best_iou, color='#f59e0b', linestyle='--', label=f'Best={best_iou:.4f}')
axes[1].set_title("IoU Curve (Resume)"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_graphs_resume.png"), dpi=150)
plt.close()

print(f"\n Resume Training Complete!  Best mIoU = {best_iou:.4f}")
