"""
test.py — Deep Thinkers | Offroad Semantic Segmentation
Standard inference (no TTA) — fast single-pass prediction.
For best results use test_tta.py instead.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
TEST_IMG_DIR = r"C:\Users\Dell\Downloads\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
MODEL_PATH   = r"C:\Users\Dell\Desktop\Hackathon\runs\best_model.pth"
OUTPUT_DIR   = r"C:\Users\Dell\Desktop\Hackathon\runs\predictions"

IMAGE_SIZE  = 384
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────
#  CLASS COLORS
# ─────────────────────────────────────────
CLASS_COLORS = np.array([
    [0,   0,   0  ],  # 0  Background
    [0,   128, 0  ],  # 1  Trees
    [0,   255, 0  ],  # 2  Lush Bushes
    [255, 255, 0  ],  # 3  Dry Grass
    [139, 69,  19 ],  # 4  Dry Bushes
    [128, 128, 128],  # 5  Ground Clutter
    [255, 0,   255],  # 6  Flowers
    [101, 67,  33 ],  # 7  Logs
    [128, 128, 0  ],  # 8  Rocks
    [210, 180, 140],  # 9  Landscape
    [0,   191, 255],  # 10 Sky
], dtype=np.uint8)

# ─────────────────────────────────────────
#  TRANSFORM
# ─────────────────────────────────────────
test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

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
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint — epoch {checkpoint.get('epoch','?')}  |  Best IoU: {checkpoint.get('best_iou','?'):.4f}")
else:
    model.load_state_dict(checkpoint)
    print("Model loaded (raw state dict)")

model = model.to(DEVICE)
model.eval()

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def colorize_mask(mask):
    h, w  = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(NUM_CLASSES):
        color[mask == cls_idx] = CLASS_COLORS[cls_idx]
    return color

# ─────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────
test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".png")])
print(f"Found {len(test_images)} test images\n")

for img_name in tqdm(test_images):
    img_path  = os.path.join(TEST_IMG_DIR, img_name)
    image     = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    aug    = test_transform(image=image_rgb)
    tensor = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        pred   = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    color_mask   = colorize_mask(pred_resized)
    side_by_side = np.hstack([image_rgb, color_mask])

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, img_name),
        cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
    )

print(f"\nAll predictions saved to: {OUTPUT_DIR}")
print("Tip: Run test_tta.py for higher-quality predictions with TTA.")
