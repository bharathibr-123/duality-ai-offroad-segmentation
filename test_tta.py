import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

TEST_IMG_DIR = r"C:\Users\Dell\Downloads\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
MODEL_PATH   = r"C:\Users\Dell\Desktop\Hackathon\runs\best_model.pth"
OUTPUT_DIR   = r"C:\Users\Dell\Desktop\Hackathon\runs\predictions_tta"

IMAGE_SIZE  = 256
DEVICE      = torch.device("cpu")
NUM_CLASSES = 11

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_COLORS = [
    (0,0,0),(0,128,0),(0,255,0),(255,255,0),
    (139,69,19),(128,128,128),(255,0,255),
    (101,67,33),(128,128,0),(210,180,140),(0,191,255),
]

# Try mobilenet_v2 first, then efficientnet
try:
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("Loaded with mobilenet_v2!")
except:
    try:
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("Loaded with efficientnet-b0!")
    except:
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("Loaded with Unet mobilenet_v2!")

model = model.to(DEVICE)
model.eval()

normalize = A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))

tta_transforms = [
    A.Compose([A.Resize(IMAGE_SIZE,IMAGE_SIZE), normalize, ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE,IMAGE_SIZE), A.HorizontalFlip(p=1.0), normalize, ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE,IMAGE_SIZE), A.RandomBrightnessContrast(0.2,0.2,p=1.0), normalize, ToTensorV2()]),
]
tta_flips = [False, True, False]

def colorize(mask):
    h,w = mask.shape
    color = np.zeros((h,w,3), dtype=np.uint8)
    for i,c in enumerate(CLASS_COLORS):
        color[mask==i] = c
    return color

def predict_tta(img_rgb):
    probs = []
    for i,t in enumerate(tta_transforms):
        aug = t(image=img_rgb)
        tensor = aug["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = torch.softmax(model(tensor), dim=1)
        if tta_flips[i]:
            out = torch.flip(out, dims=[3])
        probs.append(out.cpu().numpy())
    return np.argmax(np.mean(probs, axis=0), axis=1).squeeze(0)

test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')])
print(f"Found {len(test_images)} test images")
print("Running TTA predictions...")

for img_name in tqdm(test_images):
    img = cv2.imread(os.path.join(TEST_IMG_DIR, img_name))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w = img_rgb.shape[:2]
    pred = predict_tta(img_rgb)
    pred_resized = cv2.resize(pred.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    color = colorize(pred_resized)
    out = np.hstack([img_rgb, color])
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

print(f"\nTTA Done! Saved to: {OUTPUT_DIR}")
