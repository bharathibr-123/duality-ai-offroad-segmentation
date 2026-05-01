import os
import cv2
import numpy as np

PRED_DIR = r"C:\Users\Dell\Desktop\Hackathon\runs\predictions"

# Colors in our prediction images (right half)
CLASS_COLORS = {
    "Trees":         ([0, 100, 0],   [50, 180, 50]),
    "Lush Bushes":   ([0, 200, 0],   [50, 255, 50]),
    "Dry Grass":     ([200, 200, 0], [255, 255, 50]),
    "Dry Bushes":    ([100, 40, 0],  [180, 100, 50]),
    "Ground Clutter":([100, 100, 100],[180, 180, 180]),
    "Flowers":       ([200, 0, 200], [255, 50, 255]),
    "Logs":          ([80, 40, 10],  [140, 90, 50]),
    "Rocks":         ([100, 100, 0], [160, 160, 50]),
    "Landscape":     ([180, 140, 80],[255, 210, 170]),
    "Sky":           ([0, 150, 200], [50, 220, 255]),
}

def count_classes(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0, []
    
    # Take RIGHT half only (prediction side)
    h, w = img.shape[:2]
    pred = img[:, w//2:, :]
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    
    found = []
    for cls_name, (low, high) in CLASS_COLORS.items():
        low  = np.array(low,  dtype=np.uint8)
        high = np.array(high, dtype=np.uint8)
        mask = cv2.inRange(pred_rgb, low, high)
        if mask.sum() > 500:  # at least some pixels
            found.append(cls_name)
    
    return len(found), found

# Check images
print("Scanning predictions for best images...\n")
results = []

files = sorted(os.listdir(PRED_DIR))
# Sample every 10th image for speed
for fname in files[::10]:
    if not fname.endswith('.png'):
        continue
    path = os.path.join(PRED_DIR, fname)
    count, classes = count_classes(path)
    results.append((count, fname, classes))

# Sort by number of classes found
results.sort(reverse=True)

print("TOP 10 IMAGES WITH MOST CLASSES:")
print("="*60)
for count, fname, classes in results[:10]:
    print(f"\n{fname} — {count} classes found:")
    for c in classes:
        print(f"  ✅ {c}")

print("\n\nBEST 6 IMAGE NAMES TO USE IN DASHBOARD:")
print("="*60)
for count, fname, classes in results[:6]:
    print(fname)
