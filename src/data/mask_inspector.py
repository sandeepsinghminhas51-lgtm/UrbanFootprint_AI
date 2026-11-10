
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the output location
MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "labels"
    / "building_mask.png"
)

# Ensure the output folder exists
MASK_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create a 256 x 256 black mask
mask = Image.new("L", (256, 256), 0)

# Draw building footprints in white
draw = ImageDraw.Draw(mask)

# Building 1: rectangle
draw.rectangle((30, 30, 100, 90), fill=255)

# Building 2: rectangle
draw.rectangle((140, 40, 220, 110), fill=255)

# Building 3: polygon
draw.polygon(
    [(60, 150), (130, 140), (150, 210), (75, 220)],
    fill=255
)

# Save the mask
mask.save(MASK_PATH)

# Convert mask into a NumPy array
mask_array = np.array(mask)

# Convert pixel values into binary class labels
binary_labels = (mask_array > 0).astype(np.uint8)

# Count background and building pixels
values, counts = np.unique(
    binary_labels,
    return_counts=True
)

print("===== BUILDING MASK INSPECTOR =====")
print("Mask saved at:", MASK_PATH)
print("Mask dimensions:", mask.size)
print("Array shape:", mask_array.shape)
print("Unique saved pixel values:", np.unique(mask_array))
print("Binary class labels:", values)
print("Pixel counts:", dict(zip(values.tolist(), counts.tolist())))
print("Total pixels:", binary_labels.size)
