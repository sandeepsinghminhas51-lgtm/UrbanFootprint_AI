
from pathlib import Path

import numpy as np
from PIL import Image

# Locate the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define where our test image will be saved
IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "images"
    / "test_image.png"
)

# Create the directory if it does not exist
IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create a small RGB image: width = 4, height = 3
image = Image.new("RGB", (4, 3))

# Define 12 pixel colours
pixels = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255),
    (0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 128, 128), (255, 128, 0), (0, 128, 0), (128, 0, 128)
]

# Insert the pixel colours
image.putdata(pixels)

# Save the image
image.save(IMAGE_PATH)

# Convert image into a NumPy array
image_array = np.array(image)

# Display image information
print("===== URBANFOOTPRINT AI =====")
print("Image saved at:", IMAGE_PATH)
print("Image dimensions:", image.size)
print("Image array shape:", image_array.shape)
print("Data type:", image_array.dtype)

print(
    "Total pixels:",
    image_array.shape[0] * image_array.shape[1]
)

print("First pixel RGB values:", image_array[0, 0])
