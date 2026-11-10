
from pathlib import Path
import platform
import sys
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]

print("\n===== URBANFOOTPRINT AI: MODEL CHECK =====")

print("Python:", sys.version)
print("Platform:", platform.platform())

try:
    import torch

    print("\nPyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

except ImportError:
    print("PyTorch is not installed correctly.")

print("\n--- PROCESSED DATA CHECK ---")

required_files = [
    PROJECT_ROOT / "data/processed/images/satellite_rgb.tif",
    PROJECT_ROOT / "data/processed/masks/building_mask.tif",
    PROJECT_ROOT / "data/processed/previews/building_overlay.png",
]

for path in required_files:
    print(
        ("FOUND: " if path.exists() else "MISSING: "),
        path.relative_to(PROJECT_ROOT)
    )

print("\n--- OPTIONAL PACKAGES ---")

for package in ["rasterio", "huggingface_hub", "segmentation_models_pytorch"]:
    print(
        package,
        "INSTALLED" if importlib.util.find_spec(package) else "NOT INSTALLED"
    )

print("\n===== CHECK COMPLETED =====")
