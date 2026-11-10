from __future__ import annotations

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Paths
WEIGHTS_PATH = ROOT_DIR / "src" / "models" / "weights" / "urbanfootprint_unet_multisource.pth"
DATA_DIR = ROOT_DIR / "data"
RAW_SPACENET_DIR = DATA_DIR / "raw" / "spacenet"
SPACENET_IMAGES_DIR = RAW_SPACENET_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
JOBS_DIR = OUTPUTS_DIR / "jobs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"

# Ensure directories exist
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Default Model & Inference settings
DEFAULT_THRESHOLD = 0.75  # Optimal validation threshold identified in evaluation
PATCH_SIZE = 256
OVERLAP = 64
BASE_CHANNELS = 16
DEVICE = "cpu"

# Training / Validation metrics from existing project evaluation
VALIDATION_METRICS = {
    "model_name": "Custom U-Net (Trained from Scratch)",
    "architecture": "U-Net with DoubleConv & Skip Connections (4-level encoder-decoder)",
    "dataset": "SpaceNet 2 AOI 2 Vegas (30cm RGB Satellite Imagery)",
    "weights_file": "urbanfootprint_unet_multisource.pth",
    "optimal_threshold": 0.75,
    "combined_validation": {
        "precision": 0.7843,
        "recall": 0.8448,
        "f1": 0.8134,
        "iou": 0.6855,
        "threshold": 0.75
    },
    "comparison_samples": [
        {"threshold": 0.50, "precision": 0.5525, "recall": 0.9525, "f1": 0.6994, "iou": 0.5377},
        {"threshold": 0.60, "precision": 0.6757, "recall": 0.9105, "f1": 0.7757, "iou": 0.6336},
        {"threshold": 0.70, "precision": 0.7526, "recall": 0.8722, "f1": 0.8080, "iou": 0.6779},
        {"threshold": 0.75, "precision": 0.7843, "recall": 0.8448, "f1": 0.8134, "iou": 0.6855},
        {"threshold": 0.80, "precision": 0.8165, "recall": 0.8090, "f1": 0.8127, "iou": 0.6846},
        {"threshold": 0.85, "precision": 0.8517, "recall": 0.7596, "f1": 0.8030, "iou": 0.6709}
    ],
    "disclaimer": "Metrics represent combined validation performance on held-out SpaceNet 2 Vegas images. Prototype results do not guarantee identical performance on different sensors, regions, or resolutions."
}
