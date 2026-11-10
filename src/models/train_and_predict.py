
from pathlib import Path
import argparse
import json
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = ROOT / "data/processed/images/satellite_rgb.tif"
MASK_PATH = ROOT / "data/processed/masks/building_mask.tif"

MODEL_DIR = ROOT / "src/models/weights"
OUTPUT_DIR = ROOT / "data/processed/predictions"

CHECKPOINT_PATH = MODEL_DIR / "urbanfootprint_unet.pth"

PREDICTION_PATH = OUTPUT_DIR / "predicted_building_mask.tif"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"

PATCH_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 0.001
THRESHOLD = 0.5
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def get_patch_split(height, width):

    patches = []

    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
            patches.append((y, x))

    if len(patches) < 2:
        raise ValueError(
            "At least two complete 256x256 patches are required. "
            "Use a larger satellite image."
        )

    # Separate deterministic shuffle from augmentation randomness
    rng = random.Random(SEED)
    rng.shuffle(patches)

    split_index = max(
        1,
        min(len(patches) - 1, int(len(patches) * 0.8))
    )

    return patches[:split_index], patches[split_index:]


# ============================================================
# 3. LOAD AND VALIDATE GEOSPATIAL DATA
# ============================================================

def load_data():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Satellite image not found: {IMAGE_PATH}"
        )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"Building mask not found: {MASK_PATH}"
        )

    with rasterio.open(IMAGE_PATH) as image_src, \
         rasterio.open(MASK_PATH) as mask_src:

        # Check image and mask dimensions
        if (
            image_src.width != mask_src.width
            or image_src.height != mask_src.height
        ):
            raise ValueError(
                "Image and mask dimensions do not match."
            )

        # Check geospatial alignment
        if image_src.crs != mask_src.crs:
            raise ValueError(
                "Image and mask CRS do not match."
            )

        if image_src.transform != mask_src.transform:
            raise ValueError(
                "Image and mask transforms do not match."
            )

        if image_src.count < 3:
            raise ValueError(
                "Satellite image must contain at least 3 bands."
            )

        image = image_src.read([1, 2, 3])

        mask = mask_src.read(1)

        image_profile = image_src.profile.copy()

    # Handle invalid image values
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0
    )

    # Ensure image is uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert mask into binary format
    mask = (mask > 0).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        raise ValueError(
            "Ground-truth building mask contains no building pixels."
        )

    return image, mask, image_profile


# ============================================================
# 4. DATASET CLASS
# ============================================================

class BuildingDataset(Dataset):

    def __init__(self, image, mask, patches, augment=False):

        self.image = image
        self.mask = mask
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):

        y, x = self.patches[index]

        image = self.image[
            :,
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ].copy()

        mask = self.mask[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ].copy()

        # Data augmentation
        if self.augment:

            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()

            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()

            rotations = random.randint(0, 3)

            if rotations:
                image = np.rot90(
                    image, rotations, axes=(1, 2)
                ).copy()

                mask = np.rot90(
                    mask, rotations, axes=(0, 1)
                ).copy()

        # Convert image to tensor
        image = torch.from_numpy(
            image.astype(np.float32) / 255.0
        )

        # Convert mask to tensor
        mask = torch.from_numpy(
            mask.astype(np.float32)
        ).unsqueeze(0)

        return image, mask


# ============================================================
# 5. LIGHTWEIGHT U-NET ARCHITECTURE
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )

        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )

        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16, 2, stride=2
        )

        self.dec1 = DoubleConv(32, 16)

        self.output = nn.Conv2d(
            16, 1, kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)

        d3 = self.dec3(
            torch.cat([d3, e3], dim=1)
        )

        d2 = self.up2(d3)

        d2 = self.dec2(
            torch.cat([d2, e2], dim=1)
        )

        d1 = self.up1(d2)

        d1 = self.dec1(
            torch.cat([d1, e1], dim=1)
        )

        return self.output(d1)


# ============================================================
# 6. LOSS FUNCTION
# ============================================================

def segmentation_loss(logits, targets):

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice_loss = 1 - (
        (2 * intersection + 1)
        / (denominator + 1)
    ).mean()

    return bce + dice_loss


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def calculate_metrics(predictions, targets):

    # FIX: Use NumPy boolean arrays, not PyTorch .bool()
    predictions = np.asarray(predictions).astype(bool)
    targets = np.asarray(targets).astype(bool)

    tp = np.logical_and(
        predictions, targets
    ).sum()

    fp = np.logical_and(
        predictions, ~targets
    ).sum()

    fn = np.logical_and(
        ~predictions, targets
    ).sum()

    union = tp + fp + fn

    iou = (
        tp / union
        if union > 0 else 1.0
    )

    denominator = 2 * tp + fp + fn

    dice = (
        2 * tp / denominator
        if denominator > 0 else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall)
    }


# ============================================================
# 8. TRAIN MODEL
# ============================================================

def train_model(image, mask, train_patches, val_patches):

    print("\n========== MODEL TRAINING ==========")

    train_dataset = BuildingDataset(
        image,
        mask,
        train_patches,
        augment=True
    )

    val_dataset = BuildingDataset(
        image,
        mask,
        val_patches,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNet().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = segmentation_loss(
                outputs, masks
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                loss = segmentation_loss(
                    outputs, masks
                )

                val_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "patch_size": PATCH_SIZE,
                    "seed": SEED,
                    "train_patches": train_patches,
                    "val_patches": val_patches
                },
                CHECKPOINT_PATH
            )

    print("\nBest checkpoint saved:")
    print(CHECKPOINT_PATH)


# ============================================================
# 9. LOAD EXISTING CHECKPOINT
# ============================================================

def load_checkpoint():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "No trained checkpoint found. "
            "Run this script with --train first."
        )

    try:

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True
        )

    except TypeError:

        # Compatibility with older PyTorch versions
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

    model = UNet().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# 10. FULL IMAGE PREDICTION
# ============================================================

def predict_image(model, image, height, width):

    print("\n========== BUILDING PREDICTION ==========")

    predictions = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    image_float = image.astype(
        np.float32
    ) / 255.0

    with torch.no_grad():

        for y in range(0, height, PATCH_SIZE):

            for x in range(0, width, PATCH_SIZE):

                patch = image_float[
                    :,
                    y:min(y + PATCH_SIZE, height),
                    x:min(x + PATCH_SIZE, width)
                ]

                original_h = patch.shape[1]
                original_w = patch.shape[2]

                padded = np.zeros(
                    (3, PATCH_SIZE, PATCH_SIZE),
                    dtype=np.float32
                )

                padded[
                    :, :original_h, :original_w
                ] = patch

                tensor = torch.from_numpy(
                    padded
                ).unsqueeze(0).to(DEVICE)

                logits = model(tensor)

                probabilities = torch.sigmoid(logits)

                prediction = (
                    probabilities[0, 0]
                    .cpu()
                    .numpy() >= THRESHOLD
                ).astype(np.uint8)

                predictions[
                    y:y + original_h,
                    x:x + original_w
                ] = prediction[
                    :original_h, :original_w
                ]

    return predictions


# ============================================================
# 11. SAVE GEOSPATIAL PREDICTION
# ============================================================

def save_prediction(predictions, profile):

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        nodata=None
    )

    with rasterio.open(
        PREDICTION_PATH,
        "w",
        **output_profile
    ) as dst:

        dst.write(predictions, 1)

    print("\nPrediction saved:")
    print(PREDICTION_PATH)


# ============================================================
# 12. EVALUATE HELD-OUT PATCHES
# ============================================================

def evaluate_predictions(predictions, mask, val_patches):

    pred_parts = []
    target_parts = []

    for y, x in val_patches:

        pred_parts.append(
            predictions[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ].ravel()
        )

        target_parts.append(
            mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ].ravel()
        )

    if not pred_parts:

        raise ValueError(
            "No validation patches available for evaluation."
        )

    all_predictions = np.concatenate(pred_parts)
    all_targets = np.concatenate(target_parts)

    metrics = calculate_metrics(
        all_predictions,
        all_targets
    )

    results = {
        "metrics": metrics,
        "validation_patch_count": len(val_patches),
        "threshold": THRESHOLD,
        "patch_size": PATCH_SIZE,
        "note": (
            "Validation patches come from the same satellite tile. "
            "These results do not establish generalization to other cities."
        )
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n========== MODEL EVALUATION ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nMetrics saved:")
    print(METRICS_PATH)


# ============================================================
# 13. MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="UrbanFootprint AI building segmentation"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new U-Net model."
    )

    args = parser.parse_args()

    set_seed()

    print("\n======================================")
    print("       URBANFOOTPRINT AI")
    print("======================================")

    print("Device:", DEVICE)

    image, mask, profile = load_data()

    height, width = mask.shape

    print("\nSatellite image:", width, "x", height)
    print("Building pixels:", int(mask.sum()))

    train_patches, val_patches = get_patch_split(
        height, width
    )

    print("Training patches:", len(train_patches))
    print("Validation patches:", len(val_patches))

    if args.train:

        train_model(
            image,
            mask,
            train_patches,
            val_patches
        )

    model, checkpoint = load_checkpoint()

    print("\nCheckpoint loaded successfully.")
    print("Checkpoint epoch:", checkpoint.get("epoch", "Unknown"))

    predictions = predict_image(
        model,
        image,
        height,
        width
    )

    save_prediction(
        predictions,
        profile
    )

    # Prefer the exact validation split stored during training.
    # Older checkpoints do not contain this information.
    saved_val_patches = checkpoint.get(
        "val_patches",
        val_patches
    )

    evaluate_predictions(
        predictions,
        mask,
        saved_val_patches
    )

    print("\n======================================")
    print("       PIPELINE COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()
