
import json
import random
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "data" / "processed" / "multi_image_dataset"

REPORT_PATH = DATASET_DIR / "preprocessing_report.json"

WEIGHTS_DIR = ROOT / "src" / "models" / "weights"

OUTPUT_DIR = DATASET_DIR / "predictions"

CHECKPOINT_PATH = (
    WEIGHTS_DIR / "urbanfootprint_unet_multisource.pth"
)

METRICS_PATH = (
    DATASET_DIR / "multisource_training_metrics.json"
)

PATCH_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 0.001
THRESHOLD = 0.5

BASE_CHANNELS = 16
SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# 2. REPRODUCIBILITY AND CPU CONFIGURATION
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(
    min(4, os.cpu_count() or 1)
)


# ============================================================
# 3. LOAD AND VERIFY THE DATASET
# ============================================================

def load_source_images():

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing report not found: {REPORT_PATH}"
        )

    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    results = report.get("results", [])

    if report.get("failed", 0) != 0:
        raise RuntimeError(
            "Some image-mask pairs failed preprocessing. "
            "Resolve those failures before training."
        )

    if len(results) != 8:
        raise RuntimeError(
            f"Expected 8 processed pairs, found {len(results)}."
        )

    source_images = {}

    print("\nVERIFYING IMAGE-MASK PAIRS")
    print("------------------------------------")

    for item in results:

        image_id = str(item["image_id"])

        image_path = ROOT / item["image_file"]
        mask_path = ROOT / item["mask_file"]

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        # Read satellite image.

        with rasterio.open(image_path) as image_src:

            if image_src.count < 3:
                raise ValueError(
                    f"Image {image_id} has fewer than 3 bands."
                )

            image = image_src.read([1, 2, 3])

            image_crs = image_src.crs
            image_transform = image_src.transform

            image_height = image_src.height
            image_width = image_src.width

        # Read corresponding building mask.

        with rasterio.open(mask_path) as mask_src:

            mask = mask_src.read(1)

            if mask_src.crs != image_crs:
                raise ValueError(
                    f"CRS mismatch for image {image_id}"
                )

            if mask_src.transform != image_transform:
                raise ValueError(
                    f"Geospatial transform mismatch for image {image_id}"
                )

            if (
                mask_src.height != image_height
                or mask_src.width != image_width
            ):
                raise ValueError(
                    f"Image-mask dimensions mismatch for image {image_id}"
                )

        # Validate image and mask values.

        if not np.isfinite(image).all():
            raise ValueError(
                f"Image {image_id} contains invalid pixel values."
            )

        if not np.isfinite(mask).all():
            raise ValueError(
                f"Mask {image_id} contains invalid pixel values."
            )

        if not np.any(mask > 0):
            raise ValueError(
                f"Image {image_id} has an empty building mask."
            )

        # Convert image values to the range [0, 1].

        image = image.astype(np.float32) / 255.0

        # Convert mask to binary values.

        mask = (mask > 0).astype(np.float32)

        source_images[image_id] = {
            "image": image,
            "mask": mask,
            "image_path": image_path,
            "crs": str(image_crs)
        }

        print(
            f"Image {image_id} verified | "
            f"Dimensions: {image_width}x{image_height} | "
            f"Building pixels: {int(mask.sum())}"
        )

    print("------------------------------------")
    print("All 8 image-mask pairs verified.")

    return source_images


# ============================================================
# 4. PATCH EXTRACTION
# ============================================================

def get_patch_positions(length):

    if length <= PATCH_SIZE:
        return [0]

    positions = list(
        range(
            0,
            length - PATCH_SIZE + 1,
            PATCH_SIZE
        )
    )

    last_position = length - PATCH_SIZE

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def extract_patches(source_images, selected_ids):

    patches = []

    for image_id in selected_ids:

        source = source_images[image_id]

        image = source["image"]
        mask = source["mask"]

        height, width = mask.shape

        y_positions = get_patch_positions(height)
        x_positions = get_patch_positions(width)

        for y in y_positions:

            for x in x_positions:

                image_patch = image[
                    :,
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                mask_patch = mask[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                expected_shape = (
                    PATCH_SIZE,
                    PATCH_SIZE
                )

                if image_patch.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Incorrect image patch dimensions: {image_id}"
                    )

                if mask_patch.shape != expected_shape:
                    raise ValueError(
                        f"Incorrect mask patch dimensions: {image_id}"
                    )

                patches.append(
                    (image_patch, mask_patch, image_id)
                )

    return patches


# ============================================================
# 5. PYTORCH DATASET
# ============================================================

class BuildingPatchDataset(Dataset):

    def __init__(self, patches, augment=False):

        self.patches = patches
        self.augment = augment

    def __len__(self):

        return len(self.patches)

    def __getitem__(self, index):

        image, mask, image_id = self.patches[index]

        image = image.copy()
        mask = mask.copy()

        # Data augmentation.

        if self.augment:

            # Horizontal flip.

            if random.random() > 0.5:

                image = np.flip(image, axis=2)
                mask = np.flip(mask, axis=1)

            # Vertical flip.

            if random.random() > 0.5:

                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=0)

            # Rotation by 0, 90, 180 or 270 degrees.

            rotations = random.randint(0, 3)

            image = np.rot90(
                image,
                rotations,
                axes=(1, 2)
            )

            mask = np.rot90(
                mask,
                rotations,
                axes=(0, 1)
            )

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        # Image tensor: [3, 256, 256].

        image_tensor = torch.from_numpy(
            image
        ).float()

        # Mask tensor: [1, 256, 256].
        # The extra channel dimension is essential.

        mask_tensor = torch.from_numpy(
            mask
        ).float().unsqueeze(0)

        return image_tensor, mask_tensor


# ============================================================
# 6. U-NET ARCHITECTURE
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.layers(x)


class UNet(nn.Module):

    def __init__(self, base=BASE_CHANNELS):

        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Encoder.

        self.enc1 = DoubleConv(3, base)

        self.enc2 = DoubleConv(
            base,
            base * 2
        )

        self.enc3 = DoubleConv(
            base * 2,
            base * 4
        )

        # Bottleneck.

        self.bottleneck = DoubleConv(
            base * 4,
            base * 8
        )

        # Decoder.

        self.up3 = nn.ConvTranspose2d(
            base * 8,
            base * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            base * 8,
            base * 4
        )

        self.up2 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            base * 4,
            base * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base * 2,
            base,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            base * 2,
            base
        )

        # One output channel for binary segmentation.

        self.output_layer = nn.Conv2d(
            base,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder.

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # Bottleneck.

        b = self.bottleneck(
            self.pool(e3)
        )

        # Decoder with skip connections.

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output_layer(d1)


# ============================================================
# 7. LOSS FUNCTION: BCE + DICE
# ============================================================

def segmentation_loss(logits, targets):

    # Both tensors must have shape [B, 1, H, W].

    if logits.shape != targets.shape:

        raise ValueError(
            f"Prediction shape {logits.shape} "
            f"does not match target shape {targets.shape}"
        )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        +
        targets.sum(dim=(1, 2, 3))
    )

    dice_score = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    dice_loss = 1.0 - dice_score.mean()

    return bce + dice_loss


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

def calculate_metrics(predictions, targets):

    predictions = predictions.bool()
    targets = targets.bool()

    if predictions.shape != targets.shape:

        raise ValueError(
            "Prediction and target dimensions do not match."
        )

    true_positive = (
        predictions & targets
    ).sum().item()

    false_positive = (
        predictions & ~targets
    ).sum().item()

    false_negative = (
        ~predictions & targets
    ).sum().item()

    epsilon = 1e-8

    iou = true_positive / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    dice = (
        2 * true_positive
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    precision = true_positive / (
        true_positive
        + false_positive
        + epsilon
    )

    recall = true_positive / (
        true_positive
        + false_negative
        + epsilon
    )

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall)
    }


# ============================================================
# 9. MODEL VALIDATION
# ============================================================

def validate_model(model, loader):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            loss = segmentation_loss(
                logits,
                masks
            )

            total_loss += loss.item()

            predictions = (
                torch.sigmoid(logits) >= THRESHOLD
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                masks.cpu()
            )

    if not all_predictions:

        raise RuntimeError(
            "Validation loader contains no batches."
        )

    predictions = torch.cat(
        all_predictions,
        dim=0
    )

    targets = torch.cat(
        all_targets,
        dim=0
    )

    metrics = calculate_metrics(
        predictions,
        targets
    )

    average_loss = (
        total_loss / len(loader)
    )

    return average_loss, metrics


# ============================================================
# 10. FULL-SATELLITE-IMAGE PREDICTION
# ============================================================

def predict_full_image(model, image):

    model.eval()

    tensor = torch.from_numpy(
        image.copy()
    ).float().unsqueeze(0).to(DEVICE)

    height, width = tensor.shape[-2:]

    # U-Net uses three downsampling operations.
    # Pad dimensions to multiples of 8 if necessary.

    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8

    if pad_height > 0 or pad_width > 0:

        tensor = F.pad(
            tensor,
            (0, pad_width, 0, pad_height),
            mode="replicate"
        )

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities >= THRESHOLD
        ).float()

    prediction = prediction[
        0,
        0,
        :height,
        :width
    ]

    return prediction.cpu().numpy().astype(
        np.uint8
    )


# ============================================================
# 11. SAVE GEOREFERENCED PREDICTION
# ============================================================

def save_prediction(
    prediction,
    source_image_path,
    output_path
):

    with rasterio.open(source_image_path) as src:

        profile = src.profile.copy()

    # Preserve the source CRS and geospatial transform.

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        photometric="MINISBLACK"
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            prediction.astype(np.uint8),
            1
        )


# ============================================================
# 12. MAIN TRAINING PIPELINE
# ============================================================

def main():

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========================================")
    print(" URBANFOOTPRINT AI")
    print(" MULTI-IMAGE U-NET TRAINING")
    print("========================================")

    print("Device:", DEVICE)
    print("Model: Custom U-Net")
    print("Pretrained weights: NOT USED")
    print("Training initialization: Random")

    # --------------------------------------------------------
    # A. Load and verify dataset.
    # --------------------------------------------------------

    source_images = load_source_images()

    # --------------------------------------------------------
    # B. Split original satellite images.
    # --------------------------------------------------------

    image_ids = sorted(
        source_images.keys(),
        key=lambda value: int(value)
    )

    shuffled_ids = image_ids.copy()

    random.Random(SEED).shuffle(
        shuffled_ids
    )

    train_ids = shuffled_ids[:6]

    validation_ids = shuffled_ids[6:]

    if len(train_ids) != 6 or len(validation_ids) != 2:

        raise RuntimeError(
            "Unexpected training/validation split."
        )

    print("\nIMAGE-LEVEL DATASET SPLIT")
    print("----------------------------------------")

    print("Training image IDs:", train_ids)

    print("Validation image IDs:", validation_ids)

    print("----------------------------------------")

    # --------------------------------------------------------
    # C. Extract patches separately.
    # --------------------------------------------------------

    train_patches = extract_patches(
        source_images,
        train_ids
    )

    validation_patches = extract_patches(
        source_images,
        validation_ids
    )

    print("\nTraining patches:", len(train_patches))

    print("Validation patches:", len(validation_patches))

    # --------------------------------------------------------
    # D. Create datasets and data loaders.
    # --------------------------------------------------------

    train_dataset = BuildingPatchDataset(
        train_patches,
        augment=True
    )

    validation_dataset = BuildingPatchDataset(
        validation_patches,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # E. Create a fresh U-Net.
    # --------------------------------------------------------

    model = UNet().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float("inf")

    best_state = None

    training_history = []

    # --------------------------------------------------------
    # F. Train the model.
    # --------------------------------------------------------

    print("\nSTARTING MODEL TRAINING")
    print("========================================")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Verify tensor dimensions before training.

            if images.ndim != 4 or images.shape[1] != 3:

                raise ValueError(
                    f"Unexpected image tensor shape: {images.shape}"
                )

            if masks.ndim != 4 or masks.shape[1] != 1:

                raise ValueError(
                    f"Unexpected mask tensor shape: {masks.shape}"
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = segmentation_loss(
                logits,
                masks
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss / len(train_loader)
        )

        # Validate after every epoch.

        validation_loss, metrics = validate_model(
            model,
            validation_loader
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Training Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {validation_loss:.4f}"
        )

        print(
            f"Validation IoU: {metrics['IoU']:.4f}"
        )

        print(
            f"Validation Dice: {metrics['Dice']:.4f}"
        )

        print(
            f"Validation Precision: {metrics['Precision']:.4f}"
        )

        print(
            f"Validation Recall: {metrics['Recall']:.4f}"
        )

        training_history.append({

            "epoch": epoch + 1,

            "training_loss": train_loss,

            "validation_loss": validation_loss,

            "validation_metrics": metrics
        })

        # Keep the model with the lowest validation loss.

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print("Best validation model updated.")

    if best_state is None:

        raise RuntimeError(
            "Training failed to produce a valid checkpoint."
        )

    # --------------------------------------------------------
    # G. Restore the best model.
    # --------------------------------------------------------

    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # H. Save our independently trained checkpoint.
    # --------------------------------------------------------

    checkpoint = {

        "model_state_dict": best_state,

        "architecture": "Custom U-Net",

        "base_channels": BASE_CHANNELS,

        "training_image_ids": train_ids,

        "validation_image_ids": validation_ids,

        "patch_size": PATCH_SIZE,

        "threshold": THRESHOLD,

        "epochs": EPOCHS,

        "best_validation_loss": best_validation_loss,

        "pretrained_weights_used": False
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print("\nMODEL CHECKPOINT SAVED")

    print(CHECKPOINT_PATH)

    # --------------------------------------------------------
    # I. Predict complete held-out satellite images.
    # --------------------------------------------------------

    print("\nGENERATING HELD-OUT IMAGE PREDICTIONS")
    print("========================================")

    per_image_metrics = {}

    for image_id in validation_ids:

        source = source_images[image_id]

        prediction = predict_full_image(
            model,
            source["image"]
        )

        target = source["mask"].astype(
            np.uint8
        )

        # Evaluate each complete held-out image.

        metrics = calculate_metrics(
            torch.from_numpy(prediction),
            torch.from_numpy(target)
        )

        per_image_metrics[image_id] = metrics

        prediction_path = (
            OUTPUT_DIR
            / f"predicted_mask_image_{image_id}.tif"
        )

        save_prediction(
            prediction,
            source["image_path"],
            prediction_path
        )

        print(f"\nImage ID: {image_id}")

        print("Prediction:", prediction_path)

        print("Metrics:", metrics)

        print(
            "Predicted building pixels:",
            int(prediction.sum())
        )

    # --------------------------------------------------------
    # J. Save training and evaluation report.
    # --------------------------------------------------------

    training_report = {

        "dataset": "SpaceNet 2 AOI 2 Vegas",

        "model": "Custom U-Net trained from scratch",

        "pretrained_weights_used": False,

        "device": str(DEVICE),

        "training_image_ids": train_ids,

        "validation_image_ids": validation_ids,

        "training_patch_count": len(train_patches),

        "validation_patch_count": len(validation_patches),

        "patch_size": PATCH_SIZE,

        "batch_size": BATCH_SIZE,

        "epochs": EPOCHS,

        "learning_rate": LEARNING_RATE,

        "threshold": THRESHOLD,

        "best_validation_loss": best_validation_loss,

        "training_history": training_history,

        "held_out_full_image_metrics": per_image_metrics,

        "note": (
            "Validation uses complete original satellite images "
            "excluded from training. All images belong to "
            "SpaceNet 2 AOI 2 Vegas. Results do not establish "
            "generalization to other cities or geographic regions."
        )
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            training_report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # K. Final summary.
    # --------------------------------------------------------

    print("\n========================================")
    print(" MULTI-IMAGE TRAINING COMPLETED")
    print("========================================")

    print("\nModel checkpoint:")
    print(CHECKPOINT_PATH)

    print("\nTraining metrics:")
    print(METRICS_PATH)

    print("\nPredictions directory:")
    print(OUTPUT_DIR)

    print("\nTraining images:", len(train_ids))

    print("Validation images:", len(validation_ids))

    print("\nAll operations completed successfully.")


# ============================================================
# 13. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
