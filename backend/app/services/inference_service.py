from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import rasterio
import torch
import torch.nn as nn

from backend.app.config import (
    BASE_CHANNELS,
    DEFAULT_THRESHOLD,
    DEVICE,
    OVERLAP,
    PATCH_SIZE,
    WEIGHTS_PATH,
)
from backend.app.services.raster_service import calculate_band_limits


# ---------------------------------------------------------------------
# EXACT U-NET ARCHITECTURE (100% compatible with trained checkpoint)
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base: int = BASE_CHANNELS):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.enc1 = DoubleConv(3, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.output_layer = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output_layer(d1)


# Global model cache to avoid reloading on each request
_LOADED_MODEL: Optional[UNet] = None


def get_model() -> UNet:
    """Load and cache the trained U-Net model from the checkpoint."""
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found at: {WEIGHTS_PATH}\n"
            "Ensure the weights file exists."
        )

    device = torch.device(DEVICE)
    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    base = int(checkpoint.get("base_channels", BASE_CHANNELS))
    model = UNet(base=base).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    _LOADED_MODEL = model
    print(f"Loaded trained U-Net checkpoint from {WEIGHTS_PATH} (base channels={base})")
    return _LOADED_MODEL


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def normalize_tile(raw_tile: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    output = np.zeros(raw_tile.shape, dtype=np.float32)
    for index, (low, high) in enumerate(limits):
        band = raw_tile[index].astype(np.float32)
        valid = np.isfinite(band)
        scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
        scaled[~valid] = 0.0
        output[index] = scaled
    return output


def run_unet_inference(
    input_path: Path,
    output_mask_path: Path,
    output_prob_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    patch_size: int = PATCH_SIZE,
    overlap: int = OVERLAP,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine, Any]:
    """
    Run tiled U-Net inference on the georeferenced GeoTIFF.
    Returns: (prediction_mask, probability_map, transform, crs)
    """
    model = get_model()
    device = torch.device(DEVICE)

    with rasterio.open(input_path) as src:
        if src.count < 3:
            raise ValueError(f"Input image has {src.count} band(s). At least 3 RGB bands are required.")
        if src.crs is None:
            raise ValueError("Input GeoTIFF has no CRS. A georeferenced GeoTIFF is required.")

        height, width = src.height, src.width
        transform = src.transform
        source_crs = src.crs
        profile = src.profile.copy()

        if progress_callback:
            progress_callback(10.0, "Calculating radiometric normalization limits...")

        limits = calculate_band_limits(src)

        y_positions = tile_starts(height, patch_size, overlap)
        x_positions = tile_starts(width, patch_size, overlap)

        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        total_tiles = len(y_positions) * len(x_positions)
        tile_count = 0

        # Raised-cosine weights reduce boundary seams
        axis = np.linspace(0.05, 1.0, patch_size // 2, dtype=np.float32)
        one_axis = np.concatenate([axis, axis[::-1]])
        weight = np.outer(one_axis, one_axis).astype(np.float32)
        weight = np.maximum(weight, 0.05)

        if progress_callback:
            progress_callback(20.0, f"Starting inference across {total_tiles} sliding patches...")

        model.eval()
        with torch.inference_mode():
            for y in y_positions:
                for x in x_positions:
                    tile_count += 1
                    read_height = min(patch_size, height - y)
                    read_width = min(patch_size, width - x)

                    raw_masked = src.read(
                        [1, 2, 3],
                        window=((y, y + read_height), (x, x + read_width)),
                        boundless=False,
                        masked=True,
                    )
                    raw = np.ma.filled(raw_masked.astype(np.float32), np.nan)
                    tile = normalize_tile(raw, limits)

                    padded = np.zeros((3, patch_size, patch_size), dtype=np.float32)
                    padded[:, :read_height, :read_width] = tile

                    tensor = torch.from_numpy(padded).unsqueeze(0).to(device)
                    logits = model(tensor)
                    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                    tile_prob = probs[:read_height, :read_width]
                    tile_weight = weight[:read_height, :read_width]

                    probability_sum[y:y + read_height, x:x + read_width] += tile_prob * tile_weight
                    weight_sum[y:y + read_height, x:x + read_width] += tile_weight

                    if tile_count % 5 == 0 or tile_count == total_tiles:
                        pct = 20.0 + (tile_count / total_tiles) * 50.0  # 20% to 70%
                        if progress_callback:
                            progress_callback(pct, f"Processed patch {tile_count}/{total_tiles}")

        average_probability = probability_sum / np.maximum(weight_sum, 1e-8)
        prediction = (average_probability >= threshold).astype(np.uint8)

    # Save output mask GeoTIFF
    mask_profile = profile.copy()
    mask_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        photometric="MINISBLACK",
        nodata=0,
    )
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_mask_path, "w", **mask_profile) as dst:
        dst.write(prediction, 1)

    # Save probability GeoTIFF
    prob_profile = profile.copy()
    prob_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        compress="deflate",
        nodata=None,
    )
    output_prob_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_prob_path, "w", **prob_profile) as dst:
        dst.write(average_probability.astype(np.float32), 1)

    return prediction, average_probability, transform, source_crs


def generate_overlay_image(
    input_path: Path,
    output_overlay_path: Path,
    prediction: np.ndarray,
    output_mask_png_path: Optional[Path] = None
) -> None:
    """Generate high-contrast visual RGB overlay PNG for web display."""
    with rasterio.open(input_path) as src:
        limits = calculate_band_limits(src)
        raw = src.read([1, 2, 3])
        rgb_float = normalize_tile(raw, limits)

    rgb = (np.clip(rgb_float.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
    overlay = rgb.copy()

    building_pixels = prediction == 1
    # Red highlight: [255, 45, 45]
    red = np.array([255, 45, 45], dtype=np.float32)
    overlay[building_pixels] = (
        0.55 * overlay[building_pixels].astype(np.float32) + 0.45 * red
    ).astype(np.uint8)

    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_overlay_path, format="PNG", optimize=True)

    if output_mask_png_path:
        mask_png = np.zeros((prediction.shape[0], prediction.shape[1], 4), dtype=np.uint8)
        mask_png[building_pixels] = [255, 45, 45, 180]  # Semi-transparent red RGBA
        Image.fromarray(mask_png, mode="RGBA").save(output_mask_png_path, format="PNG", optimize=True)
