"""
UrbanFootprint AI
New georeferenced satellite-image inference using the custom multi-image U-Net.

Run from the project root:
    python src/models/predict_new_image.py "data/raw/my_satellite_image.tif"

Input requirements:
- GeoTIFF with at least 3 bands in RGB order (bands 1, 2, 3)
- A valid CRS and geospatial transform
- Raw image values; the script applies the same per-band 2nd/98th percentile
  normalization used by prepare_multi_image_dataset.py

Outputs are written under:
    data/processed/new_predictions/<input_filename>/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import transform_geom


# ---------------------------------------------------------------------
# PROJECT PATHS AND INFERENCE SETTINGS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    ROOT / "src" / "models" / "weights" / "urbanfootprint_unet_multisource.pth"
)
OUTPUT_ROOT = ROOT / "data" / "processed" / "new_predictions"

PATCH_SIZE = 256
OVERLAP = 64
THRESHOLD = 0.5
BASE_CHANNELS = 16
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# EXACT U-NET ARCHITECTURE USED DURING TRAINING
# Keep module names and layer definitions identical to the training file.
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
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

        self.up3 = nn.ConvTranspose2d(
            base * 8, base * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(
            base * 4, base * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(
            base * 2, base, kernel_size=2, stride=2
        )
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


# ---------------------------------------------------------------------
# CHECKPOINT LOADING
# ---------------------------------------------------------------------

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found:\n{CHECKPOINT_PATH}\n"
            "Do not use the older single-image checkpoint."
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
        )
    except TypeError:
        # Compatibility with PyTorch versions without weights_only.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint format is not the expected dictionary.")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'. "
            "Check that this is urbanfootprint_unet_multisource.pth."
        )

    base = int(checkpoint.get("base_channels", BASE_CHANNELS))
    model = UNet(base=base).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Architecture:", checkpoint.get("architecture", "Custom U-Net"))
    print("Base channels:", base)
    print("Training image IDs:", checkpoint.get("training_image_ids", "not recorded"))
    print("Validation image IDs:", checkpoint.get("validation_image_ids", "not recorded"))
    print("Pretrained weights used:", checkpoint.get("pretrained_weights_used", "not recorded"))

    return model


# ---------------------------------------------------------------------
# MATCH TRAINING NORMALIZATION
# Training preprocessing uses each RGB band's 2nd and 98th percentiles,
# scales to uint8 [0,255], and training divides by 255.
# ---------------------------------------------------------------------

def calculate_band_limits(src, sample_max_side: int = 2048):
    scale = min(1.0, sample_max_side / max(src.height, src.width))
    sample_height = max(1, int(round(src.height * scale)))
    sample_width = max(1, int(round(src.width * scale)))

    sample = src.read(
        [1, 2, 3],
        out_shape=(3, sample_height, sample_width),
        masked=True,
    )

    limits = []
    for band_index in range(3):
        band = np.asarray(sample[band_index].compressed(), dtype=np.float32)
        band = band[np.isfinite(band)]
        if band.size == 0:
            raise ValueError(
                f"RGB band {band_index + 1} contains no valid sample pixels."
            )

        low, high = np.percentile(band, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(
                f"RGB band {band_index + 1} has no usable intensity range."
            )
        limits.append((float(low), float(high)))

    return limits


def normalize_tile(raw_tile: np.ndarray, limits) -> np.ndarray:
    """Return float32 RGB in [0,1], with shape [3,H,W]."""
    output = np.zeros(raw_tile.shape, dtype=np.float32)

    for index, (low, high) in enumerate(limits):
        band = raw_tile[index].astype(np.float32)
        valid = np.isfinite(band)
        scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
        scaled[~valid] = 0.0
        output[index] = scaled

    return output


# ---------------------------------------------------------------------
# OVERLAPPING TILE INFERENCE
# ---------------------------------------------------------------------

def tile_starts(length: int, tile_size: int, overlap: int):
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def predict_geotiff(model, input_path: Path, output_mask_path: Path):
    with rasterio.open(input_path) as src:
        if src.count < 3:
            raise ValueError(
                f"Input has {src.count} bands; at least 3 RGB bands are required."
            )
        if src.crs is None:
            raise ValueError(
                "Input GeoTIFF has no CRS. A georeferenced GeoTIFF is required "
                "to produce geographic building polygons."
            )
        if src.width < 1 or src.height < 1:
            raise ValueError("Input raster has invalid dimensions.")

        height, width = src.height, src.width
        transform = src.transform
        source_crs = src.crs
        profile = src.profile.copy()

        print(f"Input dimensions: {width} x {height}")
        print(f"Input CRS: {source_crs}")
        print("Calculating RGB normalization limits...")
        limits = calculate_band_limits(src)

        y_positions = tile_starts(height, PATCH_SIZE, OVERLAP)
        x_positions = tile_starts(width, PATCH_SIZE, OVERLAP)

        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        total_tiles = len(y_positions) * len(x_positions)
        tile_number = 0

        # Raised-cosine-like weights reduce visible seams between overlaps.
        axis = np.linspace(0.05, 1.0, PATCH_SIZE // 2, dtype=np.float32)
        one_axis = np.concatenate([axis, axis[::-1]])
        weight = np.outer(one_axis, one_axis).astype(np.float32)
        weight = np.maximum(weight, 0.05)

        model.eval()
        with torch.inference_mode():
            for y in y_positions:
                for x in x_positions:
                    tile_number += 1

                    read_height = min(PATCH_SIZE, height - y)
                    read_width = min(PATCH_SIZE, width - x)

                    raw_masked = src.read(
                        [1, 2, 3],
                        window=((y, y + read_height), (x, x + read_width)),
                        boundless=False,
                        masked=True,
                    )
                    raw = np.ma.filled(raw_masked.astype(np.float32), np.nan)
                    tile = normalize_tile(raw, limits)

                    # Pad edge tiles to exactly 256 x 256.
                    padded = np.zeros(
                        (3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32
                    )
                    padded[:, :read_height, :read_width] = tile

                    tensor = torch.from_numpy(padded).unsqueeze(0).to(DEVICE)
                    logits = model(tensor)
                    probabilities = torch.sigmoid(logits)[0, 0]
                    probabilities = probabilities.cpu().numpy()

                    tile_prob = probabilities[:read_height, :read_width]
                    tile_weight = weight[:read_height, :read_width]

                    probability_sum[
                        y:y + read_height, x:x + read_width
                    ] += tile_prob * tile_weight

                    weight_sum[
                        y:y + read_height, x:x + read_width
                    ] += tile_weight

                    if tile_number == 1 or tile_number % 10 == 0 or tile_number == total_tiles:
                        print(f"Processed tile {tile_number}/{total_tiles}")

        average_probability = probability_sum / np.maximum(weight_sum, 1e-8)
        prediction = (average_probability >= THRESHOLD).astype(np.uint8)

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        photometric="MINISBLACK",
    )

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_mask_path, "w", **profile) as dst:
        dst.write(prediction, 1)

    print("Saved prediction mask:", output_mask_path)
    print("Predicted building pixels:", int(prediction.sum()))

    return prediction, average_probability, transform, source_crs


# ---------------------------------------------------------------------
# OVERLAY PREVIEW
# ---------------------------------------------------------------------

def save_overlay(input_path: Path, output_path: Path, prediction: np.ndarray):
    with rasterio.open(input_path) as src:
        limits = calculate_band_limits(src)
        raw = src.read([1, 2, 3])
        rgb_float = normalize_tile(raw, limits)

    rgb = (np.clip(rgb_float.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    overlay = rgb.copy()

    building = prediction == 1
    red = np.array([255, 45, 45], dtype=np.float32)
    overlay[building] = (
        0.55 * overlay[building].astype(np.float32) + 0.45 * red
    ).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print("Saved overlay:", output_path)


# ---------------------------------------------------------------------
# VECTORIZE PREDICTION INTO WGS84 GEOJSON
# ---------------------------------------------------------------------

def save_geojson(
    prediction: np.ndarray,
    transform,
    source_crs,
    output_path: Path,
    minimum_area_pixels: int = 3,
):
    features = []
    building_number = 0

    # Convert map-coordinate polygon area to source-pixel area using
    # the affine transform determinant. This avoids rasterizing every
    # individual polygon back over the entire image.
    pixel_area_map_units = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    if pixel_area_map_units <= 0:
        raise ValueError("Source raster transform has an invalid pixel area.")

    def ring_area(ring):
        area = 0.0
        for index in range(len(ring) - 1):
            x1, y1 = ring[index][0], ring[index][1]
            x2, y2 = ring[index + 1][0], ring[index + 1][1]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def polygon_map_area(polygon_coordinates):
        if not polygon_coordinates:
            return 0.0
        outer_area = ring_area(polygon_coordinates[0])
        hole_area = sum(ring_area(ring) for ring in polygon_coordinates[1:])
        return max(0.0, outer_area - hole_area)

    def geometry_area_pixels(geometry):
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])
        if geometry_type == "Polygon":
            map_area = polygon_map_area(coordinates)
        elif geometry_type == "MultiPolygon":
            map_area = sum(polygon_map_area(poly) for poly in coordinates)
        else:
            return 0
        return int(round(map_area / pixel_area_map_units))

    for geometry, value in shapes(
        prediction,
        mask=(prediction == 1),
        transform=transform,
        connectivity=4,
    ):
        if int(value) != 1:
            continue

        # Ignore tiny isolated detections likely to be pixel noise.
        # This is a prototype cleanup setting, not a learned confidence score.
        area_pixels = geometry_area_pixels(geometry)
        if area_pixels < minimum_area_pixels:
            continue

        try:
            geometry_wgs84 = transform_geom(
                source_crs,
                "EPSG:4326",
                geometry,
                precision=7,
            )
        except Exception as exc:
            print("Warning: skipped a polygon during CRS conversion:", exc)
            continue

        building_number += 1
        features.append({
            "type": "Feature",
            "properties": {
                "building_id": f"building_{building_number:06d}",
                "class": "building",
                "area_pixels": area_pixels,
                "source_image": output_path.parent.name,
            },
            "geometry": geometry_wgs84,
        })

    geojson = {
        "type": "FeatureCollection",
        "name": output_path.stem,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(geojson, file, indent=2)

    print("Saved GeoJSON:", output_path)
    print("Building polygons exported:", len(features))
    return len(features)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect building footprints in a new georeferenced GeoTIFF."
    )
    parser.add_argument(
        "image",
        help="Path to the input georeferenced satellite GeoTIFF.",
    )
    args = parser.parse_args()

    input_path = Path(args.image).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(
            "Please provide a GeoTIFF (.tif or .tiff). PNG/JPG files do not "
            "normally contain the geospatial metadata needed for GeoJSON."
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    safe_name = input_path.stem
    output_dir = OUTPUT_ROOT / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / "building_mask.tif"
    overlay_path = output_dir / "building_overlay.png"
    geojson_path = output_dir / "buildings.geojson"
    probability_path = output_dir / "building_probability.tif"

    torch.set_num_threads(min(4, os.cpu_count() or 1))

    print("\n==============================================")
    print("          URBANFOOTPRINT AI")
    print("       NEW IMAGE INFERENCE PIPELINE")
    print("==============================================")
    print("Input:", input_path)
    print("Device:", DEVICE)
    print("Model: Custom U-Net trained from scratch")
    print("Threshold:", THRESHOLD)
    print("Patch size:", PATCH_SIZE)
    print("Overlap:", OVERLAP)

    model = load_model()

    prediction, probabilities, transform, source_crs = predict_geotiff(
        model, input_path, mask_path
    )

    # Save continuous probabilities too; useful for later threshold tuning.
    with rasterio.open(input_path) as src:
        probability_profile = src.profile.copy()

    probability_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        compress="deflate",
        nodata=None,
    )
    with rasterio.open(probability_path, "w", **probability_profile) as dst:
        dst.write(probabilities.astype(np.float32), 1)
    print("Saved probability raster:", probability_path)

    save_overlay(input_path, overlay_path, prediction)

    polygon_count = save_geojson(
        prediction,
        transform,
        source_crs,
        geojson_path,
        minimum_area_pixels=3,
    )

    summary = {
        "input_image": str(input_path),
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(DEVICE),
        "patch_size": PATCH_SIZE,
        "overlap": OVERLAP,
        "threshold": THRESHOLD,
        "predicted_building_pixels": int(prediction.sum()),
        "building_polygon_count": polygon_count,
        "source_crs": str(source_crs),
        "outputs": {
            "mask": str(mask_path),
            "probability": str(probability_path),
            "overlay": str(overlay_path),
            "geojson": str(geojson_path),
        },
        "warning": (
            "Predictions are prototype results. The model was trained on a "
            "small SpaceNet 2 AOI 2 Vegas sample; performance on other cities, "
            "sensors, resolutions, or image conditions is not established."
        ),
    }

    summary_path = output_dir / "prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n==============================================")
    print("       NEW IMAGE INFERENCE COMPLETED")
    print("==============================================")
    print("Output folder:", output_dir)
    print("Mask:", mask_path)
    print("Overlay:", overlay_path)
    print("GeoJSON:", geojson_path)
    print("Polygons:", polygon_count)
    print("Summary:", summary_path)
    print("\nInspect the overlay before treating polygons as valid buildings.")


if __name__ == "__main__":
    main()
