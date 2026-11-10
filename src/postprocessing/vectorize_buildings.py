
import json
from pathlib import Path

import numpy as np
import rasterio

from rasterio.features import shapes
from rasterio.warp import transform_geom
from PIL import Image


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT / "data" / "processed" / "multi_image_dataset"
)

IMAGE_DIR = DATASET_DIR / "images"

PREDICTION_DIR = DATASET_DIR / "predictions"

VECTOR_DIR = DATASET_DIR / "vectors"

OVERLAY_DIR = DATASET_DIR / "previews"

TARGET_CRS = "EPSG:4326"

CONNECTIVITY = 4

# Keep all detected regions. We will inspect small regions
# rather than automatically deleting potentially real buildings.


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. VERIFY SOURCE IMAGE AND PREDICTION
# ============================================================

def load_and_verify(image_path, prediction_path):

    with rasterio.open(image_path) as image_src:

        if image_src.count < 3:
            raise ValueError(
                f"Source image must contain at least 3 bands: {image_path}"
            )

        image = image_src.read([1, 2, 3])

        image_crs = image_src.crs
        image_transform = image_src.transform

        image_height = image_src.height
        image_width = image_src.width

    with rasterio.open(prediction_path) as prediction_src:

        if prediction_src.count != 1:
            raise ValueError(
                f"Prediction must contain exactly one band: {prediction_path}"
            )

        prediction = prediction_src.read(1)

        if prediction_src.crs != image_crs:
            raise ValueError(
                "Prediction and source image CRS do not match."
            )

        if prediction_src.transform != image_transform:
            raise ValueError(
                "Prediction and source image transforms do not match."
            )

        if (
            prediction_src.height != image_height
            or prediction_src.width != image_width
        ):
            raise ValueError(
                "Prediction and source image dimensions do not match."
            )

        if prediction_src.crs is None:
            raise ValueError(
                "Prediction has no CRS information."
            )

    if not np.isfinite(image).all():
        raise ValueError(
            "Source image contains invalid pixel values."
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "Prediction contains invalid pixel values."
        )

    image = np.moveaxis(image, 0, -1)

    image = np.clip(
        image, 0, 255
    ).astype(np.uint8)

    binary_mask = (
        prediction > 0
    ).astype(np.uint8)

    return (
        image,
        binary_mask,
        image_crs,
        image_transform
    )


# ============================================================
# 4. GENERATE VISUAL OVERLAY
# ============================================================

def save_overlay(image, mask, image_id):

    overlay = image.copy()

    building_pixels = mask == 1

    # Red overlay for predicted building regions.

    red = np.array(
        [255, 40, 40],
        dtype=np.float32
    )

    original_pixels = overlay[
        building_pixels
    ].astype(np.float32)

    blended_pixels = (
        0.55 * original_pixels
        + 0.45 * red
    )

    overlay[building_pixels] = np.clip(
        blended_pixels,
        0,
        255
    ).astype(np.uint8)

    overlay_path = (
        OVERLAY_DIR
        / f"predicted_overlay_image_{image_id}.png"
    )

    Image.fromarray(overlay).save(
        overlay_path
    )

    print("Overlay saved:", overlay_path)


# ============================================================
# 5. CONVERT RASTER MASK INTO GEOJSON POLYGONS
# ============================================================

def polygonize_mask(
    binary_mask,
    source_crs,
    source_transform,
    image_id
):

    features = []

    # Rasterio identifies connected regions of equal value.
    # Only pixels classified as buildings are polygonized.

    polygon_generator = shapes(
        binary_mask,
        mask=(binary_mask == 1),
        transform=source_transform,
        connectivity=CONNECTIVITY
    )

    building_number = 1

    for geometry, value in polygon_generator:

        if int(value) != 1:
            continue

        # Convert geometry into standard WGS84 coordinates.

        geographic_geometry = transform_geom(
            source_crs,
            TARGET_CRS,
            geometry,
            precision=7
        )

        feature = {

            "type": "Feature",

            "id": f"{image_id}_{building_number}",

            "properties": {

                "building_id": f"{image_id}_{building_number}",

                "source_image_id": str(image_id),

                "class": "building"

            },

            "geometry": geographic_geometry
        }

        features.append(feature)

        building_number += 1

    return features


# ============================================================
# 6. SAVE INDIVIDUAL GEOJSON
# ============================================================

def save_individual_geojson(features, image_id):

    geojson = {

        "type": "FeatureCollection",

        "name": f"UrbanFootprint_Image_{image_id}",

        "features": features

    }

    output_path = (
        VECTOR_DIR
        / f"buildings_image_{image_id}.geojson"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            geojson,
            file,
            indent=4
        )

    print("Individual GeoJSON saved:", output_path)


# ============================================================
# 7. PROCESS ONE PREDICTION
# ============================================================

def process_prediction(prediction_path):

    filename = prediction_path.stem

    prefix = "predicted_mask_image_"

    if not filename.startswith(prefix):
        raise ValueError(
            f"Unexpected prediction filename: {prediction_path.name}"
        )

    image_id = filename[len(prefix):]

    image_path = (
        IMAGE_DIR
        / f"image_{image_id}.tif"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Original satellite image not found: {image_path}"
        )

    print("\n----------------------------------------")

    print("Processing image ID:", image_id)

    print("----------------------------------------")

    # Verify geospatial alignment.

    (
        image,
        binary_mask,
        source_crs,
        source_transform
    ) = load_and_verify(
        image_path,
        prediction_path
    )

    building_pixel_count = int(
        binary_mask.sum()
    )

    if building_pixel_count == 0:

        print(
            "WARNING: No building pixels predicted "
            f"for image {image_id}."
        )

    print(
        "Predicted building pixels:",
        building_pixel_count
    )

    # Create visual overlay.

    save_overlay(
        image,
        binary_mask,
        image_id
    )

    # Convert raster regions into vector polygons.

    features = polygonize_mask(
        binary_mask,
        source_crs,
        source_transform,
        image_id
    )

    # Save GeoJSON.

    save_individual_geojson(
        features,
        image_id
    )

    print(
        "Extracted building polygons:",
        len(features)
    )

    return features


# ============================================================
# 8. SAVE COMBINED GEOJSON
# ============================================================

def save_combined_geojson(all_features):

    geojson = {

        "type": "FeatureCollection",

        "name": "UrbanFootprint_AI_All_Buildings",

        "features": all_features

    }

    output_path = (
        VECTOR_DIR / "all_buildings.geojson"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            geojson,
            file,
            indent=4
        )

    print("\nCombined GeoJSON saved:")

    print(output_path)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():

    print("\n========================================")
    print(" URBANFOOTPRINT AI")
    print(" BUILDING MASK TO GEOJSON")
    print("========================================")

    create_directories()

    if not PREDICTION_DIR.exists():

        raise FileNotFoundError(
            f"Prediction directory not found: {PREDICTION_DIR}"
        )

    prediction_files = sorted(
        PREDICTION_DIR.glob(
            "predicted_mask_image_*.tif"
        ),
        key=lambda path: int(
            path.stem.replace(
                "predicted_mask_image_", ""
            )
        )
    )

    if not prediction_files:

        raise FileNotFoundError(
            "No predicted mask GeoTIFF files found."
        )

    print(
        "\nPrediction files found:",
        len(prediction_files)
    )

    all_features = []

    processed_images = 0

    for prediction_path in prediction_files:

        features = process_prediction(
            prediction_path
        )

        all_features.extend(features)

        processed_images += 1

    # Save one combined GeoJSON.

    save_combined_geojson(
        all_features
    )

    # Final summary.

    print("\n========================================")

    print(" VECTOR EXTRACTION COMPLETED")

    print("========================================")

    print(
        "Images processed:",
        processed_images
    )

    print(
        "Total building polygons:",
        len(all_features)
    )

    print(
        "Output directory:",
        VECTOR_DIR
    )

    print(
        "Overlay directory:",
        OVERLAY_DIR
    )

    print("\nAll operations completed.")


if __name__ == "__main__":

    main()
