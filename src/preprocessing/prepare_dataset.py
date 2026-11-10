
from pathlib import Path
import json
import re

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from PIL import Image


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data/raw/spacenet/images"
LABEL_DIR = PROJECT_ROOT / "data/raw/spacenet/labels"

OUTPUT_IMAGE_DIR = PROJECT_ROOT / "data/processed/images"
OUTPUT_MASK_DIR = PROJECT_ROOT / "data/processed/masks"
OUTPUT_PREVIEW_DIR = PROJECT_ROOT / "data/processed/previews"


def create_directories():
    for folder in [
        OUTPUT_IMAGE_DIR,
        OUTPUT_MASK_DIR,
        OUTPUT_PREVIEW_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def find_matching_files():
    images = sorted(
        list(IMAGE_DIR.glob("*.tif")) +
        list(IMAGE_DIR.glob("*.tiff"))
    )

    if not images:
        raise FileNotFoundError("No satellite GeoTIFF found.")

    image_path = images[0]

    match = re.search(r"_img(\d+)\.tiff?$", image_path.name.lower())

    if not match:
        raise ValueError("Could not identify satellite image number.")

    image_id = match.group(1)

    labels = [
        path for path in LABEL_DIR.glob("*.geojson")
        if re.search(
            rf"_img{re.escape(image_id)}\.geojson$",
            path.name.lower()
        )
    ]

    if len(labels) != 1:
        raise RuntimeError(
            f"Expected one annotation for image {image_id}; "
            f"found {len(labels)}."
        )

    return image_path, labels[0]


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def normalize_rgb(image):
    """
    Convert the first three satellite bands into an
    8-bit RGB image using percentile normalization.
    Original satellite data remains unchanged.
    """

    output = np.zeros(
        (image.shape[1], image.shape[2], 3),
        dtype=np.uint8
    )

    for band_index in range(3):
        band = image[band_index].astype(np.float32)

        valid = np.isfinite(band)

        if not np.any(valid):
            continue

        low = np.percentile(band[valid], 2)
        high = np.percentile(band[valid], 98)

        if high <= low:
            continue

        scaled = np.clip(
            (band - low) / (high - low),
            0,
            1
        )

        output[:, :, band_index] = (
            scaled * 255
        ).astype(np.uint8)

    return output


# --------------------------------------------------
# GEOJSON CRS HANDLING
# --------------------------------------------------

def get_geojson_crs(geojson):
    """
    SpaceNet's legacy GeoJSON can declare CRS84.
    If no CRS is declared, GeoJSON coordinates are
    interpreted as longitude/latitude (EPSG:4326).
    """

    crs_info = geojson.get("crs")

    if crs_info:
        properties = crs_info.get("properties", {})
        crs_name = (
            properties.get("name")
            or properties.get("href")
        )

        if crs_name:
            return crs_name

    return "EPSG:4326"


# --------------------------------------------------
# MAIN PREPROCESSING PIPELINE
# --------------------------------------------------

def main():
    create_directories()

    image_path, label_path = find_matching_files()

    print("\n========== URBANFOOTPRINT AI ==========")
    print("PHASE 3: IMAGE AND LABEL PREPROCESSING")

    print("\nSelected satellite image:", image_path.name)
    print("Selected annotation:", label_path.name)

    # Load genuine satellite image
    with rasterio.open(image_path) as src:

        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 image bands; found {src.count}."
            )

        if src.crs is None:
            raise ValueError(
                "Satellite image has no CRS. Cannot safely create "
                "georeferenced building masks."
            )

        print("\nSatellite image:")
        print("Dimensions:", src.width, "x", src.height)
        print("Bands:", src.count)
        print("CRS:", src.crs)

        # Read RGB bands
        raw_rgb = src.read([1, 2, 3])

        # Normalize for model-compatible RGB input
        rgb_image = normalize_rgb(raw_rgb)

        # Preserve original georeferencing
        image_profile = src.profile.copy()
        image_transform = src.transform
        image_crs = src.crs
        image_height = src.height
        image_width = src.width

    # Load genuine building annotations
    with open(label_path, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    features = geojson.get("features", [])

    if not features:
        raise ValueError("GeoJSON contains no building features.")

    label_crs = get_geojson_crs(geojson)

    print("\nBuilding annotation features:", len(features))
    print("Annotation CRS:", label_crs)

    # Transform polygons into the satellite image CRS
    shapes = []
    skipped = 0

    for feature in features:
        geometry = feature.get("geometry")

        if not geometry:
            skipped += 1
            continue

        try:
            transformed_geometry = transform_geom(
                label_crs,
                image_crs,
                geometry
            )

            shapes.append((transformed_geometry, 1))

        except Exception as error:
            skipped += 1
            print("Skipped an invalid geometry:", error)

    if not shapes:
        raise RuntimeError(
            "No valid building polygons could be transformed."
        )

    # Convert vector polygons to a pixel-level binary mask
    building_mask = rasterize(
        shapes,
        out_shape=(image_height, image_width),
        transform=image_transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=False
    )

    building_pixels = int(np.count_nonzero(building_mask))

    if building_pixels == 0:
        raise RuntimeError(
            "The generated mask is empty. Check the GeoJSON CRS "
            "and coordinate alignment before proceeding."
        )

    # --------------------------------------------------
    # SAVE PROCESSED GEOTIFF FILES
    # --------------------------------------------------

    image_output = OUTPUT_IMAGE_DIR / "satellite_rgb.tif"
    mask_output = OUTPUT_MASK_DIR / "building_mask.tif"

    image_profile.update(
        count=3,
        dtype="uint8",
        compress="deflate"
    )

    with rasterio.open(image_output, "w", **image_profile) as dst:
        dst.write(rgb_image.transpose(2, 0, 1))

    mask_profile = {
        "driver": "GTiff",
        "height": image_height,
        "width": image_width,
        "count": 1,
        "dtype": "uint8",
        "crs": image_crs,
        "transform": image_transform,
        "compress": "deflate"
    }

    with rasterio.open(mask_output, "w", **mask_profile) as dst:
        dst.write(building_mask, 1)

    # --------------------------------------------------
    # SAVE VISUAL PREVIEWS
    # --------------------------------------------------

    image_png = OUTPUT_PREVIEW_DIR / "satellite_preview.png"
    mask_png = OUTPUT_PREVIEW_DIR / "building_mask_preview.png"
    overlay_png = OUTPUT_PREVIEW_DIR / "building_overlay.png"

    Image.fromarray(rgb_image).save(image_png)
    Image.fromarray(building_mask * 255).save(mask_png)

    overlay = rgb_image.copy()

    # Highlight building pixels for visual verification
    overlay[building_mask == 1] = (
        0.55 * overlay[building_mask == 1]
        + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)

    Image.fromarray(overlay).save(overlay_png)

    print("\n========== PREPROCESSING RESULTS ==========")
    print("Valid building geometries:", len(shapes))
    print("Skipped geometries:", skipped)
    print("Building pixels:", building_pixels)
    print(
        "Building pixel percentage:",
        round(building_pixels / (image_height * image_width) * 100, 2),
        "%"
    )

    print("\nGenerated files:")
    print("Processed image:", image_output)
    print("Building mask:", mask_output)
    print("Satellite preview:", image_png)
    print("Mask preview:", mask_png)
    print("Overlay preview:", overlay_png)

    print("\nPHASE 3 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
