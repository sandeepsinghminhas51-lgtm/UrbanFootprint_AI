from pathlib import Path
import json
import re

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from PIL import Image


# ============================================================
# URBANFOOTPRINT AI
# MULTI-IMAGE SATELLITE DATASET PREPROCESSING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_IMAGE_DIR = ROOT / "data/raw/spacenet/images"
RAW_LABEL_DIR = ROOT / "data/raw/spacenet/labels"

OUTPUT_ROOT = ROOT / "data/processed/multi_image_dataset"

IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks"
PREVIEW_OUTPUT_DIR = OUTPUT_ROOT / "previews"

MANIFEST_PATH = ROOT / "data/raw/spacenet/dataset_manifest.json"
OUTPUT_REPORT = OUTPUT_ROOT / "preprocessing_report.json"


def create_directories():
    for folder in [
        IMAGE_OUTPUT_DIR,
        MASK_OUTPUT_DIR,
        PREVIEW_OUTPUT_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def extract_image_id(filename):
    match = re.search(
        r"_img(\d+)\.(?:tif|tiff|geojson|json)$",
        filename.lower()
    )
    return match.group(1) if match else None


def get_geojson_crs(data):
    crs_info = data.get("crs")

    if not crs_info:
        return "EPSG:4326"

    properties = crs_info.get("properties", {})
    crs_name = (
        properties.get("name")
        or properties.get("href")
    )

    return crs_name or "EPSG:4326"


def normalize_rgb(raw):
    """Convert three image bands into uint8 RGB."""
    output = np.zeros(
        (raw.shape[1], raw.shape[2], 3),
        dtype=np.uint8
    )

    for index in range(3):
        band = raw[index].astype(np.float32)
        valid = np.isfinite(band)

        if not np.any(valid):
            raise ValueError(
                f"RGB band {index + 1} contains no valid pixels."
            )

        low, high = np.percentile(
            band[valid],
            [2, 98]
        )

        if high <= low:
            raise ValueError(
                f"RGB band {index + 1} has no usable intensity range."
            )

        scaled = np.clip(
            (band - low) / (high - low),
            0,
            1
        )

        scaled[~valid] = 0

        output[:, :, index] = (
            scaled * 255
        ).astype(np.uint8)

    return output


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        entries = json.load(file)

    if not entries:
        raise RuntimeError("Dataset manifest is empty.")

    return entries


def process_pair(entry):

    image_name = entry["image_file"]
    label_name = entry["label_file"]
    image_id = str(entry["image_id"])

    image_path = RAW_IMAGE_DIR / image_name
    label_path = RAW_LABEL_DIR / label_name

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    # Confirm that image and label IDs agree.
    if extract_image_id(image_name) != image_id:
        raise ValueError(
            f"Image ID mismatch: {image_name}"
        )

    if extract_image_id(label_name) != image_id:
        raise ValueError(
            f"Annotation ID mismatch: {label_name}"
        )

    print(f"\nProcessing SpaceNet image ID: {image_id}")

    # --------------------------------------------------------
    # 1. READ SATELLITE IMAGE
    # --------------------------------------------------------

    with rasterio.open(image_path) as src:

        if src.count < 3:
            raise ValueError(
                f"{image_name} has fewer than three bands."
            )

        if src.crs is None:
            raise ValueError(
                f"{image_name} has no coordinate reference system."
            )

        raw_rgb = src.read([1, 2, 3])

        rgb = normalize_rgb(raw_rgb)

        height = src.height
        width = src.width
        transform = src.transform
        image_crs = src.crs

        image_profile = src.profile.copy()

    # --------------------------------------------------------
    # 2. READ BUILDING ANNOTATIONS
    # --------------------------------------------------------

    with open(label_path, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    features = geojson.get("features", [])

    if not features:
        raise ValueError(
            f"No annotation features found in {label_name}"
        )

    label_crs = get_geojson_crs(geojson)

    raster_shapes = []
    skipped = 0

    for feature in features:

        geometry = feature.get("geometry")

        if not geometry:
            skipped += 1
            continue

        if geometry.get("type") not in (
            "Polygon",
            "MultiPolygon"
        ):
            skipped += 1
            continue

        try:
            converted = transform_geom(
                label_crs,
                image_crs,
                geometry
            )

            raster_shapes.append((converted, 1))

        except Exception as error:
            skipped += 1
            print("Warning: geometry skipped:", error)

    if not raster_shapes:
        raise RuntimeError(
            f"No usable building geometries for image {image_id}."
        )

    # --------------------------------------------------------
    # 3. RASTERIZE BUILDING POLYGONS
    # --------------------------------------------------------

    mask = rasterize(
        raster_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=False
    )

    building_pixels = int(np.count_nonzero(mask))

    if building_pixels == 0:
        raise RuntimeError(
            f"Image {image_id} produced an empty building mask. "
            "Check annotation CRS and alignment."
        )

    # --------------------------------------------------------
    # 4. SAVE PROCESSED IMAGE AND MASK
    # --------------------------------------------------------

    output_image = IMAGE_OUTPUT_DIR / f"image_{image_id}.tif"
    output_mask = MASK_OUTPUT_DIR / f"mask_{image_id}.tif"

    image_profile.update(
        driver="GTiff",
        count=3,
        dtype="uint8",
        compress="deflate"
    )

    with rasterio.open(
        output_image,
        "w",
        **image_profile
    ) as dst:
        dst.write(rgb.transpose(2, 0, 1))

    mask_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": image_crs,
        "transform": transform,
        "compress": "deflate"
    }

    with rasterio.open(
        output_mask,
        "w",
        **mask_profile
    ) as dst:
        dst.write(mask, 1)

    # --------------------------------------------------------
    # 5. SAVE PREVIEW IMAGES
    # --------------------------------------------------------

    Image.fromarray(rgb).save(
        PREVIEW_OUTPUT_DIR / f"image_{image_id}.png"
    )

    Image.fromarray(mask * 255).save(
        PREVIEW_OUTPUT_DIR / f"mask_{image_id}.png"
    )

    overlay = rgb.copy()

    building = mask == 1

    overlay[building] = (
        0.55 * overlay[building]
        + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)

    Image.fromarray(overlay).save(
        PREVIEW_OUTPUT_DIR / f"overlay_{image_id}.png"
    )

    result = {
        "image_id": image_id,
        "image_file": str(output_image.relative_to(ROOT)),
        "mask_file": str(output_mask.relative_to(ROOT)),
        "width": width,
        "height": height,
        "crs": str(image_crs),
        "annotation_features": len(features),
        "usable_geometries": len(raster_shapes),
        "skipped_geometries": skipped,
        "building_pixels": building_pixels,
        "building_pixel_percentage": round(
            building_pixels / (height * width) * 100,
            4
        )
    }

    print("Dimensions:", width, "x", height)
    print("Annotation features:", len(features))
    print("Usable geometries:", len(raster_shapes))
    print("Building pixels:", building_pixels)
    print("Building coverage:", result["building_pixel_percentage"], "%")
    print("Saved image:", output_image.name)
    print("Saved mask:", output_mask.name)

    return result


def main():

    print("\n============================================")
    print("       URBANFOOTPRINT AI")
    print("       MULTI-IMAGE PREPROCESSING")
    print("============================================")

    create_directories()

    entries = load_manifest()

    print("Image/annotation pairs:", len(entries))

    results = []
    failures = []

    for entry in entries:

        try:
            result = process_pair(entry)
            results.append(result)

        except Exception as error:

            image_id = entry.get("image_id", "unknown")

            print(
                f"\nERROR processing image {image_id}: {error}"
            )

            failures.append({
                "image_id": image_id,
                "error": str(error)
            })

    report = {
        "dataset": "SpaceNet 2 AOI 2 Vegas",
        "total_pairs_in_manifest": len(entries),
        "successfully_processed": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n============================================")
    print("       PREPROCESSING SUMMARY")
    print("============================================")

    print("Total pairs:", len(entries))
    print("Successfully processed:", len(results))
    print("Failed:", len(failures))
    print("Report:", OUTPUT_REPORT)

    if failures:
        print("\nFailed image IDs:")
        for failure in failures:
            print(
                failure["image_id"],
                "->",
                failure["error"]
            )

    if not results:
        raise RuntimeError(
            "No image pairs were successfully preprocessed."
        )

    print("\nMULTI-IMAGE PREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()
