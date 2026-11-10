
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes, sieve
from rasterio.warp import transform_geom
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

THRESHOLD = 0.75
MIN_BUILDING_PIXELS = 3

IMAGE_IDS = ["1", "3"]


def create_overlay(rgb, mask, valid):

    display = np.zeros_like(rgb, dtype=np.uint8)

    for band in range(3):
        values = rgb[band][valid]

        if values.size == 0:
            continue

        low, high = np.percentile(values, [2, 98])

        if high > low:
            scaled = (
                (rgb[band].astype(np.float32) - low)
                / (high - low)
                * 255
            )
            display[band] = np.clip(scaled, 0, 255).astype(np.uint8)

    overlay = np.moveaxis(display, 0, -1).copy()

    # Highlight predicted building pixels in red.
    building_pixels = mask == 1

    overlay[building_pixels, 0] = 255
    overlay[building_pixels, 1] = (
        overlay[building_pixels, 1] * 0.45
    ).astype(np.uint8)
    overlay[building_pixels, 2] = (
        overlay[building_pixels, 2] * 0.45
    ).astype(np.uint8)

    overlay[~valid] = 0

    return overlay


def process_image(image_id):

    image_name = (
        f"SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{image_id}.tif"
    )

    source_path = (
        ROOT / "data/raw/spacenet/images" / image_name
    )

    prediction_dir = (
        ROOT
        / "data/processed/new_predictions"
        / f"SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{image_id}"
    )

    probability_path = prediction_dir / "building_probability.tif"

    output_dir = prediction_dir / "threshold_075"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [source_path, probability_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    print("\n" + "=" * 55)
    print(f"PROCESSING IMAGE {image_id}")
    print("=" * 55)

    with rasterio.open(source_path) as src:

        rgb = src.read([1, 2, 3])
        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

        height = src.height
        width = src.width

    with rasterio.open(probability_path) as src:

        if (
            src.width != width
            or src.height != height
            or src.crs != crs
            or not np.allclose(src.transform, transform)
        ):
            raise ValueError(
                f"Probability raster does not match image {image_id}"
            )

        probability = src.read(1).astype(np.float32)

    valid = np.any(rgb != 0, axis=0)

    if not np.isfinite(probability[valid]).all():
        raise ValueError("Non-finite probability values found.")

    raw_mask = (
        (probability >= THRESHOLD) & valid
    ).astype(np.uint8)

    # Remove tiny connected components.
    clean_mask = sieve(
        raw_mask,
        size=MIN_BUILDING_PIXELS,
        connectivity=8,
        out=np.zeros_like(raw_mask),
    )

    clean_mask = (
        (clean_mask == 1) & valid
    ).astype(np.uint8)

    mask_path = output_dir / "building_mask_threshold_075.tif"

    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="lzw",
        nodata=0,
    )

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(clean_mask, 1)

    # Create optimized overlay.
    overlay = create_overlay(rgb, clean_mask, valid)

    overlay_path = output_dir / "building_overlay_threshold_075.png"

    Image.fromarray(overlay).save(overlay_path)

    # Convert raster regions into geographic polygons.
    polygon_features = []

    polygon_id = 1

    for geometry, value in shapes(
        clean_mask,
        mask=clean_mask == 1,
        transform=transform,
        connectivity=8,
    ):

        if value != 1:
            continue

        if crs is not None and str(crs) != "EPSG:4326":
            geometry = transform_geom(
                crs,
                "EPSG:4326",
                geometry,
                precision=7,
            )

        polygon_features.append({
            "type": "Feature",
            "properties": {
                "building_id": polygon_id,
                "threshold": THRESHOLD,
            },
            "geometry": geometry,
        })

        polygon_id += 1

    geojson = {
        "type": "FeatureCollection",
        "features": polygon_features,
    }

    geojson_path = output_dir / "buildings_threshold_075.geojson"

    with open(geojson_path, "w", encoding="utf-8") as file:
        json.dump(geojson, file, indent=2)

    summary = {
        "image": image_name,
        "threshold": THRESHOLD,
        "minimum_component_pixels": MIN_BUILDING_PIXELS,
        "image_width": width,
        "image_height": height,
        "crs": str(crs),
        "valid_satellite_pixels": int(valid.sum()),
        "predicted_building_pixels": int(clean_mask.sum()),
        "building_polygons": len(polygon_features),
        "files": {
            "mask": mask_path.name,
            "overlay": overlay_path.name,
            "geojson": geojson_path.name,
        },
    }

    summary_path = output_dir / "prediction_summary_threshold_075.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Threshold: {THRESHOLD}")
    print(f"Valid pixels: {int(valid.sum()):,}")
    print(f"Building pixels: {int(clean_mask.sum()):,}")
    print(f"Building polygons: {len(polygon_features)}")
    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved GeoJSON: {geojson_path}")
    print("IMAGE PROCESSING COMPLETED")


def main():

    for image_id in IMAGE_IDS:
        process_image(image_id)

    print("\n" + "=" * 55)
    print("ALL OPTIMIZED PREDICTIONS COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
