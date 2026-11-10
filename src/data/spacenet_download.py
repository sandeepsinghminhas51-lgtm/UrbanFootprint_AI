from pathlib import Path
import re
import json

import boto3
from botocore import UNSIGNED
from botocore.config import Config


# =====================================================
# URBANFOOTPRINT AI — SPACE NET MULTI-IMAGE DOWNLOADER
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data/raw/spacenet/images"
LABEL_DIR = PROJECT_ROOT / "data/raw/spacenet/labels"

BUCKET = "spacenet-dataset"
BASE = "spacenet/SN2_buildings/train/AOI_2_Vegas/"

IMAGE_PREFIX = BASE + "PS-RGB/"
LABEL_PREFIX = BASE + "geojson_buildings/"

MAX_PAIRS = 8

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(signature_version=UNSIGNED)
)

paginator = s3.get_paginator("list_objects_v2")


def list_dataset_files(prefix, extensions):
    """List files in a public SpaceNet S3 folder."""
    results = []

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=prefix
    ):
        for item in page.get("Contents", []):
            key = item["Key"]

            if key.lower().endswith(extensions):
                results.append(item)

    return results


def extract_image_id(filename):
    """Extract the exact SpaceNet image number."""
    match = re.search(
        r"_img(\d+)\.(?:tif|tiff|geojson|json)$",
        filename.lower()
    )

    return match.group(1) if match else None


def download_if_needed(key, destination):
    """Download a file only when it is missing or empty."""
    if destination.exists() and destination.stat().st_size > 0:
        print("Already downloaded:", destination.name)
        return

    print("Downloading:", destination.name)

    s3.download_file(
        BUCKET,
        key,
        str(destination)
    )

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"Download failed or produced an empty file: {destination}"
        )

    print("Downloaded successfully.")


def main():

    print("\n========== URBANFOOTPRINT AI ==========")
    print("MULTI-IMAGE SPACE NET DATASET DOWNLOAD")
    print("Target image/annotation pairs:", MAX_PAIRS)

    print("\n[1/3] Discovering satellite images...")

    image_files = list_dataset_files(
        IMAGE_PREFIX,
        (".tif", ".tiff")
    )

    print("Satellite files found:", len(image_files))

    print("\n[2/3] Discovering building annotations...")

    label_files = list_dataset_files(
        LABEL_PREFIX,
        (".geojson", ".json")
    )

    print("Annotation files found:", len(label_files))

    if not image_files or not label_files:
        raise RuntimeError(
            "SpaceNet images or annotations were not found."
        )

    # Build exact ID-to-file mappings.
    images_by_id = {}

    for item in image_files:
        image_id = extract_image_id(Path(item["Key"]).name)

        if image_id is not None:
            images_by_id[image_id] = item

    labels_by_id = {}

    for item in label_files:
        label_id = extract_image_id(Path(item["Key"]).name)

        if label_id is not None:
            labels_by_id[label_id] = item

    # Only use IDs that have both an image and its matching label.
    common_ids = sorted(
        set(images_by_id) & set(labels_by_id),
        key=int
    )

    if not common_ids:
        raise RuntimeError(
            "No exact image/annotation pairs were found."
        )

    selected_ids = common_ids[:MAX_PAIRS]

    print("\nMatched pairs available:", len(common_ids))
    print("Selected image IDs:", ", ".join(selected_ids))

    manifest = []

    print("\n[3/3] Downloading matched pairs...")

    for index, image_id in enumerate(selected_ids, start=1):

        image_item = images_by_id[image_id]
        label_item = labels_by_id[image_id]

        image_name = Path(image_item["Key"]).name
        label_name = Path(label_item["Key"]).name

        image_path = IMAGE_DIR / image_name
        label_path = LABEL_DIR / label_name

        print(f"\n---------- PAIR {index}/{len(selected_ids)} ----------")
        print("Image ID:", image_id)

        download_if_needed(
            image_item["Key"],
            image_path
        )

        download_if_needed(
            label_item["Key"],
            label_path
        )

        manifest.append({
            "image_id": image_id,
            "image_file": image_name,
            "label_file": label_name,
            "image_size_bytes": image_path.stat().st_size,
            "label_size_bytes": label_path.stat().st_size
        })

    manifest_path = (
        PROJECT_ROOT / "data/raw/spacenet/dataset_manifest.json"
    )

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print("\n========== DOWNLOAD REPORT ==========")
    print("Successfully matched pairs:", len(manifest))
    print("Manifest:", manifest_path)
    print("Image directory:", IMAGE_DIR)
    print("Label directory:", LABEL_DIR)
    print("\nDATASET DOWNLOAD COMPLETED")


if __name__ == "__main__":
    main()