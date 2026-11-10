
from pathlib import Path
import rasterio
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data/raw/spacenet/images"
LABEL_DIR = PROJECT_ROOT / "data/raw/spacenet/labels"

images = list(IMAGE_DIR.glob("*.tif")) + list(IMAGE_DIR.glob("*.tiff"))
labels = list(LABEL_DIR.glob("*.geojson")) + list(LABEL_DIR.glob("*.json"))

if not images or not labels:
    raise FileNotFoundError(
        "Satellite image or building annotation is missing."
    )

image_path = images[0]
label_path = labels[0]

print("===== GEOSPATIAL INSPECTION =====")

with rasterio.open(image_path) as src:

    print("\nSATELLITE IMAGE")
    print("Filename:", image_path.name)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Data type:", src.dtypes)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Pixel resolution:", src.res)

with open(label_path, "r", encoding="utf-8") as file:
    geojson = json.load(file)

features = geojson.get("features", [])

print("\nBUILDING ANNOTATIONS")
print("Filename:", label_path.name)
print("Feature count:", len(features))
print("GeoJSON type:", geojson.get("type"))

print("\nInspection completed.")
