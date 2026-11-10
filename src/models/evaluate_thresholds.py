
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[2]

IMAGE_NAME = (
    "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img3.tif"
)

LABEL_NAME = (
    "SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img3.geojson"
)

IMAGE_PATH = ROOT / "data/raw/spacenet/images" / IMAGE_NAME

LABEL_PATH = ROOT / "data/raw/spacenet/labels" / LABEL_NAME

OUTPUT_DIR = (
    ROOT
    / "data/processed/new_predictions"
    / "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img3"
)

PROBABILITY_PATH = OUTPUT_DIR / "building_probability.tif"


def calculate_metrics(truth, prediction, valid):

    truth = (truth == 1) & valid
    prediction = (prediction == 1) & valid

    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction & valid))
    fn = int(np.sum(truth & ~prediction & valid))
    tn = int(np.sum(~truth & ~prediction & valid))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def main():

    for path in [IMAGE_PATH, LABEL_PATH, PROBABILITY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with rasterio.open(IMAGE_PATH) as src:
        rgb = src.read([1, 2, 3])
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    with rasterio.open(PROBABILITY_PATH) as src:
        if (
            src.width != width
            or src.height != height
            or src.crs != crs
            or not np.allclose(src.transform, transform)
        ):
            raise ValueError(
                "Probability raster does not match source image."
            )

        probability = src.read(1).astype(np.float32)

    valid = np.any(rgb != 0, axis=0)

    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    geometries = []

    for feature in geojson["features"]:
        geometry = feature.get("geometry")

        if geometry:
            geometries.append(shape(geometry))

    ground_truth = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )

    print("\n==============================================")
    print(" URBANFOOTPRINT AI - THRESHOLD COMPARISON")
    print("==============================================")
    print(f"Image: {IMAGE_NAME}")
    print(f"Valid satellite pixels: {int(valid.sum())}")
    print(f"Ground-truth buildings: {int(ground_truth[valid].sum())}")

    thresholds = [
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90
    ]

    results = []

    print(
        "\nThreshold | Precision | Recall | F1-score | IoU | False Positives"
    )
    print("-" * 75)

    for threshold in thresholds:

        prediction = (probability >= threshold).astype(np.uint8)

        metrics = calculate_metrics(
            ground_truth,
            prediction,
            valid
        )

        row = {
            "threshold": threshold,
            **metrics
        }

        results.append(row)

        print(
            f"{threshold:8.2f} | "
            f"{metrics['precision'] * 100:8.2f}% | "
            f"{metrics['recall'] * 100:6.2f}% | "
            f"{metrics['f1'] * 100:8.2f}% | "
            f"{metrics['iou'] * 100:5.2f}% | "
            f"{metrics['FP']:15,}"
        )

    best = max(results, key=lambda item: item["f1"])

    report_path = OUTPUT_DIR / "threshold_comparison.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "image": IMAGE_NAME,
                "valid_pixels": int(valid.sum()),
                "results": results,
                "best_f1_threshold": best,
            },
            file,
            indent=4
        )

    print("\n==============================================")
    print("BEST THRESHOLD BY F1-SCORE")
    print("==============================================")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Precision: {best['precision'] * 100:.2f}%")
    print(f"Recall:    {best['recall'] * 100:.2f}%")
    print(f"F1-score:  {best['f1'] * 100:.2f}%")
    print(f"IoU:       {best['iou'] * 100:.2f}%")
    print(f"False positives: {best['FP']:,}")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
