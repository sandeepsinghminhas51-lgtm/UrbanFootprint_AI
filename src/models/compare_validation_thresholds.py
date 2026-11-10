
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

IMAGE_1_REPORT = (
    ROOT
    / "data/processed/new_predictions"
    / "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1"
    / "threshold_comparison.json"
)

IMAGE_3_REPORT = (
    ROOT
    / "data/processed/new_predictions"
    / "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img3"
    / "threshold_comparison.json"
)

OUTPUT_REPORT = (
    ROOT
    / "data/processed/new_predictions"
    / "combined_validation_thresholds.json"
)


def load_report(path):

    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(tp, fp, fn, tn):

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

    report_1 = load_report(IMAGE_1_REPORT)
    report_3 = load_report(IMAGE_3_REPORT)

    results_1 = {
        round(row["threshold"], 2): row
        for row in report_1["results"]
    }

    results_3 = {
        round(row["threshold"], 2): row
        for row in report_3["results"]
    }

    common_thresholds = sorted(
        set(results_1.keys()) & set(results_3.keys())
    )

    if not common_thresholds:
        raise ValueError("No common thresholds found in both reports.")

    combined_results = []

    print("\n====================================================")
    print(" URBANFOOTPRINT AI - COMBINED VALIDATION EVALUATION")
    print("====================================================")

    print(
        "\nThreshold | Precision | Recall | F1-score | IoU"
    )
    print("-" * 58)

    for threshold in common_thresholds:

        first = results_1[threshold]
        third = results_3[threshold]

        tp = first["TP"] + third["TP"]
        fp = first["FP"] + third["FP"]
        fn = first["FN"] + third["FN"]
        tn = first["TN"] + third["TN"]

        metrics = calculate_metrics(tp, fp, fn, tn)

        row = {
            "threshold": threshold,
            **metrics
        }

        combined_results.append(row)

        print(
            f"{threshold:8.2f} | "
            f"{metrics['precision'] * 100:8.2f}% | "
            f"{metrics['recall'] * 100:6.2f}% | "
            f"{metrics['f1'] * 100:8.2f}% | "
            f"{metrics['iou'] * 100:5.2f}%"
        )

    best = max(
        combined_results,
        key=lambda item: item["f1"]
    )

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final_report = {
        "validation_images": [
            report_1["image"],
            report_3["image"]
        ],
        "method": "Combined confusion counts across held-out images",
        "results": combined_results,
        "best_combined_f1_threshold": best
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=4)

    print("\n====================================================")
    print(" BEST COMMON THRESHOLD")
    print("====================================================")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Precision: {best['precision'] * 100:.2f}%")
    print(f"Recall:    {best['recall'] * 100:.2f}%")
    print(f"F1-score:  {best['f1'] * 100:.2f}%")
    print(f"IoU:       {best['iou'] * 100:.2f}%")
    print(f"True positives:  {best['TP']:,}")
    print(f"False positives: {best['FP']:,}")
    print(f"False negatives: {best['FN']:,}")
    print(f"True negatives:  {best['TN']:,}")

    print(f"\nSaved report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
