
from huggingface_hub import HfApi

REPO_ID = "Abhatta7/building-footprint-segmentation"

print("\n===== MODEL CHECKPOINT INSPECTION =====")

api = HfApi()

try:
    info = api.model_info(REPO_ID, files_metadata=True)

    print("Repository:", info.id)
    print("License:", info.card_data.get("license") if info.card_data else "Not specified")

    print("\nAvailable model files:")

    for item in info.siblings:
        if item.rfilename.endswith((".pth", ".pt", ".safetensors")):
            size = getattr(item, "size", None)

            if size is not None:
                size_mb = round(size / (1024 * 1024), 2)
                print(f"{item.rfilename}: {size_mb} MB")
            else:
                print(f"{item.rfilename}: Size unavailable")

    print("\nCheckpoint inspection completed.")

except Exception as error:
    print("Inspection failed:", error)
