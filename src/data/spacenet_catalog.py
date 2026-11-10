
import boto3

from botocore import UNSIGNED
from botocore.config import Config

BUCKET_NAME = "spacenet-dataset"

BASE_PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/"

FOLDERS = {
    "SATELLITE IMAGES": BASE_PREFIX + "PS-RGB/",
    "BUILDING ANNOTATIONS": BASE_PREFIX + "geojson_buildings/"
}

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(signature_version=UNSIGNED)
)

print("===== URBANFOOTPRINT AI =====")
print("Searching for actual satellite images and labels...\n")

for folder_name, prefix in FOLDERS.items():

    print("\n", "=" * 55)
    print(folder_name)
    print("=" * 55)

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=prefix,
        MaxKeys=5
    )

    files = response.get("Contents", [])

    if not files:
        print("No files found.")
        continue

    for file in files:
        size_mb = file["Size"] / (1024 * 1024)

        print("File:", file["Key"])
        print("Size:", round(size_mb, 2), "MB")
        print("-" * 40)

print("\nFile discovery completed.")
