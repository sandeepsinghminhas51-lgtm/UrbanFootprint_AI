
from pathlib import Path

# Find the root directory of our project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Important folders required by our project
REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/sample",
    "src/preprocessing",
    "src/models",
    "src/postprocessing",
    "src/gis",
    "models",
    "outputs/masks",
    "outputs/vectors",
    "outputs/visualizations",
    "outputs/metrics",
    "backend",
    "frontend",
    "docs",
    "tests"
]

print("====================================")
print("       URBANFOOTPRINT AI")
print("====================================")

print("\nChecking project structure...\n")

for folder in REQUIRED_DIRS:

    folder_path = PROJECT_ROOT / folder

    if folder_path.exists():
        print(f"[OK] {folder}")
    else:
        print(f"[MISSING] {folder}")

print("\nProject initialization completed!")
