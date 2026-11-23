
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "parkinsons.csv"
SPIRAL_DIR = BASE_DIR / "data" / "spiral_data" / "spiral"

print("\n========== VOICE DATASET ==========")

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    if "status" in df.columns:
        print("\nClass distribution:")
        print(df["status"].value_counts())
else:
    print("ERROR: CSV not found:", CSV_PATH)

print("\n========== SPIRAL DATASET ==========")

extensions = {".png", ".jpg", ".jpeg", ".bmp"}

for split in ["training", "testing"]:
    for label in ["healthy", "parkinson"]:
        folder = SPIRAL_DIR / split / label

        if folder.exists():
            images = [
                f for f in folder.rglob("*")
                if f.is_file() and f.suffix.lower() in extensions
            ]

            print(f"{split}/{label}: {len(images)} images")
        else:
            print(f"ERROR: Missing folder: {folder}")

print("\n========== VERIFICATION COMPLETE ==========")
