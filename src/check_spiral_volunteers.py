
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "spiral_data" / "spiral"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

for class_name in ["healthy", "parkinson"]:

    all_files = []

    for split in ["training", "testing"]:

        folder = DATA_DIR / split / class_name

        files = [
            file for file in folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        all_files.extend(files)

    volunteer_ids = sorted({
        file.stem[:3] for file in all_files
    })

    print(f"\n{class_name.upper()}")
    print("Total images:", len(all_files))
    print("Unique volunteers:", len(volunteer_ids))
    print("Volunteer IDs:", ", ".join(volunteer_ids))
