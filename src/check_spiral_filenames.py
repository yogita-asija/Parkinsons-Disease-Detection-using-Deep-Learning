
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "spiral_data" / "spiral"

for split in ["training", "testing"]:

    print(f"\n========== {split.upper()} ==========")

    for class_name in ["healthy", "parkinson"]:

        folder = DATA_DIR / split / class_name

        files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        ])

        print(f"\n{class_name}: {len(files)} images")

        # Show every filename
        for file in files:
            print(file.name)

        # Extract volunteer IDs, e.g. V01 from V01HE02
        volunteer_ids = [
            file.stem[:3]
            for file in files
        ]

        counts = Counter(volunteer_ids)

        print("\nVolunteer IDs and image counts:")
        for volunteer, count in sorted(counts.items()):
            print(f"{volunteer}: {count}")

# Check whether volunteer IDs occur in both classes
print("\n========== CROSS-CLASS ID CHECK ==========")

class_ids = {}

for class_name in ["healthy", "parkinson"]:

    folder = DATA_DIR / "training" / class_name

    class_ids[class_name] = {
        f.stem[:3]
        for f in folder.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    }

common = class_ids["healthy"] & class_ids["parkinson"]

print("Healthy IDs:", sorted(class_ids["healthy"]))
print("Parkinson IDs:", sorted(class_ids["parkinson"]))
print("Common IDs:", sorted(common))
