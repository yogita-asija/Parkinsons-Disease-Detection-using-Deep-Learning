
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "spiral_data" / "spiral"

classes = ["healthy", "parkinson"]

fig, axes = plt.subplots(2, 4, figsize=(12, 7))

print("\n========== IMAGE INSPECTION ==========")

for row, class_name in enumerate(classes):

    folder = DATA_DIR / "training" / class_name

    images = sorted([
        file for file in folder.iterdir()
        if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ])

    print(f"\nClass: {class_name}")
    print("Number of images:", len(images))

    for col, image_path in enumerate(images[:4]):

        image = Image.open(image_path).convert("L")
        array = np.array(image)

        print(
            image_path.name,
            "| Size:", image.size,
            "| Min pixel:", array.min(),
            "| Max pixel:", array.max(),
            "| Mean pixel:", round(float(array.mean()), 2)
        )

        axes[row, col].imshow(array, cmap="gray")
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(ROOT / "models" / "spiral_dataset_preview.png", dpi=200)
plt.show()

print("\nPreview saved to models/spiral_dataset_preview.png")
