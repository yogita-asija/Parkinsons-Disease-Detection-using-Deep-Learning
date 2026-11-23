
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "spiral_data" / "spiral"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 60
SEED = 42

CLASS_NAMES = ["healthy", "parkinson"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ==================================================
# 2. COLLECT ALL IMAGES
# ==================================================

print("\n========== COLLECTING SPIRAL IMAGES ==========")

records = []

for class_label, class_name in enumerate(CLASS_NAMES):

    for split in ["training", "testing"]:

        folder = DATA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder}"
            )

        for image_path in sorted(folder.iterdir()):

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Extract volunteer ID, e.g. V01 from V01HE02
            volunteer_id = image_path.stem[:3]

            # Class-specific group ID.
            # Healthy V01 and Parkinson V01 are separate groups.
            group_id = f"{class_name}_{volunteer_id}"

            records.append({
                "path": image_path,
                "label": class_label,
                "class_name": class_name,
                "volunteer": volunteer_id,
                "group": group_id
            })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No spiral images found.")

if df["group"].isna().any():
    raise ValueError("Some volunteer IDs are missing.")

print("Total images:", len(df))
print("\nImage distribution:")
print(df["class_name"].value_counts())

group_df = (
    df.groupby("group")
    .agg(
        label=("label", "first"),
        class_name=("class_name", "first"),
        volunteer=("volunteer", "first"),
        image_count=("path", "count")
    )
    .reset_index()
)

print("\nUnique volunteer groups:", len(group_df))
print("\nVolunteer distribution:")
print(group_df.groupby("class_name")["group"].count())

# ==================================================
# 3. SPLIT BY VOLUNTEER
# ==================================================

print("\n========== PARTICIPANT-WISE SPLITTING ==========")

# First reserve approximately 20% of volunteers for testing
train_val_groups, test_groups = train_test_split(
    group_df,
    test_size=0.20,
    random_state=SEED,
    stratify=group_df["label"]
)

# Then reserve approximately 20% of the remaining
# volunteers for validation
train_groups, val_groups = train_test_split(
    train_val_groups,
    test_size=0.20,
    random_state=SEED,
    stratify=train_val_groups["label"]
)

train_ids = set(train_groups["group"])
val_ids = set(val_groups["group"])
test_ids = set(test_groups["group"])

# Verify that no volunteer overlaps
assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

def get_records(selected_groups):
    return df[df["group"].isin(selected_groups)].reset_index(drop=True)

train_df = get_records(train_ids)
val_df = get_records(val_ids)
test_df = get_records(test_ids)

def print_split(name, split_df):

    print(f"\n{name}")
    print("Images:", len(split_df))
    print("Volunteers:", split_df["group"].nunique())

    print(
        split_df["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

print_split("TRAINING", train_df)
print_split("VALIDATION", val_df)
print_split("TESTING", test_df)

print("\nParticipant overlap:")
print("Train vs Validation:", train_ids & val_ids)
print("Train vs Test:", train_ids & test_ids)
print("Validation vs Test:", val_ids & test_ids)

# ==================================================
# 4. LOAD AND PREPROCESS IMAGES
# ==================================================

print("\n========== PREPROCESSING IMAGES ==========")

def load_images(split_df):

    images = []
    labels = []
    groups = []

    for _, row in split_df.iterrows():

        image = Image.open(row["path"]).convert("L")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        images.append(image_array)
        labels.append(int(row["label"]))
        groups.append(row["group"])

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    groups = np.array(groups)

    X = X[..., np.newaxis]

    return X, y, groups


X_train, y_train, train_group_array = load_images(train_df)
X_val, y_val, val_group_array = load_images(val_df)
X_test, y_test, test_group_array = load_images(test_df)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

# ==================================================
# 5. VOLUNTEER-BALANCED SAMPLE WEIGHTS
# ==================================================

# Give each volunteer approximately equal total
# influence, even if they contributed different
# numbers of drawings.

group_counts = Counter(train_group_array)

sample_weights = np.array([
    1.0 / group_counts[group]
    for group in train_group_array
], dtype=np.float32)

sample_weights *= len(sample_weights) / sample_weights.sum()

# ==================================================
# 6. CREATE TENSORFLOW DATASETS
# ==================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train, sample_weights)
)

train_ds = (
    train_ds
    .shuffle(
        buffer_size=len(X_train),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==================================================
# 7. BUILD A SMALLER CNN
# ==================================================

print("\n========== BUILDING CNN ==========")

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.025),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomTranslation(0.03, 0.03)
], name="augmentation")

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 1)),

    tf.keras.layers.Rescaling(1.0 / 255),

    augmentation,

    tf.keras.layers.Conv2D(
        16, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# ==================================================
# 8. CALLBACKS
# ==================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "spiral_best_model.keras"),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.000001
    )
]

# ==================================================
# 9. TRAIN MODEL
# ==================================================

print("\n========== TRAINING CNN ==========")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==================================================
# 10. LOAD BEST VALIDATION MODEL
# ==================================================

print("\n========== LOADING BEST MODEL ==========")

model = tf.keras.models.load_model(
    MODEL_DIR / "spiral_best_model.keras"
)

# ==================================================
# 11. FINAL TEST EVALUATION
# ==================================================

print("\n========== FINAL TEST EVALUATION ==========")

test_loss, test_accuracy, test_auc = model.evaluate(
    test_ds,
    verbose=1
)

probabilities = model.predict(
    test_ds,
    verbose=1
).reshape(-1)

predictions = (probabilities >= 0.5).astype(int)

y_true = y_test.astype(int)

accuracy = accuracy_score(y_true, predictions)

cm = confusion_matrix(
    y_true,
    predictions,
    labels=[0, 1]
)

print("\nTest Loss:", round(float(test_loss), 4))
print("Test Accuracy:", round(float(accuracy), 4))
print("Test AUC:", round(float(test_auc), 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

report = classification_report(
    y_true,
    predictions,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

test_roc_auc = None

if len(np.unique(y_true)) == 2:
    test_roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("Test ROC-AUC:", round(float(test_roc_auc), 4))

# ==================================================
# 12. SAVE CONFUSION MATRIX
# ==================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spiral CNN Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "spiral_confusion_matrix.png",
    dpi=300
)

plt.close()

# ==================================================
# 13. SAVE TRAINING HISTORY
# ==================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "spiral_training_history.png",
    dpi=300
)

plt.close()

# ==================================================
# 14. SAVE SPLIT AND MODEL METADATA
# ==================================================

split_metadata = {
    "training_volunteers": sorted(train_ids),
    "validation_volunteers": sorted(val_ids),
    "testing_volunteers": sorted(test_ids),
    "training_images": len(train_df),
    "validation_images": len(val_df),
    "testing_images": len(test_df),
    "participant_overlap": 0
}

with open(
    MODEL_DIR / "spiral_split_metadata.json",
    "w"
) as file:

    json.dump(split_metadata, file, indent=4)


metadata = {
    "model": "Small CNN",
    "image_size": list(IMAGE_SIZE),
    "class_names": CLASS_NAMES,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_auc": float(test_auc),
    "test_roc_auc": (
        float(test_roc_auc)
        if test_roc_auc is not None else None
    ),
    "confusion_matrix": cm.tolist(),
    "epochs_completed": len(history.history["loss"]),
    "split_method": "Stratified volunteer-wise split",
    "participant_overlap": 0
}

with open(
    MODEL_DIR / "spiral_model_metadata.json",
    "w"
) as file:

    json.dump(metadata, file, indent=4)

# ==================================================
# 15. FINISH
# ==================================================

print("\n========== TRAINING COMPLETED ==========")

print("Best model: models/spiral_best_model.keras")
print("Confusion matrix: models/spiral_confusion_matrix.png")
print("Training history: models/spiral_training_history.png")
print("Split details: models/spiral_split_metadata.json")
print("Model metadata: models/spiral_model_metadata.json")

print("\nParticipant-wise evaluation completed.")
