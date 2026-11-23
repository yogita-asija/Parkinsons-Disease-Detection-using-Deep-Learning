
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from skimage.feature import hog

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "spiral_data" / "spiral"
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD THE EXISTING VOLUNTEER-WISE SPLIT
# --------------------------------------------------

metadata_path = MODEL_DIR / "spiral_split_metadata.json"

if not metadata_path.exists():
    raise FileNotFoundError(
        "spiral_split_metadata.json not found. "
        "Run the participant-wise CNN training script first."
    )

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

train_groups = set(metadata["training_volunteers"])
validation_groups = set(metadata["validation_volunteers"])
test_groups = set(metadata["testing_volunteers"])

all_groups = train_groups | validation_groups | test_groups

if (
    train_groups & validation_groups
    or train_groups & test_groups
    or validation_groups & test_groups
):
    raise ValueError("Volunteer overlap detected between splits.")


# --------------------------------------------------
# 3. COLLECT ALL IMAGE PATHS
# --------------------------------------------------

records = []

class_labels = {
    "healthy": 0,
    "parkinson": 1
}

for class_name, label in class_labels.items():

    for original_split in ["training", "testing"]:

        folder = DATA_DIR / original_split / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        for image_path in folder.glob("*"):

            if image_path.suffix.lower() not in [
                ".png", ".jpg", ".jpeg", ".bmp"
            ]:
                continue

            # Example filename: V01HE02.png
            # Volunteer ID: V01
            volunteer_id = image_path.stem[:3]

            group_id = f"{class_name}_{volunteer_id}"

            if group_id not in all_groups:
                raise ValueError(
                    f"Volunteer {group_id} is missing from split metadata."
                )

            if group_id in train_groups:
                split = "train"
            elif group_id in validation_groups:
                split = "validation"
            elif group_id in test_groups:
                split = "test"
            else:
                raise ValueError(f"No split assigned to {group_id}")

            records.append({
                "path": image_path,
                "label": label,
                "group": group_id,
                "split": split
            })


# Verify every image has been assigned exactly once.
image_paths = [str(record["path"].resolve()) for record in records]

if len(image_paths) != len(set(image_paths)):
    raise ValueError("Duplicate image paths detected.")

if not records:
    raise ValueError("No images found.")

print("\nDataset collection completed.")
print("Total images:", len(records))

for split_name in ["train", "validation", "test"]:

    split_records = [
        record for record in records
        if record["split"] == split_name
    ]

    groups = {record["group"] for record in split_records}

    healthy = sum(record["label"] == 0 for record in split_records)
    parkinson = sum(record["label"] == 1 for record in split_records)

    print(
        f"{split_name.title()}: "
        f"{len(split_records)} images | "
        f"{len(groups)} volunteers | "
        f"Healthy: {healthy} | Parkinson: {parkinson}"
    )


# --------------------------------------------------
# 4. EXTRACT HOG FEATURES
# --------------------------------------------------

def extract_hog_features(image_path):

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, IMAGE_SIZE)

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


print("\nExtracting HOG features...")

X = np.array([
    extract_hog_features(record["path"])
    for record in records
])

y = np.array([record["label"] for record in records])
groups = np.array([record["group"] for record in records])
splits = np.array([record["split"] for record in records])

print("Feature matrix shape:", X.shape)


# --------------------------------------------------
# 5. CREATE TRAIN, VALIDATION AND TEST SETS
# --------------------------------------------------

train_mask = splits == "train"
validation_mask = splits == "validation"
test_mask = splits == "test"

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[validation_mask], y[validation_mask]
X_test, y_test = X[test_mask], y[test_mask]

train_group_set = set(groups[train_mask])
val_group_set = set(groups[validation_mask])
test_group_set = set(groups[test_mask])

assert not (train_group_set & val_group_set)
assert not (train_group_set & test_group_set)
assert not (val_group_set & test_group_set)

print("\nVolunteer overlap check: PASSED")


# --------------------------------------------------
# 6. DEFINE MODELS
# --------------------------------------------------

models = {

    "HOG_SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "HOG_Random_Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE
    )
}


# --------------------------------------------------
# 7. TRAIN AND SELECT USING VALIDATION DATA
# --------------------------------------------------

validation_results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    val_predictions = (val_probabilities >= 0.5).astype(int)

    val_auc = roc_auc_score(y_val, val_probabilities)

    val_accuracy = accuracy_score(y_val, val_predictions)

    validation_results.append({
        "Model": model_name,
        "Validation Accuracy": val_accuracy,
        "Validation ROC-AUC": val_auc
    })

    trained_models[model_name] = model

    print("Validation Accuracy:", round(val_accuracy, 4))
    print("Validation ROC-AUC:", round(val_auc, 4))


validation_df = pd.DataFrame(validation_results)

validation_df.to_csv(
    MODEL_DIR / "spiral_hog_validation_results.csv",
    index=False
)

# Select the model using validation ROC-AUC only.
best_model_name = validation_df.sort_values(
    "Validation ROC-AUC",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)


# --------------------------------------------------
# 8. FINAL TEST EVALUATION
# --------------------------------------------------

test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= 0.5).astype(int)

test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)
test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)
test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)
test_auc = roc_auc_score(y_test, test_probabilities)

cm = confusion_matrix(y_test, test_predictions)

print("\n========== FINAL TEST RESULTS ==========")
print("Selected Model:", best_model_name)
print("Accuracy:", round(test_accuracy, 4))
print("Precision:", round(test_precision, 4))
print("Recall:", round(test_recall, 4))
print("F1 Score:", round(test_f1, 4))
print("ROC-AUC:", round(test_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Healthy", "Parkinson"],
        zero_division=0
    )
)


# --------------------------------------------------
# 9. SAVE MODELS AND RESULTS
# --------------------------------------------------

for model_name, model in trained_models.items():

    filename = model_name.lower() + ".joblib"

    joblib.dump(model, MODEL_DIR / filename)


joblib.dump(
    best_model,
    MODEL_DIR / "spiral_hog_best_model.joblib"
)

results = {
    "selected_model": best_model_name,
    "test_accuracy": float(test_accuracy),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "test_roc_auc": float(test_auc),
    "confusion_matrix": cm.tolist(),
    "training_volunteers": len(train_group_set),
    "validation_volunteers": len(val_group_set),
    "testing_volunteers": len(test_group_set),
    "volunteer_overlap": 0,
    "feature_count": int(X.shape[1]),
    "note": (
        "Exploratory results from a small dataset. "
        "Not validated for clinical diagnosis."
    )
}

with open(
    MODEL_DIR / "spiral_hog_model_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(results, file, indent=4)


test_results = pd.DataFrame([{
    "Model": best_model_name,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1 Score": test_f1,
    "ROC-AUC": test_auc
}])

test_results.to_csv(
    MODEL_DIR / "spiral_hog_test_results.csv",
    index=False
)

print("\nAll HOG models and results saved successfully.")
print("Output directory:", MODEL_DIR)
