
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "parkinsons.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

print("\n========== LOADING VOICE DATASET ==========")

df = pd.read_csv(DATA_PATH)

# Extract participant ID from recording name
df["subject"] = df["name"].str.extract(r"_(S\d+)_")

if df["subject"].isna().any():
    raise ValueError("Some participant IDs could not be extracted.")

# Verify each participant has only one diagnosis label
label_counts = df.groupby("subject")["status"].nunique()

if (label_counts > 1).any():
    raise ValueError("A participant has inconsistent diagnosis labels.")

# Do not use name or subject as model features
X = df.drop(columns=["name", "subject", "status"])
y = df["status"].astype(int)
groups = df["subject"]

feature_names = X.columns.tolist()

print("Total recordings:", len(df))
print("Unique participants:", groups.nunique())
print("\nParticipant distribution:")

subject_labels = df.groupby("subject")["status"].first()

print(
    subject_labels.value_counts()
    .sort_index()
    .rename(index={0: "Healthy", 1: "Parkinsons"})
)

print("\nRecording distribution:")
print(y.value_counts().sort_index())

# --------------------------------------------------
# 3. PARTICIPANT-WISE TRAIN/TEST SPLIT
# --------------------------------------------------

print("\n========== PARTICIPANT-WISE SPLITTING ==========")

# Four folds: approximately 25% of participants in test
outer_cv = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    outer_cv.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

groups_train = groups.iloc[train_indices]
groups_test = groups.iloc[test_indices]

# Explicitly verify no participant appears in both sets
train_subjects = set(groups_train)
test_subjects = set(groups_test)

if not train_subjects.isdisjoint(test_subjects):
    raise RuntimeError("Participant leakage detected!")

print("Training recordings:", len(X_train))
print("Testing recordings:", len(X_test))

print("Training participants:", len(train_subjects))
print("Testing participants:", len(test_subjects))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

print("\nParticipant overlap:", train_subjects.intersection(test_subjects))

# --------------------------------------------------
# 4. DEFINE MODELS
# --------------------------------------------------

def create_pipeline(classifier):

    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=3
        )),
        ("classifier", classifier)
    ])


models = {
    "Logistic Regression": create_pipeline(
        LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        )
    ),

    "SVM": create_pipeline(
        SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        )
    ),

    "Random Forest": create_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            class_weight=None
        )
    ),

    "ANN": create_pipeline(
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    )
}

# --------------------------------------------------
# 5. GROUPED CROSS-VALIDATION
# --------------------------------------------------

print("\n========== GROUPED CROSS-VALIDATION ==========")

# Three folds on training participants.
# This leaves approximately two healthy participants
# in each validation fold.
group_cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Explicitly verify group separation in every CV fold
cv_splits = list(
    group_cv.split(X_train, y_train, groups_train)
)

for fold_number, (tr_idx, val_idx) in enumerate(cv_splits, start=1):

    fold_train_groups = set(groups_train.iloc[tr_idx])
    fold_val_groups = set(groups_train.iloc[val_idx])

    if not fold_train_groups.isdisjoint(fold_val_groups):
        raise RuntimeError(
            f"Participant leakage in CV fold {fold_number}"
        )

print("Participant leakage check passed for all CV folds.")

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv_splits,
        scoring="roc_auc",
        n_jobs=1
    )

    cv_results[name] = {
        "mean_auc": float(scores.mean()),
        "std_auc": float(scores.std()),
        "fold_scores": scores.tolist()
    }

    print(
        f"{name}: "
        f"Mean ROC-AUC = {scores.mean():.4f}, "
        f"Std = {scores.std():.4f}"
    )

# --------------------------------------------------
# 6. SELECT BEST MODEL USING CV ROC-AUC
# --------------------------------------------------

best_name = max(
    cv_results,
    key=lambda name: cv_results[name]["mean_auc"]
)

print("\nBest model based on grouped CV:", best_name)

# --------------------------------------------------
# 7. TRAIN AND EVALUATE ALL MODELS
# --------------------------------------------------

print("\n========== FINAL TEST EVALUATION ==========")

comparison = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Healthy", "Parkinsons"],
            zero_division=0
        )
    )

    comparison.append({
        "Model": name,
        "CV Mean ROC-AUC": cv_results[name]["mean_auc"],
        "CV Std ROC-AUC": cv_results[name]["std_auc"],
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1,
        "Test ROC-AUC": auc
    })

    # Save individual model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        MODEL_DIR / f"voice_{filename}.joblib"
    )

# --------------------------------------------------
# 8. SAVE COMPARISON AND BEST MODEL
# --------------------------------------------------

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    by="CV Mean ROC-AUC",
    ascending=False
)

comparison_path = MODEL_DIR / "voice_model_comparison.csv"

comparison_df.to_csv(comparison_path, index=False)

# Refit selected model on the complete training set
best_model = models[best_name]
best_model.fit(X_train, y_train)

joblib.dump(
    best_model,
    MODEL_DIR / "voice_best_model.joblib"
)

# --------------------------------------------------
# 9. SAVE MODEL METADATA
# --------------------------------------------------

metadata = {
    "best_model": best_name,
    "selection_metric": "Grouped cross-validation ROC-AUC",
    "feature_names": feature_names,
    "total_recordings": int(len(df)),
    "total_participants": int(groups.nunique()),
    "training_recordings": int(len(X_train)),
    "testing_recordings": int(len(X_test)),
    "training_participants": int(len(train_subjects)),
    "testing_participants": int(len(test_subjects)),
    "train_test_participant_overlap": 0,
    "cv_folds": 3,
    "test_split_folds": 4,
    "random_state": RANDOM_STATE,
    "grouped_cv_results": cv_results
}

with open(
    MODEL_DIR / "voice_model_metadata.json",
    "w"
) as file:

    json.dump(metadata, file, indent=4)

# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n========== MODEL COMPARISON ==========")
print(comparison_df.to_string(index=False))

print("\nBest model:", best_name)

print("\nSaved files:")
print("models/voice_logistic_regression.joblib")
print("models/voice_svm.joblib")
print("models/voice_random_forest.joblib")
print("models/voice_ann.joblib")
print("models/voice_best_model.joblib")
print("models/voice_model_comparison.csv")
print("models/voice_model_metadata.json")

print("\nParticipant-wise evaluation completed successfully.")
