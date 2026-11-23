
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import json

from PIL import Image
from pathlib import Path
from skimage.feature import hog


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

VOICE_MODEL_PATH = MODEL_DIR / "voice_best_model.joblib"
VOICE_DATA_PATH = DATA_DIR / "parkinsons.csv"

SPIRAL_MODEL_PATH = MODEL_DIR / "spiral_hog_best_model.joblib"

VOICE_RESULTS_PATH = MODEL_DIR / "voice_model_comparison.csv"
SPIRAL_RESULTS_PATH = MODEL_DIR / "spiral_hog_test_results.csv"

SPIRAL_METADATA_PATH = MODEL_DIR / "spiral_hog_model_metadata.json"


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 3. CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #9CA3AF;
    margin-bottom: 30px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 4. LOAD DATA AND MODELS
# ==========================================================

@st.cache_data
def load_voice_data():

    df = pd.read_csv(VOICE_DATA_PATH)

    feature_names = [
        column for column in df.columns
        if column not in ["name", "status"]
    ]

    medians = df[feature_names].median()

    return feature_names, medians


@st.cache_resource
def load_voice_model():

    return joblib.load(VOICE_MODEL_PATH)


@st.cache_resource
def load_spiral_model():

    return joblib.load(SPIRAL_MODEL_PATH)


@st.cache_data
def load_voice_results():

    return pd.read_csv(VOICE_RESULTS_PATH)


@st.cache_data
def load_spiral_results():

    return pd.read_csv(SPIRAL_RESULTS_PATH)


@st.cache_data
def load_spiral_metadata():

    with open(SPIRAL_METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================================
# 5. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("🧠 Parkinson's Detection")

st.sidebar.caption(
    "Deep Learning & Machine Learning Project"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Voice Prediction",
        "Spiral Prediction",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Academic Project | Research Prototype"
)


# ==========================================================
# 6. HOME PAGE
# ==========================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">'
        'Parkinson\'s Disease Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An experimental machine learning system using '
        'voice measurements and spiral drawing analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This application is an academic research prototype. "
        "It is not a medical diagnostic tool."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎙️ Voice Analysis")

        st.write(
            "Analyze 22 biomedical voice measurements "
            "using a trained machine learning model."
        )

        st.write("**Model:** Random Forest")

    with col2:

        st.subheader("✍️ Spiral Drawing Analysis")

        st.write(
            "Extract visual features from uploaded spiral "
            "drawings using Histogram of Oriented Gradients."
        )

        st.write("**Model:** HOG + Random Forest")

    st.divider()

    st.subheader("How to use the application")

    st.write("1. Select Voice Prediction to enter voice measurements.")
    st.write("2. Select Spiral Prediction to upload a spiral drawing.")
    st.write("3. Visit Model Performance to review evaluation results.")

    st.caption(
        "Predictions are experimental and must not be used "
        "for medical decisions."
    )


# ==========================================================
# 7. VOICE PREDICTION
# ==========================================================

elif page == "Voice Prediction":

    st.title("🎙️ Voice-Based Parkinson's Prediction")

    st.write(
        "Enter the 22 biomedical voice measurements below."
    )

    st.warning(
        "Use measurements obtained through an appropriate "
        "voice-analysis procedure. Dataset medians are only "
        "initial values for testing the interface."
    )

    try:

        feature_names, medians = load_voice_data()
        voice_model = load_voice_model()

        if len(feature_names) != 22:
            st.error("Expected exactly 22 voice features.")
            st.stop()

        st.success("Voice prediction model loaded successfully.")

        st.subheader("Enter Voice Measurements")

        with st.form("voice_prediction_form"):

            input_values = {}

            col1, col2 = st.columns(2)

            for index, feature in enumerate(feature_names):

                target_col = col1 if index % 2 == 0 else col2

                with target_col:

                    input_values[feature] = st.number_input(
                        feature,
                        value=float(medians[feature]),
                        format="%.6f"
                    )

            submitted = st.form_submit_button(
                "Predict",
                use_container_width=True
            )

        if submitted:

            input_df = pd.DataFrame(
                [input_values],
                columns=feature_names
            )

            prediction = int(voice_model.predict(input_df)[0])

            probabilities = voice_model.predict_proba(input_df)[0]

            classes = list(voice_model.classes_)

            parkinson_index = classes.index(1)

            parkinson_probability = float(
                probabilities[parkinson_index]
            )

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Model output: Parkinson's class")
            else:
                st.success("Model output: Healthy class")

            st.metric(
                "Model probability for Parkinson's class",
                f"{parkinson_probability * 100:.2f}%"
            )

            st.caption(
                "This probability is a model output, not a "
                "clinical risk estimate or diagnostic certainty."
            )

            st.info(
                "This experimental result must not be used "
                "to diagnose or rule out Parkinson's disease."
            )

    except Exception as error:

        st.error(f"Unable to run voice prediction: {error}")


# ==========================================================
# 8. SPIRAL PREDICTION
# ==========================================================

elif page == "Spiral Prediction":

    st.title("✍️ Spiral Drawing-Based Prediction")

    st.write(
        "Upload a spiral drawing to analyze its visual features "
        "using the trained HOG + Random Forest model."
    )

    st.warning(
        "This is an experimental academic model, not a "
        "medical diagnostic tool."
    )

    try:

        spiral_model = load_spiral_model()

        st.success("Spiral prediction model loaded successfully.")

        uploaded_file = st.file_uploader(
            "Upload Spiral Drawing",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("L")

            st.subheader("Uploaded Drawing")

            st.image(
                image,
                caption="Uploaded spiral drawing",
                width=350
            )

            if st.button(
                "Analyze Spiral Drawing",
                use_container_width=True
            ):

                image_array = np.array(image)

                image_array = cv2.resize(
                    image_array,
                    (128, 128)
                )

                features = hog(
                    image_array,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys"
                )

                input_features = features.reshape(1, -1)

                if input_features.shape[1] != 8100:
                    st.error("Expected 8100 HOG features.")
                    st.stop()

                prediction = int(
                    spiral_model.predict(input_features)[0]
                )

                probabilities = spiral_model.predict_proba(
                    input_features
                )[0]

                classes = list(spiral_model.classes_)

                parkinson_index = classes.index(1)

                parkinson_probability = float(
                    probabilities[parkinson_index]
                )

                st.divider()
                st.subheader("Prediction Result")

                if prediction == 1:
                    st.error("Model output: Parkinson's class")
                else:
                    st.success("Model output: Healthy class")

                st.metric(
                    "Model probability for Parkinson's class",
                    f"{parkinson_probability * 100:.2f}%"
                )

                st.caption(
                    "This probability is a model output, not a "
                    "clinical risk estimate or diagnostic certainty."
                )

                st.info(
                    "This experimental prediction must not be "
                    "used for medical decisions."
                )

    except Exception as error:

        st.error(f"Unable to run spiral prediction: {error}")


# ==========================================================
# 9. MODEL PERFORMANCE DASHBOARD
# ==========================================================

elif page == "Model Performance":

    st.title("📊 Model Performance Dashboard")

    st.write(
        "Evaluation results from the trained voice and "
        "spiral classification models."
    )

    st.warning(
        "These are experimental results on small datasets. "
        "They do not represent clinical diagnostic performance."
    )

    try:

        voice_results = load_voice_results()
        spiral_results = load_spiral_results()
        spiral_metadata = load_spiral_metadata()

        # ----------------------------------------------
        # VOICE MODEL PERFORMANCE
        # ----------------------------------------------

        st.header("🎙️ Voice Model Performance")

        st.subheader("Model Comparison")

        voice_display = voice_results.copy()

        metric_columns = [
            "CV Mean ROC-AUC",
            "CV Std ROC-AUC",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1",
            "Test ROC-AUC"
        ]

        voice_display[metric_columns] = (
            voice_display[metric_columns] * 100
        )

        st.dataframe(
            voice_display.style.format({
                column: "{:.2f}%"
                for column in metric_columns
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Cross-validation scores use grouped splits. "
            "The CV standard deviation is shown as a percentage-point "
            "spread across folds."
        )

        st.subheader("Grouped Cross-Validation ROC-AUC")

        cv_chart = voice_results[
            ["Model", "CV Mean ROC-AUC"]
        ].set_index("Model")

        st.bar_chart(cv_chart)

        st.subheader("Voice Test Accuracy Comparison")

        accuracy_chart = voice_results[
            ["Model", "Test Accuracy"]
        ].set_index("Model")

        st.bar_chart(accuracy_chart)

        # ----------------------------------------------
        # SPIRAL MODEL PERFORMANCE
        # ----------------------------------------------

        st.divider()

        st.header("✍️ Spiral Model Performance")

        spiral_row = spiral_results.iloc[0]

        st.write(
            f"Selected Model: **{spiral_row['Model']}**"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Accuracy",
            f"{spiral_row['Accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{spiral_row['Precision'] * 100:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{spiral_row['Recall'] * 100:.2f}%"
        )

        col4.metric(
            "F1 Score",
            f"{spiral_row['F1 Score'] * 100:.2f}%"
        )

        col5.metric(
            "ROC-AUC",
            f"{spiral_row['ROC-AUC'] * 100:.2f}%"
        )

        st.subheader("Spiral Test Metrics")

        spiral_display = spiral_results.copy()

        spiral_metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]

        spiral_display[spiral_metric_columns] = (
            spiral_display[spiral_metric_columns] * 100
        )

        st.dataframe(
            spiral_display.style.format({
                column: "{:.2f}%"
                for column in spiral_metric_columns
            }),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------

        st.subheader("Spiral Confusion Matrix")

        cm = np.array(
            spiral_metadata["confusion_matrix"]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Healthy", "Actual Parkinson"],
            columns=["Predicted Healthy", "Predicted Parkinson"]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        st.caption(
            "Rows represent actual classes and columns "
            "represent predicted classes."
        )

        # ----------------------------------------------
        # DATASET AND VALIDATION INFORMATION
        # ----------------------------------------------

        st.divider()

        st.header("Dataset and Validation Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Training Volunteers",
            spiral_metadata["training_volunteers"]
        )

        col2.metric(
            "Validation Volunteers",
            spiral_metadata["validation_volunteers"]
        )

        col3.metric(
            "Testing Volunteers",
            spiral_metadata["testing_volunteers"]
        )

        st.write(
            "**Volunteer overlap between splits:**",
            spiral_metadata["volunteer_overlap"]
        )

        st.write(
            "**HOG feature count:**",
            spiral_metadata["feature_count"]
        )

        st.info(
            spiral_metadata["note"]
        )

        st.caption(
            "Voice dataset: 195 recordings from 32 participants. "
            "Spiral dataset: 102 images. Both datasets are small, "
            "and their results should be interpreted cautiously."
        )

    except Exception as error:

        st.error(
            f"Unable to load performance results: {error}"
        )


# ==========================================================
# END OF APPLICATION
# ==========================================================
