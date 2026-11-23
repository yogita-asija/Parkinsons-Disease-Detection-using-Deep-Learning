
# Parkinson's Disease Detection Using Deep Learning

An experimental machine learning application that explores Parkinson's disease classification using voice measurements and spiral drawing images.

The project combines voice-based feature analysis and image-based pattern recognition in an interactive Streamlit web application.

> **Disclaimer:** This is an academic research prototype, not a clinically validated medical diagnostic system. Its predictions must not be used for medical decisions.

## Features

- Voice-based classification using 22 biomedical voice features.
- Spiral drawing classification using Histogram of Oriented Gradients (HOG).
- Random Forest, Support Vector Machine, Logistic Regression and ANN experiments for voice data.
- CNN and traditional computer vision experiments for spiral images.
- Interactive Streamlit web application.
- Model performance dashboard with evaluation metrics and confusion matrix.
- Volunteer-wise data splitting to reduce participant leakage.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, Pillow, Scikit-image |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Plotly |
| Model Serialization | Joblib |
| Development Environment | VS Code |

## Project Architecture

```text
Parkinsons-Disease-Detection/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── parkinsons.csv
│   └── spiral_data/
│       └── spiral/
│           ├── training/
│           │   ├── healthy/
│           │   └── parkinson/
│           └── testing/
│               ├── healthy/
│               └── parkinson/
│
├── models/
│   ├── voice_best_model.joblib
│   ├── voice_random_forest.joblib
│   ├── voice_svm.joblib
│   ├── voice_logistic_regression.joblib
│   ├── voice_ann.joblib
│   ├── spiral_hog_best_model.joblib
│   ├── spiral_best_model.keras
│   ├── voice_model_metadata.json
│   ├── spiral_hog_model_metadata.json
│   └── evaluation result files
│
├── src/
│   ├── train_voice.py
│   ├── train_spiral.py
│   ├── train_spiral_hog.py
│   ├── inspect_spiral.py
│   ├── check_spiral_filenames.py
│   └── check_spiral_volunteers.py
│
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## Datasets

### 1. Voice Dataset

The voice dataset contains 195 voice recordings and 24 columns.

- 22 biomedical voice features.
- `name` identifies the recording.
- `status` represents the target class.
- 0 represents healthy.
- 1 represents Parkinson's.

The features include:

- MDVP frequency measurements.
- Jitter and shimmer measurements.
- Noise-to-Harmonics Ratio (NHR).
- Harmonics-to-Noise Ratio (HNR).
- RPDE and DFA.
- Spread and PPE measurements.

The participant-wise evaluation split contains 146 training recordings and 49 testing recordings, with no participant overlap.

### 2. Spiral Drawing Dataset

The spiral dataset contains 102 images.

| Class | Training | Testing | Total |
|---|---:|---:|---:|
| Healthy | 36 | 15 | 51 |
| Parkinson's | 36 | 15 | 51 |
| **Total** | **72** | **30** | **102** |

The experiments use volunteer-wise splitting to prevent images from the same volunteer appearing across different evaluation splits.

## Methodology

### Voice Classification

The voice classification workflow includes:

1. Loading the voice dataset.
2. Removing non-feature columns.
3. Handling missing values.
4. Feature scaling.
5. Addressing class imbalance using SMOTE within the training pipeline.
6. Grouped cross-validation.
7. Training and comparing multiple classifiers.
8. Evaluating the selected model.

Models evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Artificial Neural Network (ANN)

**Selected model:** Random Forest, based on the highest grouped cross-validation ROC-AUC.

### Spiral Image Classification

The spiral image workflow includes:

1. Loading spiral drawing images.
2. Converting images to grayscale.
3. Resizing images to 128 × 128 pixels.
4. Extracting Histogram of Oriented Gradients (HOG) features.
5. Training classification models.
6. Selecting a model using validation ROC-AUC.
7. Evaluating on the held-out test volunteers.

HOG configuration:

- Orientations: 9
- Pixels per cell: 8 × 8
- Cells per block: 2 × 2
- Block normalization: L2-Hys
- Feature count: 8100

Models evaluated:

- HOG + Support Vector Machine
- HOG + Random Forest
- Convolutional Neural Network (CNN) experiment

**Selected model:** HOG + Random Forest.

## Model Evaluation Results

### Voice Model Comparison

| Model | Grouped CV ROC-AUC | Test Accuracy | Test F1 |
|---|---:|---:|---:|
| Random Forest | 78.39% | 77.55% | 85.33% |
| ANN | 75.98% | 69.39% | 74.58% |
| SVM | 73.46% | 81.63% | 86.96% |
| Logistic Regression | 69.76% | 83.67% | 87.88% |

Random Forest was selected based on grouped cross-validation ROC-AUC, not test accuracy.

### Spiral Model Results

| Metric | Result |
|---|---:|
| Selected Model | HOG + Random Forest |
| Accuracy | 83.33% |
| Precision | 82.35% |
| Recall | 93.33% |
| F1 Score | 87.50% |
| ROC-AUC | 94.07% |

Spiral test confusion matrix:

| | Predicted Healthy | Predicted Parkinson |
|---|---:|---:|
| Actual Healthy | 6 | 3 |
| Actual Parkinson | 1 | 14 |

These results are preliminary and based on a very small number of participants. They should not be interpreted as clinical diagnostic performance.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Parkinsons-Disease-Detection-using-Deep-Learning
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Run the application

```powershell
streamlit run app/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Application Modules

### Home

Provides an overview of the project and its two analysis approaches.

### Voice Prediction

Accepts 22 biomedical voice measurements and returns the selected model's predicted class and probability output.

### Spiral Prediction

Accepts an uploaded spiral drawing, extracts HOG features and generates a classification result.

### Model Performance

Displays:

- Voice model comparison.
- Grouped cross-validation ROC-AUC.
- Voice test metrics.
- Spiral test metrics.
- Spiral confusion matrix.
- Dataset and validation information.

## Limitations

- The datasets contain relatively few participants and recordings.
- Spiral test evaluation includes only six volunteer groups.
- Model probabilities are not calibrated clinical risk estimates.
- Results may vary across different datasets and populations.
- The models have not undergone clinical validation.
- Image-level results do not necessarily represent independent patient-level performance.
- The application is intended for academic experimentation and learning.

## Future Improvements

- Collect larger and more diverse datasets.
- Perform external validation on independent participants.
- Improve image preprocessing and feature extraction.
- Explore multimodal model fusion.
- Improve model interpretability using explainable AI techniques.
- Evaluate model calibration and generalization.
- Improve the user interface and deployment workflow.

## Disclaimer

This project is developed for educational and research purposes only.

It is not intended to replace professional medical advice, diagnosis or treatment. Never use its predictions to make healthcare decisions.

## Author

Developed as an academic machine learning and deep learning project.
