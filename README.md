# Classification Model Analysis

## Problem Statement
This project implements and compares multiple classification models to predict whether a bank client will subscribe to a term deposit based on marketing campaign data. The goal is to evaluate and compare the performance of six different machine learning algorithms using comprehensive metrics including accuracy, AUC, precision, recall, F1-score, and Matthews Correlation Coefficient (MCC).

## Dataset Description

**Dataset:** Bank Marketing Dataset (UCI Machine Learning Repository)  
**Source:** https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

### Overview:
- **Total Instances:** 41,188
- **Total Features:** 21 (20 input features + 1 target variable)
- **Classification Type:** Binary Classification
- **Target Variable:** Client subscription to term deposit (yes/no)

### Features:
The dataset contains information about bank clients and their interactions with marketing campaigns:

**Client Information:**
- age, job, marital status, education level, credit default status, account balance

**Campaign Information:**
- contact type, contact day/month, contact duration, number of contacts
- days since previous contact, previous campaign outcome

**Economic Indicators:**
- employment variation rate, consumer price index, consumer confidence index
- euribor 3 month rate, number of employees

### Target Variable:
- **0:** Client did not subscribe to term deposit
- **1:** Client subscribed to term deposit

### Data Quality:
- No missing values after preprocessing
- Categorical variables encoded numerically
- Dataset is relatively imbalanced (typical for marketing data)

## Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9139 | 0.9370 | 0.7002 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.9161 | 0.9213 | 0.6357 | 0.5981 | 0.6163 | 0.5696 |
| kNN | 0.9036 | 0.8626 | 0.6095 | 0.4019 | 0.4844 | 0.4452 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| Random Forest (Ensemble) | 0.9204 | 0.9515 | 0.7556 | 0.4332 | 0.5507 | 0.5344 |
| XGBoost (Ensemble) | 0.9219 | 0.9547 | 0.6829 | 0.5733 | 0.6233 | 0.5829 |

### Observations

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | Achieves good accuracy (91.39%) and excellent AUC (93.70%), demonstrating strong ability to separate classes. High precision (70.02%) but lower recall (41.27%), indicating it's conservative in positive predictions. Best suited for scenarios where false positives are costly. Fast training and prediction make it ideal for real-time applications. |
| Decision Tree | Balanced performance with good recall (59.81%) and F1-score (61.63%), making it better at identifying positive cases than Logistic Regression. Moderate precision (63.57%) suggests some false positives. Interpretable model structure allows easy understanding of decision rules. Prone to overfitting but performs well with max_depth=10 constraint. |
| kNN | Lowest overall performance among all models with accuracy of 90.36% and AUC of 86.26%. Poor recall (40.19%) indicates it misses many positive cases. Distance-based approach struggles with imbalanced data. Computationally expensive for large datasets. Not recommended for this problem due to class imbalance sensitivity. |
| Naive Bayes | Lowest accuracy (85.36%) but highest recall (61.75%), making it excellent at identifying positive cases despite more false positives. Low precision (40.24%) means many predictions are incorrect. Strong probabilistic foundation but independence assumption may not hold for this dataset. Useful when identifying all positive cases is critical. |
| Random Forest (Ensemble) | Second-best model with excellent AUC (95.15%) and highest precision (75.56%), making predictions highly reliable. Lower recall (43.32%) suggests conservative predictions. Ensemble approach provides robustness and handles non-linear relationships well. Good balance between bias and variance. Excellent choice when precision is prioritized. |
| XGBoost (Ensemble) | **Best overall model** with highest accuracy (92.19%), AUC (95.47%), and F1-score (62.33%). Best balance between precision (68.29%) and recall (57.33%), making it most effective at identifying positive cases while maintaining reliability. Gradient boosting handles class imbalance well. Most suitable for production deployment with superior generalization ability. |

## Project Structure
```
classification-model-analysis/
│-- app.py (Streamlit web application)
│-- requirements.txt
│-- README.md
│-- model/ (saved model files and notebooks)
│-- data/ (dataset files)
```

## Installation & Deployment

### Quick Start (Single Command Deployment)

**The easiest way to run this project:**

```bash
# Clone the repository
git clone https://github.com/bips1996/classification-model-analysis.git
cd classification-model-analysis

# Install dependencies
pip install -r requirements.txt

# Run the app (automatically downloads data and trains models on first run)
streamlit run app.py
```

That's it! The app will:
1. ✅ Automatically download the dataset if not present
2. ✅ Automatically train all 6 models if not present
3. ✅ Launch the interactive web application

### Manual Setup (Optional)

If you prefer to run steps separately:

1. **Clone the repository:**
```bash
git clone https://github.com/bips1996/classification-model-analysis.git
cd classification-model-analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
```bash
python download_data.py
```

4. **Train models:**
```bash
python train_models.py
```

5. **Run the app:**
```bash
streamlit run app.py
```

## Usage

### Streamlit Web Application
The app provides an interactive interface to:
- Select from 6 trained classification models
- Upload custom CSV files or use sample data
- View comprehensive evaluation metrics
- Analyze confusion matrices and classification reports

### Development/Analysis
For exploration and development:
```bash
jupyter notebook model/data_exploration.ipynb
jupyter notebook model/train_models.ipynb
```

## Live Demo
🚀 **[Launch Streamlit App](https://2025aa05343.streamlit.app/)**

Experience the interactive classification model analysis application deployed on Streamlit Cloud.

## Repository
📦 **[GitHub Repository](https://github.com/bips1996/classification-model-analysis)**
