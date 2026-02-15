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
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| XGBoost (Ensemble) | | | | | | |

### Observations

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| XGBoost (Ensemble) | |

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
[Streamlit App Link - To be updated after deployment]

## Repository
[GitHub Repository Link - https://github.com/bips1996/classification-model-analysis]
