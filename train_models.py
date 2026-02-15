"""
Model Training Script
Trains all 6 classification models and saves them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / 'data'
MODEL_DIR = SCRIPT_DIR / 'model'

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Preprocessing and Evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a classification model and return metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy    : {metrics['Accuracy']:.4f}")
    print(f"AUC         : {metrics['AUC']:.4f}")
    print(f"Precision   : {metrics['Precision']:.4f}")
    print(f"Recall      : {metrics['Recall']:.4f}")
    print(f"F1          : {metrics['F1']:.4f}")
    print(f"MCC         : {metrics['MCC']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return metrics, y_pred, cm


def train_all_models():
    """
    Main function to train all models
    """
    print("🚀 Starting model training...")
    
    # Create model directory if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load the preprocessed data
    print("\n📂 Loading data...")
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train.csv').values.ravel()
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').values.ravel()
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature Scaling
    print("\n⚙️ Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved")
    
    # Store all results
    all_metrics = []
    confusion_matrices = []
    
    # 1. Logistic Regression
    print("\n🔄 Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_metrics, lr_pred, lr_cm = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    all_metrics.append(lr_metrics)
    confusion_matrices.append(('Logistic Regression', lr_cm))
    
    with open(MODEL_DIR / 'logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("✅ Logistic Regression saved")
    
    # 2. Decision Tree
    print("\n🔄 Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20)
    dt_model.fit(X_train, y_train)
    dt_metrics, dt_pred, dt_cm = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    all_metrics.append(dt_metrics)
    confusion_matrices.append(('Decision Tree', dt_cm))
    
    with open(MODEL_DIR / 'decision_tree.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    print("✅ Decision Tree saved")
    
    # 3. K-Nearest Neighbors
    print("\n🔄 Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_model.fit(X_train_scaled, y_train)
    knn_metrics, knn_pred, knn_cm = evaluate_model(knn_model, X_test_scaled, y_test, "K-Nearest Neighbors (kNN)")
    all_metrics.append(knn_metrics)
    confusion_matrices.append(('K-Nearest Neighbors', knn_cm))
    
    with open(MODEL_DIR / 'knn.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    print("✅ kNN saved")
    
    # 4. Naive Bayes
    print("\n🔄 Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    nb_metrics, nb_pred, nb_cm = evaluate_model(nb_model, X_test_scaled, y_test, "Naive Bayes")
    all_metrics.append(nb_metrics)
    confusion_matrices.append(('Naive Bayes', nb_cm))
    
    with open(MODEL_DIR / 'naive_bayes.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    print("✅ Naive Bayes saved")
    
    # 5. Random Forest
    print("\n🔄 Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_metrics, rf_pred, rf_cm = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    all_metrics.append(rf_metrics)
    confusion_matrices.append(('Random Forest', rf_cm))
    
    with open(MODEL_DIR / 'random_forest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✅ Random Forest saved")
    
    # 6. XGBoost
    print("\n🔄 Training XGBoost...")
    xgb_model = XGBClassifier(
        random_state=42, 
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_metrics, xgb_pred, xgb_cm = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    all_metrics.append(xgb_metrics)
    confusion_matrices.append(('XGBoost', xgb_cm))
    
    # Save using joblib for better compatibility
    joblib.dump(xgb_model, MODEL_DIR / 'xgboost.pkl')
    print("✅ XGBoost saved")
    
    # Create comparison table
    print("\n📊 Creating comparison table...")
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(MODEL_DIR / 'model_comparison.csv', index=False)
    print("✅ Comparison table saved to model/model_comparison.csv")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    # Performance comparison bar chart
    metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Performance comparison chart saved")
    
    # Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, (name, cm) in enumerate(confusion_matrices):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrices saved")
    
    print("\n" + "="*60)
    print("🎉 All models trained and saved successfully!")
    print("="*60)
    
    return results_df


if __name__ == "__main__":
    train_all_models()
