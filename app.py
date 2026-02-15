"""
Streamlit Web Application for Classification Model Analysis
Assignment 2 - Machine Learning
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Classification Model Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import subprocess
import sys
from pathlib import Path
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
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Auto-setup: Download data and train models if not present
@st.cache_data
def check_setup_status():
    """Check if data and models exist"""
    data_exists = os.path.exists('data/X_train.csv') and os.path.exists('data/y_train.csv')
    models_exist = all([
        os.path.exists('model/logistic_regression.pkl'),
        os.path.exists('model/decision_tree.pkl'),
        os.path.exists('model/knn.pkl'),
        os.path.exists('model/naive_bayes.pkl'),
        os.path.exists('model/random_forest.pkl'),
        os.path.exists('model/xgboost.pkl')
    ])
    return data_exists, models_exist

def setup_if_needed():
    """
    Automatically download data and train models if they don't exist
    """
    data_exists, models_exist = check_setup_status()
    
    if not data_exists:
        with st.spinner("📥 Downloading dataset... This may take a moment."):
            try:
                result = subprocess.run([sys.executable, 'download_data.py'], 
                                      check=True, 
                                      capture_output=True, 
                                      text=True)
                st.success("✅ Dataset downloaded successfully!")
                st.cache_data.clear()  # Clear cache to recheck
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Error downloading data: {e.stderr}")
                with st.expander("Show error details"):
                    st.code(e.stderr)
                st.stop()
    
    if not models_exist:
        st.info("🤖 Training models for the first time... This will take a few minutes.")
        progress_bar = st.progress(0, text="Initializing training...")
        
        try:
            # Run training script
            process = subprocess.Popen(
                [sys.executable, 'train_models.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Show progress
            model_count = 0
            for line in process.stdout:
                if "Training" in line:
                    model_count += 1
                    model_name = line.split("Training")[1].split("...")[0].strip()
                    progress = min(model_count / 6.0, 1.0)
                    progress_bar.progress(progress, text=f"Training {model_name}...")
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                progress_bar.progress(1.0, text="Training complete!")
                st.success("✅ All models trained successfully!")
                st.cache_data.clear()  # Clear cache to recheck
            else:
                st.error(f"❌ Error training models.")
                with st.expander("Show error details"):
                    st.code(stderr if stderr else stdout)
                st.stop()
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")
            st.stop()

# Run setup check
setup_if_needed()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Classification Model Analysis</p>', unsafe_allow_html=True)
st.markdown("### Assignment 2 - Machine Learning | M.Tech (AIML)")
st.markdown("---")

# Data Description Section
with st.expander("📋 Data Format & Description", expanded=False):
    st.markdown("""
    ### Bank Marketing Dataset - Required CSV Format
    
    Your CSV file should contain the following **20 features** (without the target column):
    
    #### 📊 Numeric Features (10)
    1. **age**: Age of the client (integer)
    2. **duration**: Last contact duration in seconds (integer)
    3. **campaign**: Number of contacts performed during this campaign (integer)
    4. **pdays**: Number of days since last contact from previous campaign (integer, 999 = never contacted)
    5. **previous**: Number of contacts before this campaign (integer)
    6. **emp.var.rate**: Employment variation rate - quarterly indicator (numeric)
    7. **cons.price.idx**: Consumer price index - monthly indicator (numeric)
    8. **cons.conf.idx**: Consumer confidence index - monthly indicator (numeric)
    9. **euribor3m**: Euribor 3 month rate - daily indicator (numeric)
    10. **nr.employed**: Number of employees - quarterly indicator (numeric)
    
    #### 📝 Categorical Features (10)
    11. **job**: Type of job (admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)
    12. **marital**: Marital status (divorced, married, single, unknown)
    13. **education**: Education level (basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown)
    14. **default**: Has credit in default? (no, yes, unknown)
    15. **housing**: Has housing loan? (no, yes, unknown)
    16. **loan**: Has personal loan? (no, yes, unknown)
    17. **contact**: Contact communication type (cellular, telephone)
    18. **month**: Last contact month (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec)
    19. **day_of_week**: Last contact day of week (mon, tue, wed, thu, fri)
    20. **poutcome**: Outcome of previous campaign (failure, nonexistent, success)
    
    #### 🎯 Optional Target Column
    - **target**: Client subscribed to term deposit? (0 = No, 1 = Yes)
    - If included, the app will calculate evaluation metrics
    - If not included, only predictions will be shown
    
    #### 💡 Tips for Creating Your Own Data:
    - All categorical features should match the exact values listed above
    - Numeric features should be within reasonable ranges (e.g., age: 18-100)
    - Use the **Download Sample Data** button to see the exact format
    - Ensure column names match exactly (case-sensitive)
    """)

# Sidebar
st.sidebar.title("🎯 Model Configuration")
st.sidebar.markdown("### Select a trained classification model")

# Model descriptions
model_info = {
    "Logistic Regression": "Linear model for binary classification. Fast and interpretable.",
    "Decision Tree": "Tree-based model that makes decisions using feature thresholds.",
    "K-Nearest Neighbors (kNN)": "Instance-based learning using distance metrics.",
    "Naive Bayes": "Probabilistic classifier based on Bayes' theorem.",
    "Random Forest": "Ensemble of decision trees for robust predictions.",
    "XGBoost": "Advanced gradient boosting ensemble method."
}

# Model selection dropdown (1 MARK)
model_options = list(model_info.keys())
selected_model_name = st.sidebar.selectbox(
    "Choose Model:",
    model_options,
    index=0,
    help="Select a classification model to evaluate"
)

# Display model description
st.sidebar.info(f"**About {selected_model_name}:**\n\n{model_info[selected_model_name]}")

# Model file mapping
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors (kNN)": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# Models that need scaled data
models_need_scaling = ["Logistic Regression", "K-Nearest Neighbors (kNN)", "Naive Bayes"]

# Load scaler
@st.cache_resource
def load_scaler():
    """Load the StandardScaler used during training"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Scaler not found. Please run the training notebook first.")
        return None

# Load model
@st.cache_resource
def load_model(model_name):
    """Load a trained model from pickle/joblib file"""
    model_file = model_files[model_name]
    model_path = f"model/{model_file}"
    
    try:
        # Use joblib for XGBoost (better compatibility)
        if model_name == "XGBoost":
            return joblib.load(model_path)
        
        # Use pickle for other models
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        st.info("💡 Please run the `train_models.ipynb` notebook first to generate model files.")
        return None

# Main content
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Upload Test Data")

# Sample data download button
if os.path.exists('data/X_test.csv'):
    with open('data/X_test.csv', 'rb') as f:
        sample_data = f.read()
    st.sidebar.download_button(
        label="📥 Download Sample Data",
        data=sample_data,
        file_name="sample_test_data.csv",
        mime="text/csv",
        help="Download sample data to see the required format"
    )

# File uploader (1 MARK)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with test data:",
    type=['csv'],
    help="Upload a CSV file containing test data with the same features as training data"
)

# Add sample data option
use_sample = st.sidebar.checkbox("Use sample test data", value=False)

# Main application logic
if uploaded_file is not None or use_sample:
    
    # Load test data
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
    else:
        # Use sample data from training
        try:
            X_test_sample = pd.read_csv('data/X_test.csv')
            y_test_sample = pd.read_csv('data/y_test.csv')
            # Combine features and target
            test_data = X_test_sample.copy()
            test_data['target'] = y_test_sample.values.ravel()
            st.info("ℹ️ Using sample test data from training set")
        except FileNotFoundError:
            st.error("❌ Sample test data not found. Please upload a CSV file.")
            st.stop()
    
    # Display data preview
    with st.expander("📊 View Uploaded Data", expanded=False):
        st.dataframe(test_data.head(10))
        st.write(f"**Dataset Shape:** {test_data.shape[0]} rows × {test_data.shape[1]} columns")
        
        # Show column names
        st.write("**Columns:**", ", ".join(test_data.columns.tolist()))
    
    # Check if target column exists (for evaluation)
    has_target = 'target' in test_data.columns
    
    if has_target:
        # Separate features and target
        X_test = test_data.drop('target', axis=1)
        y_test_true = test_data['target'].values
        st.info("✅ Target column detected - will show evaluation metrics")
    else:
        X_test = test_data
        y_test_true = None
        st.warning("⚠️ No 'target' column found - will show predictions only")
    
    # Load model and scaler
    model = load_model(selected_model_name)
    scaler = load_scaler()
    
    if model is not None:
        
        # Prepare data (scale if needed)
        if selected_model_name in models_need_scaling and scaler is not None:
            X_test_processed = scaler.transform(X_test)
            st.info(f"🔄 Features scaled for {selected_model_name}")
        else:
            X_test_processed = X_test
        
        # Make predictions
        try:
            y_pred = model.predict(X_test_processed)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            else:
                y_pred_proba = None
            
            st.success(f"✅ Predictions generated using {selected_model_name}")
            
        except Exception as e:
            st.error(f"❌ Error making predictions: {e}")
            st.stop()
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Metrics", 
            "🎯 Predictions", 
            "📊 Confusion Matrix",
            "📋 Classification Report"
        ])
        
        # TAB 1: Display Evaluation Metrics (1 MARK)
        with tab1:
            st.subheader("📈 Model Performance Metrics")
            
            if y_test_true is not None:
                # Calculate all metrics
                accuracy = accuracy_score(y_test_true, y_pred)
                precision = precision_score(y_test_true, y_pred, average='binary')
                recall = recall_score(y_test_true, y_pred, average='binary')
                f1 = f1_score(y_test_true, y_pred, average='binary')
                mcc = matthews_corrcoef(y_test_true, y_pred)
                
                # Calculate AUC if probabilities available
                if y_pred_proba is not None:
                    auc = roc_auc_score(y_test_true, y_pred_proba)
                else:
                    auc = None
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    st.metric("Precision", f"{precision:.4f}")
                
                with col2:
                    if auc is not None:
                        st.metric("AUC Score", f"{auc:.4f}")
                    else:
                        st.metric("AUC Score", "N/A")
                    st.metric("Recall", f"{recall:.4f}")
                
                with col3:
                    st.metric("F1 Score", f"{f1:.4f}")
                    st.metric("MCC Score", f"{mcc:.4f}")
                
                st.markdown("---")
                
                # Metrics explanation
                with st.expander("ℹ️ Understanding Metrics"):
                    st.markdown("""
                    - **Accuracy**: Overall correctness of predictions
                    - **AUC**: Area Under ROC Curve - model's ability to distinguish classes
                    - **Precision**: Of predicted positives, how many are actually positive
                    - **Recall**: Of actual positives, how many were predicted correctly
                    - **F1 Score**: Harmonic mean of Precision and Recall
                    - **MCC**: Matthews Correlation Coefficient - balanced measure even with imbalanced classes
                    """)
                
                # Metrics comparison chart
                st.subheader("📊 Metrics Visualization")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                    'Score': [accuracy, auc if auc else 0, precision, recall, f1, mcc]
                })
                
                fig = px.bar(
                    metrics_df, 
                    x='Metric', 
                    y='Score',
                    title=f'{selected_model_name} - Performance Metrics',
                    color='Score',
                    color_continuous_scale='Blues',
                    text='Score'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ Target values not available. Upload data with 'target' column for metrics.")
        
        # TAB 2: Predictions
        with tab2:
            st.subheader("🎯 Model Predictions")
            
            # Create predictions dataframe
            predictions_df = X_test.copy()
            predictions_df['Predicted'] = y_pred
            predictions_df['Predicted_Label'] = predictions_df['Predicted'].map({0: 'No', 1: 'Yes'})
            
            if y_test_true is not None:
                predictions_df['Actual'] = y_test_true
                predictions_df['Actual_Label'] = predictions_df['Actual'].map({0: 'No', 1: 'Yes'})
                predictions_df['Correct'] = predictions_df['Predicted'] == predictions_df['Actual']
            
            if y_pred_proba is not None:
                predictions_df['Probability'] = y_pred_proba
            
            # Show prediction summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Predictions", len(y_pred))
                st.metric("Predicted Positive (1)", int(y_pred.sum()))
            
            with col2:
                st.metric("Predicted Negative (0)", int((y_pred == 0).sum()))
                if y_test_true is not None:
                    correct_preds = (y_pred == y_test_true).sum()
                    st.metric("Correct Predictions", f"{correct_preds} ({correct_preds/len(y_pred)*100:.1f}%)")
            
            # Display predictions table
            st.dataframe(predictions_df, use_container_width=True, height=400)
            
            # Download predictions
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        # TAB 3: Confusion Matrix (1 MARK)
        with tab3:
            st.subheader("📊 Confusion Matrix")
            
            if y_test_true is not None:
                cm = confusion_matrix(y_test_true, y_pred)
                
                # Create two columns for different visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plotly heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Negative', 'Predicted Positive'],
                        y=['Actual Negative', 'Actual Positive'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Blues',
                        showscale=True
                    ))
                    fig.update_layout(
                        title=f'Confusion Matrix - {selected_model_name}',
                        xaxis_title='Predicted Label',
                        yaxis_title='Actual Label'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Normalized confusion matrix
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    fig2 = go.Figure(data=go.Heatmap(
                        z=cm_normalized,
                        x=['Predicted Negative', 'Predicted Positive'],
                        y=['Actual Negative', 'Actual Positive'],
                        text=np.around(cm_normalized, decimals=3),
                        texttemplate='%{text}',
                        colorscale='Greens',
                        showscale=True
                    ))
                    fig2.update_layout(
                        title='Normalized Confusion Matrix',
                        xaxis_title='Predicted Label',
                        yaxis_title='Actual Label'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Confusion matrix metrics
                st.markdown("### 📊 Confusion Matrix Breakdown")
                tn, fp, fn, tp = cm.ravel()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("True Negatives", tn)
                col2.metric("False Positives", fp)
                col3.metric("False Negatives", fn)
                col4.metric("True Positives", tp)
                
            else:
                st.warning("⚠️ Confusion matrix requires actual target values.")
        
        # TAB 4: Classification Report (1 MARK)
        with tab4:
            st.subheader("📋 Detailed Classification Report")
            
            if y_test_true is not None:
                # Generate classification report
                report = classification_report(y_test_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                # Style the dataframe
                st.dataframe(
                    report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                    use_container_width=True
                )
                
                # Additional insights
                st.markdown("### 🔍 Classification Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Class 0 (Negative) Performance:**")
                    st.write(f"- Precision: {report['0']['precision']:.4f}")
                    st.write(f"- Recall: {report['0']['recall']:.4f}")
                    st.write(f"- F1-Score: {report['0']['f1-score']:.4f}")
                    st.write(f"- Support: {int(report['0']['support'])}")
                
                with col2:
                    st.markdown("**Class 1 (Positive) Performance:**")
                    st.write(f"- Precision: {report['1']['precision']:.4f}")
                    st.write(f"- Recall: {report['1']['recall']:.4f}")
                    st.write(f"- F1-Score: {report['1']['f1-score']:.4f}")
                    st.write(f"- Support: {int(report['1']['support'])}")
                
            else:
                st.warning("⚠️ Classification report requires actual target values.")

else:
    # Landing page when no file is uploaded
    st.info("👈 Please upload a CSV file or select 'Use sample test data' from the sidebar to begin.")
    
    st.markdown("### 📖 How to Use This App")
    st.markdown("""
    1. **Select a Model:** Choose from 6 trained classification models in the sidebar
    2. **Upload Data:** Upload your test data CSV file (with or without target column)
    3. **View Results:** Explore predictions, metrics, confusion matrix, and detailed reports
    
    ### 📋 Required CSV Format
    Your CSV should contain the same features used during training:
    - All numerical or encoded categorical features
    - Optional: 'target' column for evaluation metrics
    
    ### 🎯 Available Models
    """)
    
    for model, desc in model_info.items():
        st.markdown(f"- **{model}**: {desc}")
    
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    
    # Try to load and display comparison if available
    try:
        comparison_df = pd.read_csv('model/model_comparison.csv')
        st.dataframe(comparison_df, use_container_width=True)
    except FileNotFoundError:
        st.info("💡 Run the training notebook to generate model comparison table.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Classification Model Analysis & Comparison</p>
</div>
""", unsafe_allow_html=True)
