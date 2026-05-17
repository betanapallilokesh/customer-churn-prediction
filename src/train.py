"""
train.py
End-to-End Customer Churn Prediction & Retention Analytics Pipeline
- Data Preprocessing & Encoding
- Stratified Train-Test Split (80/20)
- Feature Scaling
- Model Training: Baseline Logistic Regression vs. Random Forest Classifier
- Model Evaluation (ROC-AUC, Precision, Recall, F1, Accuracy)
- Visual Artifacts: ROC Curves & Feature Importance Plot
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

def run_pipeline():
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'telco_customer_churn.csv')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("[1/5] Loading and inspecting dataset...")
    df = pd.read_csv(data_path)
    print(f"      Total records: {df.shape[0]}, Features: {df.shape[1]}")
    
    # 1. Preprocessing
    print("[2/5] Performing data preprocessing and feature encoding...")
    # Drop identifier
    df_clean = df.drop(columns=['customerID'])
    
    # Encode target
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    # Handle TotalCharges numeric type
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['MonthlyCharges'] * df_clean['tenure'])
    
    # One-Hot Encoding for categorical features
    categorical_cols = df_clean.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    
    # Save feature columns list for inference
    feature_names = X.columns.tolist()
    
    # 2. Stratified Train-Test Split (maintains churn ratio across splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    
    # 3. Scaling numerical features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # 4. Training Models
    print("[3/5] Training models...")
    
    # Model A: Baseline Logistic Regression (with balanced class weights)
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # Model B: Random Forest Classifier (handles non-linear interactions)
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # 5. Evaluation
    print("[4/5] Evaluating performance metrics...")
    
    # Predictions
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "Logistic_Regression": {
            "Accuracy": round(float(accuracy_score(y_test, y_pred_lr)), 4),
            "Precision": round(float(precision_score(y_test, y_pred_lr)), 4),
            "Recall": round(float(recall_score(y_test, y_pred_lr)), 4),
            "F1_Score": round(float(f1_score(y_test, y_pred_lr)), 4),
            "ROC_AUC": round(float(roc_auc_score(y_test, y_prob_lr)), 4)
        },
        "Random_Forest": {
            "Accuracy": round(float(accuracy_score(y_test, y_pred_rf)), 4),
            "Precision": round(float(precision_score(y_test, y_pred_rf)), 4),
            "Recall": round(float(recall_score(y_test, y_pred_rf)), 4),
            "F1_Score": round(float(f1_score(y_test, y_pred_rf)), 4),
            "ROC_AUC": round(float(roc_auc_score(y_test, y_prob_rf)), 4)
        }
    }
    
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE SUMMARY")
    print("="*50)
    for model_name, m in metrics.items():
        print(f"[{model_name}]")
        print(f"  • ROC-AUC Score : {m['ROC_AUC']}")
        print(f"  • Recall        : {m['Recall']}")
        print(f"  • Precision     : {m['Precision']}")
        print(f"  • F1-Score      : {m['F1_Score']}")
        print(f"  • Accuracy      : {m['Accuracy']}\n")
    print("="*50)
    
    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 6. Generate Visual Artifacts
    print("[5/5] Generating visual artifacts...")
    
    # Plot 1: ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {metrics['Logistic_Regression']['ROC_AUC']})", color='#1f77b4', lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {metrics['Random_Forest']['ROC_AUC']})", color='#2ca02c', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing (AUC = 0.50)')
    plt.title('ROC Curve Comparison - Customer Churn Prediction', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"      Saved ROC curve to: {roc_plot_path}")
    
    # Plot 2: Top 10 Feature Importances (Random Forest)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importance_values = importances[indices]
    
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(x=top_importance_values, y=top_features, hue=top_features, palette='viridis', legend=False)
    plt.title('Top 10 Drivers of Customer Churn (Random Forest)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Relative Feature Importance Score', fontsize=12)
    plt.ylabel('Customer Feature', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    feat_plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(feat_plot_path)
    plt.close()
    print(f"      Saved Feature Importance plot to: {feat_plot_path}")
    
    # Save artifacts for inference
    artifacts = {
        'model': rf,
        'scaler': scaler,
        'feature_names': feature_names,
        'numeric_features': numeric_features
    }
    model_save_path = os.path.join(output_dir, 'churn_model_artifacts.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"      Saved model artifacts to: {model_save_path}")
    print("\nPipeline execution completed successfully!")

if __name__ == '__main__':
    run_pipeline()
