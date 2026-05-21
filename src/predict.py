"""
predict.py
Inference module for Customer Churn Risk Prediction.
Loads trained artifacts and computes churn probabilities for new customer profiles.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def load_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'output', 'churn_model_artifacts.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifacts not found at {model_path}. Please run train.py first.")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_churn(customer_data: dict):
    """
    Computes churn probability and key risk flags for a given customer profile.
    """
    artifacts = load_artifacts()
    model = artifacts['model']
    feature_names = artifacts['feature_names']
    
    # Create single-row DataFrame
    df_raw = pd.DataFrame([customer_data])
    
    # Process numeric columns
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
    # Dummy encoding aligned with training features
    df_encoded = pd.get_dummies(df_raw)
    
    # Align columns with training feature columns
    aligned_row = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in df_encoded.columns:
        if col in feature_names:
            aligned_row[col] = df_encoded[col]
            
    # Compute probability using Random Forest
    churn_prob = model.predict_proba(aligned_row)[0, 1]
    
    risk_level = "HIGH RISK" if churn_prob >= 0.60 else ("MEDIUM RISK" if churn_prob >= 0.35 else "LOW RISK")
    
    return {
        "churn_probability": round(float(churn_prob) * 100, 2),
        "risk_level": risk_level,
        "contract": customer_data.get('Contract', 'Unknown'),
        "tenure_months": customer_data.get('tenure', 0),
        "monthly_charges": customer_data.get('MonthlyCharges', 0.0)
    }

if __name__ == '__main__':
    print("Testing churn inference with sample customer profiles...\n")
    
    # Sample A: High Risk Customer (Month-to-month, short tenure, high charges, no support)
    sample_high_risk = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.50,
        'TotalCharges': 191.00
    }
    
    # Sample B: Low Risk Customer (Two year contract, long tenure, bundled security)
    sample_low_risk = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 60,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 55.00,
        'TotalCharges': 3300.00
    }
    
    res_a = predict_churn(sample_high_risk)
    res_b = predict_churn(sample_low_risk)
    
    print(f"Customer A (Short tenure, Month-to-Month):")
    print(f"  • Churn Probability: {res_a['churn_probability']}%")
    print(f"  • Risk Category:     {res_a['risk_level']}\n")
    
    print(f"Customer B (Long tenure, Two-Year Contract):")
    print(f"  • Churn Probability: {res_b['churn_probability']}%")
    print(f"  • Risk Category:     {res_b['risk_level']}")
