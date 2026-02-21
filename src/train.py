import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
import shap
from src.preprocessing import load_data, create_rfm_features, prepare_model_data, assign_rfm_segment

def main():
    print("🔄 Loading Data...")
    # Adjust path if running from root
    data_path = 'data/online_retail_II.xlsx'
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found. Please download the UCI dataset.")
        return

    df = load_data(data_path)
    rfm_df = create_rfm_features(df)

    print("🛠 Preparing Features...")
    X, y, feature_names = prepare_model_data(rfm_df)

    print("🚀 Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost Model
    model = XGBClassifier(eval_metric='logloss', scale_pos_weight=5)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n📊 Model Performance:")
    print(report)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # --- Save everything ---
    os.makedirs('models', exist_ok=True)

    # Model & feature names
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')

    # Metrics
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "test_samples": int(len(y_test)),
        "feature_importances": {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, model.feature_importances_)
        },
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # SHAP explainer (tree-based, fast)
    print("🔍 Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, 'models/shap_explainer.pkl')

    # Segmented customer data for the dashboard
    print("📂 Segmenting customers...")
    segmented_df = assign_rfm_segment(rfm_df)
    segmented_df.to_csv('models/rfm_segmented.csv', index=False)

    print("✅ Model, metrics, SHAP explainer, and segments saved to models/")

if __name__ == "__main__":
    main()
