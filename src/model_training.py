"""
Model Training and Evaluation Module for TikTok Claim Classification Project
Author: Asma Essaedi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import load_and_clean_data, prepare_features_and_target, split_data

def train_and_evaluate_models(data_path, output_dir):
    """
    Trains Logistic Regression, Random Forest, and Gradient Boosting models,
    evaluates their performance, and saves visual artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean_data(data_path)
    X, y = prepare_features_and_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)
    print("=== LOGISTIC REGRESSION (VAL SET) ===")
    print(classification_report(y_val, y_pred_lr, target_names=['Opinion', 'Claim']))
    
    # 2. Random Forest (Champion Model)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf_val = rf.predict(X_val)
    print("=== RANDOM FOREST (VAL SET) ===")
    print(classification_report(y_val, y_pred_rf_val, target_names=['Opinion', 'Claim']))
    
    # Evaluate Champion on Test Set
    y_pred_rf_test = rf.predict(X_test)
    print("=== RANDOM FOREST CHAMPION MODEL (TEST SET) ===")
    print(classification_report(y_test, y_pred_rf_test, target_names=['Opinion', 'Claim']))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf_test)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Opinion', 'Claim'], yticklabels=['Opinion', 'Claim'])
    plt.title('Random Forest Champion Model: Confusion Matrix (Test Set)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_rf.png'), dpi=300)
    plt.close()
    
    # Plot Feature Importances
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    importances.plot(kind='barh', color='#25F4EE')
    plt.title('Feature Importances - TikTok Claim Prediction', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print(f"Model artifacts and evaluation complete! Saved to {output_dir}")

if __name__ == '__main__':
    train_and_evaluate_models('../data/tiktok_dataset.csv', '../visualizations')
