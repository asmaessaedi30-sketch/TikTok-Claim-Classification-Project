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
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_and_clean_data, prepare_features_and_target, split_data

def train_and_evaluate_models(data_path, output_dir):
    """
    Trains Logistic Regression, XGBoost, and Random Forest models (without target leakage),
    evaluates performance, and saves accurate visual artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean_data(data_path)
    X, y = prepare_features_and_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 1. Logistic Regression Baseline
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)
    print("=== LOGISTIC REGRESSION (VAL SET) ===")
    print(classification_report(y_val, y_pred_lr, target_names=['Opinion', 'Claim']))
    
    # 2. Random Forest Champion Model
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf_val = rf.predict(X_val)
    print("=== RANDOM FOREST CHAMPION MODEL (VAL SET) ===")
    print(classification_report(y_val, y_pred_rf_val, target_names=['Opinion', 'Claim']))
    
    # 3. Save Confusion Matrix Artifact (Validation Set)
    cm = confusion_matrix(y_val, y_pred_rf_val)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Opinion', 'Claim'],
                yticklabels=['Opinion', 'Claim'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('Random Forest Champion Model: Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Annotate metrics on plot
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    precision = cm[1,1] / (cm[1,1] + cm[0,1])
    plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.1%} | Claim Recall: {recall:.1%} | Claim Precision: {precision:.1%}",
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_rf.png'), dpi=300)
    plt.close()
    
    # 4. Save Feature Importances Artifact
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(9, 5.5))
    ax = importances.plot(kind='barh', color='#25F4EE', edgecolor='black')
    plt.title('Feature Importances - TikTok Claim Prediction', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    
    for p in ax.patches:
        val = p.get_width()
        if val > 0:
            ax.annotate(f'{val:.1%}', (val + 0.005, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, fontweight='bold')
            
    plt.xlim(0, max(importances) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print(f"Model artifacts and evaluation complete! Saved to {output_dir}")

if __name__ == '__main__':
    train_and_evaluate_models('data/tiktok_dataset.csv', 'visualizations')
