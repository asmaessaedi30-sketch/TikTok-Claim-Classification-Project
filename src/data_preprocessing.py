"""
Data Preprocessing Module for TikTok Claim Classification Project
Author: Asma Essaedi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    """
    Loads dataset, handles missing values, and performs feature engineering.
    """
    df = pd.read_csv(file_path)
    
    # Drop 298 missing rows in claim_status and engagement metrics
    df_clean = df.dropna(subset=['claim_status']).copy()
    
    # Feature Engineering: Extract transcription text length
    df_clean['text_length'] = df_clean['video_transcription_text'].fillna('').str.len()
    
    # Binary Target: 1 for claim, 0 for opinion
    df_clean['claim_status_binary'] = (df_clean['claim_status'] == 'claim').astype(int)
    
    return df_clean

def prepare_features_and_target(df_clean):
    """
    Prepares feature matrix X and target vector y with dummy encoding.
    Excludes engagement count metrics to prevent target leakage and align
    with the champion Random Forest model defined in Notebook 04.
    """
    feature_cols = [
        'text_length', 'author_ban_status', 'verified_status'
    ]
    
    X = df_clean[feature_cols]
    X = pd.get_dummies(X, drop_first=True)
    y = df_clean['claim_status_binary']
    
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """
    Splits data into train (60%), validation (20%), and test (20%) sets.
    """
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, random_state=random_state, stratify=y_tr
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    data_path = '../data/tiktok_dataset.csv'
    df = load_and_clean_data(data_path)
    X, y = prepare_features_and_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Data split successfully:")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
