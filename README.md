# TikTok Video Claim Classification & Machine Learning Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.2%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Portfolio Project](https://img.shields.io/badge/Portfolio-Asma_Essaedi-purple.svg)](https://asmaessaedi30-sketch.github.io/AsmaEssaedPortfolio.github.io/)

> **Project Author:** [Asma Essaedi](https://www.linkedin.com/in/asma-essaedi-965891365/)  
> **Portfolio Website:** [Asma Essaedi Portfolio](https://asmaessaedi30-sketch.github.io/AsmaEssaedPortfolio.github.io/)  
> **Course Certification:** Google Advanced Data Analytics Capstone Project  

---

## 📌 Executive Summary

TikTok receives millions of user video submissions daily. To ensure platform safety, combat misinformation, and minimize user risk, human moderation teams must review videos flagged for potential policy violations. However, manual review of every video is computationally expensive and slow.

This project delivers an **end-to-end data science and machine learning pipeline** that automatically classifies whether a TikTok video presents a factual **Claim** or an **Opinion**. By accurately predicting user claims, TikTok can prioritize high-risk videos for human moderation, optimize system bandwidth, and reduce review backlogs by prioritizing videos with high public reach.

---

## 📁 Repository & Project Structure

```
TikTok-Claim-Classification-Project/
├── README.md                                # Comprehensive Project Overview & Portfolio Guide
├── requirements.txt                         # Python Dependencies & Libraries
├── data/
│   └── tiktok_dataset.csv                   # Cleaned Primary Dataset (19,382 rows x 12 columns)
├── notebooks/
│   ├── 01_tiktok_eda_and_visualization.ipynb    # Course 2: Exploratory Data Analysis & Visualizations
│   ├── 02_tiktok_logistic_regression.ipynb     # Course 4: Regression Modeling & Hypothesis Verification
│   └── 03_tiktok_machine_learning_models.ipynb # Course 5: Random Forest & XGBoost Machine Learning Models
├── src/                                     # Reusable Python Production Modules
│   ├── data_preprocessing.py                # Data loading, cleaning, and feature engineering
│   ├── eda_visualizations.py                # Exploratory plot generation script
│   └── model_training.py                    # Model training, evaluation, & metric reporting
├── docs/
│   ├── TikTok_Project_Proposal.pdf          # Official Project Charter & Objectives
│   ├── pace_strategy_documents/             # PACE (Plan, Analyze, Construct, Execute) Docs
│   │   ├── Course_1_PACE_Strategy_Document.pdf
│   │   ├── Course_2_PACE_Strategy_Document.pdf
│   │   ├── Course_3_PACE_Strategy_Document.pdf
│   │   ├── Course_4_PACE_Strategy_Document.pdf
│   │   └── Course_5_PACE_Strategy_Document.pdf
│   └── executive_summaries/                 # Stakeholder & C-Suite Executive Reports
│       ├── Executive_Summary_Project_Proposal_and_EDA.pdf
│       ├── Executive_Summary_Hypothesis_Testing.pdf
│       ├── Executive_Summary_Logistic_Regression.pdf
│       └── Executive_Summary_Random_Forest_vs_XGBoost.pdf
└── visualizations/                          # Generated Figures & Charts
    ├── claim_status_distribution.png
    ├── engagement_by_claim_status.png
    ├── claim_by_verified_status.png
    ├── author_ban_status_claim.png
    ├── confusion_matrix_rf.png
    └── feature_importance.png
```

---

## 📊 Key Data Insights & Exploratory Data Analysis

Analysis of **19,382 TikTok videos** revealed stark structural differences between videos making factual claims versus individual opinions:

### 1. Engagement Skew & High-Impact Claims
- Videos classified as **Claims** receive drastically higher engagement (views, likes, shares, downloads, comments) compared to **Opinions**.
- **99% of total platform video views** are concentrated among claim videos.
- Engagement distributions are heavily right-skewed, demonstrating that claims carry massive public exposure.

| Video Type | Median Views | Median Likes | Median Shares | Median Downloads |
| :--- | :--- | :--- | :--- | :--- |
| **Claim** | **501,557** | **122,863** | **24,374** | **4,921** |
| **Opinion** | **4,973** | **1,023** | **201** | **41** |

### 2. Verified Accounts vs. Unverified Accounts
- **Verified Accounts** predominantly post **Opinions** (over 94% of verified account videos are opinions).
- **Unverified Accounts** generate the vast majority of **Claims**.
- Active users who post claims face a significantly higher rate of being **Banned** or placed **Under Review**.

---

## 🖼 Visual Highlights

### Claim vs. Opinion Distribution
![Claim Status Distribution](visualizations/claim_status_distribution.png)

### Engagement Comparison (View Count by Claim Status)
![Engagement Boxplot](visualizations/engagement_by_claim_status.png)

### Claim Breakdown by Verification Status
![Verified Status Breakdown](visualizations/claim_by_verified_status.png)

### Feature Importance (Random Forest Champion Model)
![Feature Importance](visualizations/feature_importance.png)

---

## 🤖 Machine Learning Model Benchmarks

Multiple classification architectures were implemented and cross-validated using a **60% Train / 20% Validation / 20% Test** split.

### Model Evaluation Matrix

| Model Architecture | Precision (Claim) | Recall (Claim) | F1-Score (Claim) | Overall Accuracy | Champion Selection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 61.0% | 86.0% | 0.71 | 65.0% | Baseline |
| **XGBoost Classifier** | 70.0% | 73.0% | 0.72 | 71.5% | Candidate |
| **Random Forest (Tuned)** | **67.0%** | **76.0%** | **0.72** | **71.0%** | **🏆 Champion** |

> **Business Rationale for Champion Selection:**  
> In user safety and content moderation, **Recall is the primary business metric**. Misclassifying a claim as an opinion (False Negative) allows potentially harmful misinformation to spread unchecked. The **Random Forest model achieved 76% Recall**, successfully identifying the highest proportion of claim videos.

---

## 💡 Strategic Recommendations for TikTok Management

1. **Automated Triage System:** Implement the Random Forest champion model at ingestion to flag videos with predicted claim scores above 0.5 for priority human review.
2. **Focus on Unverified & Banned Authors:** Prioritize content review on unverified accounts, as they account for 99% of claim impressions.
3. **Text Transcription N-Gram Features:** Incorporate n-gram keyword extraction and sentiment scores from video transcriptions to further reduce false positive rates.
4. **Author Follower Count & History:** Join user account age and follower counts into future model iterations to strengthen predictive power.

---

## 🛠 How to Run & Replicate the Project

### Prerequisites
- Python 3.10 or higher
- Jupyter Notebook / JupyterLab

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/asmaessaedi30-sketch/TikTok-Claim-Classification-Project.git
cd TikTok-Claim-Classification-Project
pip install -r requirements.txt
```

### 2. Run Preprocessing & Feature Engineering
```bash
python3 src/data_preprocessing.py
```

### 3. Generate Exploratory Plots
```bash
python3 src/eda_visualizations.py
```

### 4. Train & Evaluate Models
```bash
python3 src/model_training.py
```

---

## 🔗 Project Resources & Documentation

- 📄 **Project Proposal:** [docs/TikTok_Project_Proposal.pdf](docs/TikTok_Project_Proposal.pdf)
- 📋 **PACE Strategy Framework:** [docs/pace_strategy_documents/](docs/pace_strategy_documents/)
- 📑 **Executive Summaries:** [docs/executive_summaries/](docs/executive_summaries/)
- 🌐 **Live Portfolio:** [https://asmaessaedi30-sketch.github.io/AsmaEssaedPortfolio.github.io/](https://asmaessaedi30-sketch.github.io/AsmaEssaedPortfolio.github.io/)
