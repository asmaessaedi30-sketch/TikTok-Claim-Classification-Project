"""
EDA Visualizations Module for TikTok Claim Classification Project
Author: Asma Essaedi
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_clean_data

def generate_eda_plots(data_path, output_dir):
    """
    Generates and saves core exploratory plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean_data(data_path)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Claim status distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='claim_status', hue='claim_status', palette=['#FE2C55', '#25F4EE'], legend=False)
    plt.title('Distribution of TikTok Videos: Claim vs. Opinion', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Claim Status', fontsize=12)
    plt.ylabel('Video Count', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claim_status_distribution.png'), dpi=300)
    plt.close()
    
    # 2. View Count by Claim Status
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='claim_status', y='video_view_count', hue='claim_status', palette=['#FE2C55', '#25F4EE'], legend=False)
    plt.title('Video View Count Distribution by Claim Status', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Claim Status', fontsize=12)
    plt.ylabel('View Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'engagement_by_claim_status.png'), dpi=300)
    plt.close()
    
    # 3. Verified Status vs Claim Status
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x='claim_status', hue='verified_status', palette=['#000000', '#25F4EE'])
    plt.title('Claim Status by Account Verification Status', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Claim Status', fontsize=12)
    plt.ylabel('Video Count', fontsize=12)
    plt.legend(title='Verified Status', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claim_by_verified_status.png'), dpi=300)
    plt.close()
    
    # 4. Author Ban Status vs Claim Status
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x='author_ban_status', hue='claim_status', palette=['#FE2C55', '#25F4EE'])
    plt.title('Author Ban Status Breakdown by Claim vs Opinion', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Author Ban Status', fontsize=12)
    plt.ylabel('Video Count', fontsize=12)
    plt.legend(title='Claim Status', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'author_ban_status_claim.png'), dpi=300)
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == '__main__':
    generate_eda_plots('../data/tiktok_dataset.csv', '../visualizations')
