"""
EDA Visualizations Module for TikTok Claim Classification Project
Author: Asma Essaedi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_clean_data

def generate_eda_plots(data_path, output_dir):
    """
    Generates and saves core exploratory plots with enhanced visual clarity.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean_data(data_path)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Claim status distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='claim_status', hue='claim_status', palette=['#FE2C55', '#25F4EE'], legend=False)
    plt.title('Distribution of TikTok Videos: Claim vs. Opinion', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Claim Status', fontsize=12, fontweight='bold')
    plt.ylabel('Video Count', fontsize=12, fontweight='bold')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}\n({height/len(df):.1%})', (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claim_status_distribution.png'), dpi=300)
    plt.close()
    
    # 2. View Count by Claim Status (Log Scale for Visual Clarity)
    plt.figure(figsize=(9, 5.5))
    ax = sns.boxplot(data=df, x='claim_status', y='video_view_count', hue='claim_status', palette=['#FE2C55', '#25F4EE'], legend=False)
    plt.yscale('log')
    plt.title('Video View Count Distribution by Claim Status (Log Scale)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Claim Status', fontsize=12, fontweight='bold')
    plt.ylabel('View Count (Log Scale)', fontsize=12, fontweight='bold')
    
    # Annotate medians
    medians = df.groupby('claim_status')['video_view_count'].median()
    for i, (status, median_val) in enumerate(medians.items()):
        ax.annotate(f'Median: {int(median_val):,} views', 
                    xy=(i, median_val), xytext=(i, median_val * 2.5),
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'engagement_by_claim_status.png'), dpi=300)
    plt.close()
    
    # 3. Verified Status vs Claim Status (Grouped by Account Status for Insight Clarity)
    plt.figure(figsize=(9, 5.5))
    ax = sns.countplot(data=df, x='verified_status', hue='claim_status', palette=['#FE2C55', '#25F4EE'])
    plt.title('Claim vs. Opinion Proportion by Account Verification Status', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Account Verification Status', fontsize=12, fontweight='bold')
    plt.ylabel('Video Count', fontsize=12, fontweight='bold')
    plt.legend(title='Claim Status', frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2., height + 150),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.ylim(0, df['verified_status'].value_counts().max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claim_by_verified_status.png'), dpi=300)
    plt.close()
    
    # 4. Author Ban Status vs Claim Status
    plt.figure(figsize=(9, 5.5))
    ax = sns.countplot(data=df, x='author_ban_status', hue='claim_status', palette=['#FE2C55', '#25F4EE'],
                       order=['active', 'under review', 'banned'])
    plt.title('Author Ban Status Breakdown by Claim vs. Opinion', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Author Ban Status', fontsize=12, fontweight='bold')
    plt.ylabel('Video Count', fontsize=12, fontweight='bold')
    plt.legend(title='Claim Status', frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2., height + 150),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.ylim(0, df['author_ban_status'].value_counts().max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'author_ban_status_claim.png'), dpi=300)
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == '__main__':
    generate_eda_plots('data/tiktok_dataset.csv', 'visualizations')
