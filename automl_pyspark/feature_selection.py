"""
feature_selection.py

Feature selection utilities for AutoML PySpark package.
Includes:
    - Feature importance extraction
    - Feature importance plotting
    - Saving feature importance to Excel
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_feature_importance(output_dir, user_id, mdl_ltrl, importance_df):
    """
    Draw and save a horizontal bar plot of feature importances.
    
    Args:
        output_dir: Output directory path
        user_id: User identifier
        mdl_ltrl: Model literal
        importance_df: DataFrame with feature importance data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by importance score for better visualization (highest importance first)
    importance_df = importance_df.sort_values('Importance_Score', ascending=False)
    
    # Limit to top 30 features for better readability
    if len(importance_df) > 30:
        importance_df = importance_df.head(30)  # Take top 30 highest importance
        title_suffix = " (Top 30)"
    else:
        title_suffix = f" (All {len(importance_df)} features)"
    
    # Reverse order for horizontal bar plot (so highest importance appears at top)
    importance_df = importance_df.sort_values('Importance_Score', ascending=True)
    
    # Create enhanced plot
    plt.figure(figsize=(15, max(8, len(importance_df) * 0.4)))
    
    # Create horizontal bar plot with color gradient
    bars = plt.barh(range(len(importance_df)), importance_df['Importance_Score'], 
                    align='center', alpha=0.8)
    
    # Apply color gradient based on importance
    max_importance = importance_df['Importance_Score'].max()
    for i, bar in enumerate(bars):
        # Color gradient from light blue to dark blue
        intensity = importance_df['Importance_Score'].iloc[i] / max_importance
        color = plt.cm.Blues(0.3 + 0.7 * intensity)
        bar.set_color(color)
    
    # Set labels and title
    plt.yticks(range(len(importance_df)), importance_df['name'])
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f'Feature Importance Analysis{title_suffix}', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (score, name) in enumerate(zip(importance_df['Importance_Score'], importance_df['name'])):
        plt.text(score + max_importance * 0.01, i, f'{score:.4f}', 
                va='center', fontsize=9, alpha=0.8)
    
    # Add summary statistics
    mean_importance = importance_df['Importance_Score'].mean()
    std_importance = importance_df['Importance_Score'].std()
    
    # Add text box with statistics
    stats_text = f'Statistics:\nMean: {mean_importance:.4f}\nStd: {std_importance:.4f}\nTop Feature: {importance_df.iloc[-1]["name"][:20]}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             fontsize=10)
    
    # Adjust layout for better fit
    plt.tight_layout()
    
    # Save to output directory with cross-platform path
    plot_path = os.path.join(output_dir, 'Features_selected_for_modeling.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"✅ Enhanced feature importance plot saved to: {plot_path}")
    return plot_path


def save_feature_importance(output_dir, user_id, mdl_ltrl, importance_df):
    """
    Save the top 30 feature importances to an Excel file and plot.
    
    Args:
        output_dir: Output directory path
        user_id: User identifier  
        mdl_ltrl: Model literal
        importance_df: DataFrame with feature importance data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if 'idx' in importance_df.columns:
        importance_df = importance_df.drop('idx', axis=1)
    importance_df = importance_df.head(30)
    
    # Save to output directory with cross-platform path
    excel_path = os.path.join(output_dir, 'feature_importance.xlsx')
    importance_df.to_excel(excel_path, index=False)
    
    # Create plot
    plot_path = draw_feature_importance(output_dir, user_id, mdl_ltrl, importance_df)
    
    print(f"✅ Feature importance Excel saved to: {excel_path}")
    return excel_path, plot_path


def ExtractFeatureImp(featureImp, dataset, featuresCol):
    """
    Map feature importances from a Spark model to column names.
    Returns a pandas DataFrame sorted by importance.
    """
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract += dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['Importance_Score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return varlist.sort_values('Importance_Score', ascending=False) 