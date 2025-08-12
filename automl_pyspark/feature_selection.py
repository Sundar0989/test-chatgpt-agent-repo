"""
feature_selection.py

Feature selection utilities for AutoML PySpark package.
Includes:
    - Feature importance extraction
    - Feature importance plotting
    - Saving feature importance to Excel
    - Random Forest feature selection with automatic sampling for large datasets
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
import time


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


def determine_dataset_size(df: DataFrame) -> str:
    """
    Determine if a dataset is small, medium, or large based on row count.
    
    Args:
        df: Spark DataFrame
        
    Returns:
        str: 'small', 'medium', or 'large'
    """
    try:
        row_count = df.count()
        
        if row_count < 10000:
            return 'small'
        elif row_count < 100000:
            return 'medium'
        else:
            return 'large'
    except Exception as e:
        print(f"⚠️ Could not determine dataset size: {e}")
        return 'medium'  # Default to medium if we can't determine


def get_sampling_fraction(dataset_size: str) -> float:
    """
    Get the sampling fraction for feature selection based on dataset size.
    
    Args:
        dataset_size: 'small', 'medium', or 'large'
        
    Returns:
        float: Sampling fraction (0.0 to 1.0)
    """
    sampling_fractions = {
        'small': 1.0,    # Use full data for small datasets
        'medium': 0.3,   # Use 30% for medium datasets
        'large': 0.1     # Use 10% for large datasets
    }
    return sampling_fractions.get(dataset_size, 0.3)


def random_forest_feature_selection(
    df: DataFrame,
    target_column: str,
    problem_type: str,
    spark: SparkSession,
    output_dir: str,
    user_id: str,
    model_literal: str,
    max_features: int = 50,
    importance_threshold: float = 0.01,
    use_temp_tables: bool = True,
    temp_table_manager: Any = None
) -> Tuple[DataFrame, List[str], Dict[str, Any]]:
    """
    Perform Random Forest feature selection with automatic sampling for large datasets.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        problem_type: 'classification' or 'regression'
        spark: SparkSession
        output_dir: Output directory for results
        user_id: User identifier
        model_literal: Model identifier
        max_features: Maximum number of features to select
        importance_threshold: Minimum importance score to keep a feature
        
    Returns:
        Tuple of (filtered_df, selected_features, feature_importance_info)
    """
    print(f"\n🔍 Starting Random Forest feature selection for {problem_type}...")
    
    # Use temporary tables if enabled and manager is provided
    if use_temp_tables and temp_table_manager is not None:
        print("🗄️ Using temporary tables for feature selection optimization...")
        
        # Create temp table for original data
        original_table = temp_table_manager.create_temp_table(
            data=df,
            table_name='feature_selection_original',
            stage='feature_selection',
            optimize_for='ml'
        )
        print(f"✅ Created temp table: {original_table}")
        
        # Create temp table for sampled data
        dataset_size = determine_dataset_size(df)
        sampling_fraction = get_sampling_fraction(dataset_size)
        
        if sampling_fraction < 1.0:
            print(f"🔄 Sampling {sampling_fraction:.1%} of data for feature selection...")
            sampled_df = df.sample(fraction=sampling_fraction, seed=42)
            
            # Create temp table for sampled data
            sampled_table = temp_table_manager.create_temp_table(
                data=sampled_df,
                table_name='feature_selection_sampled',
                stage='feature_selection',
                optimize_for='ml'
            )
            print(f"✅ Created temp table for sampled data: {sampled_table}")
        else:
            sampled_df = df
            print("📈 Using full dataset for feature selection")
    else:
        # Debug: Show data types before sampling
        print(f"🔍 Debug: Original DataFrame schema:")
        for col in df.columns:
            col_type = df.schema[col].dataType.typeName()
            print(f"   {col}: {col_type}")
        
        # Determine dataset size and sampling strategy
        dataset_size = determine_dataset_size(df)
        sampling_fraction = get_sampling_fraction(dataset_size)
        
        print(f"📊 Dataset size: {dataset_size} (sampling fraction: {sampling_fraction:.1%})")
        
        # Sample data if needed
        if sampling_fraction < 1.0:
            print(f"🔄 Sampling {sampling_fraction:.1%} of data for feature selection...")
            sampled_df = df.sample(fraction=sampling_fraction, seed=42)
            print(f"📈 Sampled data: {sampled_df.count()} rows")
            
            # Debug: Show data types after sampling
            print(f"🔍 Debug: Sampled DataFrame schema:")
            for col in sampled_df.columns:
                col_type = sampled_df.schema[col].dataType.typeName()
                print(f"   {col}: {col_type}")
        else:
            sampled_df = df
            print("📈 Using full dataset for feature selection")
    
    # Get feature columns (exclude target and non-numeric columns)
    # Use the sampled_df for both column detection and feature selection to avoid schema mismatches
    feature_columns = []
    for col in sampled_df.columns:  # Use sampled_df instead of df
        if col != target_column:
            # Check if column is numeric using the sampled DataFrame
            try:
                # First try to get the data type from schema
                col_type = sampled_df.schema[col].dataType.typeName()
                if col_type in ['double', 'float', 'integer', 'long', 'int', 'bigint']:
                    feature_columns.append(col)
                    print(f"✅ Added numeric column: {col} (type: {col_type})")
                else:
                    print(f"⚠️ Skipping non-numeric column: {col} (type: {col_type})")
            except Exception as e:
                print(f"⚠️ Error checking column {col}: {e}")
                # Fallback: try to cast to double
                try:
                    sampled_df.select(col).cast("double").first()
                    feature_columns.append(col)
                    print(f"✅ Added column {col} via fallback casting")
                except Exception as cast_error:
                    print(f"⚠️ Skipping non-numeric column: {col} (cast error: {cast_error})")
    
    print(f"🔢 Found {len(feature_columns)} numeric feature columns")
    
    if len(feature_columns) == 0:
        print("❌ No numeric features found for selection")
        return df, [], {}
    
    # Prepare data for Random Forest
    print("🔧 Preparing data for Random Forest...")
    
    # Create vector assembler
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="skip"
    )
    
    # Transform data
    assembled_df = assembler.transform(sampled_df)
    
    # Train Random Forest for feature selection
    print("🌳 Training Random Forest for feature selection...")
    start_time = time.time()
    
    if problem_type == 'classification':
        # Count distinct values in target for classification
        distinct_targets = sampled_df.select(target_column).distinct().count()
        
        if distinct_targets == 2:
            # Binary classification
            rf = RandomForestClassifier(
                featuresCol="features",
                labelCol=target_column,
                numTrees=10,
                maxDepth=5,
                maxBins=100,  # Increased from default 32 to handle categorical features with more values
                seed=42
            )
        else:
            # Multiclass classification
            rf = RandomForestClassifier(
                featuresCol="features",
                labelCol=target_column,
                numTrees=10,
                maxDepth=5,
                maxBins=100,  # Increased from default 32 to handle categorical features with more values
                seed=42
            )
    else:
        # Regression
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol=target_column,
            numTrees=10,
            maxDepth=5,
            maxBins=100,  # Increased from default 32 to handle categorical features with more values
            seed=42
        )
    
    # Fit the model
    model = rf.fit(assembled_df)
    
    training_time = time.time() - start_time
    print(f"✅ Random Forest trained in {training_time:.2f} seconds")
    
    # Extract feature importance
    print("📊 Extracting feature importance...")
    feature_importance = model.featureImportances
    
    # Create feature importance DataFrame
    importance_data = []
    for i, (feature, importance) in enumerate(zip(feature_columns, feature_importance)):
        importance_data.append({
            'name': feature,
            'Importance_Score': float(importance),
            'rank': i + 1
        })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('Importance_Score', ascending=False)
    
    # Select features based on importance threshold and max_features
    selected_features = []
    for _, row in importance_df.iterrows():
        if len(selected_features) >= max_features:
            break
        if row['Importance_Score'] >= importance_threshold:
            selected_features.append(row['name'])
    
    # Fallback: if no features meet the threshold, select top features anyway
    if len(selected_features) == 0:
        print(f"⚠️ No features met the importance threshold of {importance_threshold}")
        print(f"🔄 Falling back to selecting top {min(10, max_features)} features by importance")
        selected_features = importance_df.head(min(10, max_features))['name'].tolist()
    
    print(f"🎯 Selected {len(selected_features)} features out of {len(feature_columns)}")
    print(f"📊 Importance threshold: {importance_threshold}")
    print(f"📊 Max features: {max_features}")
    
    # Create filtered DataFrame with selected features
    selected_columns = [target_column] + selected_features
    filtered_df = df.select(selected_columns)
    
    # Save feature importance results
    print("💾 Saving feature importance results...")
    
    # Save to Excel
    excel_path = os.path.join(output_dir, f'feature_importance_{user_id}_{model_literal}.xlsx')
    importance_df.to_excel(excel_path, index=False)
    print(f"✅ Feature importance saved to: {excel_path}")
    
    # Create and save plot
    plot_path = draw_feature_importance(output_dir, user_id, model_literal, importance_df)
    
    # Prepare return information
    feature_importance_info = {
        'dataset_size': dataset_size,
        'sampling_fraction': sampling_fraction,
        'original_features': len(feature_columns),
        'selected_features': len(selected_features),
        'importance_threshold': importance_threshold,
        'max_features': max_features,
        'training_time': training_time,
        'excel_path': excel_path,
        'plot_path': plot_path,
        'feature_importance_df': importance_df
    }
    
    print(f"✅ Feature selection completed!")
    print(f"📊 Original features: {len(feature_columns)}")
    print(f"📊 Selected features: {len(selected_features)}")
    print(f"📊 Reduction: {((len(feature_columns) - len(selected_features)) / len(feature_columns) * 100):.1f}%")
    
    return filtered_df, selected_features, feature_importance_info


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