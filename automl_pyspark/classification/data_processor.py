"""
Data Processor

Class responsible for data preprocessing, feature selection, and data manipulation.
This class encapsulates all the data processing functionality from the original modules.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, isnan, count, monotonically_increasing_id
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier

# Import functions from original modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import using flexible import strategy for both package and direct execution
try:
    from automl_pyspark.data_manipulations import (
        missing_value_calculation, identify_variable_type, categorical_to_index,
        numerical_imputation, rename_columns, join_features_and_target,
        train_valid_test_split, assembled_vectors, scaled_dataframes,
        analyze_categorical_cardinality, convert_categorical_to_numeric
    )
    from automl_pyspark.feature_selection import ExtractFeatureImp, save_feature_importance
except ImportError:
    # Fallback to direct imports (for direct script execution)
    import sys
    import os
    
    # Add parent directory to path if not already there
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from data_manipulations import (
        missing_value_calculation, identify_variable_type, categorical_to_index,
        numerical_imputation, rename_columns, join_features_and_target,
        train_valid_test_split, assembled_vectors, scaled_dataframes,
        analyze_categorical_cardinality, convert_categorical_to_numeric
    )
    from feature_selection import ExtractFeatureImp, save_feature_importance

# Import the new data balancing module
from .data_balancing import DataBalancer


class DataProcessor:
    """
    Data processor class that handles all data preprocessing and feature engineering tasks.
    
    This class provides functionality for:
    - Data preprocessing and cleaning
    - Feature selection and engineering
    - Data splitting and scaling
    - Handling categorical and numerical variables
    """
    
    def __init__(self, spark_session: SparkSession, user_id: str, model_literal: str):

        """
        Initialize the data processor.
        
        Args:
            spark_session: PySpark SparkSession
        """
        self.spark = spark_session
        self.user_id = user_id
        self.model_literal = model_literal
        
        # Initialize data balancer
        self.data_balancer = DataBalancer(spark_session)
        self.target_label_indexer = None  # Store target label encoder for multi-class
        self.char_labels = None  # Store categorical variable encoder
        self.pipeline_model = None  # Store scaling pipeline
        self.output_dir = None  # Will be set by the AutoML class
        
        # Sequential feature selection parameters (will be set by AutoMLClassifier)
        self.sequential_threshold = 200  # Use sequential selection if features > this number
        self.chunk_size = 100  # Reduced chunk size for better JVM stability
        self.features_per_chunk = 30  # Features to select from each smaller chunk
        
        # Schema caching for recovery from Py4J connection issues
        self.last_processed_schema = None

        
    def preprocess(self, data: DataFrame, target_column: str, config: Dict) -> tuple:
        """
        Preprocess the input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            config: Configuration dictionary
            
        Returns:
            Tuple of (processed_data, selected_vars, categorical_vars, numerical_vars)
        """
        print("Starting data preprocessing...")
        
        # Apply data sampling if sample_fraction < 1.0
        sample_fraction = config.get('sample_fraction', 1.0)
        if sample_fraction < 1.0:
            print(f"📊 Applying data sampling: {sample_fraction:.2%} of original data")
            try:
                original_count = data.count()
                print(f"   Original dataset size: {original_count:,} rows")
            except Exception as e:
                print(f"   ⚠️ Cannot count original dataset size (very large BigQuery dataset): {e}")
                print(f"   📊 Proceeding with sampling without exact count...")
                original_count = None
            
            # Use seed for reproducibility
            seed = config.get('seed', 42)
            data = data.sample(fraction=sample_fraction, seed=seed)
            
            if original_count:
                try:
                    sampled_count = data.count()
                    print(f"   Sampled dataset size: {sampled_count:,} rows ({sampled_count/original_count:.2%} of original)")
                    print(f"   Using random seed: {seed}")
                except Exception as e:
                    print(f"   ⚠️ Cannot count sampled dataset size: {e}")
                    print(f"   📊 Sampling applied with seed: {seed}")
            else:
                print(f"   📊 Sampling applied with seed: {seed}")
        else:
            print(f"📊 Using full dataset (sample_fraction = {sample_fraction})")
        
        # Check if target column is string and needs encoding
        target_data_type = dict(data.dtypes)[target_column]
        
        # Detect number of classes - optimized for large BigQuery datasets
        print("🔍 Analyzing target column classes...")
        try:
            # Try the standard approach first
            num_classes = data.select(target_column).distinct().count()
            print(f"📊 Detected {num_classes} classes using distinct count")
        except Exception as e:
            print(f"⚠️ Standard distinct count failed on large dataset: {e}")
            print("🔄 Using BigQuery-optimized approach...")
            try:
                # Use groupBy which is more BigQuery-friendly
                class_counts = data.groupBy(target_column).count().limit(10).collect()
                num_classes = len(class_counts)
                print(f"📊 Detected {num_classes} classes using groupBy approach")
            except Exception as e2:
                print(f"⚠️ GroupBy approach also failed: {e2}")
                print("🔄 Using sampling approach...")
                try:
                    # Last resort: use sampling
                    sample_data = data.sample(fraction=0.001, seed=42)  # 0.1% sample
                    num_classes = sample_data.select(target_column).distinct().count()
                    print(f"📊 Detected {num_classes} classes using sample approach")
                except Exception as e3:
                    print(f"❌ All approaches failed: {e3}")
                    print("🔄 Assuming binary classification...")
                    num_classes = 2
        
        is_multiclass = num_classes > 2
        needs_target_encoding = target_data_type == 'string'
        
        if needs_target_encoding:
            print(f"Detected string target column with {num_classes} classes. Encoding {target_column} to numeric values...")
            # Apply StringIndexer to target column
            target_indexer = StringIndexer(inputCol=target_column, outputCol=target_column + "_indexed")
            self.target_label_indexer = target_indexer.fit(data)
            data = self.target_label_indexer.transform(data)
            # Replace original target column with indexed version
            data = data.drop(target_column).withColumnRenamed(target_column + "_indexed", target_column)
            print(f"Target column {target_column} encoded to numeric values")
        
        # Get feature variables (exclude target and specified exclusions)
        feature_vars = self._get_feature_variables(data, target_column, config)
        print(f"Initial feature count: {len(feature_vars)}")
        
        # Filter out date/timestamp columns automatically
        feature_vars = self._filter_date_columns(data, feature_vars)
        print(f"Features after date column filtering: {len(feature_vars)}")
        
        # Select features based on missing value threshold
        X = data.select(feature_vars)
        selected_vars = missing_value_calculation(X, config['missing_threshold'])
        print(f"Features after missing value filtering: {len(selected_vars)}")
        print(selected_vars)
        
        # Identify variable types
        X_selected = X.select(selected_vars)
        categorical_vars, numerical_vars = identify_variable_type(X_selected)
        print(f"Categorical variables: {len(categorical_vars)}")
        print(f"Numerical variables: {len(numerical_vars)}")
        
        # Analyze categorical cardinality and convert high-cardinality variables to numeric
        max_categorical_cardinality = config.get('max_categorical_cardinality', 50)
        categorical_vars_to_keep, categorical_vars_to_convert = analyze_categorical_cardinality(
            X_selected, categorical_vars, max_categorical_cardinality
        )
        
        # Convert high-cardinality categorical variables to numeric
        if categorical_vars_to_convert:
            X_selected = convert_categorical_to_numeric(X_selected, categorical_vars_to_convert)
            # Update variable lists
            numerical_vars.extend(categorical_vars_to_convert)  # These are now numeric
            categorical_vars = categorical_vars_to_keep  # Only keep low-cardinality ones
            print(f"Updated categorical variables: {len(categorical_vars)}")
            print(f"Updated numerical variables: {len(numerical_vars)}")
        
        # Apply preprocessing transformations
        X_processed, categorical_vars, numerical_vars = self._apply_preprocessing_transformations(
            X_selected, categorical_vars, numerical_vars, config['impute_value']
        )
        
        # Join with target
        Y = data.select(target_column)
        processed_data = join_features_and_target(X_processed, Y)
        print("Target and Preprocessed features joined.")
        
        print('Final Data Summary - ')
        # Add robust error handling for the data summary call
        try:
            summary = self.get_data_summary(processed_data)
            print(summary)
        except Exception as e:
            print(f"⚠️ Could not generate data summary due to Py4J error: {e}")
            print("🔄 Attempting to continue without detailed summary...")
            try:
                # Try a minimal summary without accessing schema
                print("📊 Basic summary: Data preprocessing completed successfully")
                print("   💡 Detailed schema information unavailable due to connection issues")
            except:
                print("📊 Data preprocessing completed (summary unavailable)")
        
        print("Data preprocessing completed.")
        return processed_data, selected_vars, categorical_vars, numerical_vars
    
    def select_features(self, data: DataFrame, target_column: str, max_features: int = 30) -> DataFrame:
        """
        Select the best features using sequential feature selection for large feature sets.
        
        For feature sets > 200:
        - Process features in chunks of 200
        - Select top 50 from each chunk
        - Final selection of top 30 from combined results
        
        For feature sets <= 200:
        - Direct selection of top features
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            max_features: Maximum number of features to select
            
        Returns:
            Tuple of (DataFrame with selected features, list of top features)
        """
        print(f"Feature selection: selecting top {max_features} features...")
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in data.columns if col != target_column]
        total_features = len(feature_cols)
        
        print(f"Total features available: {total_features}")
        
        # Check dataset size and apply intelligent sampling for feature selection
        try:
            row_count = data.count()
            print(f"📊 Dataset size for feature selection: {row_count:,} rows")
            
            # For very large datasets (>5M rows), use sampling for feature selection
            feature_selection_data = data
            if row_count > 5_000_000:
                sample_size = min(1_000_000, max(500_000, int(row_count * 0.05)))  # 5% or 1M max
                sample_fraction = sample_size / row_count
                
                print(f"🔄 Large dataset detected ({row_count:,} rows)")
                print(f"📊 Sampling {sample_size:,} rows ({sample_fraction:.1%}) for memory-efficient feature selection...")
                
                feature_selection_data = data.sample(withReplacement=False, fraction=sample_fraction, seed=42)
                
                # Cache the sampled data to avoid recomputation
                from pyspark import StorageLevel
                feature_selection_data.persist(StorageLevel.MEMORY_AND_DISK_SER)
                
                actual_sample_size = feature_selection_data.count()
                print(f"✅ Using {actual_sample_size:,} rows for feature selection (cached for efficiency)")
            else:
                print(f"📊 Using full dataset ({row_count:,} rows) for feature selection")
                
        except Exception as e:
            print(f"⚠️ Could not determine dataset size: {e}")
            print("📊 Proceeding with full dataset for feature selection")
            feature_selection_data = data
        
        # Sequential feature selection for large feature sets
        if total_features > self.sequential_threshold:
            print("Large feature set detected. Using sequential feature selection...")
            selected_features = self._sequential_feature_selection(
                feature_selection_data, feature_cols, target_column, max_features
            )
        else:
            print("Standard feature selection for moderate feature set...")
            selected_features = self._standard_feature_selection(
                feature_selection_data, feature_cols, target_column, max_features
            )
        
        # Create new DataFrame with selected features (use ORIGINAL full dataset)
        selected_data = data.select([target_column] + selected_features)
        
        # Clean up cached feature selection data if it was sampled
        try:
            if 'feature_selection_data' in locals() and feature_selection_data is not data:
                feature_selection_data.unpersist()
                print("🧹 Cleaned up cached feature selection data")
        except:
            pass
        
        print(f"Selected {len(selected_features)} features out of {total_features}")
        return selected_data, selected_features
    
    def _sequential_feature_selection(self, data: DataFrame, feature_cols: List[str], 
                                    target_column: str, max_features: int) -> List[str]:
        """
        Sequential feature selection for large feature sets.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            max_features: Final number of features to select
            
        Returns:
            List of selected feature names
        """
        chunk_size = self.chunk_size
        features_per_chunk = self.features_per_chunk
        all_selected_features = []
        
        print(f"Processing {len(feature_cols)} features in chunks of {chunk_size}")
        
        # Split features into chunks
        feature_chunks = [feature_cols[i:i + chunk_size] for i in range(0, len(feature_cols), chunk_size)]
        
        print(f"Created {len(feature_chunks)} chunks")
        
        # Process each chunk
        for i, chunk_features in enumerate(feature_chunks):
            print(f"\nProcessing chunk {i+1}/{len(feature_chunks)} with {len(chunk_features)} features...")
            
            # Select subset of data for this chunk
            chunk_data = data.select([target_column] + chunk_features)
            
            # Use Random Forest for all chunks with smaller chunk size for stability
            chunk_selected = self._run_feature_importance(
                chunk_data, chunk_features, target_column, features_per_chunk
            )
            
            all_selected_features.extend(chunk_selected)
            print(f"Selected {len(chunk_selected)} features from chunk {i+1}")
            
            # Clean up and give JVM a break between chunks
            if i < len(feature_chunks) - 1:  # Not the last chunk
                try:
                    chunk_data.unpersist()
                    # Force garbage collection hint to Spark
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    print("Cleaned up chunk data, pausing briefly...")
                    import time
                    time.sleep(2)  # Slightly longer pause for GC
                except:
                    pass
        
        print(f"\nCombined {len(all_selected_features)} features from all chunks")
        
        # Final feature selection on combined features
        if len(all_selected_features) > max_features:
            print(f"Running final feature selection to select top {max_features} from {len(all_selected_features)} features...")
            
            # Reuse sequential feature selection for consistency (handles chunking if needed)
            final_selected = self._sequential_feature_selection(
                data, all_selected_features, target_column, max_features
            )
        else:
            print(f"Combined features ({len(all_selected_features)}) <= max_features ({max_features}), using all")
            final_selected = all_selected_features
        
        return final_selected
    
    def _standard_feature_selection(self, data: DataFrame, feature_cols: List[str], 
                                  target_column: str, max_features: int) -> List[str]:
        """
        Standard feature selection for moderate feature sets.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            max_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        return self._run_feature_importance(data, feature_cols, target_column, max_features)
    
    def _run_feature_importance(self, data: DataFrame, feature_cols: List[str], 
                              target_column: str, num_features: int) -> List[str]:
        """
        Run Random Forest feature importance selection with improved error handling.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        print(f"\n🎯 Running Feature Importance for Classification")
        print(f"   📊 Features available: {len(feature_cols)}")
        print(f"   🎯 Features to select: {num_features}")
        print(f"   📂 Output directory: {self.output_dir}")
        print(f"   👤 User ID: {self.user_id}")
        print(f"   🏷️ Model literal: {self.model_literal}")
        print(f"   🎯 Target column: {target_column}")
        
        # Check if we have any features to work with
        if len(feature_cols) == 0:
            print("⚠️ No features available for selection!")
            return []
        
        if num_features <= 0:
            print("⚠️ num_features <= 0, returning empty list")
            return []
        
        # Use the requested number of features (should already be adjusted by AutoMLClassifier)
        actual_num_features = num_features
        print(f"📊 Will rank all {len(feature_cols)} features and select top {actual_num_features}")
        
        # For very large feature sets, use statistical fallback first
        if len(feature_cols) > 200:
            print(f"Large feature set detected ({len(feature_cols)} features). Using statistical preprocessing...")
            try:
                # Use correlation-based pre-filtering for very large sets
                selected_features = self._statistical_feature_selection(data, feature_cols, target_column, min(100, len(feature_cols)))
                if len(selected_features) <= num_features:
                    return selected_features
                feature_cols = selected_features
                print(f"Pre-filtered to {len(feature_cols)} features using statistical methods")
            except Exception as e:
                print(f"Statistical pre-filtering failed: {e}, continuing with original approach...")
        
        # Try Random Forest with improved configuration and retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}: Running Random Forest feature importance...")
                
                # Create feature vector with timeout protection
                data_with_features = assembled_vectors(data, feature_cols, target_column)
                
                # Use memory-efficient caching for large datasets
                from pyspark import StorageLevel
                
                # Choose storage level based on dataset size
                try:
                    row_count = data.count()
                    if row_count > 1_000_000:
                        # For very large datasets, use disk-based storage
                        print(f"🗄️ Large dataset ({row_count:,} rows): Using MEMORY_AND_DISK storage")
                        data_with_features.persist(StorageLevel.MEMORY_AND_DISK)
                    else:
                        # For smaller datasets, use memory
                        print(f"🧠 Medium dataset ({row_count:,} rows): Using MEMORY_ONLY storage")
                        data_with_features.persist(StorageLevel.MEMORY_ONLY)
                except:
                    # Fallback to safer option
                    print("🗄️ Using MEMORY_AND_DISK storage (safe fallback)")
                    data_with_features.persist(StorageLevel.MEMORY_AND_DISK)
                
                # Trigger persistence by counting rows
                cached_count = data_with_features.count()
                print(f"✅ Cached {cached_count:,} rows for feature importance calculation")
                
                # Use Random Forest with reduced complexity for large feature sets
                num_trees = min(10, max(5, 50 // max(1, len(feature_cols) // 50)))  # Adaptive tree count
                max_depth = min(10, max(3, 20 - len(feature_cols) // 20))  # Adaptive depth
                
                print(f"Using RandomForest with {num_trees} trees, max depth {max_depth}")
                
                rf = RandomForestClassifier(
                    featuresCol="features", 
                    labelCol=target_column, 
                    numTrees=num_trees,
                    maxDepth=max_depth,
                    subsamplingRate=0.8,  # Use subsampling to reduce memory
                    featureSubsetStrategy="sqrt"  # Use sqrt of features at each node
                )
                
                # Fit with timeout handling
                rf_model = rf.fit(data_with_features)
                
                # Extract feature importance
                feature_importance = ExtractFeatureImp(rf_model.featureImportances, data_with_features, "features")
                
                # Select top features
                top_features = feature_importance['name'].head(actual_num_features).tolist()
                
                # Save feature importance (always save when feature selection runs)
                # Enhanced saving with better error handling and verification
                output_dir = self.output_dir if self.output_dir else '.'
                print(f"💾 Attempting to save feature importance to: {output_dir}")
                print(f"📊 User ID: {self.user_id}")
                print(f"🏷️ Model literal: {self.model_literal}")
                print(f"📈 Feature importance DataFrame shape: {feature_importance.shape}")
                
                try:
                    excel_path, plot_path = save_feature_importance(output_dir, self.user_id, self.model_literal, feature_importance)
                    
                    # Verify files actually exist
                    excel_exists = os.path.exists(excel_path) if excel_path else False
                    plot_exists = os.path.exists(plot_path) if plot_path else False
                    
                    print(f"✅ Feature importance saved for classification model:")
                    print(f"   📊 Excel file: {excel_path} (exists: {excel_exists})")
                    print(f"   📈 Plot file: {plot_path} (exists: {plot_exists})")
                    
                    if not excel_exists or not plot_exists:
                        raise Exception(f"Files not created - Excel exists: {excel_exists}, Plot exists: {plot_exists}")
                        
                except Exception as e:
                    print(f"⚠️ Primary save failed: {e}")
                    print(f"🔄 Trying fallback to current directory...")
                    # Try to save to current directory as fallback
                    try:
                        excel_path, plot_path = save_feature_importance('.', self.user_id, self.model_literal, feature_importance)
                        
                        # Verify fallback files
                        excel_exists = os.path.exists(excel_path) if excel_path else False
                        plot_exists = os.path.exists(plot_path) if plot_path else False
                        
                        print(f"✅ Feature importance saved to current directory (fallback):")
                        print(f"   📊 Excel file: {excel_path} (exists: {excel_exists})")
                        print(f"   📈 Plot file: {plot_path} (exists: {plot_exists})")
                        
                        # If fallback worked, try to copy to target directory
                        if excel_exists and plot_exists and output_dir != '.':
                            try:
                                import shutil
                                target_excel = os.path.join(output_dir, 'feature_importance.xlsx')
                                target_plot = os.path.join(output_dir, 'Features_selected_for_modeling.png')
                                
                                shutil.copy2(excel_path, target_excel)
                                shutil.copy2(plot_path, target_plot)
                                
                                print(f"✅ Copied feature importance files to target directory:")
                                print(f"   📊 {target_excel}")
                                print(f"   📈 {target_plot}")
                                
                            except Exception as copy_e:
                                print(f"⚠️ Could not copy to target directory: {copy_e}")
                        
                    except Exception as e2:
                        print(f"❌ Fallback save also failed: {e2}")
                        print(f"❌ Feature importance will not be available for this classification model")
                
                # Unpersist cached data
                data_with_features.unpersist()
                
                print(f"Successfully selected {len(top_features)} features using Random Forest")
                return top_features
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # Try to clean up any cached data
                try:
                    if 'data_with_features' in locals():
                        data_with_features.unpersist()
                except:
                    pass
                
                # Check if this is a Py4J error and try to restart session
                if "Py4JNetworkError" in str(e) or "Answer from Java side is empty" in str(e):
                    print("Detected Py4J network error, attempting session restart...")
                    if self._restart_spark_session():
                        print("Session restarted, retrying...")
                        continue
                
                # On final attempt, use fallback method
                if attempt == max_retries - 1:
                    print("Random Forest failed on all attempts. Using variance-based fallback...")
                    return self._variance_based_selection(data, feature_cols, target_column, num_features)
                
                # Wait before retry to allow JVM recovery
                import time
                time.sleep(2)
        
        # Should not reach here, but fallback just in case
        print("Warning: All feature selection methods failed, returning first n features")
        return feature_cols[:num_features]
    
    def _statistical_feature_selection(self, data: DataFrame, feature_cols: List[str], 
                                     target_column: str, num_features: int) -> List[str]:
        """
        Statistical feature selection using correlation and variance filtering.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            from pyspark.sql.functions import variance, corr
            from pyspark.ml.stat import Correlation
            
            print("Running statistical feature selection...")
            
            # Remove features with zero or very low variance
            high_variance_features = []
            for col_name in feature_cols:
                try:
                    var_val = data.select(variance(col(col_name))).collect()[0][0]
                    if var_val is not None and var_val > 1e-6:  # Threshold for variance
                        high_variance_features.append(col_name)
                except:
                    continue  # Skip problematic columns
            
            print(f"Filtered {len(feature_cols)} -> {len(high_variance_features)} features by variance")
            
            if len(high_variance_features) <= num_features:
                return high_variance_features
            
            # If still too many, use correlation with target (for classification)
            try:
                # For classification, we'll use chi-square test approximation via correlation
                correlations = []
                for col_name in high_variance_features[:min(200, len(high_variance_features))]:  # Limit to prevent timeout
                    try:
                        corr_val = data.select(corr(col(col_name), col(target_column))).collect()[0][0]
                        if corr_val is not None:
                            correlations.append((col_name, abs(corr_val)))
                    except:
                        continue
                
                # Sort by correlation and take top features
                correlations.sort(key=lambda x: x[1], reverse=True)
                selected = [name for name, _ in correlations[:num_features]]
                
                print(f"Selected {len(selected)} features by correlation with target")
                return selected
                
            except Exception as e:
                print(f"Correlation analysis failed: {e}")
                # Return high variance features if correlation fails
                return high_variance_features[:num_features]
            
        except Exception as e:
            print(f"Statistical feature selection failed: {e}")
            
            # Check if this is a Py4J error and try session restart
            if "Py4JNetworkError" in str(e) or "Answer from Java side is empty" in str(e) or "Connection refused" in str(e):
                print("Detected Py4J error in statistical selection, attempting session restart...")
                if self._restart_spark_session():
                    print("Session restarted, trying simple variance calculation...")
                    try:
                        return self._simple_variance_selection(data, feature_cols, target_column, num_features)
                    except:
                        pass
            
            # Ultimate fallback: random sampling with some intelligence
            print("All statistical methods failed, using intelligent sampling fallback...")
            return self._intelligent_sampling_fallback(feature_cols, num_features)
    
    def _variance_based_selection(self, data: DataFrame, feature_cols: List[str], 
                                target_column: str, num_features: int) -> List[str]:
        """
        Simple variance-based feature selection as final fallback.
        Enhanced to also generate feature importance plots using variance scores.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            from pyspark.sql.functions import variance
            import pandas as pd
            import os
            
            print("Using variance-based feature selection as fallback...")
            
            feature_variances = []
            for col_name in feature_cols:
                try:
                    var_val = data.select(variance(col(col_name))).collect()[0][0]
                    if var_val is not None:
                        feature_variances.append((col_name, var_val))
                except:
                    feature_variances.append((col_name, 0.0))  # Assign zero variance if calculation fails
            
            # Sort by variance and take top features
            feature_variances.sort(key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in feature_variances[:num_features]]
            
            print(f"Selected {len(selected)} features by variance")
            
            # Generate feature importance plot using variance scores
            try:
                # Normalize variance scores to create "importance" scores
                total_variance = sum(var for _, var in feature_variances)
                if total_variance > 0:
                    normalized_scores = [(name, var/total_variance) for name, var in feature_variances]
                else:
                    # If all variances are 0, assign equal importance
                    normalized_scores = [(name, 1.0/len(feature_variances)) for name, _ in feature_variances]
                
                # Create DataFrame for feature importance
                feature_importance_df = pd.DataFrame(normalized_scores, columns=['name', 'Importance_Score'])
                
                print(f"💾 Generating feature importance plot using variance-based scores...")
                output_dir = self.output_dir if self.output_dir else '.'
                print(f"💾 Saving variance-based feature importance to: {output_dir}")
                
                # Save feature importance using the same function as Random Forest
                excel_path, plot_path = save_feature_importance(output_dir, self.user_id, self.model_literal, feature_importance_df)
                
                # Verify files were created
                excel_exists = os.path.exists(excel_path) if excel_path else False
                plot_exists = os.path.exists(plot_path) if plot_path else False
                
                print(f"✅ Variance-based feature importance saved:")
                print(f"   📊 Excel file: {excel_path} (exists: {excel_exists})")
                print(f"   📈 Plot file: {plot_path} (exists: {plot_exists})")
                
                if not excel_exists or not plot_exists:
                    print(f"⚠️ Some files not created, trying fallback to current directory...")
                    try:
                        excel_path, plot_path = save_feature_importance('.', self.user_id, self.model_literal, feature_importance_df)
                        print(f"✅ Variance-based feature importance saved to current directory (fallback)")
                    except Exception as save_e:
                        print(f"❌ Failed to save variance-based feature importance: {save_e}")
                
            except Exception as importance_e:
                print(f"⚠️ Could not generate feature importance plot from variance: {importance_e}")
            
            return selected
            
        except Exception as e:
            print(f"Variance-based selection failed: {e}")
            # Ultimate fallback: return first n features
            return feature_cols[:num_features]
    
    def _simple_variance_selection(self, data: DataFrame, feature_cols: List[str], 
                                 target_column: str, num_features: int) -> List[str]:
        """
        Ultra-simple variance calculation that works even with damaged Spark sessions.
        """
        try:
            print("Attempting simple variance calculation...")
            selected_features = []
            
            # Try to calculate variance for each feature individually
            for i, col_name in enumerate(feature_cols):
                if len(selected_features) >= num_features:
                    break
                    
                try:
                    # Simple check: try to get basic stats
                    result = data.select(col_name).count()
                    if result > 0:
                        selected_features.append(col_name)
                except:
                    continue  # Skip problematic columns
                    
                # Limit processing to avoid timeout
                if i > num_features * 3:  # Don't check more than 3x needed
                    break
            
            print(f"Simple variance selection completed: {len(selected_features)} features")
            return selected_features[:num_features]
            
        except Exception as e:
            print(f"Simple variance selection failed: {e}")
            return self._intelligent_sampling_fallback(feature_cols, num_features)
    
    def _intelligent_sampling_fallback(self, feature_cols: List[str], num_features: int) -> List[str]:
        """
        Intelligent sampling when all Spark operations fail.
        This doesn't require Spark and uses Python logic only.
        """
        try:
            print("Using intelligent sampling fallback...")
            
            if len(feature_cols) <= num_features:
                return feature_cols
            
            # Intelligent sampling: try to get features from different parts of the list
            # This helps if features are organized in some meaningful order
            selected = []
            step = len(feature_cols) // num_features
            
            # Sample evenly across the feature space
            for i in range(0, len(feature_cols), max(1, step)):
                if len(selected) >= num_features:
                    break
                selected.append(feature_cols[i])
            
            # Fill remaining slots with features from the beginning
            for feature in feature_cols:
                if len(selected) >= num_features:
                    break
                if feature not in selected:
                    selected.append(feature)
            
            print(f"Intelligent sampling selected {len(selected)} features")
            return selected[:num_features]
            
        except Exception as e:
            print(f"Intelligent sampling failed: {e}")
            # Absolute last resort
            return feature_cols[:num_features]
    
    def split_and_scale(self, data: DataFrame, train_size: float = 0.7, 
                       valid_size: float = 0.2, target_column: str = 'target', 
                       seed: int = 42, config: Optional[Dict] = None) -> tuple:
        """
        Split data into train/validation/test and apply scaling.
        If auto_balance is enabled and data is imbalanced, applies upsampling to training data.
        
        Args:
            data: Input DataFrame
            train_size: Proportion for training (default: 0.7)
            valid_size: Proportion for validation (default: 0.15)
            target_column: Name of target column (default: 'target')
            seed: Random seed for reproducibility (default: 42)
            config: Configuration dictionary for upsampling
            
        Returns:
            Tuple of (train_scaled, train_original_scaled, valid_scaled, test_scaled)
            - train_scaled: Upsampled training data (for model training)
            - train_original_scaled: Original training data (for evaluation metrics)
            - valid_scaled: Validation data
            - test_scaled: Test data
        """
        print("Splitting data into train/validation/test sets...")
        
        # Check for class imbalance before splitting
        print("\n📊 Analyzing class distribution on full dataset...")
        class_stats = self.data_balancer.analyze_class_distribution(data, target_column)
        imbalance_threshold = config.get('imbalance_threshold', 0.05) if config else 0.05
        is_imbalanced = self.data_balancer.detect_imbalance(class_stats, imbalance_threshold)
        
        # Split data
        train, valid, test = train_valid_test_split(data, train_size, valid_size, seed)
        
        # Keep a copy of original training data
        train_original = train
        
        # Apply upsampling to training data if needed
        if config and config.get('auto_balance', False) and is_imbalanced:
            balance_method = config.get('balance_method', 'oversample')
            max_balance_ratio = config.get('max_balance_ratio', 0.05)
            
            print(f"\n📈 Applying {balance_method} to training data...")
            train_upsampled = self.data_balancer.upsample_minority_classes(
                train, target_column, class_stats, 
                method=balance_method, max_ratio=max_balance_ratio, seed=seed, config=config
            )
            
            # Verify upsampling results
            print("\n📊 Post-upsampling class distribution:")
            post_upsample_stats = self.data_balancer.analyze_class_distribution(train_upsampled, target_column)
            
            train = train_upsampled
            print("✅ Training data upsampling completed. Validation/test data remain original.")
            print("📋 Note: Model will train on upsampled data, but training evaluation will use original data.")
        elif config and config.get('auto_balance', False):
            print("✅ No upsampling needed - classes are sufficiently balanced.")
        
        # Get feature columns
        feature_cols = [col for col in data.columns if col != target_column]
        
        # Apply scaling to upsampled training data
        train_scaled, valid_scaled, test_scaled, pipeline_model = scaled_dataframes(
            train, valid, test, feature_cols, target_column
        )
        
        # Apply scaling to original training data using the same pipeline
        train_original_scaled = pipeline_model.transform(train_original)
        
        # Store pipeline model for later use
        self.pipeline_model = pipeline_model
        
        print("Data splitting and scaling completed.")
        return train_scaled, train_original_scaled, valid_scaled, test_scaled
    
    def apply_scaling(self, data: DataFrame, target_column: Optional[str] = None) -> DataFrame:
        """
        Apply scaling to new data using the fitted pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column (None for clustering)
            
        Returns:
            Scaled DataFrame
        """
        if self.pipeline_model is None:
            if target_column is None:
                raise ValueError("Pipeline model not fitted. For clustering, ensure process_data() is called with target_column=None first.")
            else:
                raise ValueError("Pipeline model not fitted. Call split_and_scale() first.")
        
        return self.pipeline_model.transform(data)
    
    def apply_preprocessing(self, data: DataFrame, feature_vars: List[str], 
                          selected_vars: List[str], categorical_vars: List[str],
                          numerical_vars: List[str], char_labels: PipelineModel,
                          impute_value: float, target_column: Optional[str] = None,
                          target_label_indexer: Optional[Any] = None) -> DataFrame:
        """
        Apply preprocessing pipeline to new data.
        
        Args:
            data: Input DataFrame
            feature_vars: List of feature variables
            selected_vars: List of selected variables
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
            char_labels: Fitted string indexer pipeline
            impute_value: Value to use for imputation
            target_column: Name of target column to preserve (optional)
            target_label_indexer: Fitted StringIndexer for target column (optional)
            
        Returns:
            Preprocessed DataFrame
        """
        # Filter out any date columns that might be in the new data (for OOT consistency)
        available_columns = [col for col in selected_vars if col in data.columns]
        filtered_columns = self._filter_date_columns(data, available_columns)
        
        # Update columns_to_select to only include non-date columns that exist in the data
        columns_to_select = filtered_columns
        
        # Add target column back if it exists and was specified (target columns are not filtered)
        if target_column and target_column in data.columns and target_column not in columns_to_select:
            columns_to_select.append(target_column)
        
        X = data.select(columns_to_select)
        
        # Apply target column encoding if target column is present and indexer is available
        if target_column and target_column in X.columns and target_label_indexer is not None:
            X = target_label_indexer.transform(X)
            # Replace original target column with indexed version
            X = X.drop(target_column).withColumnRenamed(target_column + "_indexed", target_column)
        
        # Apply categorical encoding
        if char_labels is not None:
            X = char_labels.transform(X)
        
        # Apply numerical imputation
        X = numerical_imputation(X, numerical_vars, impute_value)
        
        # Remove original categorical columns
        X = X.select([c for c in X.columns if c not in categorical_vars])
        
        # No need to rename columns since categorical_to_index now creates _encoded suffix directly
        # X = rename_columns(X, categorical_vars)
        
        return X
    
    def analyze_class_distribution(self, data: DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyze class distribution to detect imbalanced data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Dictionary containing class distribution statistics
        """
        print(f"Analyzing class distribution for {target_column}...")
        
        # Get class counts
        class_counts = data.groupBy(target_column).count().collect()
        total_count = data.count()
        
        distribution = {}
        for row in class_counts:
            class_value = row[target_column]
            count = row['count']
            percentage = count / total_count
            distribution[str(class_value)] = {
                'count': count,
                'percentage': percentage
            }
        
        # Find minority and majority classes
        sorted_classes = sorted(distribution.items(), key=lambda x: x[1]['count'])
        minority_class = sorted_classes[0]
        majority_class = sorted_classes[-1]
        
        print(f"Class distribution:")
        for class_val, stats in sorted_classes:
            print(f"  Class {class_val}: {stats['count']} samples ({stats['percentage']:.2%})")
        
        return {
            'distribution': distribution,
            'minority_class': minority_class[0],
            'minority_count': minority_class[1]['count'],
            'minority_percentage': minority_class[1]['percentage'],
            'majority_class': majority_class[0],
            'majority_count': majority_class[1]['count'],
            'majority_percentage': majority_class[1]['percentage'],
            'total_count': total_count,
            'num_classes': len(distribution)
        }
    
    def detect_imbalance(self, class_stats: Dict[str, Any], threshold: float = 0.05) -> bool:
        """
        Detect if dataset has imbalanced classes.
        For multiclass: checks if ANY class is below threshold (not just smallest).
        
        Args:
            class_stats: Class distribution statistics from analyze_class_distribution
            threshold: Minimum percentage for a class to be considered balanced
            
        Returns:
            True if imbalanced, False otherwise
        """
        # Check if ANY class is below threshold (improved multiclass support)
        imbalanced_classes = []
        for class_value, stats in class_stats['distribution'].items():
            if stats['percentage'] < threshold:
                imbalanced_classes.append((class_value, stats['percentage']))
        
        is_imbalanced = len(imbalanced_classes) > 0
        
        if is_imbalanced:
            print(f"⚠️  Imbalanced dataset detected: {len(imbalanced_classes)} class(es) below {threshold:.1%} threshold")
            for class_val, percentage in imbalanced_classes:
                print(f"    - Class {class_val}: {percentage:.2%} (< {threshold:.1%})")
        else:
            minority_percentage = class_stats['minority_percentage']
            print(f"✅ Balanced dataset: all classes >= {threshold:.1%} (smallest: {minority_percentage:.2%})")
        
        return is_imbalanced
    
    def upsample_minority_classes(self, data: DataFrame, target_column: str, 
                                class_stats: Dict[str, Any], method: str = 'oversample',
                                max_ratio: float = 0.3, seed: int = 12345, 
                                config: Dict[str, Any] = None) -> DataFrame:
        """
        Upsample minority classes to improve balance.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            class_stats: Class distribution statistics
            method: Upsampling method ('oversample' or 'smote')
            max_ratio: Maximum ratio to balance minority class to
            seed: Random seed
            config: Configuration dictionary with optional parameters like 'smote_k_neighbors'
            
        Returns:
            Upsampled DataFrame
        """
        print(f"Upsampling minority classes using {method} method...")
        
        if method == 'oversample':
            return self._oversample_data(data, target_column, class_stats, max_ratio, seed)
        elif method == 'smote':
            k_neighbors = config.get('smote_k_neighbors', 5) if config else 5
            return self._smote_data(data, target_column, class_stats, max_ratio, seed, k_neighbors)
        else:
            print(f"Unknown upsampling method: {method}, returning original data")
            return data
    
    def _oversample_data(self, data: DataFrame, target_column: str, 
                       class_stats: Dict[str, Any], max_ratio: float, 
                       seed: int) -> DataFrame:
        """
        Oversample minority classes using random sampling with replacement.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            class_stats: Class distribution statistics
            max_ratio: Maximum ratio to balance minority class to
            seed: Random seed
            
        Returns:
            Oversampled DataFrame
        """
        majority_count = class_stats['majority_count']
        target_minority_count = int(majority_count * max_ratio)
        
        print(f"Target minority class size: {target_minority_count} samples ({max_ratio:.1%} of majority class)")
        
        # Create list to store DataFrames for union
        dataframes_to_union = []
        
        # Add original data
        dataframes_to_union.append(data)
        
        # For each minority class that needs upsampling
        for class_value, stats in class_stats['distribution'].items():
            current_count = stats['count']
            current_percentage = stats['percentage']
            
            # Check if this class needs upsampling
            if current_percentage < max_ratio and current_count < target_minority_count:
                needed_samples = target_minority_count - current_count
                
                if needed_samples > 0:
                    print(f"  Upsampling class {class_value}: {current_count} → {target_minority_count} (+{needed_samples} samples)")
                    
                    # Filter data for this class
                    class_data = data.filter(col(target_column) == class_value)
                    
                    # Calculate sample fraction for oversampling
                    # We want to sample with replacement to get needed_samples additional samples
                    sample_fraction = needed_samples / current_count
                    
                    # Sample with replacement
                    oversampled_class = class_data.sample(withReplacement=True, 
                                                        fraction=sample_fraction, 
                                                        seed=seed)
                    
                    dataframes_to_union.append(oversampled_class)
        
        # Union all DataFrames
        if len(dataframes_to_union) > 1:
            result = dataframes_to_union[0]
            for df in dataframes_to_union[1:]:
                result = result.union(df)
            
            # Shuffle the result
            result = result.orderBy(col(target_column).desc(), monotonically_increasing_id())
            
            print(f"Upsampling completed. New dataset size: {result.count()} samples")
            return result
        else:
            print("No upsampling needed.")
            return data
    
    def _smote_data(self, data: DataFrame, target_column: str, 
                   class_stats: Dict[str, Any], max_ratio: float, 
                   seed: int, k_neighbors: int = 5) -> DataFrame:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique) to balance minority classes.
        
        SMOTE generates synthetic samples by interpolating between minority class samples
        and their k-nearest neighbors in the feature space.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            class_stats: Class distribution statistics
            max_ratio: Maximum ratio to balance minority class to
            seed: Random seed
            k_neighbors: Number of nearest neighbors to consider (default: 5)
            
        Returns:
            DataFrame with synthetic minority class samples added
        """
        from pyspark.sql.functions import rand, lit, array, explode, row_number
        from pyspark.sql.window import Window
        from pyspark.sql.types import ArrayType, StructType, StructField, DoubleType
        import numpy as np
        
        print("🧬 Applying SMOTE (Synthetic Minority Oversampling Technique)...")
        
        majority_count = class_stats['majority_count']
        target_minority_count = int(majority_count * max_ratio)
        
        print(f"Target minority class size: {target_minority_count} samples ({max_ratio:.1%} of majority class)")
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in data.columns if col != target_column]
        
        # List to store DataFrames for union
        dataframes_to_union = [data]  # Start with original data
        
        # Process each minority class that needs upsampling
        for class_value, stats in class_stats['distribution'].items():
            current_count = stats['count']
            current_percentage = stats['percentage']
            
            # Check if this class needs upsampling
            if current_percentage < max_ratio and current_count < target_minority_count:
                needed_samples = target_minority_count - current_count
                
                if needed_samples > 0 and current_count >= k_neighbors:
                    print(f"  🧬 SMOTE for class {class_value}: {current_count} → {target_minority_count} (+{needed_samples} synthetic samples)")
                    
                    # Generate synthetic samples for this class
                    synthetic_samples = self._generate_smote_samples(
                        data, target_column, class_value, 
                        feature_cols, needed_samples, k_neighbors, seed
                    )
                    
                    if synthetic_samples.count() > 0:
                        dataframes_to_union.append(synthetic_samples)
                        
                elif current_count < k_neighbors:
                    print(f"  ⚠️ Class {class_value} has only {current_count} samples, less than k_neighbors={k_neighbors}")
                    print(f"     Falling back to oversampling for this class...")
                    
                    # Fall back to simple oversampling for classes with too few samples
                    class_data = data.filter(col(target_column) == class_value)
                    sample_fraction = needed_samples / current_count
                    oversampled_class = class_data.sample(withReplacement=True, 
                                                        fraction=sample_fraction, 
                                                        seed=seed)
                    dataframes_to_union.append(oversampled_class)
        
        # Union all DataFrames
        if len(dataframes_to_union) > 1:
            result = dataframes_to_union[0]
            for df in dataframes_to_union[1:]:
                result = result.union(df)
            
            # Shuffle the result
            result = result.orderBy(col(target_column).desc(), monotonically_increasing_id())
            
            print(f"🧬 SMOTE completed. New dataset size: {result.count()} samples")
            return result
        else:
            print("No SMOTE upsampling needed.")
            return data
    
    def _generate_smote_samples(self, data: DataFrame, target_column: str, 
                              class_value: str, feature_cols: List[str], 
                              num_samples: int, k_neighbors: int, seed: int) -> DataFrame:
        """
        Generate synthetic samples for a specific class using SMOTE algorithm.
        
        Args:
            data: DataFrame with features_vector column
            target_column: Name of the target column
            class_value: Class value to generate samples for
            feature_cols: List of feature column names
            num_samples: Number of synthetic samples to generate
            k_neighbors: Number of nearest neighbors to consider
            seed: Random seed
            
        Returns:
            DataFrame containing synthetic samples
        """
        from pyspark.sql.functions import udf, col, lit, rand
        from pyspark.sql.types import ArrayType, DoubleType, StringType
        from pyspark.ml.linalg import VectorUDT, Vectors, DenseVector
        import random
        
        # Filter data for the specific class
        class_data = data.filter(col(target_column) == class_value).cache()
        class_count = class_data.count()
        
        if class_count == 0:
            return data.limit(0)  # Return empty DataFrame with same schema
        
        # Collect class data to driver for neighbor calculations
        # Note: This assumes minority class data fits in memory, which is usually reasonable
        class_rows = class_data.collect()
        
        # Extract feature vectors from individual columns
        feature_vectors = []
        for row in class_rows:
            # Get feature values from individual columns
            feature_values = [row[col] for col in feature_cols]
            feature_vectors.append(np.array(feature_values))
        
        feature_vectors = np.array(feature_vectors)
        
        print(f"    Generating {num_samples} synthetic samples using {len(feature_vectors)} base samples")
        
        # Generate synthetic samples
        synthetic_rows = []
        random.seed(seed)
        np.random.seed(seed)
        
        for _ in range(num_samples):
            # Randomly select a base sample
            base_idx = random.randint(0, len(feature_vectors) - 1)
            base_sample = feature_vectors[base_idx]
            
            # Find k nearest neighbors (excluding the base sample itself)
            distances = []
            for i, sample in enumerate(feature_vectors):
                if i != base_idx:
                    # Euclidean distance
                    dist = np.sqrt(np.sum((base_sample - sample) ** 2))
                    distances.append((dist, i))
            
            # Sort by distance and take k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:min(k_neighbors, len(distances))]
            
            if k_nearest:
                # Randomly select one of the k nearest neighbors
                _, neighbor_idx = random.choice(k_nearest)
                neighbor_sample = feature_vectors[neighbor_idx]
                
                # Generate synthetic sample by interpolating
                # new_sample = base_sample + rand(0,1) * (neighbor_sample - base_sample)
                alpha = random.random()  # Random value between 0 and 1
                synthetic_sample = base_sample + alpha * (neighbor_sample - base_sample)
                
                # Create a row for the synthetic sample
                row_dict = {}
                for i, col_name in enumerate(feature_cols):
                    # Get the original data type for this column
                    field = data.schema[col_name]
                    value = synthetic_sample[i]
                    
                    # Convert to appropriate type based on original schema
                    if field.dataType.simpleString() in ['int', 'integer']:
                        row_dict[col_name] = int(round(value))
                    elif field.dataType.simpleString() in ['bigint', 'long']:
                        row_dict[col_name] = int(round(value))
                    elif field.dataType.simpleString() in ['double', 'float']:
                        row_dict[col_name] = float(value)
                    else:
                        # Default to float for other numeric types
                        row_dict[col_name] = float(value)
                
                # Set target column value with appropriate type
                target_field = data.schema[target_column]
                if target_field.dataType.simpleString() in ['int', 'integer', 'bigint', 'long']:
                    row_dict[target_column] = int(class_value)
                else:
                    row_dict[target_column] = class_value
                
                synthetic_rows.append(row_dict)
        
        # Create DataFrame from synthetic samples
        if synthetic_rows:
            # Use the original data's schema to ensure consistent data types
            original_schema = data.schema
            synthetic_df = self.spark.createDataFrame(synthetic_rows, schema=original_schema)
            return synthetic_df
        else:
            return data.limit(0)  # Return empty DataFrame with same schema

    def _get_feature_variables(self, data: DataFrame, target_column: str, config: Dict) -> List[str]:
        """
        Get list of feature variables based on configuration.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            config: Configuration dictionary
            
        Returns:
            List of feature variable names
        """
        # Get all columns except target
        all_vars = [col for col in data.columns if col != target_column]
        
        # Apply include/exclude filters
        include_vars = config.get('include_vars', [])
        exclude_vars = config.get('exclude_vars', [])
        include_prefix = config.get('include_prefix', [])
        exclude_prefix = config.get('exclude_prefix', [])
        include_suffix = config.get('include_suffix', [])
        exclude_suffix = config.get('exclude_suffix', [])
        
        # Filter by include variables
        if include_vars:
            all_vars = [var for var in all_vars if var in include_vars]
        
        # Filter by include prefixes
        if include_prefix:
            prefix_vars = []
            for prefix in include_prefix:
                prefix_vars.extend([var for var in all_vars if var.startswith(prefix)])
            all_vars = list(set(all_vars) & set(prefix_vars))
        
        # Filter by include suffixes
        if include_suffix:
            suffix_vars = []
            for suffix in include_suffix:
                suffix_vars.extend([var for var in all_vars if var.endswith(suffix)])
            all_vars = list(set(all_vars) & set(suffix_vars))
        
        # Filter by exclude variables
        if exclude_vars:
            all_vars = [var for var in all_vars if var not in exclude_vars]
        
        # Filter by exclude prefixes
        if exclude_prefix:
            for prefix in exclude_prefix:
                all_vars = [var for var in all_vars if not var.startswith(prefix)]
        
        # Filter by exclude suffixes
        if exclude_suffix:
            for suffix in exclude_suffix:
                all_vars = [var for var in all_vars if not var.endswith(suffix)]
        
        return all_vars
    
    def _filter_date_columns(self, data: DataFrame, feature_vars: List[str]) -> List[str]:
        """
        Filter out date/timestamp columns from the list of feature variables.
        This is important because these columns are often not suitable for machine learning.
        
        Args:
            data: Input DataFrame
            feature_vars: List of feature variable names
            
        Returns:
            List of feature variable names after date/timestamp filtering
        """
        from pyspark.sql.types import DateType, TimestampType
        
        date_types = ['date', 'timestamp']
        date_like_patterns = ['date', 'time', 'dt_', '_dt', 'timestamp']
        
        filtered_feature_vars = []
        filtered_date_columns = []
        
        for var in feature_vars:
            is_date_column = False
            
            try:
                # Check actual Spark data type
                field = data.schema[var]
                if isinstance(field.dataType, (DateType, TimestampType)):
                    is_date_column = True
                    filtered_date_columns.append(f"{var} (Spark {field.dataType.simpleString()})")
                
                # Also check string representation for edge cases
                field_type = field.dataType.simpleString()
                if field_type in date_types:
                    is_date_column = True
                    if var not in [col.split(' ')[0] for col in filtered_date_columns]:
                        filtered_date_columns.append(f"{var} ({field_type})")
                
                # Check for common date column naming patterns
                var_lower = var.lower()
                if any(pattern in var_lower for pattern in date_like_patterns):
                    # Additional validation: check if column contains date-like strings
                    sample_values = data.select(var).limit(5).collect()
                    if sample_values and self._looks_like_date_column(sample_values, var):
                        is_date_column = True
                        if var not in [col.split(' ')[0] for col in filtered_date_columns]:
                            filtered_date_columns.append(f"{var} (date pattern)")
                
            except (KeyError, Exception) as e:
                # If there's any issue checking the column, keep it to be safe
                pass
            
            if not is_date_column:
                filtered_feature_vars.append(var)
        
        # Log filtered columns
        if filtered_date_columns:
            print(f"🗓️ Automatically filtered out {len(filtered_date_columns)} date/timestamp columns:")
            for col_info in filtered_date_columns:
                print(f"   - {col_info}")
            print(f"💡 Date columns are excluded because they often don't provide meaningful features for ML models")
        
        return filtered_feature_vars
    
    def _looks_like_date_column(self, sample_values, column_name: str) -> bool:
        """
        Check if sample values look like dates/timestamps.
        
        Args:
            sample_values: List of Row objects with sample values
            column_name: Name of the column being checked
            
        Returns:
            bool: True if values look like dates
        """
        import re
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        ]
        
        date_like_count = 0
        total_non_null = 0
        
        for row in sample_values:
            value = row[column_name] if hasattr(row, column_name) else row[0]
            if value is not None:
                total_non_null += 1
                value_str = str(value)
                
                # Check against date patterns
                if any(re.search(pattern, value_str) for pattern in date_patterns):
                    date_like_count += 1
                # Check for long integers that might be timestamps
                elif re.match(r'^\d{10,13}$', value_str):  # Unix timestamps
                    date_like_count += 1
        
        # If more than 50% of non-null values look like dates, consider it a date column
        if total_non_null > 0:
            date_percentage = date_like_count / total_non_null
            return date_percentage > 0.5
        
        return False
    
    def _apply_preprocessing_transformations(self, data: DataFrame, 
                                           categorical_vars: List[str],
                                           numerical_vars: List[str],
                                           impute_value: float) -> Tuple[DataFrame, List[str], List[str]]:
        """
        Apply preprocessing transformations to the data.
        
        Args:
            data: Input DataFrame
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
            impute_value: Value to use for imputation
            
        Returns:
            Tuple of (transformed DataFrame, updated categorical_vars, updated numerical_vars)
        """
        # Get current data types
        current_dtypes = dict(data.dtypes)
        
        # Create local copies to avoid modifying the original lists
        local_categorical_vars = categorical_vars.copy()
        local_numerical_vars = numerical_vars.copy()
        
        # Convert all numerical columns that are currently string to numeric
        print("🔄 Converting string columns to numeric where needed...")
        for col_name in list(local_numerical_vars):  # Use list() to avoid modification during iteration
            if col_name in current_dtypes and current_dtypes[col_name] == 'string':
                print(f"   Converting '{col_name}' from string to numeric...")
                try:
                    from pyspark.sql.functions import when, isnull, length, trim, isnan, regexp_replace
                    
                    # Clean the column: remove non-numeric characters except decimal points and negative signs
                    cleaned_col = regexp_replace(col(col_name), '[^0-9.-]', '')
                    
                    # Handle empty strings and convert to double
                    data = data.withColumn(
                        col_name, 
                        when(
                            (isnull(col(col_name))) | 
                            (length(trim(col(col_name))) == 0) |
                            (trim(col(col_name)) == '') |
                            (trim(col(col_name)) == '?') |
                            (trim(col(col_name)) == 'NA') |
                            (trim(col(col_name)) == 'NULL'),
                            None
                        ).otherwise(cleaned_col.cast("double"))
                    )
                    print(f"   ✅ Successfully converted '{col_name}' to numeric")
                except Exception as e:
                    print(f"   ⚠️ Failed to convert '{col_name}' to numeric: {e}")
                    # If conversion fails, treat as categorical
                    local_categorical_vars.append(col_name)
                    local_numerical_vars.remove(col_name)
                    print(f"   📊 Moving '{col_name}' to categorical variables")
        
        # Check if target column exists and convert to numeric if it's string (for classification target encoding)
        target_column = None
        for col_name in data.columns:
            if col_name not in local_categorical_vars and col_name not in local_numerical_vars:
                target_column = col_name
                break
        
        if target_column and target_column in current_dtypes and current_dtypes[target_column] == 'string':
            print(f"🔄 Target column '{target_column}' is string type - will be handled by target encoding...")
            # For classification, we keep string targets as they will be handled by StringIndexer
        
        # Apply categorical encoding
        if local_categorical_vars:
            # Filter out high-cardinality categorical features that might cause tree-based model issues
            print(f"🔍 Checking cardinality of {len(local_categorical_vars)} categorical features...")
            filtered_categorical_vars = []
            
            for cat_var in local_categorical_vars:
                try:
                    # Optimize distinct count for large BigQuery datasets
                    try:
                        distinct_count = data.select(cat_var).distinct().count()
                        print(f"   📊 {cat_var}: {distinct_count} unique values")
                    except Exception as e:
                        print(f"   ⚠️ Could not get exact distinct count for {cat_var}: {e}")
                        print(f"   🔄 Using sample-based estimation...")
                        # Use groupBy as fallback which is more BigQuery-friendly
                        sample_distinct = data.groupBy(cat_var).count().limit(201).collect()
                        distinct_count = len(sample_distinct)
                        if len(sample_distinct) == 201:
                            print(f"   📊 {cat_var}: 200+ unique values (estimated)")
                        else:
                            print(f"   📊 {cat_var}: {distinct_count} unique values (estimated)")
                    
                    # Keep features with reasonable cardinality (up to 200 for our maxBins=256 setting)
                    if distinct_count <= 200:
                        filtered_categorical_vars.append(cat_var)
                    else:
                        print(f"   ⚠️ Skipping high-cardinality feature '{cat_var}' ({distinct_count} unique values > 200)")
                        
                except Exception as e:
                    print(f"   ⚠️ Could not check cardinality for '{cat_var}': {e}, including anyway")
                    filtered_categorical_vars.append(cat_var)
            
            print(f"📈 Categorical feature filtering: {len(local_categorical_vars)} → {len(filtered_categorical_vars)} features")
            
            if filtered_categorical_vars:
                data, char_labels = categorical_to_index(data, filtered_categorical_vars)
                self.char_labels = char_labels
                print(f"Categorical encoding step complete.")
                
                # Remove original categorical columns (only the ones we encoded)
                data = data.select([c for c in data.columns if c not in filtered_categorical_vars])
            else:
                print("   📊 No categorical features remaining after cardinality filtering")
                self.char_labels = None
        
        # Apply numerical imputation
        if local_numerical_vars:
            data = numerical_imputation(data, local_numerical_vars, impute_value)
            print(f"Numerical impuation step complete.")
        
        # No need to rename columns since categorical_to_index now creates _encoded suffix directly
        # data = rename_columns(data, categorical_vars)
        print(f"Categorical encoding complete with _encoded suffix.")
        
        return data, local_categorical_vars, local_numerical_vars
    
    def get_data_summary(self, data: DataFrame) -> Dict:
        """
        Get a summary of the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing data summary
        """
        # Try to import recovery utilities
        try:
            from spark_session_recovery import safe_dataframe_operation
            use_safe_operations = True
        except ImportError:
            use_safe_operations = False
            print("⚠️ Spark session recovery utilities not available, using basic error handling")
        
        # Optimize for large BigQuery datasets and handle Py4J connection errors
        if use_safe_operations:
            row_count = safe_dataframe_operation(data, "count", "Unknown (very large dataset)")
            column_count = safe_dataframe_operation(data, "columns", ["Unknown (connection error)"])
            dtypes = safe_dataframe_operation(data, "dtypes", {"error": "connection_error"})
            
            # Handle the case where columns operation failed
            if isinstance(column_count, list) and len(column_count) == 1 and "Unknown" in column_count[0]:
                column_count = "Unknown (connection error)"
                columns = ["Unknown (connection error)"]
            else:
                columns = column_count
                column_count = len(columns) if isinstance(columns, list) else "Unknown"
        else:
            # Fallback to original error handling
            try:
                row_count = data.count()
            except Exception as e:
                print(f"⚠️ Cannot count rows in very large dataset: {e}")
                row_count = "Unknown (very large dataset)"
            
            # Handle Py4J connection errors when accessing schema
            try:
                column_count = len(data.columns)
                columns = data.columns
                dtypes = dict(data.dtypes)
            except Exception as e:
                print(f"⚠️ Cannot access DataFrame schema due to Py4J error: {e}")
                print("🔄 Attempting to restart Spark session and retry...")
                
                # Try to restart Spark session
                if self._restart_spark_session():
                    try:
                        # Retry with fresh session
                        column_count = len(data.columns)
                        columns = data.columns
                        dtypes = dict(data.dtypes)
                        print("✅ Successfully recovered DataFrame schema after session restart")
                    except Exception as e2:
                        print(f"❌ Still cannot access DataFrame schema after session restart: {e2}")
                        # Provide fallback values
                        column_count = "Unknown (connection error)"
                        columns = ["Unknown (connection error)"]
                        dtypes = {"error": "connection_error"}
                else:
                    print("❌ Failed to restart Spark session")
                    # Provide fallback values
                    column_count = "Unknown (connection error)"
                    columns = ["Unknown (connection error)"]
                    dtypes = {"error": "connection_error"}
        
        summary = {
            'row_count': row_count,
            'column_count': column_count,
            'columns': columns,
            'dtypes': dtypes
        }
        
        # Cache the schema for recovery purposes
        if isinstance(columns, list) and len(columns) > 0 and not any("Unknown" in col for col in columns):
            self.last_processed_schema = columns
            print(f"📋 Cached schema with {len(columns)} columns for recovery")
        
        return summary

    def process_data(self, train_data: Union[str, DataFrame], target_column: Optional[str] = None,
                    oot1_data: Optional[Union[str, DataFrame]] = None,
                    oot2_data: Optional[Union[str, DataFrame]] = None,
                    **kwargs) -> Dict[str, DataFrame]:
        """
        Process data for AutoML pipeline, handling both string file paths and DataFrames.
        
        Args:
            train_data: Training data (file path or DataFrame)
            target_column: Name of the target column (None for clustering)
            oot1_data: Optional out-of-time validation data
            oot2_data: Optional second out-of-time validation data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing processed datasets
        """
        # Load data if string paths are provided
        if isinstance(train_data, str):
            print(f"📁 Loading training data from: {train_data}")
            train_data = self.spark.read.csv(train_data, header=True, inferSchema=True)
        
        if isinstance(oot1_data, str):
            print(f"📁 Loading OOT1 data from: {oot1_data}")
            oot1_data = self.spark.read.csv(oot1_data, header=True, inferSchema=True)
        
        if isinstance(oot2_data, str):
            print(f"📁 Loading OOT2 data from: {oot2_data}")
            oot2_data = self.spark.read.csv(oot2_data, header=True, inferSchema=True)
        
        # Get configuration from kwargs or use defaults
        config = kwargs.get('config', {})
        if not config:
            config = {
                'missing_threshold': kwargs.get('missing_threshold', 0.7),
                'categorical_threshold': kwargs.get('categorical_threshold', 10),
                'sample_fraction': kwargs.get('sample_fraction', 1.0),
                'seed': kwargs.get('seed', 42),
                'impute_value': kwargs.get('impute_value', -999),
                'test_size': kwargs.get('test_size', 0.2),
                'validation_size': kwargs.get('validation_size', 0.2)
            }
        
        # Preprocess training data
        if target_column:
            # For classification and regression
            processed_train, selected_vars, categorical_vars, numerical_vars = self.preprocess(
                train_data, target_column, config
            )
            
            # Store variables for later use
            self.feature_vars = [col for col in train_data.columns if col != target_column]
            self.selected_vars = selected_vars
            self.categorical_vars = categorical_vars
            self.numerical_vars = numerical_vars
            
            # Split data
            train_scaled, train_original_scaled, valid_scaled, test_scaled = self.split_and_scale(
                processed_train, 
                train_size=1-config['test_size']-config['validation_size'],
                valid_size=config['validation_size'],
                target_column=target_column,
                seed=config['seed'],
                config=config
            )
            
            # Process OOT datasets if provided
            oot1_scaled = None
            oot2_scaled = None
            
            if oot1_data is not None:
                oot1_processed = self.apply_preprocessing(
                    oot1_data, self.feature_vars, self.selected_vars,
                    self.categorical_vars, self.numerical_vars, self.char_labels,
                    config['impute_value'], target_column, self.target_label_indexer
                )
                oot1_scaled = self.apply_scaling(oot1_processed, target_column)
            
            if oot2_data is not None:
                oot2_processed = self.apply_preprocessing(
                    oot2_data, self.feature_vars, self.selected_vars,
                    self.categorical_vars, self.numerical_vars, self.char_labels,
                    config['impute_value'], target_column, self.target_label_indexer
                )
                oot2_scaled = self.apply_scaling(oot2_processed, target_column)
            
            return {
                'train': train_scaled,
                'train_original': train_original_scaled,
                'valid': valid_scaled,
                'test': test_scaled,
                'oot1': oot1_scaled,
                'oot2': oot2_scaled
            }
            
        else:
            # For clustering (no target column)
            print("🔧 Processing data for clustering (no target column)...")
            
            # Get ALL feature variables (both numeric AND categorical, not just numeric)
            exclude_cols = ['id', 'index', 'row_id', 'row_number']  # Common ID column patterns
            
            # Identify numeric and categorical variables separately
            numeric_cols = [col for col in train_data.columns 
                           if col.lower() not in exclude_cols and 
                           train_data.schema[col].dataType.typeName() in ['double', 'float', 'integer', 'long']]
            
            categorical_cols = [col for col in train_data.columns 
                               if col.lower() not in exclude_cols and 
                               train_data.schema[col].dataType.typeName() in ['string']]
            
            # Include BOTH numeric and categorical in clustering features
            self.feature_vars = numeric_cols + categorical_cols
            
            print(f"   📊 Selected {len(self.feature_vars)} total feature columns:")
            print(f"      📈 Numeric ({len(numeric_cols)}): {numeric_cols}")
            print(f"      🏷️  Categorical ({len(categorical_cols)}): {categorical_cols}")
            
            # Apply comprehensive preprocessing for clustering (including categorical encoding)
            processed_train, self.categorical_vars, self.numerical_vars = self._apply_preprocessing_transformations(
                train_data, categorical_cols, numeric_cols, config.get('impute_value', -999)
            )
            
            # Create scaling pipeline for clustering with encoded categorical variables
            print("   📏 Creating and applying scaling pipeline...")
            from pyspark.ml.feature import VectorAssembler, StandardScaler
            from pyspark.ml import Pipeline
            
            # Get all available columns for assembling (numeric + encoded categorical)
            available_cols = [col for col in processed_train.columns if col in self.numerical_vars or col.endswith('_encoded')]
            
            print(f"      🔗 Assembling features from {len(available_cols)} columns: {available_cols}")
            
            # Create feature vector and scale it
            assembler = VectorAssembler(inputCols=available_cols, outputCol='assembled_features')
            scaler = StandardScaler(inputCol='assembled_features', outputCol='features')
            pipeline = Pipeline(stages=[assembler, scaler])
            
            # Fit and transform
            self.pipeline_model = pipeline.fit(processed_train)
            processed_train = self.pipeline_model.transform(processed_train)
            
            # Select relevant columns for clustering (use actual available columns after encoding)
            # Keep features column + original numeric columns + encoded categorical columns
            available_numeric = [col for col in self.numerical_vars if col in processed_train.columns]
            available_categorical = [col for col in processed_train.columns if col.endswith('_encoded')]
            
            selected_cols = ['features'] + available_numeric + available_categorical
            processed_train = processed_train.select(selected_cols)
            
            print(f"      📊 Selected columns for clustering: {selected_cols}")
            
            # Set up variables for clustering (using actual available columns)
            self.selected_vars = available_numeric + available_categorical
            self.categorical_vars = [col.replace('_encoded', '') for col in available_categorical]  # Original names for profiling
            self.numerical_vars = available_numeric
            
            print(f"   ✅ Clustering data processed: {processed_train.count()} rows")
            
            return {
                'train': processed_train,
                'valid': None,
                'test': None,
                'oot1': None,
                'oot2': None
            }
    
    def _restart_spark_session(self):
        """
        Restart Spark session to recover from Py4J errors.
        This helps when the JVM becomes unresponsive during large operations.
        """
        try:
            print("🔄 Attempting to restart Spark session to recover from Py4J error...")
            
            # Stop current session gracefully
            if self.spark:
                try:
                    self.spark.stop()
                    print("   ✅ Stopped existing Spark session")
                except Exception as e:
                    print(f"   ⚠️ Warning while stopping session: {e}")
            
            # Wait a moment for cleanup
            import time
            time.sleep(2)
            
            # Create new session with optimized configuration including BigQuery support
            from pyspark.sql import SparkSession
            
            # Try to import optimization config
            try:
                from spark_optimization_config import get_optimized_spark_config
                spark_config = get_optimized_spark_config(include_bigquery=True)
                print("   📦 Using optimized Spark configuration with BigQuery support")
            except ImportError:
                print("   ⚠️ Spark optimization config not available, using basic config")
                spark_config = {}
            
            # Build session with BigQuery support
            builder = SparkSession.builder.appName("AutoML_DataProcessor_Recovery")
            
            # Add BigQuery connector package
            builder = builder.config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1")
            
            # Apply optimized configuration
            for key, value in spark_config.items():
                builder = builder.config(key, value)
            
            # Add additional recovery-specific configurations
            recovery_config = {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.sql.adaptive.skewJoin.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.driver.memory": "8g",  # Increased for BigQuery operations
                "spark.driver.maxResultSize": "4g",  # Increased for BigQuery results
                "spark.sql.execution.arrow.pyspark.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.minPartitionSize": "1MB",
                "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
                "spark.network.timeout": "800s",
                "spark.rpc.askTimeout": "600s",
                "spark.sql.broadcastTimeout": "36000",
                "spark.rpc.message.maxSize": "512",
                "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
            }
            
            for key, value in recovery_config.items():
                builder = builder.config(key, value)
            
            self.spark = builder.getOrCreate()
            
            # Verify BigQuery connector
            try:
                test_reader = self.spark.read.format("bigquery")
                print("   ✅ BigQuery connector verified after session restart")
            except Exception as e:
                print(f"   ⚠️ BigQuery connector verification warning: {e}")
            
            print("✅ Spark session restarted successfully with BigQuery support")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restart Spark session: {e}")
            print("💡 This may indicate a deeper system issue or insufficient resources")
            return False