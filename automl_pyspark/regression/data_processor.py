"""
Regression Data Processor

Handles data preprocessing, feature selection, and feature engineering for regression tasks.
This is a regression-specific version that uses Random Forest Regression for feature importance.
"""

import os
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, count, variance, corr
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, MinMaxScaler, RobustScaler,
    StringIndexer, OneHotEncoder, Imputer, PCA
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.stat import Correlation

# Import feature selection utilities
try:
    from ..feature_selection import ExtractFeatureImp, save_feature_importance
except ImportError:
    # For direct script execution
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from feature_selection import ExtractFeatureImp, save_feature_importance

# Import utility functions
try:
    from ..data_manipulations import (
        assembled_vectors, categorical_to_index, numerical_imputation, rename_columns,
        analyze_categorical_cardinality, convert_categorical_to_numeric
    )
except ImportError:
    try:
        from data_manipulations import (
            assembled_vectors, categorical_to_index, numerical_imputation, rename_columns,
            analyze_categorical_cardinality, convert_categorical_to_numeric
        )
    except ImportError:
        # Fallback implementation
        def assembled_vectors(data: DataFrame, feature_cols: List[str], target_column: str) -> DataFrame:
            """Create feature vector from selected columns."""
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            return assembler.transform(data)
        
        def categorical_to_index(data: DataFrame, categorical_vars: List[str]):
            """Encode categorical variables using StringIndexer."""
            if not categorical_vars:
                return data, None
            from pyspark.ml.feature import StringIndexer
            indexers = [StringIndexer(inputCol=col, outputCol=col + "_encoded", handleInvalid="keep") for col in categorical_vars]
            from pyspark.ml import Pipeline
            pipeline = Pipeline(stages=indexers)
            fitted_pipeline = pipeline.fit(data)
            return fitted_pipeline.transform(data), fitted_pipeline
        
        def numerical_imputation(data: DataFrame, numerical_vars: List[str], impute_value: float):
            """Apply numerical imputation."""
            from pyspark.sql.functions import when, isnan, isnull
            for col_name in numerical_vars:
                data = data.withColumn(col_name, when(isnan(col_name) | isnull(col_name), impute_value).otherwise(col_name))
            return data
        
        def rename_columns(data: DataFrame, categorical_vars: List[str]):
            """Rename encoded categorical columns to match expected format."""
            # Columns are already named with _encoded suffix, no renaming needed
            return data


class RegressionDataProcessor:
    """
    Data processor specifically designed for regression tasks.
    
    This class handles:
    - Data preprocessing and cleaning
    - Feature selection using Random Forest Regression
    - Data splitting and scaling
    - Feature engineering
    """
    
    def __init__(self, spark_session: SparkSession, user_id: str, model_literal: str):
        """
        Initialize the regression data processor.
        
        Args:
            spark_session: PySpark SparkSession
            user_id: User identifier for saving outputs
            model_literal: Model literal for saving outputs
        """
        self.spark = spark_session
        self.user_id = user_id
        self.model_literal = model_literal
        self.output_dir: Optional[str] = None  # Will be set by AutoMLRegressor
        
        # Data processing artifacts
        self.feature_vars = []
        self.selected_vars = []
        self.categorical_vars = []
        self.numerical_vars = []
        self.char_labels = None
        self.pipeline_model = None
        
        print(f"✅ RegressionDataProcessor initialized for user: {user_id}, model: {model_literal}")
    
    def preprocess(self, data: DataFrame, target_column: str, config: Dict) -> tuple:
        """
        Preprocess data for regression.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            config: Configuration dictionary
            
        Returns:
            Tuple of (processed_data, feature_vars, selected_vars, categorical_vars, numerical_vars)
        """
        print("🔄 Starting regression data preprocessing...")
        
        # Check if target column is string and convert to numeric for regression
        target_data_type = dict(data.dtypes)[target_column]
        if target_data_type == 'string':
            print(f"🔄 Converting string target column '{target_column}' to numeric for regression...")
            try:
                # First, check for null/empty values and handle them
                from pyspark.sql.functions import when, isnull, length, trim, isnan
                
                # Remove null, empty, and whitespace-only values
                data = data.filter(
                    ~isnull(col(target_column)) & 
                    (length(trim(col(target_column))) > 0) &
                    (trim(col(target_column)) != '')
                )
                
                print(f"📊 Filtered out null/empty values. Remaining rows: {data.count()}")
                
                # Try to convert string to numeric using cast
                data = data.withColumn(target_column, col(target_column).cast("double"))
                
                # Check for NaN values after conversion and filter them out
                data = data.filter(~isnan(col(target_column)))
                
                print(f"📊 Filtered out NaN values. Final rows: {data.count()}")
                print(f"✅ Successfully converted '{target_column}' to numeric type")
                
            except Exception as e:
                print(f"⚠️ Failed to convert '{target_column}' to numeric: {e}")
                print("🔄 Attempting to use StringIndexer for categorical target...")
                try:
                    # Use StringIndexer as fallback for categorical targets
                    from pyspark.ml.feature import StringIndexer
                    target_indexer = StringIndexer(inputCol=target_column, outputCol=target_column + "_indexed")
                    self.target_label_indexer = target_indexer.fit(data)
                    data = self.target_label_indexer.transform(data)
                    # Replace original target column with indexed version
                    data = data.drop(target_column).withColumnRenamed(target_column + "_indexed", target_column)
                    print(f"✅ Successfully encoded '{target_column}' using StringIndexer")
                except Exception as e2:
                    print(f"❌ Failed to encode target column: {e2}")
                    raise ValueError(f"Cannot convert target column '{target_column}' to numeric format for regression")
        
        # Get feature variables
        processed_data, self.feature_vars = self._get_feature_variables(data, target_column, config)
        print(f"📊 Found {len(self.feature_vars)} potential features")
        
        # Filter out date/timestamp columns automatically
        self.feature_vars = self._filter_date_columns(data, self.feature_vars)
        print(f"📊 Features after date column filtering: {len(self.feature_vars)}")
        
        # Apply preprocessing transformations
        processed_data, self.categorical_vars, self.numerical_vars = self._apply_preprocessing_transformations(
            processed_data, self.categorical_vars, self.numerical_vars, config.get('impute_value', -999)
        )
        
        # Final validation: Ensure target column has no null/NaN values for regression
        print(f"🔍 Final validation: Checking target column '{target_column}' for null/NaN values...")
        from pyspark.sql.functions import isnull, isnan
        
        # Count null/NaN values in target column
        null_count = processed_data.filter(isnull(col(target_column))).count()
        nan_count = processed_data.filter(isnan(col(target_column))).count()
        
        if null_count > 0:
            print(f"⚠️ Found {null_count} null values in target column. Filtering them out...")
            processed_data = processed_data.filter(~isnull(col(target_column)))
        
        if nan_count > 0:
            print(f"⚠️ Found {nan_count} NaN values in target column. Filtering them out...")
            processed_data = processed_data.filter(~isnan(col(target_column)))
        
        final_count = processed_data.count()
        print(f"✅ Target column validation complete. Final dataset: {final_count} rows")
        
        if final_count == 0:
            raise ValueError(f"No valid data remaining after cleaning target column '{target_column}'")
        
        # Select features using Random Forest Regression
        # Calculate actual number of features needed (min of available and configured)
        available_features = [col for col in processed_data.columns if col != target_column]
        configured_max = config.get('max_features', 30)
        actual_max_features = min(configured_max, len(available_features))
        
        print(f"📊 Available features: {len(available_features)}, Configured max: {configured_max}")
        print(f"🎯 Will select: {actual_max_features} features")
        
        self.selected_vars = self.select_features(processed_data, target_column, actual_max_features)
        
        print(f"✅ Preprocessing completed. Selected {len(self.selected_vars)} features")
        
        return processed_data, self.feature_vars, self.selected_vars, self.categorical_vars, self.numerical_vars
    
    def select_features(self, data: DataFrame, target_column: str, max_features: int = 30) -> List[str]:
        """
        Select features using Random Forest Regression feature importance.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        print(f"🎯 Selecting top {max_features} features using Random Forest Regression...")
        
        # Get numerical features for feature selection
        numerical_features = [col for col in self.numerical_vars if col != target_column]
        
        print(f"📊 Will rank all {len(numerical_features)} numerical features and select top {max_features}")
        
        # Always run feature importance (max_features should already be adjusted by caller)
        return self._run_feature_importance(data, numerical_features, target_column, max_features)
    
    def _run_feature_importance(self, data: DataFrame, feature_cols: List[str], 
                              target_column: str, num_features: int) -> List[str]:
        """
        Run Random Forest Regression feature importance selection.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
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
        
        # Try Random Forest Regression with improved configuration and retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}: Running Random Forest Regression feature importance...")
                
                # Create feature vector
                data_with_features = assembled_vectors(data, feature_cols, target_column)
                
                # Cache the data to avoid recomputation
                data_with_features.cache()
                
                # Use Random Forest Regression with reduced complexity for large feature sets
                num_trees = min(10, max(5, 50 // max(1, len(feature_cols) // 50)))  # Adaptive tree count
                max_depth = min(10, max(3, 20 - len(feature_cols) // 20))  # Adaptive depth
                
                print(f"Using RandomForestRegressor with {num_trees} trees, max depth {max_depth}")
                
                rf = RandomForestRegressor(
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
                top_features = feature_importance['name'].head(num_features).tolist()
                
                # Save feature importance (always save when feature selection runs)
                print(f"💾 Saving feature importance to output directory: {self.output_dir}")
                try:
                    # Use output_dir if available, otherwise use current directory
                    output_dir = self.output_dir if self.output_dir else '.'
                    print(f"📁 Using output directory: {output_dir}")
                    excel_path, plot_path = save_feature_importance(output_dir, self.user_id, self.model_literal, feature_importance)
                    print(f"✅ Feature importance saved for regression model:")
                    print(f"   📊 Excel file: {excel_path}")
                    print(f"   📈 Plot file: {plot_path}")
                    
                    # Verify files actually exist
                    if os.path.exists(excel_path):
                        print(f"✅ Verified Excel file exists: {excel_path}")
                    else:
                        print(f"❌ Excel file NOT found: {excel_path}")
                        
                    if plot_path and os.path.exists(plot_path):
                        print(f"✅ Verified plot file exists: {plot_path}")
                    else:
                        print(f"❌ Plot file NOT found: {plot_path}")
                        
                except Exception as e:
                    print(f"⚠️ Could not save feature importance to {self.output_dir}: {e}")
                    print(f"🔄 Trying to save to current directory as fallback...")
                    # Try to save to current directory as fallback
                    try:
                        excel_path, plot_path = save_feature_importance('.', self.user_id, self.model_literal, feature_importance)
                        print(f"✅ Feature importance saved to current directory (fallback):")
                        print(f"   📊 Excel file: {excel_path}")
                        print(f"   📈 Plot file: {plot_path}")
                    except Exception as e2:
                        print(f"❌ Failed to save feature importance even to current directory: {e2}")
                        print(f"❌ Full error details: {str(e2)}")
                
                # Unpersist cached data
                data_with_features.unpersist()
                
                print(f"Successfully selected {len(top_features)} features using Random Forest Regression")
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
                    print("Random Forest Regression failed on all attempts. Using variance-based fallback...")
                    return self._variance_based_selection(data, feature_cols, target_column, num_features)
                
                # Wait before retry to allow JVM recovery
                time.sleep(2)
        
        # Should not reach here, but fallback just in case
        print("Warning: All feature selection methods failed, returning first n features")
        return feature_cols[:num_features]
    
    def _statistical_feature_selection(self, data: DataFrame, feature_cols: List[str], 
                                     target_column: str, num_features: int) -> List[str]:
        """
        Statistical feature selection using correlation and variance filtering for regression.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            print("Running statistical feature selection for regression...")
            
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
            
            # If still too many, use correlation with target (for regression)
            try:
                # For regression, use correlation with target
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
            return feature_cols[:num_features]
    
    def _variance_based_selection(self, data: DataFrame, feature_cols: List[str], 
                                target_column: str, num_features: int) -> List[str]:
        """
        Variance-based feature selection for regression.
        
        Args:
            data: Input DataFrame
            feature_cols: List of feature column names
            target_column: Name of the target column
            num_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            print("Running variance-based feature selection...")
            
            # Calculate variance for each feature
            variances = []
            for col_name in feature_cols:
                try:
                    var_val = data.select(variance(col(col_name))).collect()[0][0]
                    if var_val is not None:
                        variances.append((col_name, var_val))
                except:
                    continue
            
            # Sort by variance and select top features
            variances.sort(key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in variances[:num_features]]
            
            print(f"Selected {len(selected)} features by variance")
            return selected
            
        except Exception as e:
            print(f"Variance-based selection failed: {e}")
            return feature_cols[:num_features]
    
    def _get_feature_variables(self, data: DataFrame, target_column: str, config: Dict) -> Tuple[DataFrame, List[str]]:
        """
        Get feature variables from the dataset and apply necessary conversions.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            config: Configuration dictionary
            
        Returns:
            Tuple of (modified_data, feature_variable_names)
        """
        # Get all columns except target
        all_columns = data.columns
        feature_columns = [col for col in all_columns if col != target_column]
        
        # Separate categorical and numerical features
        self.categorical_vars = []
        self.numerical_vars = []
        
        for col_name in feature_columns:
            # Check if column is categorical (string type)
            if data.schema[col_name].dataType.typeName() == 'string':
                self.categorical_vars.append(col_name)
            else:
                self.numerical_vars.append(col_name)
        
        print(f"📊 Feature analysis: {len(self.categorical_vars)} categorical, {len(self.numerical_vars)} numerical")
        
        # Analyze categorical cardinality and convert high-cardinality variables to numeric
        if self.categorical_vars:
            max_categorical_cardinality = config.get('max_categorical_cardinality', 50)
            categorical_vars_to_keep, categorical_vars_to_convert = analyze_categorical_cardinality(
                data, self.categorical_vars, max_categorical_cardinality
            )
            
            # Convert high-cardinality categorical variables to numeric
            if categorical_vars_to_convert:
                data = convert_categorical_to_numeric(data, categorical_vars_to_convert)
                # Update variable lists
                self.numerical_vars.extend(categorical_vars_to_convert)  # These are now numeric
                self.categorical_vars = categorical_vars_to_keep  # Only keep low-cardinality ones
                print(f"📊 Updated: {len(self.categorical_vars)} categorical, {len(self.numerical_vars)} numerical")
        
        return data, feature_columns
    
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
        
        # Check if target column exists and convert to numeric if it's string
        target_column = None
        for col_name in data.columns:
            if col_name not in local_categorical_vars and col_name not in local_numerical_vars:
                target_column = col_name
                break
        
        if target_column and target_column in current_dtypes and current_dtypes[target_column] == 'string':
            print(f"🔄 Converting string target column '{target_column}' to numeric for regression...")
            try:
                # First, check for null/empty values and handle them
                from pyspark.sql.functions import when, isnull, length, trim, isnan, regexp_replace
                
                # Clean the target column
                cleaned_col = regexp_replace(col(target_column), '[^0-9.-]', '')
                
                # Handle empty strings and convert to double
                data = data.withColumn(
                    target_column, 
                    when(
                        (isnull(col(target_column))) | 
                        (length(trim(col(target_column))) == 0) |
                        (trim(col(target_column)) == '') |
                        (trim(col(target_column)) == '?') |
                        (trim(col(target_column)) == 'NA') |
                        (trim(col(target_column)) == 'NULL'),
                        None
                    ).otherwise(cleaned_col.cast("double"))
                )
                
                # Filter out null/NaN values from target
                data = data.filter(~isnull(col(target_column)) & ~isnan(col(target_column)))
                
                print(f"📊 Final rows after target conversion: {data.count()}")
                print(f"✅ Successfully converted '{target_column}' to numeric type")
                
            except Exception as e:
                print(f"⚠️ Failed to convert '{target_column}' to numeric: {e}")
                print("🔄 Attempting to use StringIndexer for categorical target...")
                try:
                    # Use StringIndexer as fallback for categorical targets
                    target_indexer = StringIndexer(inputCol=target_column, outputCol=target_column + "_indexed")
                    fitted_indexer = target_indexer.fit(data)
                    data = fitted_indexer.transform(data)
                    # Replace original target column with indexed version
                    data = data.drop(target_column).withColumnRenamed(target_column + "_indexed", target_column)
                    print(f"✅ Successfully encoded '{target_column}' using StringIndexer")
                except Exception as e2:
                    print(f"❌ Failed to encode target column: {e2}")
                    raise ValueError(f"Cannot convert target column '{target_column}' to numeric format for regression")
        
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
                    else:
                        print(f"   📊 {cat_var}: {distinct_count} unique values")
                    
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
    
    def split_and_scale(self, data: DataFrame, train_size: float = 0.7, 
                       valid_size: float = 0.2, target_column: str = 'target', 
                       seed: int = 42, config: Optional[Dict] = None) -> tuple:
        """
        Split data into train/validation/test sets and apply scaling.
        
        Args:
            data: Input DataFrame
            train_size: Proportion for training
            valid_size: Proportion for validation
            target_column: Name of the target column
            seed: Random seed
            config: Configuration dictionary
            
        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
        print("📊 Splitting and scaling data...")
        
        # Split data
        splits = data.randomSplit([train_size, valid_size, 1 - train_size - valid_size], seed=seed)
        train_df, valid_df, test_df = splits
        
        print(f"Data split: Train={train_df.count()}, Valid={valid_df.count()}, Test={test_df.count()}")
        
        # Apply scaling if configured
        scaling_method = config.get('scaling_method', 'standard') if config else 'standard'
        
        if scaling_method != 'none':
            train_df = self.apply_scaling(train_df, target_column)
            valid_df = self.apply_scaling(valid_df, target_column)
            test_df = self.apply_scaling(test_df, target_column)
        
        return train_df, valid_df, test_df
    
    def apply_scaling(self, data: DataFrame, target_column: str) -> DataFrame:
        """
        Apply scaling to numerical features.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Scaled DataFrame
        """
        # For now, return data as-is. Scaling can be implemented as needed.
        return data

    def apply_preprocessing(self, data: DataFrame,
                            feature_vars: List[str],
                            selected_vars: List[str],
                            categorical_vars: List[str],
                            numerical_vars: List[str],
                            char_labels: Optional[PipelineModel],
                            impute_value: float,
                            target_column: Optional[str] = None,
                            target_label_indexer: Optional[Any] = None) -> DataFrame:
        """
        Apply the fitted preprocessing pipeline to a new dataset.

        This function mirrors the classification data processor's `apply_preprocessing` method.
        It starts from the raw feature columns to ensure categorical variables exist for
        encoding, applies the stored categorical encoding pipeline, performs numerical
        imputation, drops original categorical columns, and finally selects only the
        encoded feature columns present in `selected_vars` (plus the target column if present).

        Args:
            data: The raw input DataFrame to preprocess.
            feature_vars: Original list of feature names used during training (before encoding).
            selected_vars: List of selected encoded feature names from training.
            categorical_vars: List of raw categorical variable names.
            numerical_vars: List of raw numerical variable names.
            char_labels: Fitted StringIndexer pipeline used to encode categorical variables.
            impute_value: Value used for numerical imputation.
            target_column: Name of the target column (optional).
            target_label_indexer: Fitted StringIndexer for the target column (optional).

        Returns:
            Preprocessed DataFrame with columns matching `selected_vars` (and target column).
        """
        # Start by selecting the raw feature columns that exist in the new data
        raw_cols = [c for c in feature_vars if c in data.columns]

        # Filter out any obvious date/timestamp columns (simple heuristic)
        filtered_cols = []
        for col_name in raw_cols:
            try:
                dtype = dict(data.dtypes).get(col_name)
                if dtype not in ["timestamp", "date"]:
                    filtered_cols.append(col_name)
            except Exception:
                filtered_cols.append(col_name)

        columns_to_select = list(filtered_cols)
        # Include target column if provided
        if target_column and target_column in data.columns and target_column not in columns_to_select:
            columns_to_select.append(target_column)

        X = data.select(columns_to_select)

        # Apply target label encoding if necessary (rare for regression)
        if target_column and target_label_indexer is not None and target_column in X.columns:
            try:
                X = target_label_indexer.transform(X)
                X = X.drop(target_column).withColumnRenamed(f"{target_column}_indexed", target_column)
            except Exception:
                pass  # If encoding fails, assume target is already numeric

        # Apply categorical encoding using fitted pipeline
        if char_labels is not None:
            X = char_labels.transform(X)

        # Impute numerical variables
        if numerical_vars:
            X = numerical_imputation(X, numerical_vars, impute_value)

        # Drop original categorical variables
        if categorical_vars:
            X = X.select([c for c in X.columns if c not in categorical_vars])

        # Select only the selected_vars (encoded feature names) plus target column
        final_columns: List[str] = []
        for col_name in selected_vars:
            if col_name in X.columns:
                final_columns.append(col_name)
        if target_column and target_column in X.columns and target_column not in final_columns:
            final_columns.append(target_column)

        if final_columns:
            X = X.select(final_columns)

        return X
    
    def _restart_spark_session(self):
        """Restart Spark session if needed."""
        try:
            # This is a simplified restart - in practice, you might need more complex logic
            print("Attempting to restart Spark session...")
            return True
        except Exception as e:
            print(f"Failed to restart Spark session: {e}")
            return False
    
    def process_data(self, train_data: Union[str, DataFrame], target_column: Optional[str] = None,
                    oot1_data: Optional[Union[str, DataFrame]] = None,
                    oot2_data: Optional[Union[str, DataFrame]] = None,
                    **kwargs) -> Dict[str, DataFrame]:
        """
        Process data for regression training.
        
        Args:
            train_data: Training data (file path or DataFrame)
            target_column: Name of the target column
            oot1_data: Optional out-of-time validation data
            oot2_data: Optional second out-of-time validation data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing processed datasets
        """
        print("🔄 Processing data for regression...")
        
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
        
        # Preprocess training data
        processed_train, feature_vars, selected_vars, categorical_vars, numerical_vars = self.preprocess(
            train_data, target_column, kwargs
        )
        
        # Store preprocessing artifacts
        self.feature_vars = feature_vars
        self.selected_vars = selected_vars
        self.categorical_vars = categorical_vars
        self.numerical_vars = numerical_vars
        
        # Split training data
        train_size = kwargs.get('train_size', 0.7)
        valid_size = kwargs.get('valid_size', 0.2)
        seed = kwargs.get('seed', 42)
        
        train_df, valid_df, test_df = self.split_and_scale(
            processed_train, train_size, valid_size, target_column, seed, kwargs
        )
        
        # Process OOT data if provided
        oot1_df = None
        oot2_df = None
        
        if oot1_data is not None:
            print("🔄 Processing OOT1 data...")
            # Apply the same preprocessing pipeline used on training data
            oot1_df = self.apply_preprocessing(
                oot1_data,
                feature_vars,
                selected_vars,
                categorical_vars,
                numerical_vars,
                self.char_labels,
                kwargs.get('impute_value', -999),
                target_column,
                getattr(self, 'target_label_indexer', None)
            )
            # Clean OOT1 data: remove null/NaN values in target column
            if target_column and target_column in oot1_df.columns:
                print("🔍 Cleaning OOT1 target column...")
                from pyspark.sql.functions import isnull, isnan
                original_count = oot1_df.count()
                oot1_df = oot1_df.filter(~isnull(col(target_column)) & ~isnan(col(target_column)))
                clean_count = oot1_df.count()
                if original_count != clean_count:
                    print(f"   📊 OOT1: Filtered out {original_count - clean_count} rows with null/NaN target values")
        
        if oot2_data is not None:
            print("🔄 Processing OOT2 data...")
            # Apply the same preprocessing pipeline used on training data
            oot2_df = self.apply_preprocessing(
                oot2_data,
                feature_vars,
                selected_vars,
                categorical_vars,
                numerical_vars,
                self.char_labels,
                kwargs.get('impute_value', -999),
                target_column,
                getattr(self, 'target_label_indexer', None)
            )
            # Clean OOT2 data: remove null/NaN values in target column
            if target_column and target_column in oot2_df.columns:
                print("🔍 Cleaning OOT2 target column...")
                from pyspark.sql.functions import isnull, isnan
                original_count = oot2_df.count()
                oot2_df = oot2_df.filter(~isnull(col(target_column)) & ~isnan(col(target_column)))
                clean_count = oot2_df.count()
                if original_count != clean_count:
                    print(f"   📊 OOT2: Filtered out {original_count - clean_count} rows with null/NaN target values")
        
        # Create feature vectors for all datasets
        print("🔄 Creating feature vectors...")
        
        # Determine feature columns for assembling vectors. If selected_vars are available
        # (resulting from feature selection), use them; otherwise fall back to all numeric and
        # encoded categorical variables. Exclude the target column from the feature list.
        if self.selected_vars:
            all_feature_cols = [c for c in self.selected_vars if c != target_column]
        else:
            all_feature_cols = numerical_vars.copy()
            for cat_var in categorical_vars:
                all_feature_cols.append(f"{cat_var}_encoded")
        
        # Create assembler
        assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features")
        
        # Transform all datasets
        train_df = assembler.transform(train_df)
        valid_df = assembler.transform(valid_df) if valid_df is not None else None
        test_df = assembler.transform(test_df) if test_df is not None else None
        oot1_df = assembler.transform(oot1_df) if oot1_df is not None else None
        oot2_df = assembler.transform(oot2_df) if oot2_df is not None else None
        
        print("✅ Data processing completed")
        print(f"📊 Final dataset sizes:")
        print(f"   Train: {train_df.count()}")
        if valid_df:
            print(f"   Valid: {valid_df.count()}")
        if test_df:
            print(f"   Test: {test_df.count()}")
        if oot1_df:
            print(f"   OOT1: {oot1_df.count()}")
        if oot2_df:
            print(f"   OOT2: {oot2_df.count()}")
        
        return {
            'train': train_df,
            'valid': valid_df,
            'test': test_df,
            'oot1': oot1_df,
            'oot2': oot2_df
        }
    
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