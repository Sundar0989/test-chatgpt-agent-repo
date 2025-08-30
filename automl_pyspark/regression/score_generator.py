"""
Regression Score Generator

Generates scoring scripts for regression models.
"""

import os
from typing import Dict, List, Any


class RegressionScoreGenerator:
    """Generates scoring scripts for regression models."""
    
    def __init__(self, output_dir: str, user_id: str, model_id: str, model_literal: str):
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_id = model_id
        self.model_literal = model_literal
        
        # Model type to class name mappings
        self.model_class_names = {
            'linear_regression': 'LinearRegressionScorer',
            'random_forest': 'RandomForestScorer',
            'gradient_boosting': 'GradientBoostingScorer',
            'decision_tree': 'DecisionTreeScorer',
            'xgboost': 'XGBoostScorer',
            'lightgbm': 'LightGBMScorer'
        }
        
        # Model type to display name mappings
        self.model_display_names = {
            'linear_regression': 'Linear Regression',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'decision_tree': 'Decision Tree',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM'
        }
    
    def generate_scoring_code(self, config: Dict[str, Any], feature_vars: List[str],
                            selected_vars: List[str], categorical_vars: List[str],
                            numerical_vars: List[str], best_model_type: str):
        """
        Generate scoring code for all model types.
        
        Args:
            config: Model configuration
            feature_vars: List of feature variables
            selected_vars: List of selected variables
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
            best_model_type: Best model type selected
        """
        print("Generating regression scoring scripts...")
        
        # Generate scripts for all model types
        model_types = ['linear_regression', 'random_forest', 'gradient_boosting', 'decision_tree', 'xgboost', 'lightgbm']
        
        for model_type in model_types:
            if config.get(f'run_{model_type}', False):
                self._generate_model_scoring_script(
                    model_type, config, feature_vars, selected_vars,
                    categorical_vars, numerical_vars
                )
        
        print(f"Regression scoring scripts generated in {self.output_dir}")
    
    def generate_scoring_code_from_model_path(self, model_path: str, best_model_type: str):
        """
        Generate scoring code for a specific model at a custom path.
        
        Args:
            model_path: Path where the model is saved
            best_model_type: Type of the best model to generate script for
        """
        import json
        
        # Load model info from the specified path
        model_info_path = os.path.join(model_path, 'model_info.json')
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"Model info not found at {model_info_path}")
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        print(f"Generating scoring script for {best_model_type} model at {model_path}")
        
        # Generate script for the specific model type
        self._generate_model_scoring_script_with_path(
            best_model_type, 
            model_info.get('config', {}),
            model_info.get('feature_vars', []),
            model_info.get('selected_vars', []),
            model_info.get('categorical_vars', []),
            model_info.get('numerical_vars', []),
            model_path
        )
        
        print(f"Regression scoring script generated in {self.output_dir}")
    
    def _generate_model_scoring_script(self, model_type: str, config: Dict[str, Any],
                                     feature_vars: List[str], selected_vars: List[str],
                                     categorical_vars: List[str], numerical_vars: List[str]):
        """
        Generate scoring script for a specific model type.
        
        Args:
            model_type: Type of model
            config: Model configuration
            feature_vars: List of feature variables
            selected_vars: List of selected variables
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
        """
        # Get model-specific information
        class_name = self.model_class_names[model_type]
        display_name = self.model_display_names[model_type]
        
        # Generate the script content
        script_content = self._create_script_template(
            model_type, class_name, display_name, config,
            feature_vars, selected_vars, categorical_vars, numerical_vars
        )
        
        # Save the script
        script_filename = f"{model_type}_regression_scoring.py"
        script_path = os.path.join(self.output_dir, script_filename)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Generated {script_filename}")
    
    def _generate_model_scoring_script_with_path(self, model_type: str, config: Dict[str, Any],
                                               feature_vars: List[str], selected_vars: List[str],
                                               categorical_vars: List[str], numerical_vars: List[str],
                                               model_path: str):
        """
        Generate scoring script for a specific model type with custom model path.
        
        Args:
            model_type: Type of model
            config: Model configuration
            feature_vars: List of feature variables
            selected_vars: List of selected variables
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
            model_path: Path where the model is saved
        """
        # Get model-specific information
        class_name = self.model_class_names[model_type]
        display_name = self.model_display_names[model_type]
        
        # Generate the script content with custom path
        script_content = self._create_script_template_with_path(
            model_type, class_name, display_name, config,
            feature_vars, selected_vars, categorical_vars, numerical_vars, model_path
        )
        
        # Save the script
        script_filename = f"{model_type}_regression_scoring.py"
        script_path = os.path.join(self.output_dir, script_filename)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Generated {script_filename}")
    
    def _create_script_template(self, model_type: str, class_name: str, display_name: str,
                               config: Dict[str, Any], feature_vars: List[str],
                               selected_vars: List[str], categorical_vars: List[str],
                               numerical_vars: List[str]) -> str:
        """
        Create the script template for a model type.
        
        Args:
            model_type: Type of model
            class_name: Class name for the scorer
            display_name: Display name for the model
            config: Model configuration
            feature_vars: List of feature variables
            selected_vars: List of selected variables
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
            
        Returns:
            Complete script content as string
        """
        # Define paths
        hdfs_path = f"/user/{self.user_id}/mla_{self.model_literal}"
        home_path = f"/home/{self.user_id}/mla_{self.model_literal}"
        
        script = f'''"""
{display_name} Regression Model Scoring Script

This script provides scoring functionality for the {model_type} regression model.
Generated by AutoML PySpark Package
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, isnan, isnull, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.regression import (
    LinearRegressionModel, RandomForestRegressionModel,
    GBTRegressionModel, DecisionTreeRegressionModel
)

# Parameters
user_id = '{self.user_id}'
mdl_output_id = '{self.model_id}'
mdl_ltrl = '{self.model_literal}'
# Update the paths below to point to your model location
hdfs_path = '{hdfs_path}'
home_path = '{home_path}'

# Model variables - These will be loaded from model_info.json
SELECTED_VARS = {selected_vars}
CATEGORICAL_VARS = {categorical_vars}
NUMERICAL_VARS = {numerical_vars}
IMPUTE_VALUE = {config.get('impute_value', 0.0)}

class {class_name}:
    """
    {display_name} Regression Model Scorer class for making predictions on new data.
    """
    
    def __init__(self, spark_session: SparkSession = None):
        """
        Initialize the scorer.
        
        Args:
            spark_session: PySpark SparkSession (optional)
        """
        self.spark = spark_session or self._create_spark_session()
        self.char_labels = None
        self.pipeline_model = None
        self.model = None
        self.loaded = False
        
    def _create_spark_session(self) -> SparkSession:
        """Create PySpark session with optimized configuration for regression scoring."""
        return SparkSession.builder \\
            .appName("{display_name}_Regression_Scoring") \\
            .master("local[*]") \\
            .config("spark.driver.bindAddress", "127.0.0.1") \\
            .config("spark.driver.host", "127.0.0.1") \\
            .config("spark.sql.adaptive.enabled", "false") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \\
            .config("spark.sql.adaptive.skewJoin.enabled", "false") \\
            .config("spark.sql.adaptive.localShuffleReader.enabled", "false") \\
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \\
            .config("spark.driver.memory", "4g") \\
            .config("spark.driver.maxResultSize", "2g") \\
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \\
            .config("spark.network.timeout", "800s") \\
            .config("spark.rpc.askTimeout", "600s") \\
            .config("spark.sql.broadcastTimeout", "36000") \\
            .config("spark.rpc.message.maxSize", "512") \\
            .config("spark.local.dir", "/tmp") \\
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \\
            .getOrCreate()
    
    def load_model(self):
        """Load the trained model and preprocessing pipeline."""
        try:
            # Load model configuration
            model_info_path = os.path.join(home_path, 'model_info.json')
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            # Update global variables with loaded configuration
            global SELECTED_VARS, CATEGORICAL_VARS, NUMERICAL_VARS, IMPUTE_VALUE
            SELECTED_VARS = model_info.get('selected_vars', [])
            CATEGORICAL_VARS = model_info.get('categorical_vars', [])
            NUMERICAL_VARS = model_info.get('numerical_vars', [])
            IMPUTE_VALUE = model_info.get('config', {{}}).get('impute_value', 0.0)
            
            print("Model configuration loaded successfully")
            print(f"Selected variables: {{SELECTED_VARS}}")
            print(f"Categorical variables: {{CATEGORICAL_VARS}}")
            print(f"Numerical variables: {{NUMERICAL_VARS}}")
            
            # Load preprocessing pipeline (all files are in the same directory)
            self.char_labels = None
            self.pipeline_model = None
            
            # Try to load char_labels (may not exist if no categorical variables)
            try:
                self.char_labels = PipelineModel.load(os.path.join(home_path, 'char_labels'))
                print("char_labels pipeline loaded successfully")
            except Exception:
                print("Warning: Could not load char_labels pipeline (normal if no categorical variables)")
            
            # Load main pipeline model
            self.pipeline_model = PipelineModel.load(os.path.join(home_path, 'pipeline_model'))
            print("Pipeline model loaded successfully")
            
            # Load {model_type} model
            model_path = os.path.join(home_path, '{model_type}_model')
            self.model = self._load_model_by_type('{model_type}', model_path)
            
            self.loaded = True
            print("{display_name} regression model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading {model_type} regression model: {{str(e)}}")
            raise
    
    def _load_model_by_type(self, model_type: str, model_path: str):
        """Load model based on type."""
        if model_type == 'xgboost':
            # For XGBoost, load as standalone SparkXGBRegressor model
            try:
                from xgboost.spark import SparkXGBRegressorModel
                model = SparkXGBRegressorModel.load(model_path)
                print(f"Loaded XGBoost regression model as SparkXGBRegressorModel from {{model_path}}")
                return model
            except Exception as e:
                print(f"Error loading XGBoost regression model: {{str(e)}}")
                raise
        elif model_type == 'lightgbm':
            # For LightGBM, load as standalone model
            try:
                from synapse.ml.lightgbm import LightGBMRegressionModel
                model = LightGBMRegressionModel.load(model_path)
                print(f"Loaded LightGBM regression model from {{model_path}}")
                return model
            except Exception as e:
                print(f"Error loading LightGBM regression model: {{str(e)}}")
                raise
        else:
            # For standard Spark ML regression models
            model_classes = {{
                'linear_regression': LinearRegressionModel,
                'random_forest': RandomForestRegressionModel,
                'gradient_boosting': GBTRegressionModel,
                'decision_tree': DecisionTreeRegressionModel
            }}
            
            if model_type in model_classes:
                model_class = model_classes[model_type]
                model = model_class.load(model_path)
                print(f"Loaded {{model_type}} regression model from {{model_path}}")
                return model
            else:
                raise ValueError(f"Unsupported regression model type: {{model_type}}")
    
    def preprocess_data(self, data):
        """
        Preprocess input data for regression.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Check which selected variables exist in the input data
        available_columns = data.columns
        missing_vars = [var for var in SELECTED_VARS if var not in available_columns]
        
        if missing_vars:
            error_msg = f"Missing required columns in input data: {{missing_vars}}. "
            error_msg += f"Available columns: {{available_columns}}. "
            error_msg += f"Required columns: {{SELECTED_VARS}}"
            raise ValueError(error_msg)
        
        # Select features (all should exist now)
        X = data.select(SELECTED_VARS)
        
        # Apply categorical encoding only to existing categorical variables
        if self.char_labels:
            existing_cat_vars = [var for var in CATEGORICAL_VARS if var in available_columns]
            if existing_cat_vars:
                # Create a subset of char_labels pipeline for existing variables
                try:
                    X = self.char_labels.transform(X)
                except Exception as e:
                    print(f"Warning: Error applying categorical encoding: {{e}}")
                    print("Proceeding without categorical encoding.")
            else:
                print("No categorical variables found in input data, skipping encoding.")
        
        # Apply numerical imputation only to existing numerical variables
        existing_num_vars = [var for var in NUMERICAL_VARS if var in available_columns]
        if existing_num_vars:
            X = X.fillna(IMPUTE_VALUE, subset=existing_num_vars)
        
        # Remove original categorical columns that exist
        existing_cat_vars = [c for c in CATEGORICAL_VARS if c in X.columns]
        X = X.select([c for c in X.columns if c not in existing_cat_vars])
        
        # Rename encoded columns
        for cat_var in CATEGORICAL_VARS:
            index_col = cat_var + '_index'
            if index_col in X.columns:
                X = X.withColumnRenamed(index_col, cat_var)
        
        return X
    
    def score(self, data):
        """
        Score new data for regression predictions.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with predictions
        """
        if not self.loaded:
            self.load_model()
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Apply scaling pipeline
        scaled_data = self.pipeline_model.transform(processed_data)
        
        # Make predictions
        predictions = self.model.transform(scaled_data)
        
        return predictions
    
    def get_prediction_values(self, predictions):
        """
        Extract prediction values from predictions DataFrame.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            DataFrame with prediction values and any additional metrics
        """
        # For regression, we mainly care about the prediction column
        # Add any additional useful columns like confidence intervals if available
        result_columns = ['prediction']
        
        # Add residuals if actual values are present
        if 'actual' in predictions.columns or 'label' in predictions.columns:
            actual_col = 'actual' if 'actual' in predictions.columns else 'label'
            predictions = predictions.withColumn(
                'residual', 
                col('prediction') - col(actual_col)
            )
            predictions = predictions.withColumn(
                'absolute_error', 
                abs(col('prediction') - col(actual_col))
            )
            result_columns.extend(['residual', 'absolute_error'])
        
        return predictions.select(*result_columns)


def main():
    """Main function for {model_type} regression model scoring."""
    # Create scorer
    scorer = {class_name}()
    
    # Example: Load and score data
    # Replace this with your actual data loading logic
    sample_data = scorer.spark.createDataFrame([
        # Add your sample regression data here
        # Example: (feature1, feature2, feature3, ...)
    ])
    
    # Score data
    predictions = scorer.score(sample_data)
    
    # Get prediction values
    prediction_values = scorer.get_prediction_values(predictions)
    
    # Show results
    prediction_values.show()
    
    return predictions


if __name__ == "__main__":
    main()
'''
        
        return script

    def _create_script_template_with_path(self, model_type: str, class_name: str, display_name: str,
                                         config: Dict[str, Any], feature_vars: List[str],
                                         selected_vars: List[str], categorical_vars: List[str],
                                         numerical_vars: List[str], model_path: str) -> str:
        """
        Create the script template for a model type with a custom model path.
        
        Args:
            model_type: Type of model
            class_name: Class name for the scorer
            display_name: Display name for the model
            config: Model configuration
            feature_vars: List of feature variables
            selected_vars: List of selected variables
            categorical_vars: List of categorical variables
            numerical_vars: List of numerical variables
            model_path: Custom path where the model is saved
            
        Returns:
            Complete script content as string
        """
        # Use the provided model path instead of hardcoded paths
        script = f'''"""
{display_name} Regression Model Scoring Script

This script provides scoring functionality for the {model_type} regression model.
Generated by AutoML PySpark Package
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when, isnan, isnull, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.regression import (
    LinearRegressionModel, RandomForestRegressionModel,
    GBTRegressionModel, DecisionTreeRegressionModel
)

# Parameters
user_id = '{self.user_id}'
mdl_output_id = '{self.model_id}'
mdl_ltrl = '{self.model_literal}'
# Model path - Update this to point to your actual model location
model_path = '{model_path}'

# Model variables - These will be loaded from model_info.json
SELECTED_VARS = {selected_vars}
CATEGORICAL_VARS = {categorical_vars}
NUMERICAL_VARS = {numerical_vars}
IMPUTE_VALUE = {config.get('impute_value', 0.0)}

class {class_name}:
    """
    {display_name} Regression Model Scorer class for making predictions on new data.
    """
    
    def __init__(self):
        """Initialize the scorer."""
        self.spark = None
        self.char_labels = None
        self.pipeline_model = None
        self.model = None
        self.loaded = False
    
    def get_spark_session(self):
        """Get or create Spark session with optimized configuration."""
        return SparkSession.builder \\
            .appName("{class_name}_Regression_Scoring") \\
            .config("spark.driver.bindAddress", "127.0.0.1") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \\
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \\
            .config("spark.driver.memory", "4g") \\
            .config("spark.driver.maxResultSize", "2g") \\
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "1MB") \\
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \\
            .config("spark.network.timeout", "800s") \\
            .config("spark.rpc.askTimeout", "600s") \\
            .config("spark.sql.broadcastTimeout", "36000") \\
            .config("spark.rpc.message.maxSize", "512") \\
            .getOrCreate()
    
    def load_model(self):
        """Load the trained model and preprocessing pipeline."""
        try:
            # Initialize Spark session if not already done
            if self.spark is None:
                self.spark = self.get_spark_session()
            
            # Load model configuration
            model_info_path = os.path.join(model_path, 'model_info.json')
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            # Update global variables with loaded configuration
            global SELECTED_VARS, CATEGORICAL_VARS, NUMERICAL_VARS, IMPUTE_VALUE
            SELECTED_VARS = model_info.get('selected_vars', [])
            CATEGORICAL_VARS = model_info.get('categorical_vars', [])
            NUMERICAL_VARS = model_info.get('numerical_vars', [])
            IMPUTE_VALUE = model_info.get('config', {{}}).get('impute_value', 0.0)
            
            print("Model configuration loaded successfully")
            print(f"Selected variables: {{SELECTED_VARS}}")
            print(f"Categorical variables: {{CATEGORICAL_VARS}}")
            print(f"Numerical variables: {{NUMERICAL_VARS}}")
            
            # Load preprocessing pipeline (all files are in the same directory)
            self.char_labels = None
            self.pipeline_model = None
            
            # Try to load char_labels (may not exist if no categorical variables)
            try:
                self.char_labels = PipelineModel.load(os.path.join(model_path, 'char_labels'))
                print("char_labels pipeline loaded successfully")
            except Exception:
                print("Warning: Could not load char_labels pipeline (normal if no categorical variables)")
            
            # Load main pipeline model
            self.pipeline_model = PipelineModel.load(os.path.join(model_path, 'pipeline_model'))
            print("Pipeline model loaded successfully")
            
            # Load {model_type} model
            model_file_path = os.path.join(model_path, '{model_type}_model')
            self.model = self._load_model_by_type('{model_type}', model_file_path)
            
            self.loaded = True
            print("{display_name} regression model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading {model_type} regression model: {{str(e)}}")
            raise
    
    def _load_model_by_type(self, model_type: str, model_file_path: str):
        """Load model based on type."""
        if model_type == 'xgboost':
            # For XGBoost, load as standalone SparkXGBRegressor model
            try:
                from xgboost.spark import SparkXGBRegressorModel
                model = SparkXGBRegressorModel.load(model_file_path)
                print(f"Loaded XGBoost regression model from {{model_file_path}}")
                return model
            except Exception as e:
                print(f"Error loading XGBoost regression model: {{str(e)}}")
                raise
        elif model_type == 'lightgbm':
            # For LightGBM, load as standalone model
            try:
                from synapse.ml.lightgbm import LightGBMRegressionModel
                model = LightGBMRegressionModel.load(model_file_path)
                print(f"Loaded LightGBM regression model from {{model_file_path}}")
                return model
            except Exception as e:
                print(f"Error loading LightGBM regression model: {{str(e)}}")
                raise
        else:
            # For standard Spark ML regression models
            model_classes = {{
                'linear_regression': LinearRegressionModel,
                'random_forest': RandomForestRegressionModel,
                'gradient_boosting': GBTRegressionModel,
                'decision_tree': DecisionTreeRegressionModel
            }}
            
            if model_type in model_classes:
                model_class = model_classes[model_type]
                model = model_class.load(model_file_path)
                print(f"Loaded {{model_type}} regression model from {{model_file_path}}")
                return model
            else:
                raise ValueError(f"Unsupported regression model type: {{model_type}}")
    
    def preprocess_data(self, data):
        """
        Preprocess input data for regression.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Check which selected variables exist in the input data
        available_columns = data.columns
        missing_vars = [var for var in SELECTED_VARS if var not in available_columns]
        
        if missing_vars:
            error_msg = f"Missing required columns in input data: {{missing_vars}}. "
            error_msg += f"Available columns: {{available_columns}}. "
            error_msg += f"Required columns: {{SELECTED_VARS}}"
            raise ValueError(error_msg)
        
        # Select features (all should exist now)
        X = data.select(SELECTED_VARS)
        
        # Apply categorical encoding only to existing categorical variables
        if self.char_labels:
            existing_cat_vars = [var for var in CATEGORICAL_VARS if var in available_columns]
            if existing_cat_vars:
                # Create a subset of char_labels pipeline for existing variables
                try:
                    X = self.char_labels.transform(X)
                except Exception as e:
                    print(f"Warning: Error applying categorical encoding: {{e}}")
                    print("Proceeding without categorical encoding.")
            else:
                print("No categorical variables found in input data, skipping encoding.")
        
        # Apply numerical imputation only to existing numerical variables
        existing_num_vars = [var for var in NUMERICAL_VARS if var in available_columns]
        if existing_num_vars:
            X = X.fillna(IMPUTE_VALUE, subset=existing_num_vars)
        
        # Remove original categorical columns that exist
        existing_cat_vars = [c for c in CATEGORICAL_VARS if c in X.columns]
        X = X.select([c for c in X.columns if c not in existing_cat_vars])
        
        # Rename encoded columns
        for cat_var in CATEGORICAL_VARS:
            index_col = cat_var + '_index'
            if index_col in X.columns:
                X = X.withColumnRenamed(index_col, cat_var)
        
        return X
    
    def score(self, data):
        """
        Score new data for regression predictions.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with predictions
        """
        if not self.loaded:
            self.load_model()
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Apply scaling pipeline
        scaled_data = self.pipeline_model.transform(processed_data)
        
        # Make predictions
        predictions = self.model.transform(scaled_data)
        
        return predictions
    
    def get_prediction_values(self, predictions):
        """
        Extract prediction values from predictions DataFrame.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            DataFrame with prediction values and any additional metrics
        """
        # For regression, we mainly care about the prediction column
        # Add any additional useful columns like confidence intervals if available
        result_columns = ['prediction']
        
        # Add residuals if actual values are present
        if 'actual' in predictions.columns or 'label' in predictions.columns:
            actual_col = 'actual' if 'actual' in predictions.columns else 'label'
            predictions = predictions.withColumn(
                'residual', 
                col('prediction') - col(actual_col)
            )
            predictions = predictions.withColumn(
                'absolute_error', 
                abs(col('prediction') - col(actual_col))
            )
            result_columns.extend(['residual', 'absolute_error'])
        
        return predictions.select(*result_columns)


def main():
    """Main function for {model_type} regression model scoring."""
    # Create scorer
    scorer = {class_name}()
    
    # Example: Load and score data
    # Replace this with your actual data loading logic
    # sample_data = scorer.spark.createDataFrame([
    #     # Add your sample regression data here
    #     # Example: (feature1, feature2, feature3, ...)
    # ])
    
    # Score data
    # predictions = scorer.score(sample_data)
    
    # Get prediction values
    # prediction_values = scorer.get_prediction_values(predictions)
    
    # Show results
    # prediction_values.show()
    
    print("Regression scoring script is ready to use!")
    print("Load your data and call scorer.score(data) to get predictions")
    
    return scorer


if __name__ == "__main__":
    main()
'''
        
        return script 