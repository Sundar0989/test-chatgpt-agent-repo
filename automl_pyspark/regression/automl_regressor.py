"""
AutoML Regressor

Main class that orchestrates the entire AutoML pipeline for regression tasks.
This class provides a high-level interface for automated machine learning using PySpark for regression.
"""

import os
import signal
import time
from typing import Optional, Dict, Any, List, Union, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel

# Import configuration manager - handle different import contexts
try:
    from ..config_manager import ConfigManager
except ImportError:
    try:
        from config_manager import ConfigManager
    except ImportError:
        # For direct script execution
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from config_manager import ConfigManager

# Import spark optimization
try:
    from ..spark_optimization_config import get_optimized_spark_config
except ImportError:
    try:
        from spark_optimization_config import get_optimized_spark_config
    except ImportError:
        # For direct script execution
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from spark_optimization_config import get_optimized_spark_config

# Import components
from .data_processor import RegressionDataProcessor
from .model_builder import RegressionModelBuilder
from .model_validator import RegressionModelValidator
from .model_selector import RegressionModelSelector
from .score_generator import RegressionScoreGenerator
from .hyperparameter_tuner import RegressionHyperparameterTuner


class AutoMLRegressor:
    """
    Main AutoML regressor class for regression tasks.
    
    This class provides a comprehensive automated machine learning solution
    for regression tasks using PySpark, including:
    - Data preprocessing and feature engineering
    - Model building with multiple algorithms
    - Model validation and performance evaluation
    - Model selection and comparison
    - Production-ready scoring code generation
    """
    
    def __init__(self, 
                 output_dir: str = 'automl_output',
                 config_path: Optional[str] = None,
                 spark_session: Optional[SparkSession] = None,
                 preset: Optional[str] = None,
                 environment: Optional[str] = None,
                 # Backward compatibility parameters (deprecated)
                 user_id: Optional[str] = None,
                 model_id: Optional[str] = None, 
                 model_literal: Optional[str] = None,
                 **kwargs):
        """
        Initialize the AutoML regressor.
        
        Args:
            output_dir: Directory to save model outputs
            config_path: Path to YAML configuration file
            spark_session: PySpark SparkSession (optional, will create if None)
            preset: Configuration preset - 'quick' for fast training, 'comprehensive' for full training
            environment: Environment to use ('development', 'staging', 'production')
            user_id: (DEPRECATED) User identifier - use YAML config instead
            model_id: (DEPRECATED) Model identifier - use YAML config instead
            model_literal: (DEPRECATED) Model literal - use YAML config instead
            **kwargs: Additional arguments for backward compatibility
        """
        
        # Handle old-style initialization where spark_session was first argument  
        if isinstance(output_dir, SparkSession):
            print("🔄 Detected old-style initialization - converting to YAML configuration")
            spark_session = output_dir  # First arg was spark_session
            if config_path and not config_path.endswith(('.yaml', '.yml')):
                output_dir = config_path
                config_path = None
            else:
                output_dir = 'automl_output'
        
        # Store legacy parameters for later processing
        self._legacy_params = None
        if user_id is not None or model_id is not None or model_literal is not None:
            print("🔄 Detected legacy parameters - will convert to YAML configuration...")
            self._legacy_params = {
                'user_id': user_id,
                'model_id': model_id, 
                'model_literal': model_literal,
                'output_dir': output_dir
            }
            
            # Handle old-style output directory format
            if user_id and model_literal:
                output_dir = f'/home/{user_id}/mla_{model_literal}'
                print(f"🔄 Using legacy output directory format: {output_dir}")
        
        # Load configuration first (needed for Spark optimization)
        self.config_manager = ConfigManager(config_path, environment)
        
        # Initialize Spark session with performance optimization
        if spark_session is None:
            self.spark = self._create_optimized_spark_session()
        else:
            self.spark = spark_session
        
        # Store actual parameter values before cleanup
        self.actual_user_id = 'automl_user'
        self.actual_model_id = 'automl_model_id'  
        self.actual_model_literal = 'automl_model'
        
        if self._legacy_params is not None:
            print("   🔄 Converting legacy parameters to YAML configuration...")
            # Store the actual values before cleanup
            self.actual_user_id = self._legacy_params.get('user_id', 'automl_user')
            self.actual_model_id = self._legacy_params.get('model_id', 'automl_model_id')
            self.actual_model_literal = self._legacy_params.get('model_literal', 'automl_model')
            
            self._update_config_from_legacy_params(
                self._legacy_params['user_id'],
                self._legacy_params['model_id'],
                self._legacy_params['model_literal'],
                self._legacy_params['output_dir']
            )
            self._legacy_params = None  # Clean up
        
        # Apply preset configuration LAST (highest priority)
        if preset is not None:
            print(f"🎯 Applying '{preset}' preset configuration (highest priority)...")
            self._apply_preset_config(preset)
        else:
            print("📋 Using user-provided configuration (no preset specified)")
        
        self.config = self._build_config()
        
        # Set output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components with actual parameters
        self.data_processor = RegressionDataProcessor(self.spark, self.actual_user_id, self.actual_model_literal)
        # Pass output directory to data processor for saving feature importance
        self.data_processor.output_dir = self.output_dir
        self.model_builder = RegressionModelBuilder(self.spark)
        self.model_validator = RegressionModelValidator(self.spark, self.output_dir, self.actual_user_id, self.actual_model_literal)
        self.model_selector = RegressionModelSelector(self.output_dir, self.actual_user_id, self.actual_model_literal)
        self.score_generator = RegressionScoreGenerator(self.output_dir, self.actual_user_id, self.actual_model_id, self.actual_model_literal)
        self.hyperparameter_tuner = RegressionHyperparameterTuner(self.spark, self.output_dir, self.actual_user_id, self.actual_model_literal)
        
        # Pipeline artifacts
        self.feature_vars = []
        self.categorical_vars = []
        self.numerical_vars = []
        self.selected_vars = []
        self.char_labels = None
        self.pipeline_model = None
        self.best_model = None
        self.best_model_type = None
        self.model_metrics = {}
        
        # Log configuration
        self._log_configuration()
    
    def _create_optimized_spark_session(self) -> SparkSession:
        """Create Spark session with performance optimization."""
        print("🚀 Creating optimized Spark session for regression...")
        
        # Get optimized base configuration with BigQuery support
        spark_config = get_optimized_spark_config(include_bigquery=True)
        
        # Override with BigQuery-optimized memory settings
        spark_config.update({
            "spark.driver.memory": "8g",  # Increased for BigQuery operations
            "spark.driver.maxResultSize": "4g",  # Increased for BigQuery results
            "spark.executor.memory": "4g",  # Increased for BigQuery processing
        })
        
        # Build Spark session
        builder = SparkSession.builder.appName("AutoML Regression Pipeline (Optimized)")
        
        # Add BigQuery and SynapseML/LightGBM connector packages
        packages = [
            "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1",
            "com.microsoft.azure:synapseml_2.12:1.0.3"
        ]
        builder = builder.config("spark.jars.packages", ",".join(packages))
        
        for key, value in spark_config.items():
            builder = builder.config(key, value)
        
        return builder.getOrCreate()
    
    def create_bigquery_spark_session(self, driver_memory: str = "64g", include_lightgbm: bool = True) -> SparkSession:
        """
        Create a Spark session optimized for BigQuery data loading.
        Uses the proven working configuration for BigQuery.
        
        Args:
            driver_memory: Driver memory allocation (default: 64g for BigQuery)
            include_lightgbm: If True, includes SynapseML JARs for LightGBM support
            
        Returns:
            Spark session optimized for BigQuery + AutoML
        """
        try:
            print(f"🚀 Creating BigQuery-optimized Spark session for regression...")
            print(f"   💾 Driver memory: {driver_memory}")
            print(f"   🔗 BigQuery connector: v0.36.1")
            
            # Stop existing session if needed
            if hasattr(self, 'spark') and self.spark is not None:
                print("🔄 Stopping existing Spark session for BigQuery compatibility...")
                self.spark.stop()
            
            # Build Spark session with proven BigQuery configuration
            builder = SparkSession.builder.appName("AutoML BigQuery Regressor")
            
            # Add BigQuery connector package (proven working version)
            packages = ["com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1"]
            if include_lightgbm:
                packages.append("com.microsoft.azure:synapseml_2.12:1.0.3")
            
            builder = builder.config("spark.jars.packages", ",".join(packages))
            builder = builder.config("spark.driver.memory", driver_memory)
            
            # Apply AutoML optimizations
            try:
                base_config = get_optimized_spark_config(include_synapseml=include_lightgbm, include_bigquery=True)
                
                # Override with BigQuery-specific settings
                bigquery_config = {
                    "spark.driver.memory": driver_memory,
                    "spark.driver.maxResultSize": "8g",
                    "spark.executor.memory": "8g",
                    "spark.sql.execution.arrow.pyspark.enabled": "true",
                    "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
                }
                
                # Apply all configurations
                final_config = {**base_config, **bigquery_config}
                for key, value in final_config.items():
                    builder = builder.config(key, value)
                    
            except ImportError:
                print("⚠️ Spark optimization config not available, using basic config")
            
            spark = builder.getOrCreate()
            
            print(f"✅ BigQuery-optimized Spark session created")
            if include_lightgbm:
                print("   🤖 LightGBM support included")
            
            # Update our spark reference
            self.spark = spark
            
            # Update component spark references
            if hasattr(self, 'data_processor'):
                self.data_processor.spark = spark
            if hasattr(self, 'model_builder'):
                self.model_builder.spark = spark
            if hasattr(self, 'model_validator'):
                self.model_validator.spark = spark
            
            return spark
            
        except Exception as e:
            print(f"❌ Failed to create BigQuery session: {e}")
            raise
    
    def load_bigquery_data(self, 
                          project_id: str, 
                          table_id: str, 
                          limit_rows: Optional[int] = None) -> DataFrame:
        """
        Load data from BigQuery using the proven working configuration.
        
        Args:
            project_id: Your GCP project ID (e.g., "atus-prism-dev")
            table_id: Full table reference (e.g., "atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02")
            limit_rows: Limit number of rows for testing (optional)
            
        Returns:
            Spark DataFrame ready for AutoML
        """
        print(f"🔗 Loading BigQuery data for regression...")
        print(f"   📊 Project: {project_id}")
        print(f"   🎯 Table: {table_id}")
        
        # Use the proven working configuration with proper row limiting
        if limit_rows:
            print(f"   📋 Applying row limit for testing: {limit_rows:,} rows")
            print(f"      🔧 Using maxRowsPerPartition for row limiting")
            
            reader = self.spark.read \
                .format("bigquery") \
                .option("parentProject", project_id) \
                .option("viewsEnabled", "true") \
                .option("useAvroLogicalTypes", "true") \
                .option("table", table_id) \
                .option("maxRowsPerPartition", limit_rows)
        else:
            print(f"   📖 Loading full table using direct table reference")
            reader = self.spark.read \
                .format("bigquery") \
                .option("parentProject", project_id) \
                .option("viewsEnabled", "true") \
                .option("useAvroLogicalTypes", "true") \
                .option("table", table_id)
        
        df = reader.load()
        
        # Basic validation
        row_count = df.count()
        col_count = len(df.columns)
        
        print(f"✅ BigQuery data loaded successfully!")
        print(f"   📊 Shape: {row_count:,} rows × {col_count} columns")
        print(f"   📋 Columns: {df.columns[:5]}{'...' if col_count > 5 else ''}")
        
        return df
    
    def fit_from_bigquery(self,
                         project_id: str,
                         table_id: str,
                         target_column: str,
                         oot1_table_id: Optional[str] = None,
                         oot2_table_id: Optional[str] = None,
                         limit_rows: Optional[int] = None,
                         driver_memory: str = "64g",
                         **kwargs) -> 'AutoMLRegressor':
        """
        Fit AutoML regressor directly from BigQuery tables.
        Uses the proven working BigQuery configuration.
        
        Args:
            project_id: Your GCP project ID (e.g., "atus-prism-dev")
            table_id: Training data table reference (e.g., "atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02")
            target_column: Name of the target column to predict
            oot1_table_id: Out-of-time validation data 1 table reference (optional)
            oot2_table_id: Out-of-time validation data 2 table reference (optional)
            limit_rows: Limit number of rows for testing (optional)
            driver_memory: Driver memory allocation (default: 64g for BigQuery)
            **kwargs: Additional configuration parameters
            
        Returns:
            self: Fitted AutoML regressor
            
        Example:
            automl = AutoMLRegressor()
            results = automl.fit_from_bigquery(
                project_id="atus-prism-dev",
                table_id="atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02",
                target_column="your_target_column",
                limit_rows=10000  # Optional: for testing
            )
        """
        print("🚀 Starting AutoML regression with BigQuery data...")
        
        # 1. Create BigQuery-optimized Spark session
        self.create_bigquery_spark_session(driver_memory=driver_memory, include_lightgbm=True)
        
        # 2. Load training data from BigQuery
        print("📊 Loading training data from BigQuery...")
        train_data = self.load_bigquery_data(project_id, table_id, limit_rows)
        
        # 3. Load OOT data if specified
        oot1_data = None
        oot2_data = None
        
        if oot1_table_id:
            print("📅 Loading OOT1 data from BigQuery...")
            oot1_data = self.load_bigquery_data(project_id, oot1_table_id, limit_rows)
        
        if oot2_table_id:
            print("📅 Loading OOT2 data from BigQuery...")
            oot2_data = self.load_bigquery_data(project_id, oot2_table_id, limit_rows)
        
        # 4. Run AutoML fit with loaded data
        print("🤖 Starting AutoML regression training...")
        return self.fit(train_data, target_column, oot1_data, oot2_data, **kwargs)
    
    def _is_bigquery_table(self, table_reference: str) -> bool:
        """
        Check if a string is a BigQuery table reference.
        
        Args:
            table_reference: String to check
            
        Returns:
            True if it looks like a BigQuery table reference (project.dataset.table)
        """
        # BigQuery table format: project_id.dataset_id.table_id
        parts = table_reference.split('.')
        return (
            len(parts) == 3 and  # Must have exactly 3 parts
            all(part and part.replace('_', '').replace('-', '').isalnum() for part in parts) and  # Valid characters
            not table_reference.endswith('.csv') and  # Not a CSV file
            not table_reference.startswith('/') and  # Not a file path
            not table_reference.startswith('file:')  # Not a file URI
        )
    
    def _update_config_from_legacy_params(self, user_id: Optional[str], model_id: Optional[str], 
                                         model_literal: Optional[str], output_dir: str):
        """Update configuration with legacy parameters."""
        legacy_config = {}
        
        if user_id:
            legacy_config['user_id'] = user_id
        if model_id:
            legacy_config['model_id'] = model_id
        if model_literal:
            legacy_config['model_literal'] = model_literal
        if output_dir:
            legacy_config['output_directory'] = output_dir
        
        # Apply legacy config with lower priority
        self.config_manager.update_config(legacy_config)
    
    def _apply_preset_config(self, preset: str):
        """Apply preset configuration."""
        preset_configs = {
            'quick': {
                'regression': {
                    'models': {
                        'run_linear_regression': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': False,
                        'run_decision_tree': False,
                        'run_xgboost': False,
                        'run_lightgbm': False
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': False
                    }
                }
            },
            'comprehensive': {
                'regression': {
                    'models': {
                        'run_linear_regression': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': True,
                        'run_decision_tree': True,
                        'run_xgboost': True,
                        'run_lightgbm': True
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': True
                    }
                }
            }
        }
        
        if preset in preset_configs:
            self.config_manager.override_config(preset_configs[preset])
        else:
            print(f"⚠️ Unknown preset '{preset}', using default configuration")
    
    def _build_config(self) -> Dict[str, Any]:
        """Build configuration by combining global and regression-specific settings."""
        # Get flattened configuration for backwards compatibility
        config = self.config_manager.get_flat_config('regression', include_global=True)
        
        # Get global and regression configs for explicit mapping
        global_config = self.config_manager.get_global_config()
        regression_config = self.config_manager.get_task_config('regression')
        
        # Map global data processing settings to expected flat keys
        if 'data_processing' in global_config:
            data_proc = global_config['data_processing']
            config['missing_threshold'] = data_proc.get('missing_value_threshold', 0.7)
            config['categorical_threshold'] = data_proc.get('categorical_threshold', 10)
            config['sample_fraction'] = data_proc.get('sample_fraction', 1.0)
            config['seed'] = data_proc.get('random_seed', 42)
            config['impute_value'] = 0.0  # Default impute value
            
            # Variable inclusion/exclusion settings
            config['include_vars'] = data_proc.get('include_vars', [])
            config['exclude_vars'] = data_proc.get('exclude_vars', [])
            config['include_prefix'] = data_proc.get('include_prefix', [])
            config['exclude_prefix'] = data_proc.get('exclude_prefix', [])
            config['include_suffix'] = data_proc.get('include_suffix', [])
            config['exclude_suffix'] = data_proc.get('exclude_suffix', [])
        
        # Map global feature selection settings to expected flat keys
        if 'feature_selection' in global_config:
            feat_sel = global_config['feature_selection']
            config['max_features'] = feat_sel.get('max_features', 50)
            config['sequential_threshold'] = feat_sel.get('sequential_threshold', 200)
            config['chunk_size'] = feat_sel.get('chunk_size', 100)
            config['features_per_chunk'] = feat_sel.get('features_per_chunk', 30)
        
        # Map global validation settings to expected flat keys
        if 'validation' in global_config:
            validation = global_config['validation']
            config['test_size'] = validation.get('test_size', 0.2)
            config['valid_size'] = validation.get('validation_size', 0.2)
            # Calculate train_size
            config['train_size'] = 1.0 - config['test_size'] - config['valid_size']
        
        # Add model flags directly to config root
        if 'models' in regression_config:
            config.update(regression_config['models'])
        
        # Add hyperparameter tuning settings
        if 'hyperparameter_tuning' in regression_config:
            config.update(regression_config['hyperparameter_tuning'])
        
        # Add model selection settings
        if 'model_selection' in regression_config:
            config.update(regression_config['model_selection'])
        
        # Add evaluation settings
        if 'evaluation' in regression_config:
            config.update(regression_config['evaluation'])
        
        return config
    
    def _log_configuration(self):
        """Log the current configuration and available models."""
        print("\n🔧 AutoML Regression Configuration Loaded:")
        print(f"   📁 Config source: {self.config_manager.config_path}")
        
        # Validate configuration
        is_valid = self.config_manager.validate_config('regression')
        if not is_valid:
            print("⚠️ Configuration validation failed - using defaults where possible")
        
        # Log available models
        self._log_available_models()
    
    def _log_available_models(self):
        """Log which regression models are available and enabled."""
        print("\n🔧 AutoML Regression Model Configuration:")
        
        # Core models (always available)
        core_models = [
            ('Linear Regression', self.config.get('run_linear_regression', True)),
            ('Random Forest', self.config.get('run_random_forest', True)),
            ('Gradient Boosting', self.config.get('run_gradient_boosting', True)),
            ('Decision Tree', self.config.get('run_decision_tree', True))
        ]
        
        # Advanced models (conditional availability)
        from .model_builder import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
        advanced_models = [
            ('XGBoost', self.config.get('run_xgboost', False), XGBOOST_AVAILABLE),
            ('LightGBM', self.config.get('run_lightgbm', False), LIGHTGBM_AVAILABLE)
        ]
        
        enabled_count = 0
        
        print("  Core Models:")
        for name, enabled in core_models:
            status = "✅ Enabled" if enabled else "⭕ Disabled"
            print(f"    {name}: {status}")
            if enabled:
                enabled_count += 1
        
        print("  Advanced Models:")
        for name, enabled, available in advanced_models:
            if available:
                status = "✅ Enabled" if enabled else "⭕ Disabled"
                print(f"    {name}: {status}")
                if enabled:
                    enabled_count += 1
            else:
                print(f"    {name}: ❌ Not Available")
        
        print(f"\n📊 Total enabled models: {enabled_count}")
        
        if enabled_count == 0:
            print("⚠️ WARNING: No models are enabled! Please check your configuration.")
        
        # Log hyperparameter tuning status
        tuning_enabled = self.config.get('enable_hyperparameter_tuning', False)
        tuning_status = "✅ Enabled" if tuning_enabled else "⭕ Disabled"
        print(f"🎯 Hyperparameter Tuning: {tuning_status}")
    
    def fit(self, 
            train_data: Union[str, DataFrame],
            target_column: str,
            oot1_data: Optional[Union[str, DataFrame]] = None,
            oot2_data: Optional[Union[str, DataFrame]] = None,
            **kwargs) -> 'AutoMLRegressor':
        """
        Fit the AutoML pipeline on the training data.
        
        Args:
            train_data: Training DataFrame or file path
            target_column: Name of the target column
            oot1_data: Out-of-time validation data 1 (optional)
            oot2_data: Out-of-time validation data 2 (optional)
            **kwargs: Additional configuration parameters
            
        Returns:
            self: Fitted AutoML regressor
        """
        # Update configuration
        self.config.update(kwargs)
        self.config['target_column'] = target_column
        

        # Load data if string paths are provided
        if isinstance(train_data, str):
            # Check if it's a BigQuery table reference
            if self._is_bigquery_table(train_data):
                print(f"🔗 Loading training data from BigQuery: {train_data}")
                project_id = train_data.split('.')[0]
                train_data = self.spark.read \
                    .format("bigquery") \
                    .option("parentProject", project_id) \
                    .option("table", train_data) \
                    .load()
            else:
                print(f"📁 Loading training data from file: {train_data}")
                train_data = self.spark.read.csv(train_data, header=True, inferSchema=True)
        
        if isinstance(oot1_data, str):
            # Check if it's a BigQuery table reference
            if self._is_bigquery_table(oot1_data):
                print(f"🔗 Loading OOT1 data from BigQuery: {oot1_data}")
                project_id = oot1_data.split('.')[0]
                oot1_data = self.spark.read \
                    .format("bigquery") \
                    .option("parentProject", project_id) \
                    .option("viewsEnabled", "true") \
                    .option("useAvroLogicalTypes", "true") \
                    .option("table", oot1_data) \
                    .load()
            else:
                print(f"📁 Loading OOT1 data from file: {oot1_data}")
                oot1_data = self.spark.read.csv(oot1_data, header=True, inferSchema=True)
        
        if isinstance(oot2_data, str):
            # Check if it's a BigQuery table reference
            if self._is_bigquery_table(oot2_data):
                print(f"🔗 Loading OOT2 data from BigQuery: {oot2_data}")
                project_id = oot2_data.split('.')[0]
                oot2_data = self.spark.read \
                    .format("bigquery") \
                    .option("parentProject", project_id) \
                    .option("viewsEnabled", "true") \
                    .option("useAvroLogicalTypes", "true") \
                    .option("table", oot2_data) \
                    .load()
            else:
                print(f"📁 Loading OOT2 data from file: {oot2_data}")
                oot2_data = self.spark.read.csv(oot2_data, header=True, inferSchema=True)
        
        # Get timeout setting from config
        timeout_minutes = self.config.get('timeout_minutes', 0)
        if timeout_minutes <= 0:
            # Try to get from performance config
            perf_config = self.config_manager.config.get('performance', {})
            timeout_minutes = perf_config.get('timeout_minutes', 0)
        
        if timeout_minutes > 0:
            print(f"⏱️ Pipeline timeout set to {timeout_minutes} minutes")
            return self._with_timeout(self._fit_internal, timeout_minutes, train_data, target_column, oot1_data, oot2_data)
        else:
            print("⏱️ No timeout limit set - pipeline will run until completion")
            return self._fit_internal(train_data, target_column, oot1_data, oot2_data)
    
    def _with_timeout(self, func, timeout_minutes: int, *args, **kwargs):
        """Execute function with timeout."""
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"AutoML pipeline timed out after {timeout_minutes} minutes")
        
        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)  # Convert to seconds
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel the alarm
            return result
        except TimeoutError:
            print(f"⏰ Pipeline exceeded {timeout_minutes} minute timeout limit")
            raise
        finally:
            signal.alarm(0)  # Ensure alarm is cancelled
    
    def _fit_internal(self, train_data: DataFrame, target_column: str, 
                     oot1_data: Optional[DataFrame] = None, 
                     oot2_data: Optional[DataFrame] = None) -> 'AutoMLRegressor':
        """Internal fit method that performs the actual training."""
        
        # Log available model types with updated configuration
        self._log_available_models()
        
        print("Starting AutoML regression pipeline...")
        print(f"Target column: {target_column}")
        
        # Get training data shape - optimized for large BigQuery datasets
        print("📊 Analyzing dataset size...")
        try:
            total_rows = train_data.count()
            total_columns = len(train_data.columns)
            print(f"Training data shape: {total_rows:,} rows, {total_columns} columns")
        except Exception as e:
            print(f"⚠️ Could not get exact row count (large BigQuery dataset): {e}")
            print("🔄 Proceeding with column count only...")
            total_columns = len(train_data.columns)
            print(f"Training data shape: Unknown rows (very large BigQuery dataset), {total_columns} columns")
        
        # Step 1: Data preprocessing and processing
        print("\n📊 Loading and processing data...")
        
        # Create a clean config without target_column to avoid conflicts
        clean_config = {k: v for k, v in self.config.items() if k != 'target_column'}
        
        processed_data = self.data_processor.process_data(
            train_data, target_column, oot1_data, oot2_data, **clean_config
        )
        
        # Extract processed datasets
        train_df = processed_data['train']
        valid_df = processed_data.get('valid')
        test_df = processed_data.get('test')
        oot1_df = processed_data.get('oot1')
        oot2_df = processed_data.get('oot2')
        
        # Store preprocessing artifacts
        self.feature_vars = self.data_processor.feature_vars
        self.selected_vars = self.data_processor.selected_vars
        self.categorical_vars = self.data_processor.categorical_vars
        self.numerical_vars = self.data_processor.numerical_vars
        
        # Copy preprocessing pipelines from DataProcessor to AutoMLRegressor
        self.char_labels = self.data_processor.char_labels
        self.pipeline_model = self.data_processor.pipeline_model
        
        print(f"Feature variables: {self.feature_vars}")
        print(f"Selected variables: {self.selected_vars}")
        print(f"Categorical variables: {self.categorical_vars}")
        print(f"Numerical variables: {self.numerical_vars}")
        
        # Step 2: Build and evaluate models
        print("\n🤖 Building and evaluating regression models...")
        
        # Prepare datasets list for validation
        datasets = []
        dataset_names = []
        
        if train_df:
            datasets.append(train_df)
            dataset_names.append('train')
        if valid_df:
            datasets.append(valid_df)
            dataset_names.append('valid')
        if test_df:
            datasets.append(test_df)
            dataset_names.append('test')
        if oot1_df:
            datasets.append(oot1_df)
            dataset_names.append('oot1')
        if oot2_df:
            datasets.append(oot2_df)
            dataset_names.append('oot2')
        
        # Determine which models to run
        models_to_run = []
        for model_name in ['linear_regression', 'random_forest', 'gradient_boosting', 'decision_tree', 'xgboost', 'lightgbm']:
            if self.config.get(f'run_{model_name}', False):
                models_to_run.append(model_name)
        
        if not models_to_run:
            raise ValueError("No models selected for training")
        
        print(f"🎯 Training models: {models_to_run}")
        
        # Train models with dual training approach (default + hyperparameter tuned)
        trained_models = {}
        for model_type in models_to_run:
            try:
                print(f"\n🔧 Training {model_type} model...")
                
                # Use dual training approach: default vs hyperparameter-tuned
                model_result = self._build_model_with_dual_training(
                    train_df, target_column, model_type, self.selected_vars, datasets, dataset_names
                )
                
                # Store selected model and metrics
                selected_model = model_result['model']
                selected_metrics = model_result['metrics']
                comparison_info = model_result['comparison']
                
                # Store metrics with comparison information
                self.model_metrics[model_type] = selected_metrics
                self.model_metrics[model_type]['selection_info'] = comparison_info
                
                trained_models[model_type] = {
                    'model': selected_model,
                    'metrics': selected_metrics
                }
                
                # Note: Both default and tuned models are already saved separately in dual training method
                # No need to save the selected model again here to avoid redundancy
                
                print(f"✅ {model_type} model training completed - selected {comparison_info['decision']} version")
                
            except Exception as e:
                print(f"❌ Error training {model_type}: {str(e)}")
                continue
        
        if not trained_models:
            raise RuntimeError("No models were trained successfully")
        
        # Step 3: Model selection
        print("\n🎯 Model Selection...")
        self._select_best_model()
        
        # Step 4: Generate scoring code
        print("\n🔧 Generating Scoring Code...")
        self._generate_scoring_code()
        
        # Step 5: Save model configuration files
        print("\n💾 Saving model configuration files...")
        self.save_model(self.output_dir)

        # Step 6: Compute SHAP values for explainability (only for best model)
        if self.best_model is not None and self.best_model_type is not None:
            try:
                import sys
                import os
                # Add the parent directory to the path for absolute import
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from explainability import compute_shap_values
                print(f"\n🔍 Computing SHAP values for best model ({self.best_model_type}) explainability...")
                # Use the original training data for SHAP.  Feature columns
                # default to selected_vars if available, otherwise all feature_vars.
                sample_df = train_data
                feature_cols = self.selected_vars if self.selected_vars else self.feature_vars
                compute_shap_values(
                    spark=self.spark,
                    pipeline_model=self.pipeline_model,
                    model=self.best_model,
                    sample_df=sample_df,
                    feature_cols=feature_cols,
                    output_dir=self.output_dir,
                    model_type="regression",
                    max_samples=50,
                )
                print(f"✅ SHAP values computed successfully for best model ({self.best_model_type})")
            except Exception as e:
                print(f"⚠️ SHAP computation skipped for best model ({self.best_model_type}): {e}")
        else:
            print("⚠️ SHAP computation skipped: No best model available")

        print("\n✅ AutoML regression pipeline completed successfully!")

        # Print comprehensive model selection summary
        self.print_model_selection_summary()

        return self
    
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions using the best model.
        
        Args:
            data: Input DataFrame for prediction
            
        Returns:
            DataFrame with predictions
        """
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Apply preprocessing pipeline
        processed_data = self._apply_preprocessing(data)
        
        # Make predictions
        predictions = self.best_model.transform(processed_data)
        
        return predictions
    
    def _apply_preprocessing(self, data: DataFrame) -> DataFrame:
        """Apply preprocessing pipeline to new data."""
        if self.pipeline_model is None:
            raise ValueError("Preprocessing pipeline not available")
        
        return self.pipeline_model.transform(data)
    
    def save_model(self, path: str):
        """
        Save the fitted model and artifacts.
        
        Args:
            path: Directory path to save the model
        """
        if self.best_model is None and not hasattr(self, 'best_model_type'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        os.makedirs(path, exist_ok=True)
        
        # Create a simple model info dictionary (no joblib dependency)
        model_info = {
            'config': self.config,
            'feature_vars': self.feature_vars,
            'selected_vars': self.selected_vars,
            'categorical_vars': self.categorical_vars,
            'numerical_vars': self.numerical_vars,
            'best_model_type': self.best_model_type,
            'model_metrics': getattr(self, 'model_metrics', {}),
            'task_type': 'regression'
        }
        
        # Save as JSON instead of joblib
        import json
        with open(os.path.join(path, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        print(f"✅ Model info saved to {os.path.join(path, 'model_info.json')}")
        
        # Save preprocessing pipeline
        if self.char_labels:
            self.char_labels.write().overwrite().save(os.path.join(path, 'char_labels'))
            print(f"✅ Character labels saved")
        if self.pipeline_model:
            self.pipeline_model.write().overwrite().save(os.path.join(path, 'pipeline_model'))
            print(f"✅ Pipeline model saved")
        
        # Save best model using the correct path format
        if self.best_model and hasattr(self, 'model_metrics') and self.best_model_type:
            # Determine which version was selected (default or tuned)
            champion_selection_info = self.model_metrics[self.best_model_type].get('selection_info', {})
            champion_decision = champion_selection_info.get('decision', 'default')
            
            model_path = os.path.join(path, f'{self.best_model_type}_{champion_decision}_model')
            self.best_model.write().overwrite().save(model_path)
            print(f"✅ Best model ({self.best_model_type} - {champion_decision} version) saved")
        else:
            print(f"⚠️ Best model not available for saving")
        
        print(f"🎉 Complete regression model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a fitted model and artifacts.
        
        Args:
            path: Directory path containing the saved model
        """
        # Load model info
        import json
        model_info_path = os.path.join(path, 'model_info.json')
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"Model info not found at {model_info_path}")
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Restore configuration and variables
        self.config = model_info['config']
        self.feature_vars = model_info['feature_vars']
        self.selected_vars = model_info['selected_vars']
        self.categorical_vars = model_info['categorical_vars']
        self.numerical_vars = model_info['numerical_vars']
        self.best_model_type = model_info['best_model_type']
        self.model_metrics = model_info['model_metrics']
        
        # Load preprocessing pipeline
        try:
            self.char_labels = PipelineModel.load(os.path.join(path, 'char_labels'))
        except:
            self.char_labels = None
        
        try:
            self.pipeline_model = PipelineModel.load(os.path.join(path, 'pipeline_model'))
        except:
            self.pipeline_model = None
        
        # Load best model
        if self.best_model_type:
            model_path = os.path.join(path, f'{self.best_model_type}_model')
            self.best_model = self.model_builder.load_model(self.best_model_type, model_path)
        
        print(f"✅ Regression model loaded from {path}")
    
    def get_feature_importance(self, model_type: Optional[str] = None) -> Optional[Any]:
        """
        Get feature importance from the feature selection phase.
        
        Args:
            model_type: Model type (not used for feature selection importance)
            
        Returns:
            DataFrame with feature importance or None if not available
        """
        print("📊 Feature Importance for Regression Model")
        print("=" * 50)
        print("Feature importance was calculated during feature selection using Random Forest Regression")
        print(f"💾 Feature importance files should be in: {self.output_dir}")
        
        # Look for feature importance files
        importance_files_found = []
        
        try:
            import pandas as pd
            
            # Look for feature importance Excel file in output directory
            excel_patterns = [
                os.path.join(self.output_dir, "feature_importance.xlsx"),
                os.path.join(self.output_dir, "*feature_importance*.xlsx")
            ]
            
            excel_file = None
            for pattern in excel_patterns:
                if '*' in pattern:
                    import glob
                    matches = glob.glob(pattern)
                    if matches:
                        excel_file = matches[0]
                        break
                else:
                    if os.path.exists(pattern):
                        excel_file = pattern
                        break
            
            # Look for feature importance plot
            plot_patterns = [
                os.path.join(self.output_dir, "Features_selected_for_modeling.png"),
                os.path.join(self.output_dir, "*Features_selected*.png")
            ]
            
            plot_file = None
            for pattern in plot_patterns:
                if '*' in pattern:
                    import glob
                    matches = glob.glob(pattern)
                    if matches:
                        plot_file = matches[0]
                        break
                else:
                    if os.path.exists(pattern):
                        plot_file = pattern
                        break
            
            # Report findings
            if excel_file:
                importance_files_found.append(f"📊 Excel file: {excel_file}")
                
                # Load and display feature importance
                feature_importance = pd.read_excel(excel_file)
                print(f"✅ Feature importance Excel file found and loaded: {excel_file}")
                
                # Display top 10 features
                if len(feature_importance) > 0:
                    print(f"\n🎯 Top 10 Most Important Features:")
                    print("-" * 40)
                    
                    # Find importance score column
                    score_cols = [col for col in feature_importance.columns if 'importance' in col.lower() or 'score' in col.lower()]
                    name_cols = [col for col in feature_importance.columns if 'name' in col.lower() or 'feature' in col.lower()]
                    
                    if score_cols and name_cols:
                        score_col = score_cols[0]
                        name_col = name_cols[0]
                        
                        top_10 = feature_importance.nlargest(10, score_col)
                        for i, (_, row) in enumerate(top_10.iterrows(), 1):
                            print(f"{i:2d}. {row[name_col]:<25} | {row[score_col]:.6f}")
                    else:
                        print("Could not identify feature names and importance scores in the file")
                        
            else:
                print(f"⚠️ Feature importance Excel file not found in: {self.output_dir}")
                feature_importance = None
            
            if plot_file:
                importance_files_found.append(f"📈 Plot file: {plot_file}")
                print(f"✅ Feature importance plot found: {plot_file}")
            else:
                print(f"⚠️ Feature importance plot not found in: {self.output_dir}")
            
            # Summary
            if importance_files_found:
                print(f"\n📋 Feature Importance Files Found:")
                for file_info in importance_files_found:
                    print(f"   {file_info}")
                    
                print(f"\n💡 Tip: You can use these files to understand which features")
                print(f"    contributed most to your regression model's predictions.")
                
                return feature_importance if excel_file else None
            else:
                print(f"\n❌ No feature importance files found in: {self.output_dir}")
                print(f"💡 Feature importance is generated during the feature selection phase.")
                print(f"    Make sure the model training completed successfully.")
                return None
                
        except ImportError:
            print(f"⚠️ Pandas not available - cannot load feature importance Excel file")
            return None
        except Exception as e:
            print(f"❌ Error loading feature importance: {e}")
            return None
    
    def _select_best_model(self):
        """Select Champion/Challenger models using stability analysis."""
        print("🏆 Analyzing model performance for Champion/Challenger selection...")
        
        if not self.model_metrics:
            raise ValueError("No trained models found for selection")
        
        # Get dataset names for stability analysis
        dataset_names = []
        for metrics in self.model_metrics.values():
            for key in metrics.keys():
                if '_' in key:
                    dataset = key.split('_')[-1]
                    if dataset not in dataset_names and dataset in ['train', 'valid', 'test', 'oot1', 'oot2']:
                        dataset_names.append(dataset)
        
        print(f"📊 Available datasets for stability analysis: {dataset_names}")
        
        # Use model selector for Champion/Challenger selection
        selection_result = self.model_selector.select_best_model(
            self.model_metrics,
            selection_criteria='rmse',  # Primary metric for regression
            dataset_to_use='valid',     # Prefer validation for selection
            dataset_names=dataset_names
        )
        
        self.best_model_type = selection_result['model_type']
        champion_score = selection_result['performance_score']
        stability_score = selection_result['stability_score']
        
        print(f"🥇 Champion model selected: {self.best_model_type}")
        print(f"📊 Champion RMSE: {champion_score:.4f}")
        print(f"🎯 Stability score: {stability_score}")
        
        # Get Challenger info if available
        all_results = selection_result['all_results']
        if len(all_results) > 1:
            challenger = all_results.iloc[1]
            challenger_name = challenger['model_type']
            challenger_score = challenger[f"{selection_result['dataset_used']}_rmse"]
            challenger_stability = challenger['stability_score']
            print(f"🥈 Challenger model: {challenger_name} (RMSE: {challenger_score:.4f}, stability: {challenger_stability})")
        
        # Load the Champion model from the appropriate path (default or tuned)
        # Determine which version was selected (from model metrics selection_info)
        champion_selection_info = self.model_metrics[self.best_model_type].get('selection_info', {})
        champion_decision = champion_selection_info.get('decision', 'default')
        
        model_path = os.path.join(self.output_dir, f'{self.best_model_type}_{champion_decision}_model')
        try:
            self.best_model = self.model_builder.load_model(self.best_model_type, model_path)
            print(f"✅ Champion model loaded successfully: {self.best_model_type} ({champion_decision} version)")
        except Exception as e:
            print(f"⚠️ Could not reload {self.best_model_type} model: {str(e)[:100]}...")
            print(f"💡 Model training was successful - issue is only with reloading saved model")
            print(f"💡 All performance metrics and Champion/Challenger selection results are still valid")
            self.best_model = None
    
    def _generate_scoring_code(self):
        """Generate production scoring code."""
        if self.best_model_type is None:
            print("⚠️ No best model selected, skipping scoring code generation")
            return
            
        self.score_generator.generate_scoring_code(
            self.config, self.feature_vars, self.selected_vars,
            self.categorical_vars, self.numerical_vars, self.best_model_type
        )
    
    def get_model_selection_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of Champion/Challenger model selection results.
        
        Returns:
            Dictionary with detailed model selection summary
        """
        if not self.model_metrics:
            return {'error': 'No models trained yet'}
        
        # Get Champion/Challenger selection results from model selector
        try:
            # Get dataset names
            dataset_names = []
            for metrics in self.model_metrics.values():
                for key in metrics.keys():
                    if '_' in key:
                        dataset = key.split('_')[-1]
                        if dataset not in dataset_names and dataset in ['train', 'valid', 'test', 'oot1', 'oot2']:
                            dataset_names.append(dataset)
            
            # Get selection results
            selection_result = self.model_selector.select_best_model(
                self.model_metrics,
                selection_criteria='rmse',
                dataset_to_use='valid',
                dataset_names=dataset_names
            )
            
            all_results = selection_result['all_results']
            
            # Extract Champion and Challenger
            champion_model = all_results.iloc[0]['model_type']
            challenger_model = all_results.iloc[1]['model_type'] if len(all_results) > 1 else None
            
        except Exception as e:
            print(f"⚠️ Error getting Champion/Challenger selection: {e}")
            # Fallback to simple ranking
            champion_model = self.best_model_type
            challenger_model = None
            all_results = None
        
        summary = {
            'total_models_trained': len(self.model_metrics),
            'champion_model': champion_model,
            'challenger_model': challenger_model,
            'selection_criteria': 'RMSE with stability analysis (lower RMSE and stability score is better)',
            'model_performance': {},
            'ranking': []
        }
        
        # Calculate performance ranking with Champion/Challenger roles
        performance_data = []
        for model_type, metrics in self.model_metrics.items():
            # Get validation RMSE or fall back to train RMSE
            rmse = metrics.get('valid_rmse', metrics.get('train_rmse', float('inf')))
            r2 = metrics.get('valid_r2', metrics.get('train_r2', 0.0))
            mae = metrics.get('valid_mae', metrics.get('train_mae', float('inf')))
            
            # Get stability score if available
            stability_score = 0
            if all_results is not None:
                try:
                    model_row = all_results[all_results['model_type'] == model_type]
                    if not model_row.empty:
                        stability_score = int(model_row.iloc[0]['stability_score'])
                except:
                    stability_score = 0
            
            # Determine role
            if model_type == champion_model:
                role = "Champion"
                is_champion = True
            elif model_type == challenger_model:
                role = "Challenger"
                is_champion = False
            else:
                role = "Participant"
                is_champion = False
            
            performance_data.append({
                'model_type': model_type,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'stability_score': stability_score,
                'role': role,
                'is_champion': is_champion
            })
            
            summary['model_performance'][model_type] = {
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'stability_score': stability_score,
                'role': role,
                'rank': None  # Will be filled below
            }
        
        # Sort by Champion/Challenger order, then by RMSE
        performance_data.sort(key=lambda x: (
            0 if x['role'] == 'Champion' else 1 if x['role'] == 'Challenger' else 2,
            x['rmse']
        ))
        
        # Add ranking
        for rank, model_data in enumerate(performance_data, 1):
            model_type = model_data['model_type']
            summary['model_performance'][model_type]['rank'] = rank
            summary['ranking'].append({
                'rank': rank,
                'model_type': model_type,
                'rmse': model_data['rmse'],
                'r2': model_data['r2'],
                'mae': model_data['mae'],
                'stability_score': model_data['stability_score'],
                'role': model_data['role'],
                'is_champion': model_data['is_champion']
            })
        
        return summary
    
    def print_model_selection_summary(self):
        """Print a comprehensive Champion/Challenger model selection summary and save to file."""
        summary = self.get_model_selection_summary()
        
        if 'error' in summary:
            print(f"⚠️ {summary['error']}")
            return
        
        # Format the summary content
        lines = []
        lines.append("="*80)
        lines.append("🏆 CHAMPION/CHALLENGER MODEL SELECTION SUMMARY")
        lines.append("="*80)
        lines.append("")
        
        lines.append("📊 Overall Results:")
        lines.append(f"   • Total models trained: {summary['total_models_trained']}")
        lines.append(f"   • Champion model: {summary['champion_model']}")
        lines.append(f"   • Challenger model: {summary.get('challenger_model', 'N/A')}")
        lines.append(f"   • Selection criteria: {summary['selection_criteria']}")
        lines.append("")
        
        lines.append("🏆 Champion/Challenger Performance:")
        lines.append("   Role      | Model Type        | RMSE     | R²      | MAE     | Stability")
        lines.append("   ----------|-------------------|----------|---------|---------|----------")
        
        for model_info in summary['ranking']:
            role = model_info.get('role', f"Rank {model_info['rank']}")
            model_type = model_info['model_type']
            rmse = model_info['rmse']
            r2 = model_info['r2']
            mae = model_info['mae']
            stability = model_info.get('stability_score', 0)
            
            icon = "🥇" if role == "Champion" else "🥈" if role == "Challenger" else ""
            role_display = f"{icon} {role}"
            
            lines.append(f"   {role_display:9s} | {model_type:17s} | {rmse:8.4f} | {r2:7.4f} | {mae:7.4f} | {stability:9d}")
        
        lines.append("")
        lines.append("📈 Detailed Model Analysis:")
        
        for model_type, performance in summary['model_performance'].items():
            if model_type == summary['champion_model']:
                icon = "🥇"
                title = "CHAMPION"
            elif model_type == summary.get('challenger_model'):
                icon = "🥈"
                title = "CHALLENGER"
            else:
                icon = "📊"
                title = "PARTICIPANT"
            
            lines.append(f"\n   {icon} {model_type.upper()} ({title}):")
            lines.append(f"      📊 RMSE: {performance['rmse']:.4f}")
            lines.append(f"      📈 R²: {performance['r2']:.4f}")
            lines.append(f"      📉 MAE: {performance['mae']:.4f}")
            lines.append(f"      🎯 Rank: #{performance['rank']}")
            lines.append(f"      🏅 Stability Score: {performance.get('stability_score', 0)}")
        
        lines.append("")
        lines.append("="*80)
        lines.append("💡 INTERPRETATION GUIDE:")
        lines.append("   🥇 Champion: Best performing model with highest stability")
        lines.append("   🥈 Challenger: Second best model for comparison and validation")
        lines.append("   📊 RMSE: Root Mean Square Error (lower is better)")
        lines.append("   📈 R²: Coefficient of Determination (higher is better, max = 1.0)")
        lines.append("   📉 MAE: Mean Absolute Error (lower is better)")
        lines.append("   🏅 Stability Score: Number of datasets with significant performance deviation")
        lines.append("   🎯 Lower stability scores indicate more consistent performance across datasets")
        lines.append("="*80)
        
        # Print to console
        for line in lines:
            print(line)
        
        # Save to file
        try:
            summary_file = os.path.join(self.output_dir, 'champion_challenger_summary.txt')
            with open(summary_file, 'w') as f:
                f.write('\n'.join(lines))
            print(f"\n💾 Champion/Challenger summary saved to: {summary_file}")
            
            # Also save detailed metrics as JSON
            metrics_file = os.path.join(self.output_dir, 'champion_challenger_results.json')
            import json
            with open(metrics_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"💾 Detailed results saved to: {metrics_file}")
            
        except Exception as e:
            print(f"⚠️ Could not save summary files: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary information
        """
        if self.best_model is None and self.best_model_type is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summary = {
            'model_id': getattr(self, 'actual_model_id', 'regression_model'),
            'model_literal': getattr(self, 'actual_model_literal', 'automl_regression'),
            'best_model_type': self.best_model_type,
            'feature_count': len(self.selected_vars) if self.selected_vars else 0,
            'categorical_features': len(self.categorical_vars) if self.categorical_vars else 0,
            'numerical_features': len(self.numerical_vars) if self.numerical_vars else 0,
            'model_metrics': self.model_metrics,
            'output_directory': self.output_dir,
            'selection_summary': self.get_model_selection_summary()
        }
        
        return summary 

    def _build_model_with_dual_training(self, train_data: DataFrame, target_column: str, 
                                       model_type: str, top_features: List[str],
                                       datasets: List[DataFrame], dataset_names: List[str]) -> Dict[str, Any]:
        """
        Build both default and tuned models, then select the better one.
        
        Returns:
            Dictionary with selected model, metrics, and selection reasoning
        """
        print(f"\n🔄 Training both default and tuned {model_type} models...")
        
        # Build default model
        print(f"  📊 Training default {model_type} model...")
        default_model = self.model_builder.build_model(
            train_data, 'features', target_column, model_type,
            num_features=len(top_features)
        )
        
        # Validate default model
        default_metrics = self.model_validator.validate_model(
            default_model, datasets, dataset_names, target_column, f"{model_type}_default", self.output_dir
        )
        
        # Build tuned model (only if hyperparameter optimization is enabled)
        tuned_model = None
        tuned_metrics = None
        comparison_result = None
        
        if self.config.get('enable_hyperparameter_tuning', False):
            print(f"  🎯 Training hyperparameter-tuned {model_type} model...")
            
            # Optimize hyperparameters
            optimization_results = self._optimize_hyperparameters(
                train_data, target_column, model_type, len(top_features)
            )
            
            best_params = {}
            if optimization_results and 'best_params' in optimization_results:
                best_params = optimization_results['best_params']
                print(f"    🔧 Optimized parameters: {best_params}")
                print(f"    📊 Optimization score: {optimization_results.get('best_score', 'N/A')}")
                print(f"    🔄 Number of trials: {optimization_results.get('n_trials', 'N/A')}")
            else:
                print(f"    ⚠️ No optimization results returned. Using default parameters.")
                print(f"    🔍 Optimization results: {optimization_results}")
            
            # Build tuned model
            tuned_model = self.model_builder.build_model(
                train_data, 'features', target_column, model_type,
                num_features=len(top_features),
                **best_params
            )
            
            # Validate tuned model
            tuned_metrics = self.model_validator.validate_model(
                tuned_model, datasets, dataset_names, target_column, f"{model_type}_tuned", self.output_dir
            )
            
            # Compare models and select the better one
            comparison_result = self._compare_models(default_metrics, tuned_metrics, model_type)
            
            # Log detailed comparison
            self._log_model_comparison(model_type, comparison_result)
            
        else:
            print(f"  📝 Hyperparameter optimization disabled - using default {model_type} model")
            comparison_result = {
                'decision': 'default',
                'reasons': ['Hyperparameter optimization is disabled'],
                'default_score': self._extract_score(default_metrics, 'rmse')
            }
        
        # Select the final model and metrics
        if comparison_result['decision'] == 'tuned' and tuned_model is not None:
            selected_model = tuned_model
            selected_metrics = tuned_metrics
            selected_type = f"{model_type}_tuned"
        else:
            selected_model = default_model
            selected_metrics = default_metrics
            selected_type = f"{model_type}_default"
        
        # Save both models separately (for transparency and comparison)
        try:
            default_model_path = os.path.join(self.output_dir, f'{model_type}_default_model')
            self.model_builder.save_model(default_model, default_model_path)
            print(f"   💾 Default {model_type} model saved to {default_model_path}")
        except Exception as e:
            print(f"   ⚠️ Could not save default {model_type} model: {e}")
        
        if tuned_model is not None:
            try:
                tuned_model_path = os.path.join(self.output_dir, f'{model_type}_tuned_model')
                self.model_builder.save_model(tuned_model, tuned_model_path)
                print(f"   💾 Tuned {model_type} model saved to {tuned_model_path}")
            except Exception as e:
                print(f"   ⚠️ Could not save tuned {model_type} model: {e}")
        
        return {
            'model': selected_model,
            'metrics': selected_metrics,
            'model_type': selected_type,
            'comparison': comparison_result,
            'default_metrics': default_metrics,
            'tuned_metrics': tuned_metrics,
            'default_model': default_model,
            'tuned_model': tuned_model
        }
    
    def _optimize_hyperparameters(self, train_data: DataFrame, target_column: str, 
                                 model_type: str, feature_count: int) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a given model type.
        
        Args:
            train_data: Training DataFrame
            target_column: Name of the target column
            model_type: Type of model to optimize
            feature_count: Number of features
            
        Returns:
            Dictionary containing the best parameters and metrics
        """
        # Apply additional Spark optimizations if this is the first hyperparameter tuning
        if not hasattr(self, '_tuning_optimizations_applied'):
            try:
                from spark_optimization_config import apply_tuning_optimizations
                print("🎯 Applying enhanced Spark optimizations for hyperparameter tuning session...")
                
                # Use aggressive optimizations for tree-based models
                aggressive = model_type in ['gradient_boosting', 'random_forest', 'xgboost', 'lightgbm']
                apply_tuning_optimizations(self.spark, aggressive=aggressive)
                
                self._tuning_optimizations_applied = True
                print("✅ Enhanced optimizations applied - large task binary warnings should be significantly reduced")
                
            except Exception as e:
                print(f"⚠️  Could not apply enhanced tuning optimizations: {e}")
        
        # Use the hyperparameter tuner for optimization
        return self.hyperparameter_tuner.tune_hyperparameters(
            model_type, train_data, target_column, self.config
        )
    
    def _compare_models(self, default_metrics: Dict, tuned_metrics: Dict, model_type: str) -> Dict[str, Any]:
        """
        Intelligently compare default vs hyperparameter-tuned models.
        
        Args:
            default_metrics: Performance metrics for default model
            tuned_metrics: Performance metrics for tuned model
            model_type: Type of model being compared
            
        Returns:
            Dict with selection decision and reasoning
        """
        # Selection criteria from configuration
        improvement_threshold = self.config.get('improvement_threshold', 0.05)  # 5% improvement for regression
        overfitting_threshold = self.config.get('overfitting_threshold', 0.1)   # 10% overfitting threshold
        
        # Primary metric for regression is RMSE (lower is better)
        primary_metric = 'rmse'
        
        # Extract metrics for comparison
        default_score = self._extract_score(default_metrics, primary_metric)
        tuned_score = self._extract_score(tuned_metrics, primary_metric)
        
        # Calculate improvement (for RMSE, lower is better)
        improvement = default_score - tuned_score  # Positive = improvement
        improvement_pct = (improvement / default_score) * 100 if default_score != 0 else 0
        
        is_better = tuned_score < default_score  # Lower RMSE is better
        
        # Check for overfitting (train vs validation performance gap)
        default_overfitting = self._check_overfitting(default_metrics, primary_metric)
        tuned_overfitting = self._check_overfitting(tuned_metrics, primary_metric)
        
        # Decision logic
        reasons = []
        decision = "default"
        
        if not is_better:
            reasons.append(f"Tuned model RMSE ({tuned_score:.4f}) is not better than default ({default_score:.4f})")
            decision = "default"
        elif improvement_pct < (improvement_threshold * 100):
            reasons.append(f"Performance improvement ({improvement_pct:.2f}%) is below threshold ({improvement_threshold*100}%)")
            decision = "default"
        elif tuned_overfitting > default_overfitting + overfitting_threshold:
            reasons.append(f"Tuned model shows more overfitting (gap: {tuned_overfitting:.3f} vs {default_overfitting:.3f})")
            decision = "default"
        else:
            reasons.append(f"Tuned model shows significant improvement ({improvement_pct:.2f}% better RMSE)")
            if tuned_overfitting <= default_overfitting:
                reasons.append("Tuned model generalizes as well or better than default")
            decision = "tuned"
        
        return {
            'decision': decision,
            'default_score': default_score,
            'tuned_score': tuned_score,
            'improvement_pct': improvement_pct,
            'default_overfitting': default_overfitting,
            'tuned_overfitting': tuned_overfitting,
            'reasons': reasons,
            'primary_metric': 'RMSE'
        }
    
    def _extract_score(self, metrics: Dict, metric_name: str) -> float:
        """Extract primary metric score from metrics dictionary."""
        # Prefer validation, fall back to train
        for dataset in ['valid', 'train', 'test', 'oot1', 'oot2']:
            metric_key = f"{dataset}_{metric_name}"
            if metric_key in metrics:
                return metrics[metric_key]
        
        # If no dataset-specific metric found, try direct key
        if metric_name in metrics:
            return metrics[metric_name]
        
        # Return worst possible score for RMSE
        return float('inf')
    
    def _check_overfitting(self, metrics: Dict, metric_name: str) -> float:
        """Check for overfitting by comparing train vs validation performance."""
        if 'train' not in [key.split('_')[-1] for key in metrics.keys()] or \
           'valid' not in [key.split('_')[-1] for key in metrics.keys()]:
            return 0.0
        
        train_score = self._extract_score({'train_' + metric_name: metrics.get('train_' + metric_name, float('inf'))}, metric_name)
        valid_score = self._extract_score({'valid_' + metric_name: metrics.get('valid_' + metric_name, float('inf'))}, metric_name)
        
        if train_score == 0 or train_score == float('inf'):
            return 0.0
        
        # For RMSE (lower is better), overfitting = (valid_rmse - train_rmse) / train_rmse
        return (valid_score - train_score) / train_score
    
    def _prepare_oot_datasets(self, oot1_data: Optional[DataFrame], 
                             oot2_data: Optional[DataFrame], 
                             target_column: str) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Prepare out-of-time datasets using the fitted preprocessing pipeline."""
        oot1_scaled = None
        oot2_scaled = None
        
        if oot1_data is not None:
            print("🔄 Processing OOT1 data...")
            try:
                # Apply the same preprocessing pipeline that was fitted on training data
                oot1_processed = self._apply_preprocessing(oot1_data)
                print(f"✅ OOT1 preprocessing completed. Columns: {oot1_processed.columns}")
                
                # Apply scaling using the fitted pipeline
                oot1_scaled = self.data_processor.apply_scaling(oot1_processed, target_column)
                print(f"✅ OOT1 scaling completed. Final columns: {oot1_scaled.columns}")
                
            except Exception as e:
                print(f"❌ Error processing OOT1 data: {str(e)}")
                print(f"Available columns in OOT1: {oot1_data.columns}")
                print(f"Expected feature columns: {self.feature_vars}")
                print(f"Expected categorical columns: {self.categorical_vars}")
                print(f"Expected numerical columns: {self.numerical_vars}")
                print("⚠️ OOT1 data will be skipped due to preprocessing errors")
                oot1_scaled = None
        
        if oot2_data is not None:
            print("🔄 Processing OOT2 data...")
            try:
                # Apply the same preprocessing pipeline that was fitted on training data
                oot2_processed = self._apply_preprocessing(oot2_data)
                print(f"✅ OOT2 preprocessing completed. Columns: {oot2_processed.columns}")
                
                # Apply scaling using the fitted pipeline
                oot2_scaled = self.data_processor.apply_scaling(oot2_processed, target_column)
                print(f"✅ OOT2 scaling completed. Final columns: {oot2_scaled.columns}")
                
            except Exception as e:
                print(f"❌ Error processing OOT2 data: {str(e)}")
                print(f"Available columns in OOT2: {oot2_data.columns}")
                print(f"Expected feature columns: {self.feature_vars}")
                print(f"Expected categorical columns: {self.categorical_vars}")
                print(f"Expected numerical columns: {self.numerical_vars}")
                print("⚠️ OOT2 data will be skipped due to preprocessing errors")
                oot2_scaled = None
        
        return oot1_scaled, oot2_scaled
    
    def _log_model_comparison(self, model_type: str, comparison: Dict[str, Any]):
        """Log detailed model comparison results."""
        print(f"\n🤔 Model Selection Analysis for {model_type.upper()}:")
        print(f"   📈 Default model {comparison['primary_metric']}: {comparison['default_score']:.4f}")
        
        if 'tuned_score' in comparison:
            print(f"   🎯 Tuned model {comparison['primary_metric']}: {comparison['tuned_score']:.4f}")
            print(f"   📊 Performance improvement: {comparison['improvement_pct']:.2f}%")
            print(f"   🎪 Default overfitting gap: {comparison['default_overfitting']:.3f}")
            print(f"   🎪 Tuned overfitting gap: {comparison['tuned_overfitting']:.3f}")
        
        print(f"   ✅ DECISION: Using {comparison['decision'].upper()} model")
        print(f"   💭 Reasoning:")
        for reason in comparison['reasons']:
            print(f"      • {reason}") 
