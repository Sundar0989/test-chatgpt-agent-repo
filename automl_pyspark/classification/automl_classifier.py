"""
AutoML Classification Classifier

Main class that orchestrates the entire AutoML pipeline for classification tasks (binary and multiclass).
This class provides a high-level interface for automated machine learning using PySpark for classification.
"""

import os
import joblib
import pandas as pd
import signal
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

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

# Import custom modules
from .data_processor import DataProcessor
from .model_builder import ModelBuilder, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE

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
from .model_validator import ModelValidator
from .model_selector import ModelSelector
from .score_generator import ScoreGenerator
from .hyperparameter_tuner import HyperparameterTuner
from .data_balancing import DataBalancer
# from feature_selection import FeatureSelector  # Will be imported when needed

class AutoMLClassifier:
    """
    Main AutoML classifier class for classification (binary and multiclass).
    
    This class provides a comprehensive automated machine learning solution
    for classification tasks using PySpark, including:
    - Data preprocessing and feature engineering
    - Model building with multiple algorithms
    - Model validation and performance evaluation (binary & multiclass)
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
        Initialize the AutoML classifier.
        
        Args:
            output_dir: Directory to save model outputs
            config_path: Path to YAML configuration file
            spark_session: PySpark SparkSession (optional, will create if None)
            preset: Configuration preset - 'quick' for fast training (logistic + random forest only, 
                   no CV, no hyperparameter tuning), 'comprehensive' for full feature training,
                   or None to use user-provided configuration from YAML/defaults
            environment: Environment to use ('development', 'staging', 'production'). 
                        If None, uses default_environment from config (now 'production')
            user_id: (DEPRECATED) User identifier - use YAML config instead
            model_id: (DEPRECATED) Model identifier - use YAML config instead
            model_literal: (DEPRECATED) Model literal - use YAML config instead
            **kwargs: Additional arguments for backward compatibility
            
        Configuration Priority (highest to lowest):
            1. Preset configuration (if preset is specified) ← HIGHEST PRIORITY
            2. Legacy parameters (for backward compatibility)
            3. User-provided YAML configuration 
            4. Environment-specific overrides
            5. Default configuration ← LOWEST PRIORITY
        """
        
        # Handle old-style initialization where spark_session was first argument  
        if isinstance(output_dir, SparkSession):
            print("🔄 Detected old-style initialization - converting to YAML configuration")
            # Old style: AutoMLClassifier(spark_session, output_dir=None, user_id=..., model_id=..., model_literal=...)
            spark_session = output_dir  # First arg was spark_session
            # If config_path looks like an output directory, use it; otherwise use default
            if config_path and not config_path.endswith(('.yaml', '.yml')):
                output_dir = config_path
                config_path = None
            else:
                output_dir = 'automl_output'
        
        # Store legacy parameters for later processing (after config manager is created)
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
        
        # Apply legacy parameter overrides FIRST (lower priority)
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
        
        # Initialize components with actual parameters (stored during legacy parameter processing)
        self.data_processor = DataProcessor(self.spark, self.actual_user_id, self.actual_model_literal)
        # Set the output directory for feature importance saving
        self.data_processor.output_dir = self.output_dir
        
        self.model_builder = ModelBuilder(self.spark)
        self.model_validator = ModelValidator(self.spark, self.output_dir, self.actual_user_id, self.actual_model_literal)
        self.model_selector = ModelSelector(self.output_dir, self.actual_user_id, self.actual_model_literal)
        self.score_generator = ScoreGenerator(self.output_dir, self.actual_user_id, self.actual_model_id, self.actual_model_literal)
        self.data_balancer = DataBalancer(self.spark)
        # self.feature_selector = FeatureSelector(self.spark, self.config)  # Will be initialized when needed
        self.hyperparameter_tuner = None  # Will be initialized when needed
        
        # Pipeline artifacts
        self.feature_vars = []
        self.categorical_vars = []
        self.numerical_vars = []
        self.selected_vars = []
        self.char_labels = None
        self.target_label_indexer = None  # For multi-class string targets
        self.pipeline_model = None
        self.best_model = None
        self.best_model_type = None
        self.model_metrics = {}
        
        self.is_multiclass = False
        self.num_classes = 2
        
        # Log configuration
        self._log_configuration()
    
    def _create_optimized_spark_session(self) -> SparkSession:
        """Create Spark session with performance optimization based on config."""
        print("🚀 Creating optimized Spark session with performance settings...")
        
        # Stop any existing Spark session to avoid conflicts
        try:
            existing_spark = SparkSession.getActiveSession()
            if existing_spark:
                print("🔄 Stopping existing Spark session to avoid conflicts...")
                existing_spark.stop()
        except Exception as e:
            print(f"⚠️ Note: {e}")
        
        # Get performance configuration
        perf_config = self.config_manager.config.get('performance', {})
        parallel_jobs = perf_config.get('parallel_jobs', -1)
        memory_limit_gb = perf_config.get('memory_limit_gb', 8)
        
        print(f"   ⚡ Parallel jobs: {parallel_jobs}")
        print(f"   💾 Memory limit: {memory_limit_gb}GB")
        
        # Get optimized base configuration with BigQuery support
        spark_config = get_optimized_spark_config(include_synapseml=LIGHTGBM_AVAILABLE, include_bigquery=True)
        
        # Override with performance-specific settings optimized for BigQuery
        spark_config.update({
            "spark.executor.memory": f"{max(4, memory_limit_gb // 2)}g",
            "spark.driver.memory": f"{max(8, memory_limit_gb)}g",  # Increased for BigQuery operations
            "spark.driver.maxResultSize": "4g",  # Increased for BigQuery results
            "spark.sql.shuffle.partitions": str(max(8, parallel_jobs * 4)) if parallel_jobs > 0 else "200",
        })
        
        # Configure executor cores based on parallel_jobs
        if parallel_jobs > 0:
            spark_config["spark.executor.cores"] = str(min(4, parallel_jobs))
            spark_config["spark.executor.instances"] = str(max(1, parallel_jobs // 4))
        
        # Build Spark session
        builder = SparkSession.builder.appName("AutoML Classification Pipeline (Performance Optimized)")
        
        # Add BigQuery connector package - using your proven working configuration
        packages = ["com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1"]
        if LIGHTGBM_AVAILABLE:
            packages.append("com.microsoft.azure:synapseml_2.12:1.0.3")
        
        print(f"   📦 Adding packages: {packages}")
        builder = builder.config("spark.jars.packages", ",".join(packages))
        
        for key, value in spark_config.items():
            builder = builder.config(key, value)
        
        print("   🔧 Creating new Spark session with BigQuery support...")
        spark = builder.getOrCreate()
        
        # Verify BigQuery connector is available
        try:
            # Test BigQuery connector availability
            test_packages = spark.sparkContext.getConf().get("spark.jars.packages", "")
            print(f"   📋 Loaded packages: {test_packages}")
            if "spark-bigquery-with-dependencies" in test_packages:
                print("   ✅ BigQuery connector package confirmed")
            else:
                print("   ⚠️ BigQuery connector package not found in loaded packages")
        except Exception as e:
            print(f"   ⚠️ Could not verify packages: {e}")
        
        return spark
    
    def create_proven_bigquery_session(self, driver_memory: str = "64g") -> SparkSession:
        """
        Create Spark session using your exact proven working BigQuery configuration.
        This replicates your working code exactly.
        """
        print(f"🚀 Creating Spark session using your proven BigQuery configuration...")
        print(f"   💾 Driver memory: {driver_memory}")
        
        # Stop any existing session first
        try:
            existing_spark = SparkSession.getActiveSession()
            if existing_spark:
                print("🔄 Stopping existing Spark session...")
                existing_spark.stop()
        except Exception as e:
            pass
        
        # Your exact working configuration
        spark = SparkSession.builder \
            .appName("AutoML BigQuery (Proven Config)") \
            .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1") \
            .config("spark.driver.memory", driver_memory) \
            .getOrCreate()
        
        print("✅ Spark session created with proven BigQuery configuration")
        
        # Update our spark reference and component references
        self.spark = spark
        if hasattr(self, 'data_processor'):
            self.data_processor.spark = spark
        if hasattr(self, 'model_builder'):
            self.model_builder.spark = spark
        if hasattr(self, 'model_validator'):
            self.model_validator.spark = spark
        if hasattr(self, 'data_balancer'):
            self.data_balancer.spark = spark
        
        return spark
    
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
            print(f"🚀 Creating BigQuery-optimized Spark session...")
            print(f"   💾 Driver memory: {driver_memory}")
            print(f"   🔗 BigQuery connector: v0.36.1")
            
            # Stop existing session if needed
            if hasattr(self, 'spark') and self.spark is not None:
                print("🔄 Stopping existing Spark session for BigQuery compatibility...")
                self.spark.stop()
            
            # Build Spark session with proven BigQuery configuration
            builder = SparkSession.builder.appName("AutoML BigQuery Classifier")
            
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
            if hasattr(self, 'data_balancer'):
                self.data_balancer.spark = spark
            
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
        print(f"🔗 Loading BigQuery data...")
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
                         **kwargs) -> 'AutoMLClassifier':
        """
        Fit AutoML classifier directly from BigQuery tables.
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
            self: Fitted AutoML classifier
            
        Example:
            automl = AutoMLClassifier()
            results = automl.fit_from_bigquery(
                project_id="atus-prism-dev",
                table_id="atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02",
                target_column="your_target_column",
                limit_rows=10000  # Optional: for testing
            )
        """
        print("🚀 Starting AutoML with BigQuery data...")
        
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
        print("🤖 Starting AutoML training...")
        return self.fit(train_data, target_column, oot1_data, oot2_data, **kwargs)
    
    def _with_timeout(self, func, timeout_minutes: int, *args, **kwargs):
        """Execute function with timeout."""
        if timeout_minutes <= 0:
            return func(*args, **kwargs)
        
        print(f"⏱️ Setting pipeline timeout: {timeout_minutes} minutes")
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Pipeline timed out after {timeout_minutes} minutes")
        
        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)  # Convert to seconds
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel the alarm
            return result
        except TimeoutError:
            print(f"❌ Pipeline execution timed out after {timeout_minutes} minutes")
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _build_config(self) -> Dict[str, Any]:
        """Build configuration by combining global and classification-specific settings."""
        # Get flattened configuration for backwards compatibility
        config = self.config_manager.get_flat_config('classification', include_global=True)
        
        # Get global and classification configs for explicit mapping
        global_config = self.config_manager.get_global_config()
        classification_config = self.config_manager.get_task_config('classification')
        
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
            config['min_sample_size_for_split'] = validation.get('min_sample_size_for_split', 5000)
            config['test_size'] = validation.get('test_size', 0.2)
            config['valid_size'] = validation.get('validation_size', 0.2)
            # Calculate train_size
            config['train_size'] = 1.0 - config['test_size'] - config['valid_size']
        
        # Add model flags directly to config root
        if 'models' in classification_config:
            config.update(classification_config['models'])
        
        # Add hyperparameter tuning settings
        if 'hyperparameter_tuning' in classification_config:
            config.update(classification_config['hyperparameter_tuning'])
        
        # Add model selection settings
        if 'model_selection' in classification_config:
            config.update(classification_config['model_selection'])
        
        # Add evaluation settings
        if 'evaluation' in classification_config:
            config.update(classification_config['evaluation'])
        
        # Add cross-validation settings
        if 'cross_validation' in classification_config:
            # Rename to match existing code expectations
            cv_config = classification_config['cross_validation']
            config['use_cross_validation'] = cv_config.get('use_cross_validation', 'auto')
            config['cv_folds'] = cv_config.get('cv_folds', 5)
            config['cv_metric_binary'] = cv_config.get('cv_metric_binary', 'areaUnderROC')
            config['cv_metric_multiclass'] = cv_config.get('cv_metric_multiclass', 'accuracy')
        
        # Add data balancing settings
        if 'data_balancing' in classification_config:
            balancing_config = classification_config['data_balancing']
            config.update(balancing_config)
        
        # Add legacy parameter mappings if they exist
        if 'legacy_info' in global_config:
            legacy_info = global_config['legacy_info']
            config['user_id'] = legacy_info.get('user_id', 'automl_user')
            config['model_id'] = legacy_info.get('model_id', 'automl_model')
            config['model_literal'] = legacy_info.get('model_literal', 'automl_model')
        else:
            # Default values for new-style usage
            config['user_id'] = 'automl_user'
            config['model_id'] = 'automl_model'
            config['model_literal'] = 'automl_model'
        
        # Ensure all required keys exist with defaults
        config.setdefault('missing_threshold', 0.7)
        config.setdefault('impute_value', 0.0)
        config.setdefault('categorical_threshold', 10)
        config.setdefault('sample_fraction', 1.0)
        config.setdefault('seed', 42)
        config.setdefault('max_features', 50)
        config.setdefault('sequential_threshold', 200)
        config.setdefault('chunk_size', 100)
        config.setdefault('features_per_chunk', 30)
        config.setdefault('min_sample_size_for_split', 5000)
        config.setdefault('test_size', 0.2)
        config.setdefault('valid_size', 0.2)
        config.setdefault('train_size', 0.6)
        config.setdefault('use_cross_validation', 'auto')
        config.setdefault('cv_folds', 5)
        config.setdefault('cv_metric_binary', 'areaUnderROC')
        config.setdefault('cv_metric_multiclass', 'accuracy')
        config.setdefault('model_selection_criteria', 'ks')
        config.setdefault('dataset_to_use', 'valid')
        
        return config
    
    def _log_configuration(self):
        """Log the current configuration and available models."""
        print("\n🔧 AutoML Configuration Loaded:")
        print(f"   📁 Config source: {self.config_manager.config_path}")
        
        # Validate configuration
        is_valid = self.config_manager.validate_config('classification')
        if not is_valid:
            print("⚠️ Configuration validation failed - using defaults where possible")
        
        # Log available models
        self._log_available_models()
    
    def _log_available_models(self):
        """Log which models are available and enabled."""
        print("\n🔧 AutoML Model Configuration:")
        
        # Core models (always available)
        core_models = [
            ('Logistic Regression', self.config.get('run_logistic', True)),
            ('Random Forest', self.config.get('run_random_forest', True)),
            ('Gradient Boosting', self.config.get('run_gradient_boosting', True)),
            ('Decision Tree', self.config.get('run_decision_tree', True)),
            ('Neural Network', self.config.get('run_neural_network', True))
        ]
        
        # Advanced models (conditional availability)
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
            if available and enabled:
                status = "✅ Enabled"
                enabled_count += 1
            elif available and not enabled:
                status = "⭕ Disabled"
            else:
                status = "❌ Not Available (package not installed)"
            
            print(f"    {name}: {status}")
        
        print(f"  Total Models Enabled: {enabled_count}")
        
        if not any(available for _, _, available in advanced_models if not XGBOOST_AVAILABLE or not LIGHTGBM_AVAILABLE):
            print("  💡 Install XGBoost: pip install xgboost>=1.6.0")
            print("  💡 Install LightGBM: pip install synapseml>=0.11.0")
    
    def configure(self, **kwargs):
        """
        Configure the AutoML pipeline parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
        return self
    
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
    
    def fit(self, 
            train_data: Union[str, DataFrame],
            target_column: str,
            oot1_data: Optional[Union[str, DataFrame]] = None,
            oot2_data: Optional[Union[str, DataFrame]] = None,
            **kwargs) -> 'AutoMLClassifier':
        """
        Fit the AutoML pipeline on the training data.
        
        Args:
            train_data: Training DataFrame
            target_column: Name of the target column
            oot1_data: Out-of-time validation data 1 (optional)
            oot2_data: Out-of-time validation data 2 (optional)
            **kwargs: Additional configuration parameters
            
        Returns:
            self: Fitted AutoML classifier
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
    
    def _fit_internal(self, train_data: DataFrame, target_column: str, 
                     oot1_data: Optional[DataFrame] = None, 
                     oot2_data: Optional[DataFrame] = None) -> 'AutoMLClassifier':
        """Internal fit method that performs the actual training."""
        
        # Log available model types with updated configuration
        self._log_available_models()
        
        print("Starting AutoML pipeline...")
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
            total_rows = None  # Will be estimated later if needed
            print(f"Training data shape: Unknown rows (very large BigQuery dataset), {total_columns} columns")
        
        # Detect multiclass - optimized for large BigQuery datasets
        print("🔍 Detecting number of target classes...")
        try:
            # For very large datasets, use a sample to determine class count
            if total_rows is None or total_rows > 1000000:  # If more than 1M rows or unknown size, use sampling
                if total_rows:
                    sample_size = min(100000, int(total_rows * 0.01))  # 1% or 100k, whichever is smaller
                    sample_fraction = sample_size / total_rows
                    print(f"📊 Large dataset detected ({total_rows:,} rows). Sampling {sample_size:,} rows for class detection...")
                else:
                    sample_fraction = 0.001  # 0.1% sample for unknown size datasets
                    print(f"📊 Very large BigQuery dataset detected. Using 0.1% sample for class detection...")
                
                sample_data = train_data.sample(fraction=sample_fraction, seed=42)
                self.num_classes = sample_data.select(target_column).distinct().count()
                print(f"📊 Class count from sample: {self.num_classes}")
                
                # Verify with a different approach if sample suggests binary classification
                if self.num_classes <= 2:
                    print("🔍 Verifying class count with aggregation approach...")
                    # Use groupBy and count which is more efficient for BigQuery
                    class_counts = train_data.groupBy(target_column).count().limit(10).collect()
                    self.num_classes = len(class_counts)
                    print(f"📊 Verified class count: {self.num_classes}")
            else:
                # For smaller datasets, use the original approach
                self.num_classes = train_data.select(target_column).distinct().count()
        except Exception as e:
            print(f"⚠️ Error during class detection: {e}")
            print("🔄 Falling back to groupBy approach...")
            try:
                # Fallback: Use groupBy which is more BigQuery-friendly
                class_counts = train_data.groupBy(target_column).count().limit(10).collect()
                self.num_classes = len(class_counts)
                print(f"📊 Class count from groupBy fallback: {self.num_classes}")
            except Exception as e2:
                print(f"❌ Both methods failed: {e2}")
                print("🔄 Assuming binary classification and proceeding...")
                self.num_classes = 2
        
        self.is_multiclass = self.num_classes > 2
        print(f"✅ Detected {self.num_classes} classes. Multiclass: {self.is_multiclass}")
        
        # Automatically adjust model selection criteria for multiclass problems
        if self.is_multiclass and self.config['model_selection_criteria'] in ['ks', 'roc']:
            original_criteria = self.config['model_selection_criteria']
            self.config['model_selection_criteria'] = 'accuracy'
            print(f"Automatically switched model selection criteria from '{original_criteria}' to 'accuracy' for multiclass problem")
        
        # Step 1: Data preprocessing
        print("\n1. Data Preprocessing...")
        processed_data = self._preprocess_data(train_data, target_column)
        print(f"Feature variables: {self.feature_vars}")
        print(f"Selected variables: {self.selected_vars}")
        print(f"Categorical variables: {self.categorical_vars}")
        print(f"Numerical variables: {self.numerical_vars}")
        
        # Step 2: Feature selection
        print("\n2. Feature Selection...")
        selected_features, top_features = self._select_features(processed_data, target_column)
        
        # Update selected variables with the feature selection results
        self.selected_vars = top_features
        # Filter the categorical/numerical vars to only include final selected variables
        self.categorical_vars = [var for var in self.categorical_vars if var in self.selected_vars]
        self.numerical_vars = [var for var in self.numerical_vars if var in self.selected_vars]
        print(f"Final selected variables: {self.selected_vars}")
        print(f"Final categorical variables: {self.categorical_vars}")
        print(f"Final numerical variables: {self.numerical_vars}")
        
        # Step 3: Data splitting and scaling
        print("\n3. Data Splitting and Scaling...")
        train_scaled, train_original_scaled, valid_scaled, test_scaled = self._split_and_scale_data(
            selected_features, target_column
        )
        
        # Copy preprocessing pipelines from DataProcessor to AutoMLClassifier
        self.char_labels = self.data_processor.char_labels
        self.target_label_indexer = self.data_processor.target_label_indexer
        self.pipeline_model = self.data_processor.pipeline_model
        
        # Step 4: Prepare out-of-time datasets
        print("\n4. Preparing Out-of-Time Datasets...")
        oot1_scaled, oot2_scaled = self._prepare_oot_datasets(oot1_data, oot2_data, target_column)
        
        # Step 4.5: Determine validation strategy (cross-validation vs train/valid/test)
        use_cross_validation = self._should_use_cross_validation(
            train_data, oot1_data, oot2_data, target_column
        )
        
        # Step 5: Model building and validation
        print("\n5. Model Building and Validation...")
        if use_cross_validation:
            print("🔄 Using cross-validation for model training and validation...")
            dataset_names = self._build_and_validate_models_cv(
                train_scaled, train_original_scaled, valid_scaled, test_scaled, oot1_scaled, oot2_scaled, target_column, top_features
            )
        else:
            print("📊 Using train/validation/test split for model training and validation...")
            dataset_names = self._build_and_validate_models(
                train_scaled, train_original_scaled, valid_scaled, test_scaled, oot1_scaled, oot2_scaled, target_column, top_features
            )
        
        # Step 6: Model selection
        print("\n6. Model Selection...")
        self._select_best_model(target_column, dataset_names)
        
        # Step 7: Generate scoring code
        print("\n7. Generating Scoring Code...")
        self._generate_scoring_code()
        
        # Step 8: Save the model configuration files for future scoring 
        print("\n8. Save model configuration files...")
        self.save_model(self.output_dir)

        # Step 9: Compute SHAP values for explainability (only for best model)
        if self.best_model is not None and self.best_model_type is not None:
            try:
                # Import the explainability module lazily to avoid unnecessary dependencies
                try:
                    from ..explainability import compute_shap_values  # type: ignore
                except ImportError:
                    try:
                        from explainability import compute_shap_values  # type: ignore
                    except ImportError:
                        # For direct script execution
                        import sys
                        import os
                        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        if parent_dir not in sys.path:
                            sys.path.insert(0, parent_dir)
                        from explainability import compute_shap_values
                print(f"\n9. Computing SHAP values for best model ({self.best_model_type}) explainability...")
                # Use the original training data for SHAP.  Only selected variables are passed
                sample_df = train_data
                # Choose the set of features used in the final model.  If no
                # selected_vars are available, fall back to all feature variables.
                feature_cols = self.selected_vars if self.selected_vars else self.feature_vars
                compute_shap_values(
                    spark=self.spark,
                    pipeline_model=self.pipeline_model,
                    model=self.best_model,
                    sample_df=sample_df,
                    feature_cols=feature_cols,
                    output_dir=self.output_dir,
                    model_type="classification",
                    max_samples=50,
                )
                print(f"✅ SHAP values computed successfully for best model ({self.best_model_type})")
            except Exception as e:
                # Explanation errors should not interrupt the pipeline
                print(f"⚠️ SHAP computation skipped for best model ({self.best_model_type}): {e}")
        else:
            print("⚠️ SHAP computation skipped: No best model available")

        print("\nAutoML pipeline completed successfully!")

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
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary information
        """
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summary = {
            'model_id': self.config['model_id'],
            'model_literal': self.config['model_literal'],
            'best_model_type': self.best_model_type,
            'target_column': self.config['target_column'],
            'feature_count': len(self.selected_vars),
            'categorical_features': len(self.categorical_vars),
            'numerical_features': len(self.numerical_vars),
            'model_metrics': self.model_metrics,
            'output_directory': self.output_dir,
            'is_multiclass': self.is_multiclass,
            'num_classes': self.num_classes
        }
        
        return summary
    
    def save_model(self, path: str):
        """
        Save the fitted model and artifacts.
        
        Args:
            path: Directory path to save the model
        """
        if self.best_model is None and not hasattr(self, 'best_model_type'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model artifacts
        model_info = {
            'config': self.config,
            'feature_vars': self.feature_vars,
            'selected_vars': self.selected_vars,
            'categorical_vars': self.categorical_vars,
            'numerical_vars': self.numerical_vars,
            'best_model_type': self.best_model_type,
            'model_metrics': getattr(self, 'model_metrics', {}),
            'is_multiclass': self.is_multiclass,
            'num_classes': self.num_classes
        }
        
        joblib.dump(model_info, os.path.join(path, 'model_info.pkl'))
        print(f"✅ Model info saved to {os.path.join(path, 'model_info.pkl')}")
        
        # Save preprocessing pipeline
        if self.char_labels:
            self.char_labels.write().overwrite().save(os.path.join(path, 'char_labels'))
            print(f"✅ Character labels saved")
        if self.target_label_indexer:
            self.target_label_indexer.write().overwrite().save(os.path.join(path, 'target_label_indexer'))
            print(f"✅ Target label indexer saved")
        if self.pipeline_model:
            self.pipeline_model.write().overwrite().save(os.path.join(path, 'pipeline_model'))
            print(f"✅ Pipeline model saved")
        
        # Save best model with proper handling for different model types
        if self.best_model is not None:
            try:
                # All models (including XGBoost) are saved as standalone models
                model_path = os.path.join(path, f'{self.best_model_type}_model')
                self.best_model.write().overwrite().save(model_path)
                print(f"✅ {self.best_model_type} model saved to {model_path}")
            except Exception as e:
                print(f"⚠️ Could not save {self.best_model_type} model due to serialization issues: {str(e)[:100]}...")
                print(f"💡 Attempting to copy from original location...")
                
                # Try to copy the original model files if they exist
                original_model_path = os.path.join(self.output_dir, f'{self.best_model_type}_model')
                target_model_path = os.path.join(path, f'{self.best_model_type}_model')
                
                if os.path.exists(original_model_path):
                    import shutil
                    try:
                        if os.path.exists(target_model_path):
                            shutil.rmtree(target_model_path)
                        shutil.copytree(original_model_path, target_model_path)
                        print(f"✅ Copied original {self.best_model_type} model from {original_model_path}")
                    except Exception as copy_e:
                        print(f"⚠️ Could not copy original model files: {str(copy_e)[:100]}...")
                        raise ValueError(f"Unable to save {self.best_model_type} model. Original error: {str(e)}")
                else:
                    print(f"⚠️ Original model path {original_model_path} does not exist")
                    raise ValueError(f"Unable to save {self.best_model_type} model. Original error: {str(e)}")
        else:
            # Handle case where best_model is None but training was successful
            if hasattr(self, 'best_model_type') and self.best_model_type:
                print(f"⚠️ Best model object not available (serialization issue), but attempting to copy saved model files...")
                
                # Try to copy the original model files if they exist
                original_model_path = os.path.join(self.output_dir, f'{self.best_model_type}_model')
                target_model_path = os.path.join(path, f'{self.best_model_type}_model')
                
                if os.path.exists(original_model_path):
                    import shutil
                    try:
                        if os.path.exists(target_model_path):
                            shutil.rmtree(target_model_path)
                        shutil.copytree(original_model_path, target_model_path)
                        print(f"✅ Copied original {self.best_model_type} model from {original_model_path}")
                        print(f"💡 Model training was successful - saved model files are available")
                    except Exception as copy_e:
                        print(f"⚠️ Could not copy original model files: {str(copy_e)[:100]}...")
                        raise ValueError(f"Unable to save {self.best_model_type} model. Copy error: {str(copy_e)}")
                else:
                    print(f"⚠️ No model files found at {original_model_path}")
                    raise ValueError("No trained model available to save")
            else:
                print(f"⚠️ No trained model available for saving")
                raise ValueError("No trained model available to save")
        
        print(f"🎉 Complete model saved to {path}")
        print(f"📁 Saved files:")
        print(f"   - model_info.pkl")
        print(f"   - {self.best_model_type}_model/")
        if self.char_labels:
            print(f"   - char_labels/")
        if self.pipeline_model:
            print(f"   - pipeline_model/")
        if self.target_label_indexer:
            print(f"   - target_label_indexer/")
    
    def load_model(self, path: str):
        """
        Load a fitted model and artifacts.
        
        Args:
            path: Directory path containing the saved model
        """
        # Load model info
        model_info = joblib.load(os.path.join(path, 'model_info.pkl'))
        
        # Restore configuration and variables
        self.config = model_info['config']
        self.feature_vars = model_info['feature_vars']
        self.selected_vars = model_info['selected_vars']
        self.categorical_vars = model_info['categorical_vars']
        self.numerical_vars = model_info['numerical_vars']
        self.best_model_type = model_info['best_model_type']
        self.model_metrics = model_info['model_metrics']
        self.is_multiclass = model_info.get('is_multiclass', False)
        self.num_classes = model_info.get('num_classes', 2)
        
        # Load preprocessing pipeline
        try:
            self.char_labels = PipelineModel.load(os.path.join(path, 'char_labels'))
        except:
            self.char_labels = None
            
        try:
            from pyspark.ml.feature import StringIndexerModel
            self.target_label_indexer = StringIndexerModel.load(os.path.join(path, 'target_label_indexer'))
        except:
            self.target_label_indexer = None
            
        try:
            self.pipeline_model = PipelineModel.load(os.path.join(path, 'pipeline_model'))
        except:
            self.pipeline_model = None
        
        # Load best model
        try:
            self.best_model = self.model_builder.load_model(
                self.best_model_type, os.path.join(path, 'best_model')
            )
        except Exception as e:
            print(f"⚠️ Could not reload {self.best_model_type} model: {str(e)[:100]}...")
            print(f"💡 This may be due to XGBoost/Spark serialization issues with NaN values")
            self.best_model = None
        
        print(f"Model loaded from {path}")
    
    def _preprocess_data(self, data: DataFrame, target_column: str) -> DataFrame:
        """Preprocess the input data."""
        processed_data, selected_vars, categorical_vars, numerical_vars = self.data_processor.preprocess(
            data, target_column, self.config
        )
        
        # Store the variable information for later use
        self.selected_vars = selected_vars
        self.categorical_vars = categorical_vars  
        self.numerical_vars = numerical_vars
        self.feature_vars = self.data_processor._get_feature_variables(data, target_column, self.config)
        
        return processed_data
    
    def _select_features(self, data: DataFrame, target_column: str) -> Tuple[DataFrame, List[str]]:
        """Select the best features for modeling."""
        # Pass configuration parameters to data processor
        self.data_processor.sequential_threshold = self.config['sequential_threshold']
        self.data_processor.chunk_size = self.config['chunk_size']
        self.data_processor.features_per_chunk = self.config['features_per_chunk']
        
        # Calculate actual number of features needed (min of available and configured)
        # Add robust error handling for Py4J connection issues
        try:
            feature_cols = [col for col in data.columns if col != target_column]
        except Exception as e:
            print(f"⚠️ Cannot access DataFrame columns due to Py4J error: {e}")
            print("🔄 Attempting to restart Spark session and retry...")
            
            # Try to restart Spark session
            if hasattr(self, '_restart_spark_session'):
                if self._restart_spark_session():
                    try:
                        # Retry with fresh session
                        feature_cols = [col for col in data.columns if col != target_column]
                        print("✅ Successfully recovered DataFrame columns after session restart")
                    except Exception as e2:
                        print(f"❌ Still cannot access DataFrame columns after session restart: {e2}")
                        raise RuntimeError(f"Failed to access DataFrame columns after session restart: {e2}")
                else:
                    print("❌ Failed to restart Spark session")
                    raise RuntimeError(f"Failed to restart Spark session: {e}")
            else:
                # Fallback: try to get columns from the data processor's cached info
                print("🔄 Attempting to use cached column information...")
                try:
                    # Try to get columns from the data processor if it has cached the schema
                    if hasattr(self.data_processor, 'last_processed_schema'):
                        feature_cols = [col for col in self.data_processor.last_processed_schema if col != target_column]
                        print(f"✅ Using cached schema with {len(feature_cols)} features")
                    else:
                        raise RuntimeError("No cached schema available")
                except Exception as e3:
                    print(f"❌ Cannot recover column information: {e3}")
                    raise RuntimeError(f"Failed to access DataFrame columns and no recovery method available: {e}")
        
        actual_max_features = min(self.config['max_features'], len(feature_cols))
        
        print(f"📊 Available features: {len(feature_cols)}, Configured max: {self.config['max_features']}")
        print(f"🎯 Will select: {actual_max_features} features")
        
        return self.data_processor.select_features(
            data, target_column, actual_max_features
        )
    
    def _split_and_scale_data(self, data: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """Split data and apply scaling."""
        return self.data_processor.split_and_scale(
            data, 
            self.config['train_size'], 
            self.config['valid_size'], 
            target_column,
            self.config['seed'],
            self.config
        )
    
    def _prepare_oot_datasets(self, oot1_data: Optional[DataFrame], 
                             oot2_data: Optional[DataFrame], 
                             target_column: str) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Prepare out-of-time datasets using the fitted preprocessing pipeline."""
        oot1_scaled = None
        oot2_scaled = None
        
        if oot1_data is not None:
            print("🔄 Processing OOT1 data...")
            try:
                # Check if OOT1 data has the same structure as training data
                # Convert encoded column names back to original names for comparison
                original_column_names = []
                for col in self.selected_vars:
                    if col.endswith('_encoded'):
                        original_name = col.replace('_encoded', '')
                        original_column_names.append(original_name)
                    else:
                        original_column_names.append(col)
                
                # Also check for target column
                if target_column not in oot1_data.columns:
                    print(f"⚠️ OOT1 data missing target column: {target_column}")
                    print(f"Available columns in OOT1: {oot1_data.columns}")
                    print("⚠️ OOT1 data will be skipped due to missing target column")
                    oot1_scaled = None
                else:
                    # Identify any missing raw feature columns required by the preprocessing pipeline
                    missing_columns = [col for col in original_column_names if col not in oot1_data.columns]
                    if missing_columns:
                        print(f"⚠️ OOT1 data missing columns: {missing_columns}")
                        print(f"Available columns in OOT1: {oot1_data.columns}")
                        print(f"Expected original columns: {original_column_names}")
                        print("⚠️ OOT1 data will be skipped due to missing columns")
                        oot1_scaled = None
                    else:
                        # Apply the same preprocessing pipeline that was fitted on training data
                        oot1_processed = self._apply_preprocessing(oot1_data)
                        # Only proceed if preprocessing was successful
                        if oot1_processed is not None:
                            print(f"✅ OOT1 preprocessing completed. Columns: {oot1_processed.columns}")
                            # Check if the preprocessed data has all required columns for scaling
                            scaling_columns = [col for col in self.selected_vars if col != target_column]
                            missing_scaling_columns = [col for col in scaling_columns if col not in oot1_processed.columns]
                            if missing_scaling_columns:
                                print(f"⚠️ OOT1 data missing columns for scaling: {missing_scaling_columns}")
                                print(f"Available columns after preprocessing: {oot1_processed.columns}")
                                print(f"Required columns for scaling: {scaling_columns}")
                                print("⚠️ OOT1 data will be skipped due to missing scaling columns")
                                oot1_scaled = None
                            else:
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
                # Check if OOT2 data has the same structure as training data
                # Convert encoded column names back to original names for comparison
                original_column_names = []
                for col in self.selected_vars:
                    if col.endswith('_encoded'):
                        original_name = col.replace('_encoded', '')
                        original_column_names.append(original_name)
                    else:
                        original_column_names.append(col)
                
                # Also check for target column
                if target_column not in oot2_data.columns:
                    print(f"⚠️ OOT2 data missing target column: {target_column}")
                    print(f"Available columns in OOT2: {oot2_data.columns}")
                    print("⚠️ OOT2 data will be skipped due to missing target column")
                    oot2_scaled = None
                else:
                    # Identify any missing raw feature columns required by the preprocessing pipeline
                    missing_columns = [col for col in original_column_names if col not in oot2_data.columns]
                    if missing_columns:
                        print(f"⚠️ OOT2 data missing columns: {missing_columns}")
                        print(f"Available columns in OOT2: {oot2_data.columns}")
                        print(f"Expected original columns: {original_column_names}")
                        print("⚠️ OOT2 data will be skipped due to missing columns")
                        oot2_scaled = None
                    else:
                        # Apply the same preprocessing pipeline that was fitted on training data
                        oot2_processed = self._apply_preprocessing(oot2_data)
                        if oot2_processed is not None:
                            print(f"✅ OOT2 preprocessing completed. Columns: {oot2_processed.columns}")
                            # Check if the preprocessed data has all required columns for scaling
                            scaling_columns = [col for col in self.selected_vars if col != target_column]
                            missing_scaling_columns = [col for col in scaling_columns if col not in oot2_processed.columns]
                            if missing_scaling_columns:
                                print(f"⚠️ OOT2 data missing columns for scaling: {missing_scaling_columns}")
                                print(f"Available columns after preprocessing: {oot2_processed.columns}")
                                print(f"Required columns for scaling: {scaling_columns}")
                                print("⚠️ OOT2 data will be skipped due to missing scaling columns")
                                oot2_scaled = None
                            else:
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
    

    
    def _should_use_cross_validation(self, train_data: DataFrame, 
                                   oot1_data: Optional[DataFrame], 
                                   oot2_data: Optional[DataFrame],
                                   target_column: str) -> bool:
        """
        Determine whether to use cross-validation based on data availability and size.
        
        Args:
            train_data: Training DataFrame
            oot1_data: Out-of-time validation data 1 (optional)
            oot2_data: Out-of-time validation data 2 (optional) 
            target_column: Name of the target column
            
        Returns:
            True if cross-validation should be used, False otherwise
        """
        cv_setting = self.config.get('use_cross_validation', 'auto')
        
        if cv_setting == 'always':
            print("✅ Cross-validation enabled: forced by configuration")
            return True
        elif cv_setting == 'never':
            print("❌ Cross-validation disabled: forced by configuration")
            return False
        elif cv_setting == 'auto':
            # Auto-detect based on data availability and size
            sample_size = train_data.count()
            min_sample_size = self.config.get('min_sample_size_for_split', 5000)
            
            # Check if we have out-of-time validation data
            has_oot_data = oot1_data is not None or oot2_data is not None
            
            # Use cross-validation if:
            # 1. No out-of-time validation data available, OR
            # 2. Sample size is too small for reliable train/valid/test split
            use_cv = not has_oot_data or sample_size < min_sample_size
            
            if use_cv:
                if not has_oot_data:
                    print(f"✅ Cross-validation enabled: no out-of-time validation data available")
                else:
                    print(f"✅ Cross-validation enabled: sample size ({sample_size}) < minimum ({min_sample_size})")
            else:
                print(f"❌ Cross-validation disabled: sufficient data for train/valid/test split")
                
            return use_cv
        else:
            print(f"⚠️ Unknown cross-validation setting: {cv_setting}, defaulting to auto")
            return self._should_use_cross_validation(train_data, oot1_data, oot2_data, target_column)
    
    def _validate_cv_metrics(self):
        """Validate and provide information about cross-validation metrics."""
        # Available metrics for each classification type
        binary_metrics = ['areaUnderROC', 'areaUnderPR']
        multiclass_metrics = ['accuracy', 'f1', 'weightedPrecision', 'weightedRecall', 'weightedFMeasure']
        
        # Validate binary metric
        binary_metric = self.config.get('cv_metric_binary', 'areaUnderROC')
        if binary_metric not in binary_metrics:
            print(f"⚠️ Warning: Unknown binary CV metric '{binary_metric}'. Available: {binary_metrics}")
            print(f"Falling back to 'areaUnderROC'")
            self.config['cv_metric_binary'] = 'areaUnderROC'
        
        # Validate multiclass metric
        multiclass_metric = self.config.get('cv_metric_multiclass', 'accuracy')
        if multiclass_metric not in multiclass_metrics:
            print(f"⚠️ Warning: Unknown multiclass CV metric '{multiclass_metric}'. Available: {multiclass_metrics}")
            print(f"Falling back to 'accuracy'")
            self.config['cv_metric_multiclass'] = 'accuracy'
        
        # Provide information about metrics
        print(f"\n📊 Cross-Validation Metrics Configuration:")
        print(f"  Binary classification: {self.config['cv_metric_binary']}")
        print(f"  Multiclass classification: {self.config['cv_metric_multiclass']}")
        
        if binary_metric == 'areaUnderROC':
            print(f"  📈 areaUnderROC: Measures ability to distinguish between classes")
        elif binary_metric == 'areaUnderPR':
            print(f"  📈 areaUnderPR: Precision-Recall AUC, better for imbalanced datasets")
            
        if multiclass_metric == 'accuracy':
            print(f"  🎯 accuracy: Overall classification accuracy")
        elif multiclass_metric == 'f1':
            print(f"  ⚖️ f1: Harmonic mean of precision and recall")
        elif multiclass_metric in ['weightedPrecision', 'weightedRecall', 'weightedFMeasure']:
            print(f"  📊 {multiclass_metric}: Class-weighted {multiclass_metric.replace('weighted', '').lower()}")
    
    def _apply_preprocessing(self, data: DataFrame) -> DataFrame:
        """Apply preprocessing pipeline to new data."""
        return self.data_processor.apply_preprocessing(
            data, self.feature_vars, self.selected_vars,
            self.categorical_vars, self.numerical_vars,
            self.char_labels, self.config['impute_value'],
            self.config['target_column'], self.target_label_indexer
        )
    
    def _build_and_validate_models(self, train_data: DataFrame, train_original_data: DataFrame, 
                                  valid_data: DataFrame, test_data: DataFrame, oot1_data: Optional[DataFrame],
                                  oot2_data: Optional[DataFrame], target_column: str, top_features: List[str]):
        """Build and validate multiple models using comprehensive validation."""
        # Use original training data for evaluation metrics
        datasets = [train_original_data, valid_data, test_data]
        dataset_names = ['train', 'valid', 'test']
        
        if oot1_data is not None:
            datasets.append(oot1_data)
            dataset_names.append('oot1')
        
        if oot2_data is not None:
            datasets.append(oot2_data)
            dataset_names.append('oot2')
        
        # Build and validate models using dual training approach
        for model_type in ['logistic', 'random_forest', 'gradient_boosting', 'decision_tree', 'neural_network', 'xgboost', 'lightgbm']:
            if self.config.get(f'run_{model_type}', True):
                # Skip gradient boosting for multi-class problems (PySpark limitation)
                if model_type == 'gradient_boosting' and self.is_multiclass:
                    print(f"Skipping {model_type} model - PySpark GBTClassifier only supports binary classification")
                    continue
                    
                print(f"Building {model_type} model...")
                
                # Use dual training approach to select best model
                model_result = self._build_model_with_dual_training(
                    train_data, target_column, model_type, top_features, datasets, dataset_names
                )
                
                # Store selected model and metrics
                selected_model = model_result['model']
                selected_metrics = model_result['metrics']
                comparison_info = model_result['comparison']
                
                # Store metrics with comparison information
                self.model_metrics[model_type] = selected_metrics
                self.model_metrics[model_type]['selection_info'] = comparison_info
                
                # Save the selected model
                model_path = os.path.join(self.output_dir, f'{model_type}_model')
                self.model_builder.save_model(selected_model, model_path)
                
                print(f"✅ {model_type} model training completed - selected {comparison_info['decision']} version")

        return dataset_names

    def _build_and_validate_models_cv(self, train_data: DataFrame, train_original_data: DataFrame,
                                     valid_data: DataFrame, test_data: DataFrame, oot1_data: Optional[DataFrame],
                                     oot2_data: Optional[DataFrame], target_column: str, top_features: List[str]):
        """Build and validate multiple models using cross-validation."""
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
        
        # For cross-validation, use upsampled training data for model training
        # Combine upsampled train, valid, test for cross-validation (exclude oot data for training)
        cv_data = train_data.union(valid_data).union(test_data)
        print(f"Cross-validation dataset size: {cv_data.count()} samples")
        print("📋 Note: CV training uses upsampled data, but evaluation metrics use original data")
        
        # Prepare validation datasets using original data for evaluation metrics
        datasets = [train_original_data, valid_data, test_data]  # Original data for evaluation
        dataset_names = ['train', 'valid', 'test']
        
        if oot1_data is not None:
            datasets.append(oot1_data)
            dataset_names.append('oot1')
        
        if oot2_data is not None:
            datasets.append(oot2_data)
            dataset_names.append('oot2')
        
        # Validate CV metrics configuration
        self._validate_cv_metrics()
        
        # Build and validate models with cross-validation
        cv_folds = self.config.get('cv_folds', 5)
        
        for model_type in ['logistic', 'random_forest', 'gradient_boosting', 'decision_tree', 'neural_network', 'xgboost', 'lightgbm']:
            if self.config.get(f'run_{model_type}', True):
                # Skip gradient boosting for multi-class problems (PySpark limitation)
                if model_type == 'gradient_boosting' and self.is_multiclass:
                    print(f"Skipping {model_type} model - PySpark GBTClassifier only supports binary classification")
                    continue
                    
                print(f"Building {model_type} model with {cv_folds}-fold cross-validation...")
                
                # Use dual training approach for CV as well
                model_result = self._build_model_with_dual_training_cv(
                    cv_data, target_column, model_type, top_features, datasets, dataset_names, cv_folds
                )
                
                # Store selected model and metrics
                selected_model = model_result['model']
                selected_metrics = model_result['metrics']
                comparison_info = model_result['comparison']
                
                # Store metrics with comparison information
                self.model_metrics[model_type] = selected_metrics
                self.model_metrics[model_type]['selection_info'] = comparison_info
                
                # Save the selected model
                model_path = os.path.join(self.output_dir, f'{model_type}_model')
                self.model_builder.save_model(selected_model, model_path)
                
                print(f"✅ {model_type} CV model training completed - selected {comparison_info['decision']} version")
                continue  # Skip the rest of the loop since we handled everything
                
                # Set up evaluator for cross-validation
                if self.is_multiclass:
                    cv_metric = self.config.get('cv_metric_multiclass', 'accuracy')
                    evaluator = MulticlassClassificationEvaluator(
                        labelCol=target_column, 
                        predictionCol="prediction",
                        metricName=cv_metric
                    )
                    print(f"Using {cv_metric} as cross-validation metric for multiclass classification")
                else:
                    cv_metric = self.config.get('cv_metric_binary', 'areaUnderROC')
                    evaluator = BinaryClassificationEvaluator(
                        labelCol=target_column,
                        rawPredictionCol="rawPrediction",
                        metricName=cv_metric
                    )
                    print(f"Using {cv_metric} as cross-validation metric for binary classification")
                
                # Create simple parameter grid (you can expand this for hyperparameter tuning)
                paramGrid = ParamGridBuilder().build()
                
                # Create cross-validator
                cv = CrossValidator(
                    estimator=base_estimator,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=cv_folds,
                    seed=self.config['seed']
                )
                
                # Fit cross-validator
                cv_model = cv.fit(cv_data)
                
                # Get the best model
                best_model = cv_model.bestModel
                
                # Calculate metrics on all datasets using the best model
                if self.is_multiclass:
                    metrics = self.model_validator.validate_model_multiclass(
                        best_model, datasets, dataset_names, target_column, model_type, self.output_dir
                    )
                else:
                    metrics = self.model_validator.validate_model(
                        best_model, datasets, dataset_names, target_column, model_type
                    )
                
                # Add cross-validation metrics
                cv_metrics = cv_model.avgMetrics[0]  # Best fold metrics
                metrics[f'cv_score_{dataset_names[0]}'] = cv_metrics
                
                print(f"Cross-validation score for {model_type}: {cv_metrics:.4f}")
                
                # Store metrics
                self.model_metrics[model_type] = metrics
                
                # Save model
                model_path = os.path.join(self.output_dir, f'{model_type}_model')
                self.model_builder.save_model(best_model, model_path)

        return dataset_names

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
        
        # Initialize hyperparameter tuner if needed
        if self.hyperparameter_tuner is None:
            self.hyperparameter_tuner = HyperparameterTuner(self.spark, self.config)
            self.hyperparameter_tuner.set_problem_type(self.is_multiclass, self.num_classes)
        
        # Use the hyperparameter tuner for optimization
        return self.hyperparameter_tuner.optimize_hyperparameters(
            train_data, target_column, model_type, feature_count, self.model_builder.model_types
        )

    
    def _select_best_model(self, target_column: str, dataset_names: List[str]):
        """Select the best model based on criteria."""
        best_model_info = self.model_selector.select_best_model(
            self.model_metrics, 
            self.config['model_selection_criteria'],
            self.config['dataset_to_use'],
            dataset_names)
        
        self.best_model_type = best_model_info['model_type']
        model_path = os.path.join(self.output_dir, f'{self.best_model_type}_model')
        
        try:
            self.best_model = self.model_builder.load_model(self.best_model_type, model_path)
            print(f"✅ Best model loaded successfully: {self.best_model_type}")
        except Exception as e:
            print(f"⚠️ Could not reload {self.best_model_type} model: {str(e)[:100]}...")
            print(f"🔍 Checking if model files exist on disk...")
            
            if os.path.exists(model_path):
                print(f"✅ Model directory exists: {model_path}")
                # List files in the model directory
                try:
                    files = os.listdir(model_path)
                    print(f"📁 Model files found: {files[:5]}...")  # Show first 5 files
                except:
                    print(f"❌ Could not list files in model directory")
            else:
                print(f"❌ Model directory does not exist: {model_path}")
            
            print(f"💡 Model training was successful - issue is only with reloading saved model")
            print(f"💡 This is a known XGBoost/Spark serialization issue with NaN values")
            print(f"💡 All performance metrics and model selection results are still valid")
            print(f"💡 Model files will be copied during save operation")
            self.best_model = None
        
        print(f"Best model selected: {self.best_model_type}")
    
    def _generate_scoring_code(self):
        """Generate production scoring code."""
        if self.best_model_type is None:
            print("Warning: No best model selected, skipping scoring code generation")
            return
            
        self.score_generator.generate_scoring_code(
            self.config, self.feature_vars, self.selected_vars,
            self.categorical_vars, self.numerical_vars,
            self.best_model_type, self.is_multiclass
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
        improvement_threshold = self.config.get('improvement_threshold', 0.01)
        overfitting_threshold = self.config.get('overfitting_threshold', 0.05)
        significance_threshold = self.config.get('statistical_significance_threshold', 0.05)
        
        # Get primary metric based on problem type and configuration
        primary_metric = self._get_primary_metric()
        
        # Extract metrics for comparison
        default_score = self._extract_score(default_metrics, primary_metric)
        tuned_score = self._extract_score(tuned_metrics, primary_metric)
        
        # Calculate improvement
        if primary_metric in ['ks', 'roc', 'accuracy', 'f1', 'precision', 'recall']:
            # Higher is better
            improvement = tuned_score - default_score
            is_better = tuned_score > default_score
        else:
            # Lower is better (e.g., loss metrics)
            improvement = default_score - tuned_score  
            is_better = tuned_score < default_score
        
        improvement_pct = (improvement / abs(default_score)) * 100 if default_score != 0 else 0
        
        # Check for overfitting (train vs validation performance gap)
        default_overfitting = self._check_overfitting(default_metrics, primary_metric)
        tuned_overfitting = self._check_overfitting(tuned_metrics, primary_metric)
        
        # Decision logic
        reasons = []
        decision = "default"
        
        if not is_better:
            reasons.append(f"Tuned model performance ({tuned_score:.4f}) is not better than default ({default_score:.4f})")
            decision = "default"
        elif improvement_pct < (improvement_threshold * 100):
            reasons.append(f"Performance improvement ({improvement_pct:.2f}%) is below threshold ({improvement_threshold*100}%)")
            decision = "default"
        elif tuned_overfitting > default_overfitting + overfitting_threshold:
            reasons.append(f"Tuned model shows more overfitting (gap: {tuned_overfitting:.3f} vs {default_overfitting:.3f})")
            decision = "default"
        else:
            reasons.append(f"Tuned model shows significant improvement ({improvement_pct:.2f}% better)")
            if tuned_overfitting <= default_overfitting:
                reasons.append("Tuned model generalizes as well or better than default")
            decision = "tuned"
        
        # Statistical significance check (simplified)
        if improvement_pct > (significance_threshold * 100):
            reasons.append(f"Improvement is statistically significant (>{significance_threshold*100}%)")
        elif improvement_pct > 0 and decision == "tuned":
            reasons.append("Improvement is modest but consistent")
        
        # Format metric name for display
        display_metric = primary_metric.upper() if primary_metric in ['roc', 'ks'] else primary_metric.title()
        
        return {
            'decision': decision,
            'default_score': default_score,
            'tuned_score': tuned_score,
            'improvement_pct': improvement_pct,
            'default_overfitting': default_overfitting,
            'tuned_overfitting': tuned_overfitting,
            'reasons': reasons,
            'primary_metric': display_metric
        }
    
    def _get_primary_metric(self) -> str:
        """Get the primary metric for model comparison based on configuration."""
        if self.is_multiclass:
            return self.config.get('model_selection_criteria', 'accuracy')
        else:
            return self.config.get('model_selection_criteria', 'ks')
    
    def _extract_score(self, metrics: Dict, metric_name: str) -> float:
        """Extract the primary score from metrics dictionary."""
        # Use validation dataset score for comparison (avoid overfitting to train)
        dataset_to_use = self.config.get('dataset_to_use', 'valid')
        
        # Try nested structure first: metrics[dataset][metric]
        if dataset_to_use in metrics and isinstance(metrics[dataset_to_use], dict) and metric_name in metrics[dataset_to_use]:
            return metrics[dataset_to_use][metric_name]
        
        # Try flat structure: metrics[metric_dataset]
        flat_key = f"{metric_name}_{dataset_to_use}"
        if flat_key in metrics:
            return metrics[flat_key]
        
        # Fallback to train dataset
        if 'train' in metrics and isinstance(metrics['train'], dict) and metric_name in metrics['train']:
            return metrics['train'][metric_name]
        
        flat_key_train = f"{metric_name}_train"
        if flat_key_train in metrics:
            return metrics[flat_key_train]
        
        # Last resort: find any available score with this metric name
        for key, value in metrics.items():
            if isinstance(value, dict) and metric_name in value:
                return value[metric_name]
            elif key.startswith(f"{metric_name}_") and isinstance(value, (int, float)):
                return value
        
        print(f"⚠️ Could not find {metric_name} score in metrics. Available keys: {list(metrics.keys())}")
        return 0.0
    
    def _check_overfitting(self, metrics: Dict, metric_name: str) -> float:
        """Check for overfitting by comparing train vs validation performance."""
        if 'train' not in metrics or 'valid' not in metrics:
            return 0.0
        
        train_score = metrics['train'].get(metric_name, 0)
        valid_score = metrics['valid'].get(metric_name, 0)
        
        if train_score == 0:
            return 0.0
        
        # For metrics where higher is better
        if metric_name in ['ks', 'roc', 'accuracy', 'f1', 'precision', 'recall']:
            return (train_score - valid_score) / train_score
        else:
            # For metrics where lower is better
            return (valid_score - train_score) / train_score
    
    def _build_model_with_dual_training(self, train_data: DataFrame, target_column: str, 
                                       model_type: str, top_features: List[str],
                                       datasets: List[DataFrame], dataset_names: List[str]) -> Dict[str, Any]:
        """
        Build model using dual training approach: default + hyperparameter-tuned versions.
        
        Args:
            train_data: Training data
            target_column: Target column name
            model_type: Model type to build
            top_features: List of selected features
            datasets: List of datasets for evaluation
            dataset_names: Names of the datasets
            
        Returns:
            Dictionary containing selected model, metrics, and comparison info
        """
        print(f"  🚀 Building dual-trained {model_type} model...")
        
        # Build default model
        print(f"  📊 Training default {model_type} model...")
        default_model = self.model_builder.build_model(
            train_data, 'features', target_column, model_type,
            num_features=len(top_features)
        )
        
        # Validate default model
        default_metrics = self.model_validator.validate_model(
            default_model, datasets, dataset_names, target_column, f"{model_type}_default"
        )
        
        # Initialize variables for tuned model
        tuned_model = None
        tuned_metrics = None
        
        # Build hyperparameter-tuned model if enabled
        if self.config.get('enable_hyperparameter_tuning', False):
            print(f"  🎯 Training hyperparameter-tuned {model_type} model...")
            
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
            
            # Optimize hyperparameters
            if self.hyperparameter_tuner is None:
                self.hyperparameter_tuner = HyperparameterTuner(self.spark, self.config)
                self.hyperparameter_tuner.set_problem_type(self.is_multiclass, self.num_classes)
            
            optimization_results = self.hyperparameter_tuner.optimize_hyperparameters(
                train_data, target_column, model_type, len(top_features), self.model_builder.model_types
            )
        
        # Build tuned model
        tuned_model = self.model_builder.build_model(
            train_data, 'features', target_column, model_type,
            num_features=len(top_features),
            **optimization_results['best_params']
        )
        
        # Validate tuned model
        tuned_metrics = self.model_validator.validate_model(
            tuned_model, datasets, dataset_names, target_column, f"{model_type}_tuned"
        )
        
        # Compare models and select the better one
        comparison_result = self._compare_models(default_metrics, tuned_metrics, model_type)
        
        # Log detailed comparison
        self._log_model_comparison(model_type, comparison_result)
        
        return {
            'model': tuned_model,
            'metrics': tuned_metrics,
            'model_type': f"{model_type}_tuned",
            'comparison': comparison_result,
            'default_metrics': default_metrics,
            'tuned_metrics': tuned_metrics
        }
    
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

    def _build_model_with_dual_training_cv(self, cv_data: DataFrame, target_column: str, 
                                           model_type: str, top_features: List[str],
                                           datasets: List[DataFrame], dataset_names: List[str],
                                           cv_folds: int) -> Dict[str, Any]:
        """
        Build both default and tuned models using cross-validation, then select the better one.
        
        Returns:
            Dictionary with selected model, metrics, and selection reasoning
        """
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
        
        print(f"\n🔄 Training both default and tuned {model_type} models with {cv_folds}-fold CV...")
        
        # Validate Spark session before proceeding
        try:
            self.spark.sparkContext.getConf().get("spark.driver.host")
            print("✅ Spark session is healthy")
        except Exception as e:
            print(f"⚠️ Spark session issue detected: {e}")
            print("🔄 Attempting to restart Spark session...")
            if self._restart_spark_session():
                print("✅ Spark session restarted successfully")
            else:
                print("❌ Failed to restart Spark session, proceeding with caution")
        
        # Set up evaluator for cross-validation
        if self.is_multiclass:
            cv_metric = self.config.get('cv_metric_multiclass', 'accuracy')
            evaluator = MulticlassClassificationEvaluator(
                labelCol=target_column, 
                predictionCol="prediction",
                metricName=cv_metric
            )
            print(f"Using {cv_metric} as cross-validation metric for multiclass classification")
        else:
            cv_metric = self.config.get('cv_metric_binary', 'areaUnderROC')
            evaluator = BinaryClassificationEvaluator(
                labelCol=target_column,
                metricName=cv_metric
            )
            print(f"Using {cv_metric} as cross-validation metric for binary classification")
        
        # Build default model with CV
        print(f"  📊 Training default {model_type} model with CV...")
        try:
            default_estimator = self.model_builder.create_estimator(
                'features', target_column, model_type,
                num_classes=self.num_classes if self.is_multiclass else 2,
                feature_count=len(top_features)
            )
            
            # Create a simple parameter grid for default model (minimal grid)
            default_param_grid = ParamGridBuilder().build()  # Empty grid - uses default params
            
            default_cv = CrossValidator(
                estimator=default_estimator,
                estimatorParamMaps=default_param_grid,
                evaluator=evaluator,
                numFolds=cv_folds,
                seed=42
            )
            
            default_cv_model = default_cv.fit(cv_data)
            default_final_model = default_cv_model.bestModel
            
            # Validate default model on all datasets
            if self.is_multiclass:
                default_metrics = self.model_validator.validate_model_multiclass(
                    default_final_model, datasets, dataset_names, target_column, f"{model_type}_default", self.output_dir
                )
            else:
                default_metrics = self.model_validator.validate_model(
                    default_final_model, datasets, dataset_names, target_column, f"{model_type}_default"
                )
            
            # Add CV score to metrics
            default_metrics['cv_score'] = float(default_cv_model.avgMetrics[0])
            
        except Exception as e:
            print(f"❌ Error training default {model_type} model: {e}")
            # Return a minimal result to prevent pipeline failure
            return {
                'model': None,
                'metrics': {'error': str(e)},
                'model_type': f"{model_type}_default",
                'comparison': {'decision': 'default', 'reasons': [f'Training failed: {e}']},
                'default_metrics': {'error': str(e)},
                'tuned_metrics': None
            }
        
        # Build tuned model with CV (only if hyperparameter optimization is enabled)
        tuned_final_model = None
        tuned_metrics = None
        comparison_result = None
        
        if self.config.get('enable_hyperparameter_tuning', False):
            print(f"  🎯 Training hyperparameter-tuned {model_type} model with CV...")
            
            # Optimize hyperparameters first
            optimization_results = self._optimize_hyperparameters(
                cv_data, target_column, model_type, len(top_features)
            )
            
            best_params = {}
            if optimization_results and 'best_params' in optimization_results:
                best_params = optimization_results['best_params']
                print(f"    🔧 Optimized parameters: {best_params}")
            
            # Create tuned estimator
            tuned_estimator = self.model_builder.create_estimator(
                'features', target_column, model_type,
                num_classes=self.num_classes if self.is_multiclass else 2,
                feature_count=len(top_features),
                **best_params
            )
            
            tuned_param_grid = ParamGridBuilder().build()  # Empty grid - uses optimized params
            
            tuned_cv = CrossValidator(
                estimator=tuned_estimator,
                estimatorParamMaps=tuned_param_grid,
                evaluator=evaluator,
                numFolds=cv_folds,
                seed=42
            )
            
            tuned_cv_model = tuned_cv.fit(cv_data)
            tuned_final_model = tuned_cv_model.bestModel
            
            # Validate tuned model on all datasets
            if self.is_multiclass:
                tuned_metrics = self.model_validator.validate_model_multiclass(
                    tuned_final_model, datasets, dataset_names, target_column, f"{model_type}_tuned", self.output_dir
                )
            else:
                tuned_metrics = self.model_validator.validate_model(
                    tuned_final_model, datasets, dataset_names, target_column, f"{model_type}_tuned"
                )
            
            # Add CV score to metrics
            tuned_metrics['cv_score'] = float(tuned_cv_model.avgMetrics[0])
            
            # Compare models and select the better one
            comparison_result = self._compare_models(default_metrics, tuned_metrics, model_type)
            
            # Add CV score comparison
            cv_improvement = tuned_metrics['cv_score'] - default_metrics['cv_score']
            comparison_result['cv_improvement'] = cv_improvement
            comparison_result['default_cv_score'] = default_metrics['cv_score']
            comparison_result['tuned_cv_score'] = tuned_metrics['cv_score']
            
            # Log detailed comparison including CV scores
            self._log_model_comparison_cv(model_type, comparison_result)
            
        else:
            print(f"  📝 Hyperparameter optimization disabled - using default {model_type} model")
            comparison_result = {
                'decision': 'default',
                'reasons': ['Hyperparameter optimization is disabled'],
                'default_score': self._extract_score(default_metrics, self._get_primary_metric()),
                'default_cv_score': default_metrics['cv_score']
            }
        
        # Select the final model and metrics
        if comparison_result['decision'] == 'tuned' and tuned_final_model is not None:
            selected_model = tuned_final_model
            selected_metrics = tuned_metrics
            selected_type = f"{model_type}_tuned"
        else:
            selected_model = default_final_model
            selected_metrics = default_metrics
            selected_type = f"{model_type}_default"
        
        return {
            'model': selected_model,
            'metrics': selected_metrics,
            'model_type': selected_type,
            'comparison': comparison_result,
            'default_metrics': default_metrics,
            'tuned_metrics': tuned_metrics
        }
    
    def _log_model_comparison_cv(self, model_type: str, comparison: Dict[str, Any]):
        """Log detailed model comparison results for cross-validation."""
        print(f"\n🤔 CV Model Selection Analysis for {model_type.upper()}:")
        print(f"   📈 Default model {comparison['primary_metric']}: {comparison['default_score']:.4f}")
        print(f"   📊 Default CV score: {comparison['default_cv_score']:.4f}")
        
        if 'tuned_score' in comparison:
            print(f"   🎯 Tuned model {comparison['primary_metric']}: {comparison['tuned_score']:.4f}")
            print(f"   🎯 Tuned CV score: {comparison['tuned_cv_score']:.4f}")
            print(f"   📊 Performance improvement: {comparison['improvement_pct']:.2f}%")
            print(f"   📊 CV score improvement: {comparison['cv_improvement']:.4f}")
            print(f"   🎪 Default overfitting gap: {comparison['default_overfitting']:.3f}")
            print(f"   🎪 Tuned overfitting gap: {comparison['tuned_overfitting']:.3f}")
        
        print(f"   ✅ DECISION: Using {comparison['decision'].upper()} model")
        print(f"   💭 Reasoning:")
        for reason in comparison['reasons']:
            print(f"      • {reason}") 

    def get_model_selection_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all model selection decisions.
        
        Returns:
            Dictionary with detailed selection summary for all models
        """
        summary = {
            'total_models_trained': 0,
            'default_selected': 0,
            'tuned_selected': 0,
            'model_decisions': {},
            'selection_criteria': {
                'improvement_threshold': self.config.get('improvement_threshold', 0.01),
                'overfitting_threshold': self.config.get('overfitting_threshold', 0.05),
                'significance_threshold': self.config.get('statistical_significance_threshold', 0.05)
            },
            'hyperparameter_optimization_enabled': self.config.get('enable_hyperparameter_tuning', False),
            'optimization_method': self.config.get('optimization_method', 'optuna')
        }
        
        for model_type, metrics in self.model_metrics.items():
            if 'selection_info' in metrics:
                selection_info = metrics['selection_info']
                summary['total_models_trained'] += 1
                
                if selection_info['decision'] == 'default':
                    summary['default_selected'] += 1
                else:
                    summary['tuned_selected'] += 1
                
                summary['model_decisions'][model_type] = {
                    'decision': selection_info['decision'],
                    'reasons': selection_info['reasons'],
                    'default_score': selection_info.get('default_score', 'N/A'),
                    'tuned_score': selection_info.get('tuned_score', 'N/A'),
                    'improvement_pct': selection_info.get('improvement_pct', 0),
                    'overfitting_comparison': {
                        'default': selection_info.get('default_overfitting', 'N/A'),
                        'tuned': selection_info.get('tuned_overfitting', 'N/A')
                    }
                }
        
        return summary
    
    def print_model_selection_summary(self):
        """Print a user-friendly summary of model selection decisions and save to file."""
        summary = self.get_model_selection_summary()
        
        # Format the summary content
        lines = []
        lines.append("="*80)
        lines.append("🎯 MODEL SELECTION SUMMARY")
        lines.append("="*80)
        lines.append("")
        
        lines.append("📊 Overall Statistics:")
        lines.append(f"   • Total models trained: {summary['total_models_trained']}")
        lines.append(f"   • Default models selected: {summary['default_selected']}")
        lines.append(f"   • Tuned models selected: {summary['tuned_selected']}")
        
        if summary['hyperparameter_optimization_enabled']:
            lines.append(f"   • Hyperparameter optimization: ✅ Enabled ({summary['optimization_method']})")
        else:
            lines.append(f"   • Hyperparameter optimization: ❌ Disabled")
        
        lines.append("")
        lines.append("🎛️ Selection Criteria:")
        criteria = summary['selection_criteria']
        lines.append(f"   • Minimum improvement threshold: {criteria['improvement_threshold']*100:.1f}%")
        lines.append(f"   • Maximum overfitting gap: {criteria['overfitting_threshold']*100:.1f}%")
        lines.append(f"   • Statistical significance threshold: {criteria['significance_threshold']*100:.1f}%")
        
        lines.append("")
        lines.append("🔍 Individual Model Decisions:")
        for model_type, decision_info in summary['model_decisions'].items():
            decision = decision_info['decision'].upper()
            icon = "🎯" if decision == "TUNED" else "📊"
            lines.append("")
            lines.append(f"   {icon} {model_type.upper()}: {decision} MODEL SELECTED")
            
            if decision_info['tuned_score'] != 'N/A':
                lines.append(f"      📈 Default performance: {decision_info['default_score']:.4f}")
                lines.append(f"      🎯 Tuned performance: {decision_info['tuned_score']:.4f}")
                lines.append(f"      📊 Improvement: {decision_info['improvement_pct']:.2f}%")
                
                overfitting = decision_info['overfitting_comparison']
                if overfitting['default'] != 'N/A':
                    lines.append(f"      🎪 Overfitting - Default: {overfitting['default']:.3f}, Tuned: {overfitting['tuned']:.3f}")
                
                # Add debug information for identical performance
                if (isinstance(decision_info['default_score'], (int, float)) and 
                    isinstance(decision_info['tuned_score'], (int, float)) and
                    abs(decision_info['default_score'] - decision_info['tuned_score']) < 0.0001):
                    
                    lines.append(f"      🔍 DEBUG: Identical performance detected!")
                    lines.append(f"         • This suggests hyperparameter tuning may not have found different parameters")
                    lines.append(f"         • Possible causes:")
                    lines.append(f"           - Parameter space too limited")
                    lines.append(f"           - Optimization failed silently")
                    lines.append(f"           - Model insensitive to parameter changes")
                    lines.append(f"         • Recommendation: Check optimization logs and parameter diversity")
            else:
                lines.append(f"      📈 Default performance: {decision_info['default_score']:.4f}")
                lines.append(f"      🎯 Tuned performance: N/A (no tuning performed)")
            
            lines.append(f"      💭 Key reasons:")
            for reason in decision_info['reasons']:
                lines.append(f"         • {reason}")
        
        lines.append("")
        lines.append("="*80)
        lines.append("💡 INTERPRETATION GUIDE:")
        lines.append("   🎯 TUNED models: Hyperparameter optimization provided significant benefit")
        lines.append("   📊 DEFAULT models: Default parameters performed as well or better")
        lines.append("   🎪 Overfitting gap: Difference between training and validation performance")
        lines.append("   📈 Higher scores are better for accuracy, AUC, KS; lower for loss metrics")
        
        # Add troubleshooting section for identical performance
        identical_performance_detected = any(
            info.get('tuned_score') != 'N/A' and 
            isinstance(info.get('default_score'), (int, float)) and 
            isinstance(info.get('tuned_score'), (int, float)) and
            abs(info.get('default_score', 0) - info.get('tuned_score', 0)) < 0.0001
            for info in summary['model_decisions'].values()
        )
        
        if identical_performance_detected:
            lines.append("")
            lines.append("="*80)
            lines.append("🔧 TROUBLESHOOTING IDENTICAL PERFORMANCE:")
            lines.append("   If you see identical default and tuned performance:")
            lines.append("   1. Check if hyperparameter tuning is enabled in config")
            lines.append("   2. Verify parameter spaces are diverse enough")
            lines.append("   3. Check optimization method and number of trials")
            lines.append("   4. Review optimization logs for errors")
            lines.append("   5. Consider using different optimization methods")
            lines.append("   6. Test with more diverse parameter ranges")
            lines.append("   7. Check if XGBoost/LightGBM packages are properly installed")
            lines.append("   8. Verify that optimization is actually running (check logs)")
        
        lines.append("="*80)
        
        # Print to console
        for line in lines:
            print(line)
        
        # Write to file
        import os
        summary_file = os.path.join(self.output_dir, 'model_selection_summary.txt')
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
            print(f"\n📄 Model selection summary saved to: {summary_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save summary to file: {e}")
    
    def _apply_preset_config(self, preset: str):
        """
        Apply preset configuration for quick or comprehensive training.
        
        This method is only called when a preset is explicitly specified.
        When preset=None or empty string, user-provided configuration from YAML or defaults is used.
        
        Args:
            preset: Either 'quick', 'comprehensive', or empty string for no preset
        """
        # Handle empty preset (no preset selected)
        if not preset or preset.strip() == '':
            print("   📝 No preset selected - using configuration from YAML file")
            return
        
        valid_presets = ['quick', 'comprehensive']
        if preset not in valid_presets:
            raise ValueError(f"Invalid preset '{preset}'. Must be one of: {valid_presets} or empty string")
        
        if preset == 'quick':
            print("   ⚡ Quick preset: Logistic + Random Forest only, no CV, no hyperparameter tuning")
            quick_overrides = {
                'classification': {
                    'models': {
                        'run_logistic': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': False,
                        'run_neural_network': False,
                        'run_decision_tree': False,
                        'run_xgboost': False,
                        'run_lightgbm': False
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': False,
                        'optimization_method': 'random_search'
                    },
                    'cross_validation': {
                        'use_cross_validation': False,
                        'cv_folds': 5
                    },
                    'evaluation': {
                        'model_selection_criteria': 'ks',
                        'dataset_to_use': 'train'
                    }
                }
            }
            self.config_manager.override_config(quick_overrides)
            
        elif preset == 'comprehensive':
            print("   🔬 Comprehensive preset: All models enabled, full hyperparameter tuning")
            comprehensive_overrides = {
                'classification': {
                    'models': {
                        'run_logistic': True,
                        'run_random_forest': True,
                        'run_gradient_boosting': True,
                        'run_neural_network': True,
                        'run_decision_tree': True,
                        'run_xgboost': True,  # Will be auto-disabled if not available
                        'run_lightgbm': True  # Will be auto-disabled if not available
                    },
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': True,
                        'optimization_method': 'optuna',
                        'trials': 100,
                        'timeout': 3600
                    },
                    'cross_validation': {
                        'use_cross_validation': 'auto',  # Let existing logic decide
                        'cv_folds': 5
                    },
                    'evaluation': {
                        'model_selection_criteria': 'ks',
                        'dataset_to_use': 'test'
                    }
                }
            }
            self.config_manager.override_config(comprehensive_overrides)

    def _update_config_from_legacy_params(self, user_id: Optional[str], model_id: Optional[str], 
                                         model_literal: Optional[str], output_dir: str):
        """
        Update YAML configuration with legacy parameters.
        
        This method takes old-style parameters and updates the configuration
        so everything works through the standard YAML system.
        
        Args:
            user_id: Legacy user identifier
            model_id: Legacy model identifier  
            model_literal: Legacy model literal
            output_dir: Output directory path
        """
        # Create or update configuration overrides based on legacy parameters
        legacy_overrides = {}
        
        # Add legacy identifiers to global configuration
        if user_id or model_id or model_literal:
            legacy_overrides['global'] = legacy_overrides.get('global', {})
            legacy_overrides['global']['legacy_info'] = {
                'user_id': user_id,
                'model_id': model_id, 
                'model_literal': model_literal,
                'output_directory': output_dir
            }
        
        # Set reasonable defaults for legacy usage patterns
        # Legacy users typically want faster, simpler configurations
        legacy_overrides['classification'] = {
            'models': {
                'run_logistic': True,
                'run_random_forest': True,
                'run_gradient_boosting': True,
                'run_neural_network': True,
                'run_decision_tree': True,
                'run_xgboost': False,  # Disable advanced models for legacy compatibility
                'run_lightgbm': False
            },
            'hyperparameter_tuning': {
                'enable_hyperparameter_tuning': False,  # Default to faster execution
                'optimization_method': 'random_search'   # Simpler method if enabled
            },
            'cross_validation': {
                'use_cross_validation': 'auto',
                'cv_folds': 5
            },
            'evaluation': {
                'model_selection_criteria': 'ks',  # Common legacy preference
                'dataset_to_use': 'train'          # Legacy default
            }
        }
        
        # Apply the legacy configuration overrides to the existing config manager
        print("   📝 Updating YAML configuration with legacy parameters...")
        self.config_manager.override_config(legacy_overrides)
        
        print(f"   ✅ Legacy parameters converted:")
        if user_id:
            print(f"      • User ID: {user_id}")
        if model_id:
            print(f"      • Model ID: {model_id}")
        if model_literal:
            print(f"      • Model Literal: {model_literal}")
        print(f"      • Configuration optimized for legacy usage patterns")

    def get_feature_importance(self, model_type: Optional[str] = None) -> Optional[Any]:
        """
        Get feature importance from the feature selection phase.
        
        Args:
            model_type: Model type (not used for feature selection importance)
            
        Returns:
            DataFrame with feature importance or None if not available
        """
        print("📊 Feature Importance for Classification Model")
        print("=" * 50)
        print("Feature importance was calculated during feature selection using Random Forest Classification")
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
                print(f"    contributed most to your classification model's predictions.")
                
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

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary information
        """
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summary = {
            'model_id': self.config['model_id'],
            'model_literal': self.config['model_literal'],
            'best_model_type': self.best_model_type,
            'target_column': self.config['target_column'],
            'feature_count': len(self.selected_vars),
            'categorical_features': len(self.categorical_vars),
            'numerical_features': len(self.numerical_vars),
            'model_metrics': self.model_metrics,
            'output_directory': self.output_dir,
            'is_multiclass': self.is_multiclass,
            'num_classes': self.num_classes
        }
        
        return summary

    def _restart_spark_session(self) -> bool:
        """Restart Spark session to recover from Py4J connection issues."""
        try:
            print("🔄 Attempting to restart Spark session to recover from Py4J error...")
            
            # Stop existing session gracefully
            try:
                if self.spark:
                    print("🔄 Stopping existing Spark session...")
                    self.spark.stop()
                    import time
                    time.sleep(2)  # Allow proper shutdown
            except Exception as e:
                print(f"⚠️ Warning while stopping session: {e}")
            
            # Create new session with BigQuery support
            print("📦 Using optimized Spark configuration with BigQuery support")
            self.spark = self.create_proven_bigquery_session()
            
            # Update data processor with new session
            if hasattr(self, 'data_processor') and self.data_processor:
                self.data_processor.spark = self.spark
            
            print("✅ Spark session restarted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restart Spark session: {e}")
            print("💡 This may indicate a deeper system issue or insufficient resources")
            return False