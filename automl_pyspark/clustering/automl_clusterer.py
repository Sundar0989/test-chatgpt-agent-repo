"""
AutoML Clusterer

Main class for automated clustering tasks.
"""

import os
import joblib
import numpy as np
from typing import Optional, Dict, Any, List, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel

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
try:
    from ..classification.data_processor import DataProcessor
    from ..config_manager import ConfigManager
except ImportError:
    # For direct script execution
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from classification.data_processor import DataProcessor
    from config_manager import ConfigManager

from .model_builder import ClusteringModelBuilder
from .model_validator import ClusteringModelValidator
from .model_selector import ClusteringModelSelector
from .score_generator import ClusteringScoreGenerator
from .hyperparameter_tuner import ClusteringHyperparameterTuner


class AutoMLClusterer:
    """AutoML Clusterer for automated clustering tasks."""
    
    def __init__(self, 
                 output_dir: str = 'automl_output',
                 config_path: Optional[str] = None,
                 spark_session: Optional[SparkSession] = None,
                 preset: Optional[str] = None,
                 environment: Optional[str] = None,
                 # Backward compatibility parameters
                 user_id: Optional[str] = None,
                 model_id: Optional[str] = None, 
                 model_literal: Optional[str] = None,
                 **kwargs):
        """Initialize the AutoML clusterer."""
        
        # Handle legacy parameters like other classes
        self.actual_user_id = user_id or 'automl_user'
        self.actual_model_id = model_id or 'automl_model_id'
        self.actual_model_literal = model_literal or 'automl_model'
        
        # Initialize Spark session with BigQuery optimization
        if spark_session is None:
            spark_config = get_optimized_spark_config(include_bigquery=True)
            
            # Override with BigQuery-optimized memory settings
            spark_config.update({
                "spark.driver.memory": "8g",  # Increased for BigQuery operations
                "spark.driver.maxResultSize": "4g",  # Increased for BigQuery results
                "spark.executor.memory": "4g",  # Increased for BigQuery processing
            })
            
            builder = SparkSession.builder.appName("AutoML Clustering Pipeline (Optimized)")
            
            # Add BigQuery and SynapseML connector packages
            packages = [
                "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1",
                "com.microsoft.azure:synapseml_2.12:1.0.3"
            ]
            builder = builder.config("spark.jars.packages", ",".join(packages))
            
            for key, value in spark_config.items():
                builder = builder.config(key, value)
            self.spark = builder.getOrCreate()
        else:
            self.spark = spark_session
        
        # Load configuration
        self.config_manager = ConfigManager(config_path, environment)
        
        # Apply preset configuration if specified
        if preset:
            print(f"📋 Applying preset configuration: '{preset}'")
            self.config_manager.apply_preset(preset)
        
        self.config = self.config_manager.get_config()
        
        # Ensure clustering configuration exists with sensible defaults
        if 'clustering' not in self.config or not self.config['clustering']:
            print("🔧 No clustering configuration found - applying defaults...")
            self.config['clustering'] = {
                'models': {
                    'run_kmeans': True,
                    'run_bisecting_kmeans': True,
                    'run_dbscan': False,
                    'run_gaussian_mixture': False,
                    'run_hierarchical': False
                },
                'hyperparameter_tuning': {
                    'enable_hyperparameter_tuning': False,
                    'optimization_method': 'random_search'
                },
                'evaluation': {
                    'model_selection_criteria': 'silhouette'
                }
            }
            print("   ✅ Default clustering configuration applied")
        
        # Ensure models are enabled if they exist but are empty
        elif not self.config['clustering'].get('models'):
            print("🔧 No clustering models configured - enabling defaults...")
            self.config['clustering']['models'] = {
                'run_kmeans': True,
                'run_bisecting_kmeans': True,
                'run_dbscan': False,
                'run_gaussian_mixture': False,
                'run_hierarchical': False
            }
            print("   ✅ Default clustering models enabled")
        
        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor(self.spark, self.actual_user_id, self.actual_model_literal)
        self.model_builder = ClusteringModelBuilder(self.spark)
        self.model_validator = ClusteringModelValidator(self.spark, self.output_dir, self.actual_user_id, self.actual_model_literal)
        self.model_selector = ClusteringModelSelector(self.output_dir, self.actual_user_id, self.actual_model_literal, evaluation_method='silhouette')
        self.score_generator = ClusteringScoreGenerator(self.output_dir, self.actual_user_id, self.actual_model_id, self.actual_model_literal)
        self.hyperparameter_tuner = ClusteringHyperparameterTuner(self.spark, self.output_dir, self.actual_user_id, self.actual_model_literal)
        
        # Initialize model artifacts
        self.feature_vars = []
        self.selected_vars = []
        self.categorical_vars = []
        self.numerical_vars = []
        self.best_model = None
        self.best_model_type = None
        self.model_metrics = {}
        self.pipeline_model = None
        
        print(f"✅ AutoML Clusterer initialized successfully")
    
    def _should_use_cross_validation(self, train_df: DataFrame, valid_df: DataFrame, test_df: DataFrame) -> bool:
        """
        Determine whether to use cross-validation or train/valid/test split.
        
        Args:
            train_df: Training dataset
            valid_df: Validation dataset (can be None)
            test_df: Test dataset (can be None)
            
        Returns:
            bool: True if cross-validation should be used
        """
        # Use cross-validation if no separate validation/test sets are provided
        if valid_df is None and test_df is None:
            print("📊 No validation/test sets provided - using cross-validation")
            return True
            
        # Check if training dataset is small (cross-validation works better with small datasets)
        train_count = train_df.count()
        if train_count < 1000:
            print(f"📊 Small training dataset ({train_count} samples) - using cross-validation")
            return True
            
        # Use train/valid/test split for larger datasets with separate validation sets
        print(f"📊 Using train/validation/test split for clustering evaluation")
        return False
    
    def _validate_with_cross_validation(self, model, train_df: DataFrame, model_type: str, 
                                      k_range: List[int] = None, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Validate clustering model using cross-validation.
        
        Args:
            model: Trained clustering model
            train_df: Training dataset
            model_type: Type of clustering model
            k_range: Range of k values tested
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation metrics
        """
        print(f"   📊 Performing {cv_folds}-fold cross-validation for {model_type}...")
        
        try:
            from pyspark.sql.functions import monotonically_increasing_id, rand
            
            # Add random column for splitting
            df_with_id = train_df.withColumn("id", monotonically_increasing_id()).withColumn("rand", rand(seed=42))
            
            # Perform k-fold cross-validation
            cv_metrics = []
            fold_size = 1.0 / cv_folds
            
            for fold in range(cv_folds):
                print(f"     Fold {fold + 1}/{cv_folds}...")
                
                # Split data for this fold
                start_range = fold * fold_size
                end_range = (fold + 1) * fold_size
                
                # Create train and validation sets for this fold
                cv_train = df_with_id.filter((df_with_id.rand < start_range) | (df_with_id.rand >= end_range))
                cv_val = df_with_id.filter((df_with_id.rand >= start_range) & (df_with_id.rand < end_range))
                
                # Remove helper columns
                cv_train = cv_train.drop("id", "rand")
                cv_val = cv_val.drop("id", "rand")
                
                # Train model on cv_train and validate on cv_val
                if model_type == 'kmeans':
                    from pyspark.ml.clustering import KMeans
                    best_k = model._automl_metadata.get('best_k', 3) if hasattr(model, '_automl_metadata') else 3
                    cv_model = KMeans(featuresCol='features', predictionCol="prediction", k=best_k, seed=42)
                elif model_type == 'bisecting_kmeans':
                    from pyspark.ml.clustering import BisectingKMeans
                    best_k = model._automl_metadata.get('best_k', 3) if hasattr(model, '_automl_metadata') else 3
                    cv_model = BisectingKMeans(featuresCol='features', predictionCol="prediction", k=best_k, seed=42)
                elif model_type == 'gaussian_mixture':
                    from pyspark.ml.clustering import GaussianMixture
                    best_k = model._automl_metadata.get('best_k', 3) if hasattr(model, '_automl_metadata') else 3
                    cv_model = GaussianMixture(featuresCol='features', predictionCol="prediction", k=best_k, seed=42)
                elif model_type == 'dbscan':
                    # DBSCAN doesn't use k parameter, use the best parameters from the model
                    best_params = model._automl_metadata.get('best_params', {'eps': 0.5, 'minPts': 5}) if hasattr(model, '_automl_metadata') else {'eps': 0.5, 'minPts': 5}
                    cv_model = self.model_builder._create_dbscan_model(best_params['eps'], best_params['minPts'])
                else:
                    print(f"     ⚠️ Cross-validation not implemented for {model_type}")
                    continue
                
                # Fit and predict
                fitted_cv_model = cv_model.fit(cv_train)
                cv_predictions = fitted_cv_model.transform(cv_val)
                
                # Calculate metrics for this fold
                fold_metrics = self.model_validator._calculate_comprehensive_metrics(cv_predictions, model_type)
                fold_metrics['fold'] = fold + 1
                fold_metrics['train_samples'] = cv_train.count()
                fold_metrics['val_samples'] = cv_val.count()
                
                cv_metrics.append(fold_metrics)
                
            # Aggregate cross-validation results
            cv_summary = self._aggregate_cv_results(cv_metrics)
            cv_summary['cv_folds'] = cv_folds
            cv_summary['validation_type'] = 'cross_validation'
            
            print(f"   ✅ Cross-validation completed: {cv_summary.get('mean_silhouette_score', 'N/A'):.4f} ± {cv_summary.get('std_silhouette_score', 'N/A'):.4f}")
            
            return cv_summary
            
        except Exception as e:
            print(f"   ⚠️ Error in cross-validation: {str(e)}")
            import traceback
            print(f"   🐛 Cross-validation error traceback:")
            traceback.print_exc()
            return {'error': str(e), 'validation_type': 'cross_validation'}
    
    def _aggregate_cv_results(self, cv_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results across folds."""
        
        aggregated = {}
        
        # Metrics to aggregate
        metrics_to_aggregate = ['silhouette_score', 'inertia', 'calinski_harabasz_score', 'davies_bouldin_score']
        
        for metric in metrics_to_aggregate:
            values = [fold.get(metric) for fold in cv_metrics if fold.get(metric) is not None]
            if values:
                aggregated[f'mean_{metric}'] = float(np.mean(values))
                aggregated[f'std_{metric}'] = float(np.std(values))
                aggregated[f'min_{metric}'] = float(np.min(values))
                aggregated[f'max_{metric}'] = float(np.max(values))
        
        # Add fold-specific results
        aggregated['fold_results'] = cv_metrics
        aggregated['n_folds'] = len(cv_metrics)
        
        return aggregated
    
    def _save_cv_metrics_for_compatibility(self, cv_metrics: Dict[str, Any], model_type: str):
        """Save cross-validation metrics in standard format for compatibility with reporting system."""
        
        print(f"   📄 Saving CV compatibility metrics for {model_type}...")
        
        try:
            # Create standard metrics format from cross-validation results
            standard_metrics = {}
            
            # Use mean values as the primary metrics
            if 'mean_silhouette_score' in cv_metrics:
                standard_metrics['silhouette_score'] = cv_metrics['mean_silhouette_score']
            if 'mean_inertia' in cv_metrics:
                standard_metrics['inertia'] = cv_metrics['mean_inertia']
            if 'mean_calinski_harabasz_score' in cv_metrics:
                standard_metrics['calinski_harabasz_score'] = cv_metrics['mean_calinski_harabasz_score']
            if 'mean_davies_bouldin_score' in cv_metrics:
                standard_metrics['davies_bouldin_score'] = cv_metrics['mean_davies_bouldin_score']
            
            # Add cross-validation specific metadata
            standard_metrics['validation_type'] = 'cross_validation'
            standard_metrics['cv_folds'] = cv_metrics.get('cv_folds', 'unknown')
            standard_metrics['metric_source'] = 'cross_validation'
            
            # Add standard deviations for uncertainty quantification
            if 'std_silhouette_score' in cv_metrics:
                standard_metrics['silhouette_score_std'] = cv_metrics['std_silhouette_score']
            if 'std_inertia' in cv_metrics:
                standard_metrics['inertia_std'] = cv_metrics['std_inertia']
            
            # Generate plots for the training data (for visualization)
            try:
                print(f"   🎨 Generating plots for CV compatibility...")
                print(f"      📊 Best model available: {self.best_model is not None}")
                print(f"      📊 Training data available: {self.train_df is not None}")
                
                if self.best_model is None:
                    print(f"      ❌ No best model available for plot generation")
                    standard_metrics['validation_plots'] = []
                elif self.train_df is None:
                    print(f"      ❌ No training data available for plot generation")
                    standard_metrics['validation_plots'] = []
                else:
                    predictions = self.best_model.transform(self.train_df)
                    print(f"      📊 Generated predictions for plot generation")
                    
                    # Use test_k_range if available, otherwise get from model metadata
                    k_range_for_plots = None
                    if hasattr(self.best_model, '_automl_metadata'):
                        k_range_for_plots = self.best_model._automl_metadata.get('k_range_tested')
                        print(f"      📊 Using k_range from model metadata: {k_range_for_plots}")
                    
                    print(f"      🎨 Calling plot generation method...")
                    plot_files = self.model_validator._generate_clustering_plots(predictions, model_type, k_range_for_plots)
                    print(f"      📊 Plot generation completed!")
                    print(f"      📂 Generated {len(plot_files)} CV plots: {plot_files}")
                    standard_metrics['validation_plots'] = plot_files
                    
                    # Also check if plot files actually exist
                    existing_plots = []
                    for plot_file in plot_files:
                        if os.path.exists(plot_file):
                            existing_plots.append(plot_file)
                            print(f"      ✅ Plot exists: {plot_file}")
                        else:
                            print(f"      ❌ Plot missing: {plot_file}")
                    
                    print(f"      📈 {len(existing_plots)}/{len(plot_files)} plots actually saved")
                    
            except Exception as e:
                print(f"   ⚠️ Could not generate plots for CV: {str(e)}")
                import traceback
                print(f"   🐛 CV plot error traceback:")
                traceback.print_exc()
                standard_metrics['validation_plots'] = []
            
            # Save using the standard method
            self.model_validator._save_metrics(standard_metrics, model_type)
            
            print(f"   📄 Cross-validation metrics saved in standard format")
            
        except Exception as e:
            print(f"   ⚠️ Error saving CV compatibility metrics: {str(e)}")
    
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
            print(f"🚀 Creating BigQuery-optimized Spark session for clustering...")
            print(f"   💾 Driver memory: {driver_memory}")
            print(f"   🔗 BigQuery connector: v0.36.1")
            
            # Stop existing session if needed
            if hasattr(self, 'spark') and self.spark is not None:
                print("🔄 Stopping existing Spark session for BigQuery compatibility...")
                self.spark.stop()
            
            # Build Spark session with proven BigQuery configuration
            builder = SparkSession.builder.appName("AutoML BigQuery Clusterer")
            
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
            if hasattr(self, 'hyperparameter_tuner'):
                self.hyperparameter_tuner.spark = spark
            
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
        print(f"🔗 Loading BigQuery data for clustering...")
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
                         k_range: Optional[List[int]] = None,
                         limit_rows: Optional[int] = None,
                         driver_memory: str = "64g",
                         validation_data: Optional[Union[str, DataFrame]] = None,
                         test_data: Optional[Union[str, DataFrame]] = None,
                         oot1_data: Optional[Union[str, DataFrame]] = None,
                         oot2_data: Optional[Union[str, DataFrame]] = None,
                         use_cross_validation: bool = False,
                         **kwargs) -> 'AutoMLClusterer':
        """
        Fit AutoML clusterer directly from BigQuery tables with train/validation/test support.
        Uses the proven working BigQuery configuration.
        
        Args:
            project_id: Your GCP project ID (e.g., "atus-prism-dev")
            table_id: Training data table reference (e.g., "atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02")
            k_range: Range of k values to test for clustering (optional)
            limit_rows: Limit number of rows for testing (optional)
            driver_memory: Driver memory allocation (default: 64g for BigQuery)
            validation_data: Optional separate validation data
            test_data: Optional separate test data
            oot1_data: Optional out-of-time dataset 1
            oot2_data: Optional out-of-time dataset 2
            use_cross_validation: Whether to use cross-validation instead of train/val/test split
            **kwargs: Additional configuration parameters
            
        Returns:
            self: Fitted AutoML clusterer
            
        Example:
            # Basic usage with automatic data splitting
            automl = AutoMLClusterer()
            results = automl.fit_from_bigquery(
                project_id="atus-prism-dev",
                table_id="atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02",
                k_range=[2, 3, 4, 5],  # Test different cluster numbers
                limit_rows=10000,  # Optional: for testing
                train_ratio=0.7,  # 70% for training
                valid_ratio=0.15,  # 15% for validation  
                test_ratio=0.15   # 15% for testing
            )
            
            # Using cross-validation
            automl = AutoMLClusterer()
            results = automl.fit_from_bigquery(
                project_id="atus-prism-dev",
                table_id="atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02",
                k_range=[2, 3, 4, 5],
                use_cross_validation=True,
                cv_folds=5
            )
        """
        print("🚀 Starting AutoML clustering with BigQuery data...")
        
        # 1. Create BigQuery-optimized Spark session
        self.create_bigquery_spark_session(driver_memory=driver_memory, include_lightgbm=True)
        
        # 2. Load training data from BigQuery
        print("📊 Loading training data from BigQuery...")
        train_data = self.load_bigquery_data(project_id, table_id, limit_rows)
        
        # 3. Run AutoML fit with loaded data
        print("🤖 Starting AutoML clustering...")
        return self.fit(train_data, k_range=k_range, validation_data=validation_data, 
                       test_data=test_data, oot1_data=oot1_data, oot2_data=oot2_data,
                       use_cross_validation=use_cross_validation, **kwargs)
    
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
    
    def fit(self, train_data: Union[str, DataFrame], 
            k_range: Optional[List[int]] = None,
            validation_data: Optional[Union[str, DataFrame]] = None,
            test_data: Optional[Union[str, DataFrame]] = None,
            oot1_data: Optional[Union[str, DataFrame]] = None,
            oot2_data: Optional[Union[str, DataFrame]] = None,
            use_cross_validation: bool = False,
            use_elbow_method: bool = True,
            evaluation_method: str = 'silhouette',
            **kwargs) -> 'AutoMLClusterer':
        """
        Fit the AutoML clusterer on the data with proper train/validation/test splits.
        
        Args:
            train_data: Training data (file path or DataFrame)
            k_range: Range of k values to test for clustering
            validation_data: Optional separate validation data
            test_data: Optional separate test data
            oot1_data: Optional out-of-time dataset 1
            oot2_data: Optional out-of-time dataset 2
            use_cross_validation: Whether to use cross-validation instead of train/val/test split
            use_elbow_method: Whether to use elbow method for optimal k selection
            evaluation_method: Method to evaluate clustering quality ('silhouette', 'calinski_harabasz', 'davies_bouldin')
            **kwargs: Additional parameters
        """
        print("Starting AutoML clustering pipeline with train/validation/test splits...")
        
        # Store configuration parameters
        self.use_elbow_method = use_elbow_method
        self.evaluation_method = evaluation_method
        
        print(f"🎯 Clustering Configuration:")
        print(f"   • Use Elbow Method: {use_elbow_method}")
        print(f"   • Evaluation Method: {evaluation_method}")
        print(f"   • K Range: {k_range}")
        
        # Update model selector with evaluation method
        if hasattr(self, 'model_selector'):
            self.model_selector.evaluation_method = evaluation_method
            print(f"   📊 Model selector updated to use {evaluation_method} evaluation method")
        
        # Load data if string path is provided and detect BigQuery tables
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
        
        # Process data (no target column for clustering)
        print("\n📊 Loading and processing data...")
        
        # For clustering, we don't need to enforce train/val/test splits through target_column
        # but we can still benefit from data preprocessing
        processed_data = self.data_processor.process_data(
            train_data, target_column=None, **kwargs
        )
        
        # Extract datasets - for clustering without target, all data goes to 'train' by default
        # We'll manually split if needed
        full_processed_data = processed_data['train']
        
        # Determine data splitting strategy
        if validation_data is not None or test_data is not None or oot1_data is not None or oot2_data is not None:
            # Use provided validation/test/oot sets
            print("📊 Using provided validation/test/OOT datasets...")
            train_df = full_processed_data
            
            # Process validation data if provided
            if validation_data is not None:
                if isinstance(validation_data, str):
                    validation_data = self.spark.read.option("header", "true").option("inferSchema", "true").csv(validation_data)
                val_processed = self.data_processor.pipeline_model.transform(validation_data)
                valid_df = val_processed
            else:
                valid_df = None
                
            # Process test data if provided
            if test_data is not None:
                if isinstance(test_data, str):
                    test_data = self.spark.read.option("header", "true").option("inferSchema", "true").csv(test_data)
                test_processed = self.data_processor.pipeline_model.transform(test_data)
                test_df = test_processed
            else:
                test_df = None
                
            # Process OOT1 data if provided
            if oot1_data is not None:
                if isinstance(oot1_data, str):
                    oot1_data = self.spark.read.option("header", "true").option("inferSchema", "true").csv(oot1_data)
                oot1_processed = self.data_processor.pipeline_model.transform(oot1_data)
                oot1_df = oot1_processed
            else:
                oot1_df = None
                
            # Process OOT2 data if provided
            if oot2_data is not None:
                if isinstance(oot2_data, str):
                    oot2_data = self.spark.read.option("header", "true").option("inferSchema", "true").csv(oot2_data)
                oot2_processed = self.data_processor.pipeline_model.transform(oot2_data)
                oot2_df = oot2_processed
            else:
                oot2_df = None
                
        elif use_cross_validation:
            # Use cross-validation approach
            print("📊 Using cross-validation approach...")
            train_df = full_processed_data
            valid_df = None
            test_df = None
            
        else:
            # Split the data into train/validation/test
            print("📊 Splitting data into train/validation/test sets...")
            
            # Default split ratios
            train_ratio = kwargs.get('train_ratio', 0.7)
            valid_ratio = kwargs.get('valid_ratio', 0.15)
            test_ratio = kwargs.get('test_ratio', 0.15)
            
            # Ensure ratios sum to 1
            total_ratio = train_ratio + valid_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.01:
                print(f"⚠️ Adjusting ratios to sum to 1.0 (was {total_ratio})")
                train_ratio = train_ratio / total_ratio
                valid_ratio = valid_ratio / total_ratio
                test_ratio = test_ratio / total_ratio
            
            # Split the data
            seed = kwargs.get('seed', 42)
            train_df, temp_df = full_processed_data.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
            
            if valid_ratio > 0 and test_ratio > 0:
                # Split temp into validation and test
                valid_test_ratio = valid_ratio / (valid_ratio + test_ratio)
                valid_df, test_df = temp_df.randomSplit([valid_test_ratio, 1 - valid_test_ratio], seed=seed)
            elif valid_ratio > 0:
                valid_df = temp_df
                test_df = None
            elif test_ratio > 0:
                valid_df = None
                test_df = temp_df
            else:
                valid_df = None
                test_df = None
            
            # No OOT data when using automatic splitting
            oot1_df = None
            oot2_df = None
                
        # Print dataset sizes
        train_count = train_df.count()
        print(f"✅ Training set: {train_count:,} samples")
        
        if valid_df is not None:
            valid_count = valid_df.count()
            print(f"✅ Validation set: {valid_count:,} samples")
        else:
            print("ℹ️ No validation set")
            
        if test_df is not None:
            test_count = test_df.count()
            print(f"✅ Test set: {test_count:,} samples")
        else:
            print("ℹ️ No test set")
            
        if oot1_df is not None:
            oot1_count = oot1_df.count()
            print(f"✅ OOT1 set: {oot1_count:,} samples")
        else:
            print("ℹ️ No OOT1 set")
            
        if oot2_df is not None:
            oot2_count = oot2_df.count()
            print(f"✅ OOT2 set: {oot2_count:,} samples")
        else:
            print("ℹ️ No OOT2 set")
        
        # Store preprocessing artifacts
        self.feature_vars = self.data_processor.feature_vars
        self.selected_vars = self.data_processor.selected_vars
        self.categorical_vars = self.data_processor.categorical_vars
        self.numerical_vars = self.data_processor.numerical_vars
        self.pipeline_model = self.data_processor.pipeline_model
        
        # Store datasets for validation
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.oot1_df = oot1_df
        self.oot2_df = oot2_df
        self.use_cross_validation = use_cross_validation
        
        # Get clustering configuration
        clustering_config = self.config.get('clustering', {})
        models_config = clustering_config.get('models', {})
        
        # Train clustering models
        print("\n🤖 Building clustering models...")
        print(f"   📋 Models configuration: {models_config}")
        
        enabled_models = [name for name, enabled in models_config.items() if enabled]
        print(f"   ✅ Enabled models: {enabled_models}")
        
        if not enabled_models:
            print("   ❌ No models are enabled in configuration!")
            raise RuntimeError("No clustering models are enabled in configuration")
        
        trained_models = {}
        
        for model_name, should_run in models_config.items():
            print(f"   🔍 Checking model: {model_name} = {should_run}")
            if should_run:
                model_type = model_name.replace('run_', '')
                try:
                    print(f"\n🔧 Training {model_type} model on training data...")
                    
                    # Tune hyperparameters on training data only
                    # Merge clustering config with advanced params (contains Streamlit optuna_trials setting)
                    clustering_config = self.config.get('clustering', {}).copy()
                    
                    # Advanced params are flattened into kwargs by background_job_manager
                    hp_settings = {k: v for k, v in kwargs.items() 
                                 if k in ['optuna_trials', 'optuna_timeout', 'random_trials', 'hp_method', 'enable_hyperparameter_tuning'] and v is not None}
                    if hp_settings:
                        if 'hyperparameter_tuning' not in clustering_config:
                            clustering_config['hyperparameter_tuning'] = {}
                        clustering_config['hyperparameter_tuning'].update(hp_settings)
                        print(f"   📋 Using Streamlit hyperparameter settings: {hp_settings}")
                    
                    # Extract clustering hyperparameter ranges from advanced_params (passed via kwargs)
                    clustering_hp_ranges = kwargs.get('clustering_hp_ranges', {})
                    if clustering_hp_ranges:
                        clustering_config['clustering_hp_ranges'] = clustering_hp_ranges
                        print(f"   📋 Using Streamlit clustering hyperparameter ranges: {clustering_hp_ranges}")
                    
                    tuning_result = self.hyperparameter_tuner.tune_hyperparameters(
                        model_type, train_df, clustering_config, k_range
                    )
                    
                    # Always test all k values to enable proper elbow plot generation
                    test_k_range = k_range or [2, 3, 4, 5]
                    
                    if tuning_result.get('best_params'):
                        print(f"   🎯 Using optimized parameters: {tuning_result['best_params']}")
                        # Still test all k values but with optimized other parameters
                        other_params = {k: v for k, v in tuning_result['best_params'].items() if k != 'k'}
                        model = self.model_builder.build_model(
                            train_df, 'features', model_type, 
                            k_range=test_k_range,
                            use_elbow_method=self.use_elbow_method,
                            **other_params
                        )
                    else:
                        # Build model with default parameters, testing all k values
                        model = self.model_builder.build_model(
                            train_df, 'features', model_type, 
                            k_range=test_k_range,
                            use_elbow_method=self.use_elbow_method
                        )
                    
                    # ALWAYS perform multi-dataset validation when datasets are available
                    print(f"   📊 Validating {model_type} model on different datasets...")
                    
                    # Prepare datasets for validation
                    validation_datasets = [train_df]
                    dataset_names = ['train']
                    
                    if valid_df is not None:
                        validation_datasets.append(valid_df)
                        dataset_names.append('validation')
                        
                    if test_df is not None:
                        validation_datasets.append(test_df)
                        dataset_names.append('test')
                        
                    if oot1_df is not None:
                        validation_datasets.append(oot1_df)
                        dataset_names.append('oot1')
                        
                    if oot2_df is not None:
                        validation_datasets.append(oot2_df)
                        dataset_names.append('oot2')
                    
                    # Validate on all datasets
                    metrics = self.model_validator.validate_model_multiple_datasets(
                        model, validation_datasets, dataset_names, model_type, self.output_dir, k_range=test_k_range
                    )
                    
                    # ADDITIONALLY perform cross-validation if requested or for small datasets
                    should_use_cv = use_cross_validation or self._should_use_cross_validation(train_df, valid_df, test_df)
                    
                    if should_use_cv:
                        print(f"   📊 Additionally performing cross-validation for {model_type}...")
                        cv_folds = kwargs.get('cv_folds', 5)
                        cv_metrics = self._validate_with_cross_validation(
                            model, train_df, model_type, k_range=test_k_range, cv_folds=cv_folds
                        )
                        
                        # Add cross-validation results to the metrics
                        metrics['cross_validation'] = cv_metrics
                        print(f"   ✅ Both multi-dataset and cross-validation completed for {model_type}")
                    
                    # Add model selection explanation
                    if hasattr(model, '_automl_metadata'):
                        metadata = model._automl_metadata
                        print(f"   📊 Model Selection Summary:")
                        print(f"      • Tested k values: {metadata['k_range_tested']}")
                        print(f"      • Best k chosen: {metadata['best_k']}")
                        print(f"      • Best silhouette score: {metadata['best_score']:.4f}")
                        print(f"      • Reason: Highest silhouette score among {len(metadata['all_results'])} tested models")
                        
                        # Add detailed results to metrics
                        metrics['k_selection_details'] = {
                            'k_range_tested': metadata['k_range_tested'],
                            'best_k': metadata['best_k'],
                            'best_score': metadata['best_score'],
                            'all_k_results': [
                                {'k': r['k'], 'silhouette_score': r['silhouette_score']} 
                                for r in metadata['all_results']
                            ]
                        }
                    
                    # Add tuning information to metrics
                    metrics.update({
                        'hyperparameter_tuning': tuning_result,
                        'optimization_method': tuning_result.get('optimization_method', 'default'),
                        'n_trials': tuning_result.get('n_trials', 0)
                    })
                    
                    trained_models[model_type] = {
                        'model': model,
                        'metrics': metrics
                    }
                    
                    print(f"✅ {model_type} model completed")
                    
                except Exception as e:
                    print(f"❌ Error training {model_type}: {str(e)}")
                    print(f"   🐛 Full error details:")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not trained_models:
            raise RuntimeError("No clustering models were trained successfully")
        
        # Select best model
        print("\n🏆 Selecting best clustering model...")
        try:
            selection_result = self.model_selector.select_best_model(
                trained_models, clustering_config.get('evaluation', {})
            )
            
            self.best_model_type = selection_result['best_model_type']
            
            # Safety check for best_model_type
            if self.best_model_type is None or self.best_model_type not in trained_models:
                available_models = list(trained_models.keys())
                if available_models:
                    # Fallback to first available model
                    self.best_model_type = available_models[0]
                    print(f"⚠️ Model selection failed, using fallback model: {self.best_model_type}")
                else:
                    raise RuntimeError("No trained models available for selection")
            
            self.best_model = trained_models[self.best_model_type]['model']
            self.model_metrics = selection_result['metrics']
            
        except Exception as e:
            print(f"❌ Error in model selection: {str(e)}")
            # Emergency fallback: use the first available model
            if trained_models:
                fallback_model_type = list(trained_models.keys())[0]
                print(f"🚨 Using emergency fallback model: {fallback_model_type}")
                self.best_model_type = fallback_model_type
                self.best_model = trained_models[fallback_model_type]['model']
                self.model_metrics = trained_models[fallback_model_type]['metrics']
            else:
                raise RuntimeError("No clustering models were trained successfully and no fallback available")
        
        # Store all trained models for saving
        self.trained_models = trained_models
        
        # Save additional cross-validation metrics if they were generated
        print(f"🔍 Checking for cross-validation results...")
        for model_type, model_data in trained_models.items():
            metrics = model_data['metrics']
            if 'cross_validation' in metrics:
                print(f"   📊 Found cross-validation results for {model_type}")
                cv_metrics = metrics['cross_validation']
                
                # Create supplementary CV metrics file
                cv_filename = f"{model_type}_cross_validation_metrics.json"
                cv_filepath = os.path.join(self.output_dir, cv_filename)
                
                try:
                    import json
                    with open(cv_filepath, 'w') as f:
                        json.dump(cv_metrics, f, indent=2)
                    print(f"   📄 Cross-validation metrics saved: {cv_filename}")
                except Exception as e:
                    print(f"   ⚠️ Error saving CV metrics for {model_type}: {str(e)}")
            else:
                print(f"   ℹ️ No cross-validation results for {model_type}")
        
        # IMPORTANT: Generate plots and comprehensive metrics for ALL trained models
        print(f"\n🎨 Generating final validation plots and metrics for all models...")
        for model_type, model_data in trained_models.items():
            try:
                model = model_data['model']
                print(f"   🎨 Processing {model_type} model...")
                
                if model and self.train_df:
                    # Generate comprehensive validation datasets
                    validation_datasets = [self.train_df]
                    dataset_names = ['train']
                    
                    if valid_df is not None:
                        validation_datasets.append(valid_df)
                        dataset_names.append('validation')
                        
                    if test_df is not None:
                        validation_datasets.append(test_df)
                        dataset_names.append('test')
                        
                    if oot1_df is not None:
                        validation_datasets.append(oot1_df)
                        dataset_names.append('oot1')
                        
                    if oot2_df is not None:
                        validation_datasets.append(oot2_df)
                        dataset_names.append('oot2')
                    
                    print(f"      📊 Validating {model_type} on {len(validation_datasets)} datasets: {dataset_names}")
                    
                    # Generate plots and metrics for each dataset
                    all_plot_files = []
                    dataset_metrics = {}
                    
                    for dataset, dataset_name in zip(validation_datasets, dataset_names):
                        print(f"      🔍 Processing {dataset_name} dataset for {model_type}...")
                        
                        # Generate predictions
                        predictions = model.transform(dataset)
                        
                        # Calculate metrics for this dataset
                        metrics_for_dataset = self.model_validator._calculate_comprehensive_metrics(predictions, model_type)
                        metrics_for_dataset['dataset'] = dataset_name
                        metrics_for_dataset['sample_count'] = dataset.count()
                        dataset_metrics[dataset_name] = metrics_for_dataset
                        
                        # Generate plots for this dataset
                        k_range_for_plots = None
                        if hasattr(model, '_automl_metadata'):
                            k_range_for_plots = model._automl_metadata.get('k_range_tested')
                        
                        dataset_plots = self.model_validator._generate_clustering_plots_for_dataset(
                            predictions, model_type, dataset_name, k_range_for_plots
                        )
                        all_plot_files.extend(dataset_plots)
                        
                        print(f"         ✅ {dataset_name}: {len(dataset_plots)} plots, {len(metrics_for_dataset)} metrics")
                    
                    # Create comprehensive metrics structure
                    final_metrics = {
                        'datasets': dataset_metrics,
                        'validation_plots': all_plot_files,
                        'validation_type': 'comprehensive_multi_dataset',
                        'model_type': model_type,
                        'total_datasets': len(validation_datasets)
                    }
                    
                    # Add model metadata
                    if hasattr(model, '_automl_metadata'):
                        metadata = model._automl_metadata
                        final_metrics.update({
                            'best_k': metadata.get('best_k'),
                            'best_score': metadata.get('best_score'),
                            'k_range_tested': metadata.get('k_range_tested')
                        })
                        
                        # Use validation dataset score if available, otherwise use model's best score
                        if 'validation' in dataset_metrics:
                            final_metrics['silhouette_score'] = dataset_metrics['validation'].get('silhouette_score', metadata.get('best_score'))
                        elif 'test' in dataset_metrics:
                            final_metrics['silhouette_score'] = dataset_metrics['test'].get('silhouette_score', metadata.get('best_score'))
                        else:
                            final_metrics['silhouette_score'] = dataset_metrics['train'].get('silhouette_score', metadata.get('best_score'))
                    
                    # Save comprehensive metrics to standard format
                    self.model_validator._save_metrics(final_metrics, model_type)
                    
                    print(f"   ✅ {model_type}: Generated {len(all_plot_files)} plots for {len(validation_datasets)} datasets")
                    print(f"      📄 Comprehensive metrics saved for {model_type}")
                    
                else:
                    print(f"   ⚠️ Cannot process {model_type} - missing model or data")
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {model_type}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Enhanced reporting of model selection
        print(f"\n🥇 Best Overall Model: {self.best_model_type}")
        
        # Show comparison of all models
        print(f"\n📊 Model Comparison Summary:")
        for model_type, model_data in trained_models.items():
            model = model_data['model']
            metrics = model_data['metrics']
            
            is_best = "🥇" if model_type == self.best_model_type else "  "
            print(f"{is_best} {model_type}:")
            
            if hasattr(model, '_automl_metadata'):
                metadata = model._automl_metadata
                print(f"      • Best k: {metadata['best_k']}")
                print(f"      • Silhouette Score: {metadata['best_score']:.4f}")
                print(f"      • Tested k values: {metadata['k_range_tested']}")
            
            # Show other metrics if available
            if 'silhouette_score' in metrics:
                print(f"      • Final Silhouette: {metrics['silhouette_score']:.4f}")
        
        # Show why this model was chosen
        if hasattr(self.best_model, '_automl_metadata'):
            best_metadata = self.best_model._automl_metadata
            print(f"\n🎯 Selection Rationale:")
            print(f"   • {self.best_model_type} achieved the highest silhouette score ({best_metadata['best_score']:.4f})")
            print(f"   • Optimal number of clusters: {best_metadata['best_k']}")
            print(f"   • This was determined by testing k = {best_metadata['k_range_tested']}")
            print(f"   • Elbow plots and detailed analysis available in output directory")
        
        # Generate scoring scripts for ALL trained models
        print("\n📝 Generating scoring scripts...")
        if hasattr(self, 'trained_models') and self.trained_models:
            for model_type in self.trained_models.keys():
                print(f"📝 Generating clustering scoring code for {model_type}")
                self.score_generator.generate_scoring_code(
                    self.config, self.feature_vars, self.selected_vars,
                    self.categorical_vars, self.numerical_vars, model_type
                )
            print(f"✅ Generated {len(self.trained_models)} clustering scoring scripts in {self.output_dir}")
        else:
            # Fallback for best model only
            self.score_generator.generate_scoring_code(
                self.config, self.feature_vars, self.selected_vars,
                self.categorical_vars, self.numerical_vars, self.best_model_type
            )
        
        # Save model
        print(f"\n💾 Saving model...")
        self.save_model(self.output_dir)

        # Step: Compute SHAP values for explainability (only for best model)
        if self.best_model is not None and self.best_model_type is not None:
            try:
                import sys
                import os
                # Add the parent directory to the path for absolute import
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from explainability import compute_shap_values
                print(f"\n🔍 Computing SHAP values for best model ({self.best_model_type}) explainability...")
                # Use the processed training DataFrame for SHAP.  Use
                # selected_vars if available, otherwise all feature_vars.
                sample_df = train_df
                feature_cols = self.selected_vars if self.selected_vars else self.feature_vars
                compute_shap_values(
                    spark=self.spark,
                    pipeline_model=self.pipeline_model,
                    model=self.best_model,
                    sample_df=sample_df,
                    feature_cols=feature_cols,
                    output_dir=self.output_dir,
                    model_type="clustering",
                    max_samples=50,
                )
                print(f"✅ SHAP values computed successfully for best model ({self.best_model_type})")
            except Exception as e:
                if "No data available" in str(e):
                    print("⚠️ Insufficient data for SHAP computation (no training data or feature columns available)")
                else:
                    print(f"⚠️ SHAP computation skipped for best model ({self.best_model_type}): {e}")
        else:
            print("⚠️ SHAP computation skipped: No best model available")

        # Store the original training data path for profiling
        self.train_data_path = train_data

        # Automatically generate cluster profiling
        try:
            print(f"\n🔬 Generating Cluster Profiling...")
            profiling_results = self.profile_clusters(
                original_data_path=self.train_data_path,
                save_results=True
            )
            print(f"✅ Cluster profiling completed and saved!")
            self.profiling_results = profiling_results
        except Exception as e:
            print(f"⚠️ Could not generate cluster profiling: {str(e)}")
            self.profiling_results = None

        print("\nAutoML clustering pipeline completed successfully!")
        return self
    
    def save_model(self, path: str):
        """Save the fitted clustering model."""
        if self.best_model is None:
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
            'task_type': 'clustering'
        }
        
        joblib.dump(model_info, os.path.join(path, 'model_info.pkl'))
        print(f"✅ Clustering model info saved")
        
        # Save preprocessing pipeline
        if self.pipeline_model:
            self.pipeline_model.write().overwrite().save(os.path.join(path, 'preprocessing_pipeline'))
            print(f"✅ Preprocessing pipeline saved")
        
        # Save categorical encoder if available (consistent with regression/classification)
        if hasattr(self.data_processor, 'char_labels') and self.data_processor.char_labels:
            self.data_processor.char_labels.write().overwrite().save(os.path.join(path, 'char_labels'))
            print(f"✅ Categorical encoder saved")
        
        # Save ALL trained models (not just the best one)
        if hasattr(self, 'trained_models') and self.trained_models:
            print(f"   🔄 Saving all {len(self.trained_models)} trained clustering models...")
            
            for model_type, model_data in self.trained_models.items():
                model = model_data['model']
                try:
                    print(f"   🔄 Saving {model_type} clustering model...")
                    model_path = os.path.join(path, f'{model_type}_model')
                    print(f"   📂 Model path: {model_path}")
                    
                    # Handle different model types
                    if model_type == 'dbscan':
                        # DBSCAN models are sklearn models, save as pickle
                        import pickle
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        print(f"✅ {model_type} clustering model saved as pickle to {model_path}")
                    else:
                        # Standard PySpark models
                        model.write().overwrite().save(model_path)
                        print(f"✅ {model_type} clustering model saved to {model_path}")
                    
                    # Verify the model was actually saved
                    if os.path.exists(model_path):
                        if model_type == 'dbscan':
                            print(f"   📄 Model file created: {model_path}")
                        else:
                            model_files = os.listdir(model_path)
                            print(f"   📄 Model files created: {len(model_files)} files")
                        
                        # Mark best model
                        if model_type == self.best_model_type:
                            print(f"   🥇 {model_type} is the BEST model (silhouette: {self.model_metrics.get('silhouette_score', 'N/A')})")
                    else:
                        print(f"   ⚠️ Model was not created: {model_path}")
                        
                except Exception as e:
                    print(f"❌ FAILED to save {model_type} clustering model!")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Error message: {str(e)}")
                    print(f"   Model path attempted: {model_path}")
                    print(f"   Model type: {type(model)}")
                    # Continue with other models instead of failing completely
                    continue
                    
        elif self.best_model is not None:
            # Fallback: save only best model if trained_models not available
            try:
                print(f"   🔄 Saving {self.best_model_type} clustering model (best only)...")
                model_path = os.path.join(path, f'{self.best_model_type}_model')
                print(f"   📂 Model path: {model_path}")
                
                # Handle different model types
                if self.best_model_type == 'dbscan':
                    # DBSCAN models are sklearn models, save as pickle
                    import pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.best_model, f)
                    print(f"✅ {self.best_model_type} clustering model saved as pickle to {model_path}")
                else:
                    # Standard PySpark models
                    self.best_model.write().overwrite().save(model_path)
                    print(f"✅ {self.best_model_type} clustering model saved to {model_path}")
                
                # Verify the model was actually saved
                if os.path.exists(model_path):
                    if self.best_model_type == 'dbscan':
                        print(f"   📄 Model file created: {model_path}")
                    else:
                        model_files = os.listdir(model_path)
                        print(f"   📄 Model files created: {len(model_files)} files")
                else:
                    print(f"   ⚠️ Model was not created: {model_path}")
                    
            except Exception as e:
                print(f"❌ FAILED to save clustering model!")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {str(e)}")
                print(f"   Model path attempted: {model_path}")
                print(f"   Model type: {type(self.best_model)}")
                # Don't suppress the error - raise it so we can see what's failing
                raise e
        else:
            print(f"   ⚠️ No best_model to save (best_model is None)")
            print(f"   🔍 Debug info:")
            print(f"      • best_model_type: {getattr(self, 'best_model_type', 'Not set')}")
            print(f"      • Available attributes: {[attr for attr in dir(self) if 'model' in attr.lower()]}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the trained clustering model."""
        if self.best_model is None:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "task_type": "clustering",
            "best_model_type": self.best_model_type,
            "metrics": self.model_metrics,
            "feature_count": len(self.selected_vars),
            "categorical_features": len(self.categorical_vars),
            "numerical_features": len(self.numerical_vars),
            "output_directory": self.output_dir
        } 

    def profile_clusters(self, original_data_path=None, include_target_analysis=True, save_results=True):
        """
        Generate comprehensive cluster profiling analysis.
        
        Args:
            original_data_path: Path to original data (to include dropped variables like target)
            include_target_analysis: Whether to analyze target variable vs clusters
            save_results: Whether to save profiling results to output directory
            
        Returns:
            Dictionary with comprehensive profiling results
        """
        
        if not self.best_model: # Changed from self.fitted to self.best_model
            raise ValueError("AutoML model must be fitted before profiling. Call fit() first.")
        
        print("\n🔬 Generating Cluster Profiling Analysis")
        print("=" * 45)
        
        # Use original data path if provided, otherwise use training data path
        data_path = original_data_path or getattr(self, 'train_data_path', None)
        
        if isinstance(data_path, str):
            # Load original data to get all variables
            original_df = self.spark.read.csv(data_path, header=True, inferSchema=True)
        else:
            # If data_path is already a DataFrame
            original_df = data_path
        
        # Get cluster predictions on original feature space
        print("📊 Applying clustering to original data...")
        
        # Process data through the same pipeline
        processed_data = self.data_processor.process_data(
            data_path, target_column=None
        )
        train_df = processed_data['train']
        
        # Apply best model to get cluster assignments
        cluster_predictions = self.best_model.transform(train_df)
        
        # Combine original data with cluster assignments
        original_pandas = original_df.toPandas()
        cluster_pandas = cluster_predictions.select("prediction").toPandas()
        
        # Create combined dataset
        combined_data = original_pandas.copy()
        combined_data['cluster'] = cluster_pandas['prediction']
        
        print(f"✅ Combined data: {len(combined_data)} samples, {len(combined_data.columns)} variables")
        
        # Auto-detect variable types
        numeric_vars = [col for col in combined_data.columns 
                       if col != 'cluster' and 
                       combined_data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        categorical_vars = [col for col in combined_data.columns 
                           if col != 'cluster' and 
                           combined_data[col].dtype in ['object', 'string', 'category']]
        
        print(f"📈 Found {len(numeric_vars)} numeric variables: {numeric_vars}")
        print(f"📊 Found {len(categorical_vars)} categorical variables: {categorical_vars}")
        
        # Generate comprehensive profiling
        profiling_results = self._generate_cluster_profiles(
            data=combined_data,
            numeric_vars=numeric_vars,
            categorical_vars=categorical_vars,
            include_target_analysis=include_target_analysis
        )
        
        # Save results if requested
        if save_results:
            self._save_profiling_results(profiling_results, combined_data)
        
        return profiling_results
    
    def _generate_cluster_profiles(self, data, numeric_vars, categorical_vars, include_target_analysis=True):
        """Generate comprehensive cluster profiling analysis."""
        
        print(f"\n📊 Cluster Profiling Results:")
        print("=" * 35)
        
        cluster_col = 'cluster'
        
        # Cluster sizes and summary
        cluster_sizes = data[cluster_col].value_counts().sort_index()
        total_samples = len(data)
        
        print(f"\n🎯 Cluster Summary:")
        for cluster in sorted(cluster_sizes.index):
            size = cluster_sizes[cluster]
            percentage = (size / total_samples) * 100
            print(f"   Cluster {cluster}: {size:,} samples ({percentage:.1f}%)")
        
        # Initialize results dictionary
        results = {
            'cluster_sizes': cluster_sizes,
            'total_samples': total_samples,
            'numeric_profiles': {},
            'categorical_profiles': {},
            'distinctive_variables': {},
            'cluster_insights': {}
        }
        
        # Profile numeric variables
        if numeric_vars:
            print(f"\n📈 Numeric Variable Profiles:")
            print("-" * 30)
            
            for var in numeric_vars:
                var_profile = self._profile_numeric_variable(data, var, cluster_col, cluster_sizes)
                results['numeric_profiles'][var] = var_profile
        
        # Profile categorical variables  
        if categorical_vars:
            print(f"\n📊 Categorical Variable Profiles:")
            print("-" * 35)
            
            for var in categorical_vars:
                var_profile = self._profile_categorical_variable(data, var, cluster_col, cluster_sizes)
                results['categorical_profiles'][var] = var_profile
        
        # Find most distinctive variables
        if numeric_vars:
            distinctive_scores = self._calculate_distinctive_variables(data, numeric_vars, cluster_col, cluster_sizes)
            results['distinctive_variables'] = distinctive_scores
            
            print(f"\n🎯 Most Distinctive Variables:")
            print("-" * 30)
            print("📈 Numeric variables (by cluster separation):")
            
            sorted_vars = sorted(distinctive_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (var, score) in enumerate(sorted_vars):
                stars = "⭐" * min(3, int(score * 10))
                print(f"   {i+1}. {var}: {score:.3f} {stars}")
        
        # Generate cluster insights
        cluster_insights = self._generate_cluster_insights(data, cluster_col, numeric_vars, categorical_vars)
        results['cluster_insights'] = cluster_insights
        
        print(f"\n💡 Cluster Insights:")
        print("-" * 20)
        for cluster, insights in cluster_insights.items():
            print(f"\n🎯 Cluster {cluster}:")
            for insight in insights:
                print(f"   • {insight}")
        
        return results
    
    def _profile_numeric_variable(self, data, var, cluster_col, cluster_sizes):
        """Profile a single numeric variable across clusters."""
        
        print(f"\n🔢 {var}:")
        overall_mean = data[var].mean()
        overall_std = data[var].std()
        print(f"   Overall: {overall_mean:.3f} ± {overall_std:.3f}")
        
        var_profile = {
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'cluster_stats': {}
        }
        
        for cluster in sorted(cluster_sizes.index):
            cluster_data = data[data[cluster_col] == cluster]
            cluster_mean = cluster_data[var].mean()
            cluster_std = cluster_data[var].std()
            cluster_median = cluster_data[var].median()
            cluster_min = cluster_data[var].min()
            cluster_max = cluster_data[var].max()
            
            pct_diff = ((cluster_mean - overall_mean) / overall_mean) * 100
            
            direction = "📈" if pct_diff > 5 else "📉" if pct_diff < -5 else "➡️"
            print(f"   Cluster {cluster}: {cluster_mean:.3f} ± {cluster_std:.3f} ({direction} {pct_diff:+.1f}%)")
            
            var_profile['cluster_stats'][cluster] = {
                'mean': cluster_mean,
                'std': cluster_std,
                'median': cluster_median,
                'min': cluster_min,
                'max': cluster_max,
                'pct_diff': pct_diff,
                'count': len(cluster_data)
            }
        
        return var_profile
    
    def _profile_categorical_variable(self, data, var, cluster_col, cluster_sizes):
        """Profile a single categorical variable across clusters."""
        
        print(f"\n🏷️  {var}:")
        
        # Get top 3 categories overall
        overall_counts = data[var].value_counts()
        top_3_categories = overall_counts.head(3).index.tolist()
        
        print(f"   Top 3 categories: {top_3_categories}")
        
        var_profile = {
            'top_categories': top_3_categories,
            'cluster_distributions': {}
        }
        
        # Create summary table
        print(f"\n   Distribution by Cluster:")
        summary_data = []
        
        # Profile top 3 categories
        for category in top_3_categories:
            row = {'Category': category}
            for cluster in sorted(cluster_sizes.index):
                cluster_data = data[data[cluster_col] == cluster]
                count = (cluster_data[var] == category).sum()
                percentage = (count / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
                row[f'Cluster_{cluster}'] = f"{percentage:.1f}% ({count})"
                
                # Store in profile
                if cluster not in var_profile['cluster_distributions']:
                    var_profile['cluster_distributions'][cluster] = {}
                var_profile['cluster_distributions'][cluster][category] = {
                    'count': count,
                    'percentage': percentage
                }
            summary_data.append(row)
        
        # Add "Other" category
        other_row = {'Category': 'Other'}
        for cluster in sorted(cluster_sizes.index):
            cluster_data = data[data[cluster_col] == cluster]
            other_count = (~cluster_data[var].isin(top_3_categories)).sum()
            other_percentage = (other_count / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
            other_row[f'Cluster_{cluster}'] = f"{other_percentage:.1f}% ({other_count})"
            
            # Store in profile
            var_profile['cluster_distributions'][cluster]['Other'] = {
                'count': other_count,
                'percentage': other_percentage
            }
        summary_data.append(other_row)
        
        # Display table
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Key insights
        print(f"\n   🔍 Key Insights:")
        insights = []
        for cluster in sorted(cluster_sizes.index):
            cluster_data = data[data[cluster_col] == cluster]
            if len(cluster_data) > 0:
                cluster_dist = cluster_data[var].value_counts(normalize=True)
                
                dominant_category = cluster_dist.index[0]
                dominant_percentage = cluster_dist.iloc[0] * 100
                
                if dominant_percentage > 60:
                    insight = f"Cluster {cluster}: Dominated by {dominant_category} ({dominant_percentage:.1f}%)"
                    print(f"   • {insight}")
                    insights.append(insight)
                else:
                    insight = f"Cluster {cluster}: Most common {dominant_category} ({dominant_percentage:.1f}%)"
                    print(f"   • {insight}")
                    insights.append(insight)
        
        var_profile['insights'] = insights
        return var_profile
    
    def _calculate_distinctive_variables(self, data, numeric_vars, cluster_col, cluster_sizes):
        """Calculate which variables are most distinctive across clusters."""
        
        distinctive_scores = {}
        
        for var in numeric_vars:
            cluster_means = []
            for cluster in sorted(cluster_sizes.index):
                cluster_data = data[data[cluster_col] == cluster]
                if len(cluster_data) > 0:
                    cluster_means.append(cluster_data[var].mean())
            
            if len(cluster_means) > 1:
                import numpy as np
                cv = np.std(cluster_means) / np.mean(cluster_means) if np.mean(cluster_means) != 0 else 0
                distinctive_scores[var] = cv
        
        return distinctive_scores
    
    def _generate_cluster_insights(self, data, cluster_col, numeric_vars, categorical_vars):
        """Generate high-level insights for each cluster."""
        
        cluster_sizes = data[cluster_col].value_counts().sort_index()
        insights = {}
        
        for cluster in sorted(cluster_sizes.index):
            cluster_data = data[data[cluster_col] == cluster]
            cluster_insights = []
            
            # Size insight
            size = len(cluster_data)
            total_size = len(data)
            pct = (size / total_size) * 100
            cluster_insights.append(f"Size: {size} samples ({pct:.1f}% of data)")
            
            # Numeric insights - find most distinctive characteristics
            if numeric_vars:
                import numpy as np
                for var in numeric_vars:
                    overall_mean = data[var].mean()
                    cluster_mean = cluster_data[var].mean()
                    pct_diff = ((cluster_mean - overall_mean) / overall_mean) * 100
                    
                    if abs(pct_diff) > 20:  # Significant difference
                        direction = "higher" if pct_diff > 0 else "lower"
                        cluster_insights.append(f"{var}: {abs(pct_diff):.0f}% {direction} than average")
            
            # Categorical insights - find dominant categories
            if categorical_vars:
                for var in categorical_vars:
                    if len(cluster_data) > 0:
                        cluster_dist = cluster_data[var].value_counts(normalize=True)
                        if len(cluster_dist) > 0:
                            dominant_category = cluster_dist.index[0]
                            dominant_pct = cluster_dist.iloc[0] * 100
                            
                            if dominant_pct > 70:
                                cluster_insights.append(f"{var}: Mostly {dominant_category} ({dominant_pct:.0f}%)")
            
            insights[cluster] = cluster_insights
        
        return insights
    
    def _save_profiling_results(self, profiling_results, data):
        """Save profiling results to output directory."""
        
        if not hasattr(self, 'output_dir') or not self.output_dir: # Changed from self.output_path to self.output_dir
            print("⚠️ No output path set - skipping profiling save")
            return
        
        import os
        import json
        import pandas as pd
        
        print(f"\n💾 Saving Profiling Results...")
        
        try:
            # Save profiling summary as JSON
            profiling_summary = {
                'cluster_sizes': profiling_results['cluster_sizes'].to_dict(),
                'total_samples': profiling_results['total_samples'],
                'distinctive_variables': profiling_results['distinctive_variables'],
                'cluster_insights': profiling_results['cluster_insights']
            }
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                return obj
            
            # Clean the summary for JSON serialization
            clean_summary = json.loads(json.dumps(profiling_summary, default=convert_numpy))
            
            summary_path = os.path.join(self.output_dir, 'cluster_profiling_summary.json') # Changed from self.output_path to self.output_dir
            with open(summary_path, 'w') as f:
                json.dump(clean_summary, f, indent=2)
            print(f"   ✅ Profiling summary saved to {summary_path}")
            
            # Save detailed numeric profiles as CSV
            if profiling_results['numeric_profiles']:
                numeric_data = []
                for var, profile in profiling_results['numeric_profiles'].items():
                    for cluster, stats in profile['cluster_stats'].items():
                        row = {
                            'variable': var,
                            'cluster': cluster,
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'median': stats['median'],
                            'min': stats['min'],
                            'max': stats['max'],
                            'pct_diff_from_overall': stats['pct_diff'],
                            'count': stats['count']
                        }
                        numeric_data.append(row)
                
                numeric_df = pd.DataFrame(numeric_data)
                numeric_path = os.path.join(self.output_dir, 'cluster_numeric_profiles.csv')
                numeric_df.to_csv(numeric_path, index=False)
                print(f"   ✅ Numeric profiles saved to {numeric_path}")
            
            # Save detailed categorical profiles as CSV
            if profiling_results['categorical_profiles']:
                categorical_data = []
                for var, profile in profiling_results['categorical_profiles'].items():
                    for cluster, distributions in profile['cluster_distributions'].items():
                        for category, stats in distributions.items():
                            row = {
                                'variable': var,
                                'cluster': cluster,
                                'category': category,
                                'count': stats['count'],
                                'percentage': stats['percentage'],
                                'is_top_3': category in profile['top_categories']
                            }
                            categorical_data.append(row)
                
                categorical_df = pd.DataFrame(categorical_data)
                categorical_path = os.path.join(self.output_dir, 'cluster_categorical_profiles.csv')
                categorical_df.to_csv(categorical_path, index=False)
                print(f"   ✅ Categorical profiles saved to {categorical_path}")
                
                # Also save categorical summary in Excel format with multiple sheets
                try:
                    with pd.ExcelWriter(os.path.join(self.output_dir, 'cluster_categorical_profiles.xlsx'), engine='xlsxwriter') as writer:
                        # Create a summary sheet
                        summary_rows = []
                        for var, profile in profiling_results['categorical_profiles'].items():
                            for category in profile['top_categories'] + ['Other']:
                                row = {'Variable': var, 'Category': category}
                                for cluster in sorted(profile['cluster_distributions'].keys()):
                                    if category in profile['cluster_distributions'][cluster]:
                                        stats = profile['cluster_distributions'][cluster][category]
                                        row[f'Cluster_{cluster}_Count'] = stats['count']
                                        row[f'Cluster_{cluster}_Percentage'] = f"{stats['percentage']:.1f}%"
                                summary_rows.append(row)
                        
                        summary_df = pd.DataFrame(summary_rows)
                        summary_df.to_excel(writer, sheet_name='Categorical_Summary', index=False)
                        
                        # Create individual sheets for each categorical variable
                        for var, profile in profiling_results['categorical_profiles'].items():
                            var_data = []
                            for category in profile['top_categories'] + ['Other']:
                                row = {'Category': category}
                                for cluster in sorted(profile['cluster_distributions'].keys()):
                                    if category in profile['cluster_distributions'][cluster]:
                                        stats = profile['cluster_distributions'][cluster][category]
                                        row[f'Cluster_{cluster}'] = f"{stats['percentage']:.1f}% ({stats['count']})"
                                var_data.append(row)
                            
                            var_df = pd.DataFrame(var_data)
                            var_df.to_excel(writer, sheet_name=f'{var[:30]}', index=False)  # Limit sheet name length
                        
                    print(f"   ✅ Categorical profiles Excel saved to cluster_categorical_profiles.xlsx")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not create Excel file: {str(e)}")
            
            # Save full dataset with cluster assignments
            data_path = os.path.join(self.output_dir, 'clustered_data.csv')
            data.to_csv(data_path, index=False)
            print(f"   ✅ Clustered dataset saved to {data_path}")
            
            # Create a human-readable profiling report
            self._create_profiling_report(profiling_results)
            
        except Exception as e:
            print(f"   ⚠️ Error saving profiling results: {str(e)}")
    
    def _create_profiling_report(self, profiling_results):
        """Create a human-readable profiling report."""
        
        import os
        
        report_path = os.path.join(self.output_dir, 'cluster_profiling_report.txt') # Changed from self.output_path to self.output_dir
        
        with open(report_path, 'w') as f:
            f.write("🔬 CLUSTER PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Cluster summary
            f.write("🎯 CLUSTER SUMMARY\n")
            f.write("-" * 20 + "\n")
            for cluster, size in profiling_results['cluster_sizes'].items():
                pct = (size / profiling_results['total_samples']) * 100
                f.write(f"Cluster {cluster}: {size:,} samples ({pct:.1f}%)\n")
            
            # Most distinctive variables
            if profiling_results['distinctive_variables']:
                f.write(f"\n🎯 MOST DISTINCTIVE VARIABLES\n")
                f.write("-" * 30 + "\n")
                sorted_vars = sorted(profiling_results['distinctive_variables'].items(), 
                                   key=lambda x: x[1], reverse=True)
                for i, (var, score) in enumerate(sorted_vars[:5]):
                    f.write(f"{i+1}. {var}: {score:.3f}\n")
            
            # Cluster insights
            f.write(f"\n💡 CLUSTER INSIGHTS\n")
            f.write("-" * 20 + "\n")
            for cluster, insights in profiling_results['cluster_insights'].items():
                f.write(f"\nCluster {cluster}:\n")
                for insight in insights:
                    f.write(f"  • {insight}\n")
        
        print(f"   ✅ Human-readable report saved to {report_path}") 
