"""
Clustering Hyperparameter Tuner

Contains hyperparameter optimization functionality for clustering models.
Supports Optuna, Random Search, and Grid Search optimization methods.
"""

import os
import json
import random
import itertools
from typing import Dict, Any, List, Optional, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import ClusteringEvaluator

# Import clustering models
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture

# For DBSCAN implementation
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional imports
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import time # Added for chunked processing
import numpy as np


class ClusteringHyperparameterTuner:
    """
    Hyperparameter tuning for clustering models using various optimization methods.
    """
    
    def __init__(self, spark_session: SparkSession, output_dir: str, user_id: str, model_literal: str):
        """
        Initialize the clustering hyperparameter tuner.
        
        Args:
            spark_session: PySpark SparkSession
            output_dir: Directory to save tuning results
            user_id: User identifier
            model_literal: Model identifier
        """
        self.spark = spark_session
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
        
        # Create tuning results directory
        self.tuning_dir = os.path.join(output_dir, 'hyperparameter_tuning')
        os.makedirs(self.tuning_dir, exist_ok=True)
        
        print(f"✅ ClusteringHyperparameterTuner initialized")
        if OPTUNA_AVAILABLE:
            print("   📦 Optuna available for optimization")
    
    def tune_hyperparameters(self, model_type: str, train_data: DataFrame, 
                           config: Dict[str, Any], k_range: List[int] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for a clustering model.
        
        Args:
            model_type: Type of clustering model to tune
            train_data: Training data
            config: Tuning configuration
            k_range: Range of k values to test for clustering
            
        Returns:
            Dictionary containing best parameters and performance metrics
        """
        print(f"\n🔧 Tuning hyperparameters for {model_type} clustering...")
        
        hp_config = config.get('hyperparameter_tuning', {})
        
        if not hp_config.get('enable_hyperparameter_tuning', False):
            print("   ⏭️  Hyperparameter tuning disabled, using default parameters")
            return self._get_default_params(model_type, k_range or [3])
        
        # Store config for later use in parameter suggestion
        self.current_config = config
        
        optimization_method = hp_config.get('optimization_method', 'optuna')
        
        if optimization_method == 'optuna' and OPTUNA_AVAILABLE:
            return self._tune_with_optuna(model_type, train_data, hp_config, k_range)
        elif optimization_method == 'random_search':
            return self._tune_with_random_search(model_type, train_data, hp_config, k_range)
        elif optimization_method == 'grid_search':
            return self._tune_with_grid_search(model_type, train_data, hp_config, k_range)
        else:
            print(f"   ⚠️  Optimization method '{optimization_method}' not available, using defaults")
            return self._get_default_params(model_type, k_range or [3])
    
    def _tune_with_optuna(self, model_type: str, train_data: DataFrame, 
                         config: Dict[str, Any], k_range: List[int] = None) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna optimization."""
        
        if not OPTUNA_AVAILABLE:
            print("   ❌ Optuna not available, falling back to default parameters")
            return self._get_default_params(model_type, k_range or [3])
        
        print(f"   🎯 Using Optuna optimization for {model_type}")
        
        # Get optuna settings from hyperparameter_tuning section or top-level config
        hp_config = config.get('hyperparameter_tuning', {})
        n_trials = hp_config.get('optuna_trials', config.get('optuna_trials', 50))
        timeout = hp_config.get('optuna_timeout', config.get('optuna_timeout', 300))
        
        print(f"   🎯 Optuna trials configured: {n_trials}")
        
        # Track trial count for cleanup
        trial_counter = {'count': 0}
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',  # Maximize silhouette score for clustering
            sampler=TPESampler(seed=42)
        )
        
        # Define objective function
        def objective(trial):
            trial_counter['count'] += 1
            current_trial = trial_counter['count']
            
            params = self._suggest_optuna_params(trial, model_type, k_range)
            score = self._evaluate_params(model_type, params, train_data)
            
            # Clean up every 5 trials to reduce memory pressure
            if current_trial % 5 == 0:
                try:
                    # Force garbage collection hint to Spark
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    print(f"   🧹 Cleaned up after trial {current_trial}")
                    time.sleep(1)  # Brief pause for GC
                except:
                    pass
            
            return score
        
        # Run optimization
        try:
            print(f"   📊 Running {n_trials} Optuna trials with cleanup every 5 trials")
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"   ✅ Optuna completed: {n_trials} trials, best silhouette: {best_score:.4f}")
            
            # Convert Optuna trials to JSON-serializable format
            serializable_trials = []
            for trial in study.trials:
                trial_data = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state),
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                }
                serializable_trials.append(trial_data)
            
            # Save tuning results
            self._save_tuning_results(model_type, 'optuna', best_params, best_score, serializable_trials)
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_method': 'optuna',
                'n_trials': len(study.trials)
            }
            
        except Exception as e:
            print(f"   ❌ Optuna optimization failed: {str(e)}")
            return self._get_default_params(model_type, k_range or [3])
    
    def _tune_with_random_search(self, model_type: str, train_data: DataFrame,
                                config: Dict[str, Any], k_range: List[int] = None) -> Dict[str, Any]:
        """Tune hyperparameters using random search."""
        
        print(f"   🎲 Using Random Search optimization for {model_type}")
        
        # Get random search settings from hyperparameter_tuning section or top-level config
        hp_config = config.get('hyperparameter_tuning', {})
        n_trials = hp_config.get('random_trials', config.get('random_search_trials', 20))
        
        print(f"   🎯 Random search trials configured: {n_trials}")
        param_space = self._get_param_space(model_type, k_range)
        
        best_params = None
        best_score = -1.0  # Silhouette score ranges from -1 to 1
        all_results = []
        
        # Chunk trials to reduce memory pressure and broadcasting warnings
        chunk_size = 5  # Process 5 trials at a time
        trial_chunks = [list(range(i, min(i + chunk_size, n_trials))) for i in range(0, n_trials, chunk_size)]
        
        print(f"   📊 Processing {n_trials} trials in {len(trial_chunks)} chunks of {chunk_size} trials each")
        
        for chunk_idx, trial_chunk in enumerate(trial_chunks):
            print(f"   🔄 Processing chunk {chunk_idx + 1}/{len(trial_chunks)} (trials {trial_chunk[0] + 1}-{trial_chunk[-1] + 1})")
            
            chunk_results = []
            
            for trial in trial_chunk:
                # Sample random parameters
                params = {}
                for param_name, param_range in param_space.items():
                    if isinstance(param_range, list):
                        params[param_name] = random.choice(param_range)
                    elif isinstance(param_range, tuple) and len(param_range) == 2:
                        if isinstance(param_range[0], int):
                            params[param_name] = random.randint(param_range[0], param_range[1])
                        else:
                            params[param_name] = random.uniform(param_range[0], param_range[1])
                
                # Evaluate parameters
                score = self._evaluate_params(model_type, params, train_data)
                chunk_results.append({'params': params, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                print(f"     Trial {trial + 1}/{n_trials}: Silhouette = {score:.4f}")
            
            # Add chunk results to overall results
            all_results.extend(chunk_results)
            
            # Clean up and give JVM a break between chunks
            if chunk_idx < len(trial_chunks) - 1:  # Not the last chunk
                try:
                    # Force garbage collection hint to Spark
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    print("   🧹 Cleaned up chunk data, pausing briefly...")
                    time.sleep(2)  # Slightly longer pause for GC
                except:
                    pass
        
        print(f"   ✅ Random Search completed: best silhouette = {best_score:.4f}")
        
        # Save results
        self._save_tuning_results(model_type, 'random_search', best_params, best_score, all_results)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_method': 'random_search',
            'n_trials': n_trials
        }
    
    def _tune_with_grid_search(self, model_type: str, train_data: DataFrame,
                              config: Dict[str, Any], k_range: List[int] = None) -> Dict[str, Any]:
        """Tune hyperparameters using grid search."""
        
        print(f"   📊 Using Grid Search optimization for {model_type}")
        
        param_space = self._get_param_space(model_type, k_range, grid_search=True)
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        print(f"   📈 Testing {total_combinations} parameter combinations")
        
        best_params = None
        best_score = -1.0
        all_results = []
        
        # Chunk combinations to reduce memory pressure and broadcasting warnings
        chunk_size = 5  # Process 5 combinations at a time
        combination_chunks = [param_combinations[i:i + chunk_size] for i in range(0, len(param_combinations), chunk_size)]
        
        print(f"   📊 Processing {total_combinations} combinations in {len(combination_chunks)} chunks of {chunk_size} combinations each")
        
        for chunk_idx, combination_chunk in enumerate(combination_chunks):
            print(f"   🔄 Processing chunk {chunk_idx + 1}/{len(combination_chunks)} (combinations {chunk_idx * chunk_size + 1}-{min((chunk_idx + 1) * chunk_size, total_combinations)})")
            
            chunk_results = []
            
            for i, combination in enumerate(combination_chunk):
                params = dict(zip(param_names, combination))
                
                # Evaluate parameters
                score = self._evaluate_params(model_type, params, train_data)
                chunk_results.append({'params': params, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                global_idx = chunk_idx * chunk_size + i + 1
                if global_idx % max(1, total_combinations // 10) == 0:
                    print(f"     Progress: {global_idx}/{total_combinations} ({100 * global_idx / total_combinations:.1f}%)")
            
            # Add chunk results to overall results
            all_results.extend(chunk_results)
            
            # Clean up and give JVM a break between chunks
            if chunk_idx < len(combination_chunks) - 1:  # Not the last chunk
                try:
                    # Force garbage collection hint to Spark
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    print("   🧹 Cleaned up chunk data, pausing briefly...")
                    time.sleep(2)  # Slightly longer pause for GC
                except:
                    pass
        
        print(f"   ✅ Grid Search completed: best silhouette = {best_score:.4f}")
        
        # Save results
        self._save_tuning_results(model_type, 'grid_search', best_params, best_score, all_results)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_method': 'grid_search',
            'n_trials': total_combinations
        }
    
    def _suggest_optuna_params(self, trial, model_type: str, k_range: List[int] = None) -> Dict[str, Any]:
        """Suggest parameters for Optuna optimization using config ranges."""
        
        # Use k_range from config if available, otherwise default
        k_values = k_range or [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
        
        if model_type == 'kmeans':
            # Get parameter ranges from config if available
            kmeans_config = self._get_config_param_ranges('kmeans')
            
            # Prioritize function parameter k_range over config.yaml k_range
            final_k_range = k_values if k_range else kmeans_config.get('k_range', k_values)
            print(f"   🎯 Using k_range: {final_k_range}")
            
            # Ensure tolerance values are floats (handle config.yaml string conversion)
            tol_range = kmeans_config.get('tol_range', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
            if tol_range and isinstance(tol_range[0], str):
                try:
                    tol_range = [float(t) for t in tol_range]
                except (ValueError, TypeError):
                    print(f"   ⚠️  Warning: Could not convert tolerance range {tol_range} to floats, using defaults")
                    tol_range = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            
            return {
                'k': trial.suggest_categorical('k', final_k_range),
                'maxIter': trial.suggest_categorical('maxIter', kmeans_config.get('max_iter_range', [20, 50, 100, 200])),
                'tol': trial.suggest_categorical('tol', tol_range),
                'initMode': trial.suggest_categorical('initMode', kmeans_config.get('init_modes', ['k-means||', 'random']))
            }
        
        elif model_type == 'bisecting_kmeans':
            # Get parameter ranges from config if available
            bisecting_config = self._get_config_param_ranges('bisecting_kmeans')
            
            # Prioritize function parameter k_range over config.yaml k_range
            final_k_range = k_values if k_range else bisecting_config.get('k_range', k_values)
            print(f"   🎯 Using k_range: {final_k_range}")
            
            return {
                'k': trial.suggest_categorical('k', final_k_range),
                'maxIter': trial.suggest_categorical('maxIter', bisecting_config.get('max_iter_range', [20, 50, 100, 200])),
                'minDivisibleClusterSize': trial.suggest_categorical('minDivisibleClusterSize', 
                                                                   bisecting_config.get('min_divisible_range', [0.5, 1.0, 2.0, 3.0, 5.0]))
            }
        
        else:
            return {'k': trial.suggest_categorical('k', k_values)}
    
    def _get_config_param_ranges(self, model_type: str) -> Dict[str, Any]:
        """Get parameter ranges from config file or job-specific config."""
        param_ranges = {}
        
        # First, try to get from job-specific config (passed from Streamlit)
        if hasattr(self, 'current_config') and self.current_config:
            clustering_hp_ranges = self.current_config.get('clustering_hp_ranges', {})
            if clustering_hp_ranges and model_type in clustering_hp_ranges:
                param_ranges = clustering_hp_ranges[model_type]
                print(f"   📋 Using job-specific parameter ranges for {model_type}")
        
        # If no job-specific ranges, try config.yaml
        if not param_ranges:
            try:
                import yaml
                # Try to load current config  
                with open('config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                
                config_param_ranges = config.get('clustering', {}).get('hyperparameter_tuning', {}).get('parameter_ranges', {})
                param_ranges = config_param_ranges.get(model_type, {})
                if param_ranges:
                    print(f"   📋 Using config.yaml parameter ranges for {model_type}")
            except Exception as e:
                print(f"   ⚠️  Could not load config parameter ranges: {str(e)}")
        
        # If still no ranges, use defaults
        if not param_ranges:
            print(f"   📋 Using default parameter ranges for {model_type}")
        
        return param_ranges
    
    def _get_param_space(self, model_type: str, k_range: List[int] = None, grid_search: bool = False) -> Dict[str, Any]:
        """Get parameter space for random/grid search."""
        
        k_values = k_range or [2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        if grid_search:
            # Smaller parameter space for grid search
            if model_type == 'kmeans':
                return {
                    'k': k_values,
                    'maxIter': [20, 50, 100],
                    'initMode': ['k-means||', 'random']
                }
            elif model_type == 'bisecting_kmeans':
                return {
                    'k': k_values,
                    'maxIter': [20, 50, 100],
                    'minDivisibleClusterSize': [1.0, 2.0, 3.0]
                }
            elif model_type == 'gaussian_mixture':
                return {
                    'k': k_values,
                    'maxIter': [50, 100, 200],
                    'tol': [1e-5, 1e-4, 1e-3],
                    'regParam': [0.001, 0.01, 0.1]
                }
            elif model_type == 'dbscan':
                return {
                    'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
                    'minPts': [3, 5, 7, 10]
                }
            else:
                return {'k': k_values}
        
        else:
            # Larger parameter space for random search
            if model_type == 'kmeans':
                return {
                    'k': k_values,
                    'maxIter': (10, 100),
                    'tol': (1e-6, 1e-2),
                    'initMode': ['k-means||', 'random']
                }
            elif model_type == 'bisecting_kmeans':
                return {
                    'k': k_values,
                    'maxIter': (10, 100),
                    'minDivisibleClusterSize': (0.5, 5.0)
                }
            elif model_type == 'gaussian_mixture':
                return {
                    'k': k_values,
                    'maxIter': (50, 300),
                    'tol': (1e-6, 1e-3),
                    'regParam': (0.001, 1.0)
                }
            elif model_type == 'dbscan':
                return {
                    'eps': (0.1, 2.0),
                    'minPts': (3, 20)
                }
            else:
                return {'k': k_values}
    
    def _evaluate_params(self, model_type: str, params: Dict[str, Any], train_data: DataFrame) -> float:
        """Evaluate a set of parameters and return silhouette score."""
        
        try:
            # Build model with parameters
            model = self._build_model_with_params(model_type, params, train_data)
            
            # Make predictions
            predictions = model.transform(train_data)
            
            # Calculate silhouette score
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)
            
            return silhouette
            
        except Exception as e:
            print(f"     ⚠️  Error evaluating parameters: {str(e)[:100]}...")
            return -1.0  # Return worst possible silhouette score
    
    def _build_model_with_params(self, model_type: str, params: Dict[str, Any], train_data: DataFrame):
        """Build a clustering model with specific parameters."""
        
        if model_type == 'kmeans':
            # Convert tolerance to float if it's a string (handles JSON serialization issues)
            tol_value = params.get('tol', 1e-4)
            if isinstance(tol_value, str):
                try:
                    tol_value = float(tol_value)
                except (ValueError, TypeError):
                    print(f"   ⚠️  Warning: Could not convert tolerance '{tol_value}' to float, using default 1e-4")
                    tol_value = 1e-4
            
            kmeans = KMeans(
                featuresCol='features',
                k=params.get('k', 3),
                maxIter=params.get('maxIter', 20),
                tol=tol_value,
                initMode=params.get('initMode', 'k-means||'),
                seed=42
            )
            return kmeans.fit(train_data)
        
        elif model_type == 'bisecting_kmeans':
            bisecting = BisectingKMeans(
                featuresCol='features',
                k=params.get('k', 3),
                maxIter=params.get('maxIter', 20),
                minDivisibleClusterSize=params.get('minDivisibleClusterSize', 1.0),
                seed=42
            )
            return bisecting.fit(train_data)
        
        elif model_type == 'gaussian_mixture':
            # Convert tolerance to float if it's a string
            tol_value = params.get('tol', 1e-4)
            if isinstance(tol_value, str):
                try:
                    tol_value = float(tol_value)
                except (ValueError, TypeError):
                    print(f"   ⚠️  Warning: Could not convert tolerance '{tol_value}' to float, using default 1e-4")
                    tol_value = 1e-4
            
            gaussian = GaussianMixture(
                featuresCol='features',
                k=params.get('k', 3),
                maxIter=params.get('maxIter', 100),
                tol=tol_value,
                regParam=params.get('regParam', 0.01),
                seed=42
            )
            return gaussian.fit(train_data)
        
        elif model_type == 'dbscan':
            # Build a DBSCAN model using scikit‑learn.  Capture the Spark
            # session from the training DataFrame to avoid referencing
            # undefined attributes on the wrapper.  This wrapper mirrors
            # the one in ClusteringModelBuilder._create_dbscan_model.
            if not SKLEARN_AVAILABLE:
                raise ImportError("DBSCAN requires sklearn. Install with: pip install scikit-learn")

            eps = params.get('eps', 0.5)
            minPts = params.get('minPts', 5)
            spark_session = train_data.sparkSession

            class DBSCANWrapper:
                def __init__(self, eps_val: float, min_pts: int):
                    self.eps = eps_val
                    self.minPts = min_pts
                    self.model = None
                    self.scaler = StandardScaler()

                def fit(self, data: DataFrame):
                    # Extract features and fit DBSCAN
                    features_list = [row.features.toArray() for row in data.select("features").collect()]
                    features_array = np.array(features_list)
                    features_scaled = self.scaler.fit_transform(features_array)
                    self.model = DBSCAN(eps=self.eps, min_samples=self.minPts)
                    self.model.fit(features_scaled)
                    return self

                def transform(self, data: DataFrame) -> DataFrame:
                    from pyspark.sql.functions import monotonically_increasing_id
                    # Extract and scale features
                    features_list = [row.features.toArray() for row in data.select("features").collect()]
                    features_array = np.array(features_list)
                    features_scaled = self.scaler.transform(features_array)
                    # Predict clusters
                    predictions = self.model.fit_predict(features_scaled)
                    # Build predictions DataFrame with row indices
                    pred_rows = [(int(pred),) for pred in predictions]
                    pred_df = spark_session.createDataFrame(pred_rows, ["prediction"])
                    pred_df = pred_df.withColumn("row_index", monotonically_increasing_id())
                    # Attach row indices to original data
                    data_with_index = data.withColumn("row_index", monotonically_increasing_id())
                    # Join predictions to original data and drop index
                    result = data_with_index.join(pred_df, on="row_index").drop("row_index")
                    return result

            dbscan = DBSCANWrapper(eps, minPts)
            return dbscan.fit(train_data)
        
        else:
            raise ValueError(f"Unsupported clustering model type: {model_type}")
    
    def _get_default_params(self, model_type: str, k_values: List[int]) -> Dict[str, Any]:
        """Get default parameters for a model type."""
        
        default_k = k_values[len(k_values) // 2] if k_values else 3
        
        return {
            'best_params': {'k': default_k},
            'best_score': None,
            'optimization_method': 'default',
            'n_trials': 0
        }
    
    def _save_tuning_results(self, model_type: str, method: str, best_params: Dict[str, Any],
                           best_score: float, all_results: List[Dict[str, Any]]):
        """Save hyperparameter tuning results."""
        
        results = {
            'model_type': model_type,
            'optimization_method': method,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(all_results),
            'all_results': all_results[:100]  # Limit to first 100 results
        }
        
        filename = f"{model_type}_{method}_tuning_results.json"
        filepath = os.path.join(self.tuning_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   💾 Tuning results saved to {filename}")
        except Exception as e:
            print(f"   ⚠️  Could not save tuning results: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of models available for hyperparameter tuning."""
        
        models = ['kmeans', 'bisecting_kmeans', 'gaussian_mixture']
        
        if SKLEARN_AVAILABLE:
            models.append('dbscan')
        
        return models 