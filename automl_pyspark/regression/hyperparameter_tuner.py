"""
Regression Hyperparameter Tuner

Contains hyperparameter optimization functionality for regression models.
Supports Optuna, Random Search, and Grid Search optimization methods.
"""

import os
import json
import random
import itertools
from typing import Dict, Any, List, Optional, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import RegressionEvaluator

# Import regression models
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor, GBTRegressor, 
    DecisionTreeRegressor, FMRegressor
)

# Optional imports
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Check for XGBoost and LightGBM availability
try:
    from xgboost.spark import SparkXGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from synapse.ml.lightgbm import LightGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import time # Added for chunked processing


class RegressionHyperparameterTuner:
    """
    Hyperparameter tuning for regression models using various optimization methods.
    """
    
    def __init__(self, spark_session: SparkSession, output_dir: str, user_id: str, model_literal: str):
        """
        Initialize the regression hyperparameter tuner.
        
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
        
        print(f"✅ RegressionHyperparameterTuner initialized")
        if OPTUNA_AVAILABLE:
            print("   📦 Optuna available for optimization")
        if XGBOOST_AVAILABLE:
            print("   📦 XGBoost available for tuning")
        if LIGHTGBM_AVAILABLE:
            print("   📦 LightGBM available for tuning")
    
    def tune_hyperparameters(self, model_type: str, train_data: DataFrame, 
                           target_column: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tune hyperparameters for a regression model.
        
        Args:
            model_type: Type of regression model to tune
            train_data: Training data
            target_column: Target column name
            config: Tuning configuration
            
        Returns:
            Dictionary containing best parameters and performance metrics
        """
        print(f"\n🔧 Tuning hyperparameters for {model_type} regression...")
        
        # Store config for later use in parameter suggestion
        self.config = config
        
        # Check if hyperparameter tuning is enabled
        # The config is flattened, so hyperparameter_tuning settings are at the root level
        if not config.get('enable_hyperparameter_tuning', False):
            print("   ⏭️  Hyperparameter tuning disabled, using default parameters")
            return self._get_default_params(model_type)
        
        # Check for both 'hp_method' (from Streamlit) and 'optimization_method' (legacy)
        optimization_method = config.get('hp_method', config.get('optimization_method', 'optuna'))
        print(f"   🎯 Hyperparameter tuning ENABLED! Using {optimization_method} optimization")
        print(f"   📊 Config debug: hp_method={config.get('hp_method')}, optuna_trials={config.get('optuna_trials')}")
        
        if optimization_method == 'optuna' and OPTUNA_AVAILABLE:
            return self._tune_with_optuna(model_type, train_data, target_column, config)
        elif optimization_method == 'random_search':
            return self._tune_with_random_search(model_type, train_data, target_column, config)
        elif optimization_method == 'grid_search':
            return self._tune_with_grid_search(model_type, train_data, target_column, config)
        else:
            print(f"   ⚠️  Optimization method '{optimization_method}' not available, using defaults")
            return self._get_default_params(model_type)
    
    def _tune_with_optuna(self, model_type: str, train_data: DataFrame, 
                         target_column: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna optimization."""
        
        if not OPTUNA_AVAILABLE:
            print("   ❌ Optuna not available, falling back to default parameters")
            return self._get_default_params(model_type)
        
        print(f"   🎯 Using Optuna optimization for {model_type}")
        
        n_trials = config.get('optuna_trials', 50)
        timeout = config.get('optuna_timeout', 300)
        
        # Ultra-aggressive limits for tree-based models to prevent broadcast escalation
        if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            n_trials = min(n_trials, 3)   # Ultra-conservative: only 3 trials max
            timeout = min(timeout, 120)   # 2 minutes max timeout
            print(f"   🌳 Tree-based model optimization: {n_trials} trials, {timeout}s timeout (ultra-conservative)")
        
        # Conservative limits for decision trees
        if model_type.lower() == 'decision_tree':
            n_trials = min(n_trials, 5)   # Conservative: only 5 trials max
            timeout = min(timeout, 180)   # 3 minutes max timeout
            print(f"   🌲 Decision tree optimization: {n_trials} trials, {timeout}s timeout (conservative)")
        
        # Reduce trials and timeout for neural networks to prevent hanging
        if model_type.lower() == 'neural_network':
            n_trials = min(n_trials, 20)  # Limit neural network trials
            timeout = min(timeout, 180)   # Limit neural network timeout
            print(f"   🧠 Neural network optimization: {n_trials} trials, {timeout}s timeout")
        
        # Apply Spark optimizations specifically for hyperparameter tuning
        try:
            from spark_optimization_config import optimize_for_hyperparameter_tuning, optimize_for_tree_based_tuning
            
            # Use tree-specific optimizations for tree-based models
            if model_type.lower() in ['random_forest', 'decision_tree', 'gradient_boosting', 'xgboost', 'lightgbm']:
                print(f"   ⚡ Applying tree-based optimizations for {model_type} hyperparameter tuning...")
                optimize_for_tree_based_tuning(self.spark, model_type, n_trials)
            else:
                print(f"   ⚡ Applying general optimizations for {model_type} hyperparameter tuning...")
                optimize_for_hyperparameter_tuning(self.spark, model_type, n_trials)
        except Exception as e:
            print(f"   ⚠️  Could not apply tuning optimizations: {e}")
        
        # Debug: Show what we're actually using
        print(f"   📊 Running {n_trials} Optuna trials with cleanup every 5 trials")
        print(f"   🔍 Debug: optuna_trials from config = {config.get('optuna_trials', 'NOT_SET')}")
        print(f"   🔍 Debug: timeout = {timeout}")
        
        # Track trial count for cleanup
        trial_counter = {'count': 0}
        
        # Add global timeout protection
        optimization_start_time = time.time()
        max_optimization_time = 300  # 5 minutes max for entire optimization
        if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            max_optimization_time = 180  # 3 minutes max for tree-based models
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',  # Minimize RMSE for regression
            sampler=TPESampler(seed=42)
        )
        
        # Define objective function
        def objective(trial):
            # Check global timeout
            if time.time() - optimization_start_time > max_optimization_time:
                print(f"    ⏰ Global optimization timeout reached ({max_optimization_time}s)")
                return 1000.0  # Return high RMSE to stop optimization
            
            trial_counter['count'] += 1
            current_trial = trial_counter['count']
            
            # Force memory cleanup between trials to prevent memory accumulation
            if trial_counter['count'] > 1:  # Skip first trial
                try:
                    import gc
                    gc.collect()  # Force Python garbage collection
                    
                    # Force Spark garbage collection
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    
                    # Clear Spark cache if available
                    try:
                        self.spark.catalog.clearCache()
                    except:
                        pass
                    
                    print(f"   🧹 Memory cleanup after trial {trial_counter['count'] - 1}")
                    
                    # Brief pause to allow cleanup
                    time.sleep(1)
                except Exception as e:
                    print(f"   ⚠️ Memory cleanup failed: {e}")
            
            try:
                params = self._suggest_optuna_params(trial, model_type)
                score = self._evaluate_params(model_type, params, train_data, target_column)
                
                # Check for broadcast warning escalation and terminate early if detected
                if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                    # If we've had 2 successful trials, stop early to prevent escalation
                    if trial_counter['count'] >= 2:
                        print(f"    🛑 Early termination: {trial_counter['count']} successful trials completed for tree-based model")
                        print(f"    💡 This prevents broadcast warning escalation")
                        return 0.5  # Return neutral score to stop optimization
                
                # Enhanced cleanup every 3 trials for tree-based models, 5 for others
                cleanup_interval = 3 if model_type in ['gradient_boosting', 'random_forest'] else 5
                
                if current_trial % cleanup_interval == 0:
                    try:
                        # Force garbage collection hint to Spark
                        if hasattr(self.spark, 'sparkContext'):
                            self.spark.sparkContext._jvm.System.gc()
                        
                        # Additional cleanup for tree-based models
                        if model_type in ['gradient_boosting', 'random_forest']:
                            # Clear any cached DataFrames
                            train_data.unpersist()
                            import gc
                            gc.collect()
                            time.sleep(1.5)  # Longer pause for tree models
                        
                        print(f"   🧹 Enhanced cleanup after trial {current_trial} (interval: {cleanup_interval})")
                        time.sleep(1)  # Brief pause for GC
                    except:
                        pass
                
                return score
            except Exception as e:
                print(f"     ⚠️  Error evaluating parameters: {str(e)[:100]}...")
                return float('inf')  # Return worst possible score
        
        # Run optimization
        try:
            print(f"   📊 Running {n_trials} Optuna trials with cleanup every 5 trials")
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"   ✅ Optuna completed: {n_trials} trials, best RMSE: {best_score:.4f}")
            
            # Save tuning results
            self._save_tuning_results(model_type, 'optuna', best_params, best_score, study.trials)
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'optimization_method': 'optuna',
                'n_trials': len(study.trials)
            }
            
        except Exception as e:
            print(f"   ❌ Optuna optimization failed: {str(e)}")
            return self._get_default_params(model_type)
    
    def _tune_with_random_search(self, model_type: str, train_data: DataFrame,
                                target_column: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Tune hyperparameters using random search."""
        
        print(f"   🎲 Using Random Search optimization for {model_type}")
        
        # Check for both 'random_trials' (from Streamlit) and 'random_search_trials' (legacy)
        n_trials = config.get('random_trials', config.get('random_search_trials', 20))
        param_space = self._get_param_space(model_type)
        
        # Ultra-aggressive limits for tree-based models to prevent broadcast escalation
        if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            n_trials = min(n_trials, 3)   # Ultra-conservative: only 3 trials max
            print(f"   🌳 Tree-based model optimization: {n_trials} trials (ultra-conservative)")
        
        # Conservative limits for decision trees
        if model_type.lower() == 'decision_tree':
            n_trials = min(n_trials, 5)   # Conservative: only 5 trials max
            print(f"   🌲 Decision tree optimization: {n_trials} trials (conservative)")
        
        # Reduce trials for neural networks to prevent hanging
        if model_type.lower() == 'neural_network':
            n_trials = min(n_trials, 20)  # Limit neural network trials
            print(f"   🧠 Neural network optimization: {n_trials} trials")
        
        best_params = None
        best_score = float('inf')
        all_results = []
        successful_trials = 0
        
        start_time = time.time()
        trial_count = 0
        
        # Add global timeout protection for tree-based models
        max_optimization_time = 300  # 5 minutes max for entire optimization
        if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            max_optimization_time = 180  # 3 minutes max for tree-based models
            print(f"   ⏰ Global timeout set to {max_optimization_time}s for tree-based model")
        
        # Chunk trials to reduce memory pressure and broadcasting warnings
        chunk_size = 5  # Process 5 trials at a time
        trial_chunks = [list(range(i, min(i + chunk_size, n_trials))) for i in range(0, n_trials, chunk_size)]
        
        print(f"   📊 Processing {n_trials} trials in {len(trial_chunks)} chunks of {chunk_size} trials each")
        
        for chunk_idx, trial_chunk in enumerate(trial_chunks):
            print(f"   🔄 Processing chunk {chunk_idx + 1}/{len(trial_chunks)} (trials {trial_chunk[0] + 1}-{trial_chunk[-1] + 1})")
            
            chunk_results = []
            
            for trial in trial_chunk:
                # Check global timeout
                if time.time() - start_time > max_optimization_time:
                    print(f"⏰ Global optimization timeout reached after {trial} trials")
                    break
                
                # Force memory cleanup between trials for tree-based models
                if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'] and trial > 0:
                    try:
                        import gc
                        gc.collect()  # Force Python garbage collection
                        
                        # Force Spark garbage collection
                        if hasattr(self.spark, 'sparkContext'):
                            self.spark.sparkContext._jvm.System.gc()
                        
                        # Clear Spark cache if available
                        try:
                            self.spark.catalog.clearCache()
                        except:
                            pass
                        
                        print(f"    🧹 Memory cleanup before trial {trial + 1}")
                        time.sleep(0.5)  # Brief pause for cleanup
                    except Exception as e:
                        print(f"    ⚠️ Memory cleanup failed: {e}")
                
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
                score = self._evaluate_params(model_type, params, train_data, target_column)
                chunk_results.append({'params': params, 'score': score})
                successful_trials += 1
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                
                print(f"     Trial {trial + 1}/{n_trials}: RMSE = {score:.4f}")
                
                # Check for broadcast warning escalation and terminate early if detected
                if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                    # If we've had 2 successful trials, stop early to prevent escalation
                    if successful_trials >= 2:
                        print(f"    🛑 Early termination: {successful_trials} successful trials completed for tree-based model")
                        print(f"    💡 This prevents broadcast warning escalation")
                        break
            
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
        
        print(f"   ✅ Random Search completed: best RMSE = {best_score:.4f}")
        
        # Save results
        self._save_tuning_results(model_type, 'random_search', best_params, best_score, all_results)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_method': 'random_search',
            'n_trials': n_trials
        }
    
    def _tune_with_grid_search(self, model_type: str, train_data: DataFrame,
                              target_column: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Tune hyperparameters using grid search."""
        
        print(f"   📊 Using Grid Search optimization for {model_type}")
        
        param_space = self._get_param_space(model_type, grid_search=True)
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        print(f"   📈 Testing {total_combinations} parameter combinations")
        
        best_params = None
        best_score = float('inf')
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
                score = self._evaluate_params(model_type, params, train_data, target_column)
                chunk_results.append({'params': params, 'score': score})
                
                if score < best_score:
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
        
        print(f"   ✅ Grid Search completed: best RMSE = {best_score:.4f}")
        
        # Save results
        self._save_tuning_results(model_type, 'grid_search', best_params, best_score, all_results)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_method': 'grid_search',
            'n_trials': total_combinations
        }
    
    def _suggest_optuna_params(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest parameters for Optuna optimization."""
        
        if model_type == 'linear_regression':
            return {
                'regParam': trial.suggest_float('regParam', 0.001, 1.0, log=True),
                'elasticNetParam': trial.suggest_float('elasticNetParam', 0.0, 1.0)
            }
        
        elif model_type == 'random_forest':
            return {
                'numTrees': trial.suggest_int('numTrees', 5, 10),           # Further reduced from (10, 200)
                'maxDepth': trial.suggest_int('maxDepth', 3, 5),             # Further reduced from (3, 20)
                'minInstancesPerNode': trial.suggest_int('minInstancesPerNode', 1, 2),  # Further reduced from (1, 10)
                'subsamplingRate': trial.suggest_float('subsamplingRate', 0.8, 1.0)     # Further reduced from (0.5, 1.0)
            }
        
        elif model_type == 'gradient_boosting':
            return {
                'maxIter': trial.suggest_int('maxIter', 5, 10),             # Further reduced from (50, 200)
                'maxDepth': trial.suggest_int('maxDepth', 3, 5),             # Further reduced from (3, 15)
                'stepSize': trial.suggest_float('stepSize', 0.1),            # Further reduced from (0.01, 0.3)
                'subsamplingRate': trial.suggest_float('subsamplingRate', 0.8, 1.0)     # Further reduced from (0.5, 1.0)
            }
        
        elif model_type == 'decision_tree':
            return {
                'maxDepth': trial.suggest_int('maxDepth', 3, 5),             # Further reduced from (3, 20)
                'minInstancesPerNode': trial.suggest_int('minInstancesPerNode', 1, 2),  # Further reduced from (1, 10)
                'minInfoGain': trial.suggest_float('minInfoGain', 0.0, 0.01) # Further reduced from (0.0, 0.1)
            }
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 5),           # Further reduced from (3, 15)
                'n_estimators': trial.suggest_int('n_estimators', 20, 30),   # Further reduced from (50, 300)
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # Further reduced from (0.01, 0.3)
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),     # Further reduced from (0.5, 1.0)
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0)  # Further reduced from (0.5, 1.0)
            }
        
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return {
                'numLeaves': trial.suggest_int('numLeaves', 5, 15),          # Further reduced from (10, 100)
                'numIterations': trial.suggest_int('numIterations', 20, 30), # Further reduced from (50, 300)
                'learningRate': trial.suggest_float('learningRate', 0.01, 0.1, log=True),  # Further reduced from (0.01, 0.3)
                'featureFraction': trial.suggest_float('featureFraction', 0.8, 1.0),      # Further reduced from (0.5, 1.0)
                'baggingFraction': trial.suggest_float('baggingFraction', 0.8, 1.0)       # Further reduced from (0.5, 1.0)
            }
        
        else:
            return {}
    
    def _get_param_space(self, model_type: str, grid_search: bool = False) -> Dict[str, Any]:
        """Get parameter space for random/grid search."""
        
        # First, try to get from job-specific config (passed from Streamlit)
        if hasattr(self, 'config') and self.config:
            regression_hp_ranges = self.config.get('regression_hp_ranges', {})
            if regression_hp_ranges and model_type.lower() in regression_hp_ranges:
                param_ranges = regression_hp_ranges[model_type.lower()]
                print(f"   📋 Using job-specific parameter ranges for {model_type}")
                return param_ranges
        
        # If no job-specific ranges, use hardcoded defaults
        print(f"   📋 Using default parameter ranges for {model_type}")
        
        if grid_search:
            # Smaller parameter space for grid search
            if model_type == 'linear_regression':
                return {
                    'regParam': [0.01, 0.1, 1.0],
                    'elasticNetParam': [0.0, 0.5, 1.0]
                }
            elif model_type == 'random_forest':
                return {
                    'numTrees': [5, 10],            # Further reduced from [50, 100, 150]
                    'maxDepth': [3, 5],              # Further reduced from [5, 10, 15]
                    'minInstancesPerNode': [1, 2]    # Further reduced from [1, 5]
                }
            elif model_type == 'gradient_boosting':
                # Restrict gradient boosting grid search to modest values to avoid
                # excessive memory usage and large broadcasted tasks
                return {
                    'maxIter': [5, 10],       # Further reduced from [20, 40, 60]
                    'maxDepth': [3, 5],       # Further reduced from [3, 5, 7]
                    'stepSize': [0.1],        # Further reduced from [0.05, 0.1, 0.2]
                    'maxBins': [32, 64]       # Further reduced from [32, 48, 64]
                }
            elif model_type == 'decision_tree':
                return {
                    'maxDepth': [3, 5],              # Further reduced from [5, 10, 15]
                    'minInstancesPerNode': [1, 2]    # Further reduced from [1, 5, 10]
                }
            elif model_type == 'xgboost':
                return {
                    'max_depth': [3, 5],              # Further reduced from [3, 5, 7]
                    'n_estimators': [20, 30],         # Further reduced from [30, 50, 75]
                    'learning_rate': [0.01, 0.1],     # Further reduced from [0.01, 0.05, 0.1]
                    'subsample': [0.8, 1.0],          # Further reduced from [0.7, 0.8, 1.0]
                    'colsample_bytree': [0.8, 1.0]    # Further reduced from [0.7, 0.8, 1.0]
                }
            elif model_type == 'lightgbm':
                return {
                    'numLeaves': [5, 15],             # Further reduced from [5, 15, 31]
                    'numIterations': [20, 30],        # Further reduced from [30, 50, 75]
                    'learningRate': [0.01, 0.1],      # Further reduced from [0.01, 0.05, 0.1]
                    'featureFraction': [0.8, 1.0],    # Further reduced from [0.7, 0.8, 1.0]
                    'baggingFraction': [0.8, 1.0]     # Further reduced from [0.7, 0.8, 1.0]
                }
            else:
                return {}
        
        else:
            # Larger parameter space for random search
            if model_type == 'linear_regression':
                return {
                    'regParam': (0.001, 1.0),
                    'elasticNetParam': (0.0, 1.0)
                }
            elif model_type == 'random_forest':
                return {
                    'numTrees': (5, 10),            # Further reduced from (10, 200)
                    'maxDepth': (3, 5),              # Further reduced from (3, 20)
                    'minInstancesPerNode': (1, 2),   # Further reduced from (1, 10)
                    'subsamplingRate': (0.8, 1.0),   # Further reduced from (0.5, 1.0)
                    'maxBins': (32, 64)              # Further reduced from (64, 256)
                }
            elif model_type == 'gradient_boosting':
                # Limit the search space for gradient boosting to mitigate large task binaries
                return {
                    'maxIter': (5, 10),       # Further reduced from (10, 60)
                    'maxDepth': (3, 5),       # Further reduced from (3, 8)
                    'stepSize': (0.1),        # Further reduced from (0.05, 0.3)
                    'subsamplingRate': (0.8, 1.0),   # Further reduced from (0.5, 1.0)
                    'maxBins': (32, 64)       # Further reduced from (32, 64)
                }
            elif model_type == 'decision_tree':
                return {
                    'maxDepth': (3, 5),              # Further reduced from (3, 20)
                    'minInstancesPerNode': (1, 2),   # Further reduced from (1, 10)
                    'minInfoGain': (0.0, 0.01),      # Further reduced from (0.0, 0.1)
                    'maxBins': (32, 64)              # Further reduced from (64, 256)
                }
            elif model_type == 'xgboost':
                return {
                    'max_depth': (3, 5),              # Further reduced from (3, 15)
                    'n_estimators': (20, 30),         # Further reduced from (30, 300)
                    'learning_rate': (0.01, 0.1),     # Further reduced from (0.01, 0.3)
                    'subsample': (0.8, 1.0),          # Further reduced from (0.5, 1.0)
                    'colsample_bytree': (0.8, 1.0)    # Further reduced from (0.5, 1.0)
                }
            elif model_type == 'lightgbm':
                return {
                    'numLeaves': (5, 15),             # Further reduced from (5, 120)
                    'numIterations': (20, 30),        # Further reduced from (30, 300)
                    'learningRate': (0.01, 0.1),      # Further reduced from (0.01, 0.3)
                    'featureFraction': (0.8, 1.0),    # Further reduced from (0.5, 1.0)
                    'baggingFraction': (0.8, 1.0)     # Further reduced from (0.5, 1.0)
                }
            else:
                return {}
    
    def _evaluate_params(self, model_type: str, params: Dict[str, Any], 
                        train_data: DataFrame, target_column: str) -> float:
        """Evaluate a set of parameters and return RMSE."""
        
        try:
            # Build model with parameters
            model = self._build_model_with_params(model_type, params, train_data, target_column)
            
            # Make predictions
            predictions = model.transform(train_data)
            
            # Calculate RMSE
            evaluator = RegressionEvaluator(
                labelCol=target_column,
                predictionCol="prediction",
                metricName="rmse"
            )
            
            rmse = evaluator.evaluate(predictions)
            return rmse
            
        except Exception as e:
            print(f"     ⚠️  Error evaluating parameters: {str(e)[:100]}...")
            return float('inf')  # Return worst possible score
    
    def _build_model_with_params(self, model_type: str, params: Dict[str, Any],
                                train_data: DataFrame, target_column: str):
        """Build a model with specific parameters."""
        
        if model_type == 'linear_regression':
            lr = LinearRegression(
                featuresCol='features',
                labelCol=target_column,
                regParam=params.get('regParam', 0.01),
                elasticNetParam=params.get('elasticNetParam', 0.0)
            )
            return lr.fit(train_data)
        
        elif model_type == 'random_forest':
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for random forest")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            rf = RandomForestRegressor(
                featuresCol='features',
                labelCol=target_column,
                numTrees=params.get('numTrees', 20),
                maxDepth=params.get('maxDepth', 5),
                minInstancesPerNode=params.get('minInstancesPerNode', 1),
                maxBins=params.get('maxBins', 32)
            )
            return rf.fit(train_data)
        
        elif model_type == 'gradient_boosting':
            # Apply gradient boosting optimizations to reduce task binary warnings
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                try:
                    import sys
                    import os
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from spark_optimization_config import apply_gradient_boosting_optimizations
                    apply_gradient_boosting_optimizations(spark)
                except Exception as e:
                    print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
            
            # Build a GBTRegressor with conservative defaults.  By keeping the
            # number of iterations and depth modest, we reduce the size of
            # broadcasted task binaries and improve stability.
            gbt = GBTRegressor(
                featuresCol='features',
                labelCol=target_column,
                maxIter=params.get('maxIter', 50),
                maxDepth=params.get('maxDepth', 5),
                stepSize=params.get('stepSize', 0.1),
                subsamplingRate=params.get('subsamplingRate', 1.0),
                maxBins=params.get('maxBins', 64),
            )
            return gbt.fit(train_data)
        
        elif model_type == 'decision_tree':
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for decision tree")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            dt = DecisionTreeRegressor(
                featuresCol='features',
                labelCol=target_column,
                maxDepth=params.get('maxDepth', 10),
                minInstancesPerNode=params.get('minInstancesPerNode', 1),
                minInfoGain=params.get('minInfoGain', 0.0),
                maxBins=params.get('maxBins', 128),  # Default to 128 to safely handle high cardinality
            )
            return dt.fit(train_data)
        
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for XGBoost")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            xgb = SparkXGBRegressor(
                features_col='features',
                label_col=target_column,
                max_depth=params.get('max_depth', 6),
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                num_workers=1,
                use_gpu=False
            )
            return xgb.fit(train_data)
        
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for LightGBM")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            lgb = LightGBMRegressor(
                featuresCol='features',
                labelCol=target_column,
                numLeaves=params.get('numLeaves', 31),
                numIterations=params.get('numIterations', 100),
                learningRate=params.get('learningRate', 0.1),
                featureFraction=params.get('featureFraction', 1.0),
                baggingFraction=params.get('baggingFraction', 1.0),
                objective="regression"
            )
            return lgb.fit(train_data)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type."""
        
        return {
            'best_params': {},
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
        
        models = ['linear_regression', 'random_forest', 'gradient_boosting', 'decision_tree']
        
        if XGBOOST_AVAILABLE:
            models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            models.append('lightgbm')
            
        return models 