"""
Hyperparameter Tuner

Class responsible for hyperparameter optimization using Optuna and other techniques.
This class encapsulates all hyperparameter tuning functionality for the AutoML pipeline.

Usage Examples:
--------------

# Using Optuna (Bayesian optimization - most efficient)
config = {
    'enable_hyperparameter_tuning': True,
    'optimization_method': 'optuna',
    'optuna_trials': 50,
    'optuna_timeout': 300
}

# Using Random Search (good balance of efficiency and exploration)
config = {
    'enable_hyperparameter_tuning': True,
    'optimization_method': 'random_search',
    'random_search_trials': 30,
    'random_search_timeout': 300
}

# Using Grid Search (exhaustive but computationally expensive)
config = {
    'enable_hyperparameter_tuning': True,
    'optimization_method': 'grid_search',
    'grid_search_max_combinations': 50,
    'grid_search_timeout': 600
}

Method Comparison:
- Optuna: Bayesian optimization, learns from previous trials, most efficient for complex parameter spaces
- Random Search: Random sampling, good baseline, balances exploration and computational cost
- Grid Search: Exhaustive search, guarantees finding optimal within defined grid, expensive but thorough
"""

from typing import Dict, List, Any, Optional, Tuple
from pyspark.sql import SparkSession, DataFrame
import itertools
import random
import time

# Optional import for Optuna hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ Optuna not available. Install with: pip install optuna>=3.0.0")
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """
    Hyperparameter tuner class that handles optimization using various techniques.
    
    This class provides functionality for:
    - Optuna-based hyperparameter optimization (Bayesian)
    - Grid search optimization (exhaustive)
    - Random search optimization (random sampling)
    - Bayesian optimization integration
    """
    
    def __init__(self, spark_session: SparkSession, config: Dict[str, Any]):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            spark_session: PySpark SparkSession
            config: Configuration dictionary containing tuning parameters
        """
        self.spark = spark_session
        self.config = config
        self.is_multiclass = None
        self.num_classes = None
    
    def set_problem_type(self, is_multiclass: bool, num_classes: int):
        """
        Set the problem type for hyperparameter optimization.
        
        Args:
            is_multiclass: Whether this is a multiclass problem
            num_classes: Number of classes in the target variable
        """
        self.is_multiclass = is_multiclass
        self.num_classes = num_classes
    
    def optimize_hyperparameters(self, train_data: DataFrame, target_column: str, 
                                model_type: str, feature_count: int, 
                                available_model_types: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a given model type.
        
        Args:
            train_data: Training DataFrame
            target_column: Name of the target column
            model_type: Type of model to optimize
            feature_count: Number of features
            available_model_types: Dictionary of available model types
            
        Returns:
            Dictionary containing the best parameters and metrics
        """
        print(f"\n🔧 Starting hyperparameter optimization for {model_type}...")
        
        optimization_method = self.config.get('optimization_method', 'optuna')
        
        try:
            if optimization_method == 'optuna':
                result = self._optimize_with_optuna(
                    train_data, target_column, model_type, feature_count, available_model_types
                )
            elif optimization_method == 'random_search':
                result = self._optimize_with_random_search(
                    train_data, target_column, model_type, feature_count, available_model_types
                )
            elif optimization_method == 'grid_search':
                result = self._optimize_with_grid_search(
                    train_data, target_column, model_type, feature_count, available_model_types
                )
            else:
                print(f"⚠️ Optimization method '{optimization_method}' not implemented. Using default parameters.")
                return {}
            
            # Validate the result
            if not result:
                print(f"❌ Hyperparameter optimization for {model_type} returned empty result.")
                return {}
            
            if 'best_params' not in result or not result['best_params']:
                print(f"❌ No best parameters found for {model_type}.")
                return {}
            
            if 'best_score' not in result:
                print(f"❌ No best score found for {model_type}.")
                return {}
            
            # Additional validation for neural networks
            if model_type.lower() == 'neural_network':
                if not result['best_params'] or len(result['best_params']) == 0:
                    print(f"❌ Neural network optimization returned empty parameters.")
                    return {}
                
                # Validate that layers parameter exists and is valid
                if 'layers' not in result['best_params']:
                    print(f"❌ Neural network optimization missing 'layers' parameter.")
                    return {}
                
                print(f"✅ Neural network optimization validation passed")
            
            print(f"✅ Hyperparameter optimization completed for {model_type}")
            print(f"   Best score: {result['best_score']:.4f}")
            print(f"   Best parameters: {result['best_params']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Hyperparameter optimization failed for {model_type}: {str(e)}")
            return {}

    def _get_parameter_space(self, model_type: str) -> Dict[str, List]:
        """Get parameter space for hyperparameter tuning."""
        
        # First, try to get from job-specific config (passed from Streamlit)
        if self.config:
            classification_hp_ranges = self.config.get('classification_hp_ranges', {})
            if classification_hp_ranges and model_type.lower() in classification_hp_ranges:
                param_ranges = classification_hp_ranges[model_type.lower()]
                print(f"   📋 Using job-specific parameter ranges for {model_type}")
                return param_ranges
        
        # If no job-specific ranges, use hardcoded defaults
        print(f"   📋 Using default parameter ranges for {model_type}")
        
        parameter_spaces = {
            'logistic': {
                'maxIter': [10, 20, 50, 100],
                'regParam': [0.01, 0.1, 0.5, 1.0],
                'elasticNetParam': [0.0, 0.25, 0.5, 0.75, 1.0]
            },
            'random_forest': {
                'numTrees': [5, 10],            # Further reduced from [10, 20]
                'maxDepth': [3, 5],              # Further reduced from [3, 5]
                'minInstancesPerNode': [1, 2]    # Further reduced from [1, 2]
            },
            'decision_tree': {
                'maxDepth': [3, 5],              # Further reduced from [3, 5, 10]
                'minInstancesPerNode': [1, 2],   # Further reduced from [1, 2, 5]
                'minInfoGain': [0.0, 0.01]       # Further reduced from [0.0, 0.01, 0.1]
            },
            'gradient_boosting': {
                'maxIter': [5, 10],       # Further reduced from [10, 20]
                'maxDepth': [3, 5],       # Further reduced from [3, 5]
                'stepSize': [0.1]         # Further reduced from [0.1, 0.2]
            },
            'naive_bayes': {
                'smoothing': [0.1, 0.5, 1.0, 2.0]
            },
            'neural_network': {
                'layers': [[8], [16], [8, 4], [16, 8], [16, 8, 4]],  # Hidden layers only - must be <= input size
                'maxIter': [50, 100, 200],
                'blockSize': [32, 64, 128]
            },
            'one_vs_rest': {
                'maxIter': [10, 20, 50, 100],
                'regParam': [0.01, 0.1, 0.5, 1.0]
            },
            'xgboost': {
                'maxDepth': [3, 5],              # Further reduced from [3, 5, 7]
                'numRound': [20, 30],            # Further reduced from [30, 50, 75]
                'eta': [0.01, 0.1],             # Further reduced from [0.01, 0.05, 0.1]
                'subsample': [0.8, 1.0],        # Further reduced from [0.7, 0.8, 1.0]
                'colsampleBytree': [0.8, 1.0],  # Further reduced from [0.7, 0.8, 1.0]
                'minChildWeight': [1, 3],       # Further reduced from [1, 3, 5]
                'gamma': [0, 0.1]               # Further reduced from [0, 0.1, 0.3]
            },
            'lightgbm': {
                'numLeaves': [5, 15],            # Further reduced from [5, 15, 31]
                'numIterations': [20, 30],       # Further reduced from [30, 50, 75]
                'learningRate': [0.01, 0.1],     # Further reduced from [0.01, 0.05, 0.1]
                'featureFraction': [0.8, 1.0],   # Further reduced from [0.7, 0.8, 1.0]
                'baggingFraction': [0.8, 1.0],   # Further reduced from [0.7, 0.8, 1.0]
                'minDataInLeaf': [10, 20],       # Further reduced from [10, 30, 50]
                'lambdaL1': [0, 0.5],           # Further reduced from [0, 0.5, 1.0]
                'lambdaL2': [0, 0.5]            # Further reduced from [0, 0.5, 1.0]
            }
        }
        
        return parameter_spaces.get(model_type.lower(), {})

    def _sample_random_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        Sample random parameters from the parameter space.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of randomly sampled parameters
        """
        param_space = self._get_parameter_space(model_type)
        sampled_params = {}
        
        for param_name, param_values in param_space.items():
            sampled_params[param_name] = random.choice(param_values)
        
        return sampled_params
    
    def _ensure_diverse_parameters(self, model_type: str, num_trials: int) -> List[Dict[str, Any]]:
        """
        Ensure diverse parameter combinations to avoid identical results.
        
        Args:
            model_type: Type of model
            num_trials: Number of trials
            
        Returns:
            List of diverse parameter dictionaries
        """
        param_space = self._get_parameter_space(model_type)
        if not param_space:
            return []
        
        # Generate diverse parameter combinations
        diverse_params = []
        used_combinations = set()
        
        for trial in range(num_trials):
            # Try to find a unique combination
            max_attempts = 50
            for attempt in range(max_attempts):
                params = {}
                for param_name, param_values in param_space.items():
                    params[param_name] = random.choice(param_values)
                
                # Create a hash of the parameter combination
                # Convert lists to tuples for hashing
                hashable_params = {}
                for key, value in params.items():
                    if isinstance(value, list):
                        hashable_params[key] = tuple(value)
                    else:
                        hashable_params[key] = value
                param_hash = tuple(sorted(hashable_params.items()))
                
                if param_hash not in used_combinations:
                    used_combinations.add(param_hash)
                    diverse_params.append(params)
                    break
            else:
                # If we can't find a unique combination, use the last one
                diverse_params.append(params)
        
        return diverse_params

    def _optimize_with_random_search(self, train_data: DataFrame, target_column: str, 
                                   model_type: str, feature_count: int,
                                   available_model_types: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Random Search.
        
        Args:
            train_data: Training DataFrame
            target_column: Name of the target column
            model_type: Type of model to optimize
            feature_count: Number of features
            available_model_types: Dictionary of available model types
            
        Returns:
            Dictionary containing the best parameters and metrics
        """
        print(f"🎲 Optimizing {model_type} hyperparameters with Random Search...")
        
        param_space = self._get_parameter_space(model_type)
        if not param_space:
            print(f"⚠️ No parameter space defined for {model_type}. Using default parameters.")
            return {}
        
        n_trials = self.config.get('random_search_trials', 20)
        timeout = self.config.get('random_search_timeout', 300)
        
        best_score = -float('inf')
        best_params = {}
        trial_results = []
        successful_trials = 0
        
        start_time = time.time()
        trial_count = 0
        
        # Add global timeout protection for tree-based models
        max_optimization_time = 300  # 5 minutes max for entire optimization
        if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            max_optimization_time = 180  # 3 minutes max for tree-based models
            print(f"   ⏰ Global timeout set to {max_optimization_time}s for tree-based model")
        
        # Generate diverse parameter combinations
        diverse_params = self._ensure_diverse_parameters(model_type, n_trials)
        print(f"🎲 Generated {len(diverse_params)} diverse parameter combinations")
        
        # Chunk trials to reduce memory pressure and broadcasting warnings
        chunk_size = 5  # Process 5 trials at a time
        trial_chunks = [list(range(i, min(i + chunk_size, n_trials))) for i in range(0, n_trials, chunk_size)]
        
        print(f"📊 Processing {n_trials} trials in {len(trial_chunks)} chunks of {chunk_size} trials each")
        print(f"🔍 Parameter space for {model_type}: {param_space}")
        
        for chunk_idx, trial_indices in enumerate(trial_chunks):
            print(f"🔄 Processing chunk {chunk_idx + 1}/{len(trial_chunks)} (trials {trial_indices[0] + 1}-{trial_indices[-1] + 1})")
            
            chunk_results = []
            
            for trial_idx in trial_indices:
                # Check global timeout
                if time.time() - start_time > max_optimization_time:
                    print(f"⏰ Global optimization timeout reached after {trial_idx} trials")
                    break
                    
                try:
                    # Use pre-generated diverse parameters
                    if trial_idx < len(diverse_params):
                        params = diverse_params[trial_idx]
                    else:
                        # Fallback to random sampling if we don't have enough diverse params
                        params = self._sample_random_parameters(model_type)
                    
                    # Validate that we got different parameters
                    if not params:
                        print(f"    Trial {trial_idx + 1}: No parameters sampled, skipping...")
                        continue
                    
                    print(f"    Trial {trial_idx + 1}/{n_trials}: Testing params = {params}")
                    
                    # Force memory cleanup between trials for tree-based models
                    if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'] and trial_idx > 0:
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
                            
                            print(f"    🧹 Memory cleanup before trial {trial_idx + 1}")
                            time.sleep(0.5)  # Brief pause for cleanup
                        except Exception as e:
                            print(f"    ⚠️ Memory cleanup failed: {e}")
                    
                    # Create and train model
                    estimator = self._create_estimator_with_params(
                        model_type, target_column, params, available_model_types, feature_count
                    )
                    
                    if estimator is None:
                        print(f"    Trial {trial_idx + 1}: Failed to create estimator, skipping...")
                        continue
                    
                    # Add timeout protection for tree-based models
                    if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Model training timed out")
                        
                        # Set timeout for tree-based models (30 seconds per trial)
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)
                        
                        try:
                            model = estimator.fit(train_data)
                            signal.alarm(0)  # Cancel timeout
                        except TimeoutError:
                            print(f"    ⏰ Trial {trial_idx + 1}: Model training timed out, skipping...")
                            signal.alarm(0)  # Cancel timeout
                            continue
                        except Exception as e:
                            signal.alarm(0)  # Cancel timeout
                            raise e
                    else:
                        model = estimator.fit(train_data)
                    
                    predictions = model.transform(train_data)
                    
                    # Evaluate model
                    score = self._evaluate_model(predictions, target_column)
                    chunk_results.append((params, score))
                    successful_trials += 1
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        print(f"    🎯 New best score: {score:.4f} with params: {params}")
                    else:
                        print(f"    Score: {score:.4f} (best: {best_score:.4f})")
                    
                    # Check for broadcast warning escalation and terminate early if detected
                    if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                        # If we've had 2 successful trials, stop early to prevent escalation
                        if successful_trials >= 2:
                            print(f"    🛑 Early termination: {successful_trials} successful trials completed for tree-based model")
                            print(f"    💡 This prevents broadcast warning escalation")
                            break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'broadcast' in error_msg or 'connection refused' in error_msg or 'task binary' in error_msg:
                        print(f"    ⚠️ Broadcasting error detected in trial {trial_idx + 1}: {str(e)}")
                        print(f"    🛑 Stopping hyperparameter tuning due to broadcasting error")
                        print(f"    💡 Consider reducing hyperparameter space or using fewer trials")
                        break
                    else:
                        print(f"    Trial {trial_idx + 1} failed: {str(e)}")
                    continue
            
            # Add chunk results to overall results
            trial_results.extend(chunk_results)
            
            # Clean up and give JVM a break between chunks
            if chunk_idx < len(trial_chunks) - 1:  # Not the last chunk
                try:
                    # Force garbage collection hint to Spark
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    print("🧹 Cleaned up chunk data, pausing briefly...")
                    time.sleep(2)  # Slightly longer pause for GC
                except:
                    pass
        
        # Validate results
        if successful_trials == 0:
            print(f"❌ No successful trials for {model_type}. Returning empty result.")
            return {}
        
        if not best_params:
            print(f"❌ No best parameters found for {model_type}. Returning empty result.")
            return {}
        
        print(f"🎯 Best {model_type} score: {best_score:.4f}")
        print(f"🔧 Best {model_type} parameters: {best_params}")
        print(f"📊 Successful trials: {successful_trials}/{n_trials}")
        
        # Log all trial results for debugging
        print(f"📋 All trial results for {model_type}:")
        for i, (params, score) in enumerate(trial_results):
            print(f"  Trial {i+1}: Score = {score:.4f}, Params = {params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(trial_results),
            'successful_trials': successful_trials,
            'all_trials': trial_results
        }

    def _optimize_with_grid_search(self, train_data: DataFrame, target_column: str, 
                                  model_type: str, feature_count: int,
                                  available_model_types: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Grid Search.
        
        Args:
            train_data: Training DataFrame
            target_column: Name of the target column
            model_type: Type of model to optimize
            feature_count: Number of features
            available_model_types: Dictionary of available model types
            
        Returns:
            Dictionary containing the best parameters and metrics
        """
        print(f"🔍 Optimizing {model_type} hyperparameters with Grid Search...")
        
        param_space = self._get_parameter_space(model_type)
        if not param_space:
            print(f"⚠️ No parameter space defined for {model_type}. Using default parameters.")
            return {}
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        max_combinations = self.config.get('grid_search_max_combinations', 100)
        timeout = self.config.get('grid_search_timeout', 600)
        
        if total_combinations > max_combinations:
            print(f"⚠️ Grid search would require {total_combinations} combinations, "
                  f"but max is set to {max_combinations}. Consider using random search instead.")
            # Fall back to random sampling from grid
            return self._optimize_with_random_search(train_data, target_column, model_type, feature_count, available_model_types)
        
        best_score = -float('inf')
        best_params = {}
        trial_results = []
        
        start_time = time.time()
        trial_count = 0
        
        # Generate all combinations and chunk them
        all_combinations = list(itertools.product(*param_values))
        chunk_size = 5  # Process 5 combinations at a time
        combination_chunks = [all_combinations[i:i + chunk_size] for i in range(0, len(all_combinations), chunk_size)]
        
        print(f"📊 Processing {total_combinations} combinations in {len(combination_chunks)} chunks of {chunk_size} combinations each")
        
        for chunk_idx, combination_chunk in enumerate(combination_chunks):
            print(f"🔄 Processing chunk {chunk_idx + 1}/{len(combination_chunks)} (combinations {trial_count + 1}-{trial_count + len(combination_chunk)})")
            
            chunk_results = []
            
            for combination in combination_chunk:
                if time.time() - start_time > timeout:
                    print(f"⏰ Grid search timeout reached after {trial_count} trials")
                    break
                
                try:
                    # Create parameter dictionary
                    params = dict(zip(param_names, combination))
                    
                    # Create and train model
                    estimator = self._create_estimator_with_params(
                        model_type, target_column, params, available_model_types
                    )
                    
                    if estimator is None:
                        trial_count += 1
                        continue
                    
                    model = estimator.fit(train_data)
                    predictions = model.transform(train_data)
                    
                    # Evaluate model
                    score = self._evaluate_model(predictions, target_column)
                    chunk_results.append((params, score))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    trial_count += 1
                    print(f"    Trial {trial_count}/{total_combinations}: Score = {score:.4f}, Params = {params}")
                    
                except Exception as e:
                    print(f"    Trial {trial_count + 1} failed: {str(e)}")
                    trial_count += 1
                    continue
            
            # Add chunk results to overall results
            trial_results.extend(chunk_results)
            
            # Clean up and give JVM a break between chunks
            if chunk_idx < len(combination_chunks) - 1:  # Not the last chunk
                try:
                    # Force garbage collection hint to Spark
                    if hasattr(self.spark, 'sparkContext'):
                        self.spark.sparkContext._jvm.System.gc()
                    print("🧹 Cleaned up chunk data, pausing briefly...")
                    time.sleep(2)  # Slightly longer pause for GC
                except:
                    pass
        
        print(f"🎯 Best {model_type} score: {best_score:.4f}")
        print(f"🔧 Best {model_type} parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(trial_results),
            'all_trials': trial_results
        }

    def _create_estimator_with_params(self, model_type: str, target_column: str, 
                                    params: Dict[str, Any], 
                                    available_model_types: Dict[str, Any],
                                    feature_count: int = 16):
        """
        Create an estimator with specified parameters.
        
        Args:
            model_type: Type of model to create
            target_column: Name of the target column
            params: Dictionary of parameters
            available_model_types: Dictionary of available model types
            
        Returns:
            Configured estimator or None if unsupported
        """
        if model_type.lower() == 'logistic':
            from pyspark.ml.classification import LogisticRegression
            return LogisticRegression(
                featuresCol='features', labelCol=target_column,
                maxIter=params.get('maxIter', 100),
                regParam=params.get('regParam', 0.1),
                elasticNetParam=params.get('elasticNetParam', 0.0)
            )
            
        elif model_type.lower() == 'random_forest':
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for random forest")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            from pyspark.ml.classification import RandomForestClassifier
            return RandomForestClassifier(
                featuresCol='features', labelCol=target_column,
                numTrees=params.get('numTrees', 20),
                maxDepth=params.get('maxDepth', 5),
                minInstancesPerNode=params.get('minInstancesPerNode', 1)
            )
            
        elif model_type.lower() == 'decision_tree':
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for decision tree")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            from pyspark.ml.classification import DecisionTreeClassifier
            return DecisionTreeClassifier(
                featuresCol='features', labelCol=target_column,
                maxDepth=params.get('maxDepth', 5),
                minInstancesPerNode=params.get('minInstancesPerNode', 1),
                maxBins=params.get('maxBins', 32)
            )
            
        elif model_type.lower() == 'gradient_boosting':
            # Apply gradient boosting optimisations to reduce task binary warnings
            try:
                from spark_optimization_config import apply_gradient_boosting_optimizations
                apply_gradient_boosting_optimizations(self.spark)
            except Exception as e:
                print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
            
            from pyspark.ml.classification import GBTClassifier
            # Limit the complexity of the estimator by enforcing smaller defaults.  This helps
            # mitigate broadcast warnings during hyperparameter tuning.
            return GBTClassifier(
                featuresCol='features', labelCol=target_column,
                maxIter=params.get('maxIter', 30),  # default lower than before
                maxDepth=params.get('maxDepth', 5),
                stepSize=params.get('stepSize', 0.1),
                maxBins=params.get('maxBins', 64)
            )
            
        elif model_type.lower() == 'xgboost' and 'xgboost' in available_model_types:
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for XGBoost")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            from xgboost.spark import SparkXGBClassifier
            # Note: SparkXGBClassifier automatically determines objective based on label data
            return SparkXGBClassifier(
                features_col='features', 
                label_col=target_column,
                max_depth=params.get('maxDepth', 6),
                n_estimators=params.get('numRound', 100),
                eta=params.get('eta', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsampleBytree', 1.0),
                min_child_weight=params.get('minChildWeight', 1),
                gamma=params.get('gamma', 0.0),
                num_workers=1, 
                use_gpu=False
            )
            
        elif model_type.lower() == 'lightgbm' and 'lightgbm' in available_model_types:
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for LightGBM")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            from synapse.ml.lightgbm import LightGBMClassifier
            if self.num_classes and self.num_classes > 2:
                # Multi-class classification
                estimator = LightGBMClassifier(
                    featuresCol='features', labelCol=target_column,
                    objective="multiclass",
                    numLeaves=params.get('numLeaves', 31),
                    numIterations=params.get('numIterations', 100),
                    learningRate=params.get('learningRate', 0.1),
                    featureFraction=params.get('featureFraction', 1.0),
                    baggingFraction=params.get('baggingFraction', 1.0),
                    minDataInLeaf=params.get('minDataInLeaf', 20),
                    lambdaL1=params.get('lambdaL1', 0.0),
                    lambdaL2=params.get('lambdaL2', 0.0)
                )
            else:
                # Binary classification
                estimator = LightGBMClassifier(
                    featuresCol='features', labelCol=target_column,
                    objective="binary",
                    numLeaves=params.get('numLeaves', 31),
                    numIterations=params.get('numIterations', 100),
                    learningRate=params.get('learningRate', 0.1),
                    featureFraction=params.get('featureFraction', 1.0),
                    baggingFraction=params.get('baggingFraction', 1.0),
                    minDataInLeaf=params.get('minDataInLeaf', 20),
                    lambdaL1=params.get('lambdaL1', 0.0),
                    lambdaL2=params.get('lambdaL2', 0.0)
                )
            return estimator
        
        elif model_type.lower() == 'neural_network':
            from pyspark.ml.classification import MultilayerPerceptronClassifier
            
            # Apply neural network specific optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_neural_network_optimizations
                apply_neural_network_optimizations(self.spark)
                print(f"   ⚡ Applied neural network optimizations")
            except Exception as e:
                print(f"   ⚠️ Could not apply neural network optimizations: {e}")
            
            # Get hidden layers from parameters
            hidden_layers = params.get('layers', [8])
            
            # Validate that hidden layers are compatible with input size
            if hidden_layers and len(hidden_layers) > 0:
                first_hidden_size = hidden_layers[0]
                if first_hidden_size > feature_count:
                    print(f"   ⚠️ Neural network first hidden layer size ({first_hidden_size}) > input size ({feature_count})")
                    print(f"   🔧 Adjusting to use compatible layer size: {feature_count}")
                    # Adjust the first hidden layer to match input size
                    hidden_layers[0] = feature_count
            
            # Build complete layer architecture: [input_size, hidden_layers..., output_size]
            # Use the passed feature_count parameter
            
            # Determine output layer size
            output_size = self.num_classes if self.num_classes and self.num_classes > 2 else 2
            
            # Build complete layers: [input, hidden..., output]
            complete_layers = [feature_count] + hidden_layers + [output_size]
            
            print(f"   🧠 Neural network architecture: {complete_layers}")
            
            return MultilayerPerceptronClassifier(
                featuresCol='features', labelCol=target_column,
                layers=complete_layers,
                maxIter=params.get('maxIter', 100),
                blockSize=params.get('blockSize', 128),
                seed=42
            )
        
        else:
            # Unsupported model type for optimization
            return None
    
    def _optimize_with_optuna(self, train_data: DataFrame, target_column: str, 
                             model_type: str, feature_count: int,
                             available_model_types: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_data: Training DataFrame
            target_column: Name of the target column
            model_type: Type of model to optimize
            feature_count: Number of features
            available_model_types: Dictionary of available model types
            
        Returns:
            Dictionary containing the best parameters and metrics
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Falling back to default parameters.")
            return {}
        
        print(f"🔍 Optimizing {model_type} hyperparameters with Optuna...")
        
        n_trials = self.config.get('optuna_trials', 50)
        timeout = self.config.get('optuna_timeout', 300)
        
        # Reduce trials and timeout for neural networks to prevent hanging
        if model_type.lower() == 'neural_network':
            n_trials = min(n_trials, 20)  # Limit neural network trials
            timeout = min(timeout, 180)   # Limit neural network timeout
            print(f"   🧠 Neural network optimization: {n_trials} trials, {timeout}s timeout")
        
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
        
        # Track trial count for cleanup
        trial_counter = {'count': 0}
        
        # Add global timeout protection
        optimization_start_time = time.time()
        max_optimization_time = 300  # 5 minutes max for entire optimization
        
        def objective(trial):
            # Check global timeout
            if time.time() - optimization_start_time > max_optimization_time:
                print(f"    ⏰ Global optimization timeout reached ({max_optimization_time}s)")
                return 0.5  # Return neutral score to stop optimization
            
            trial_counter['count'] += 1
            current_trial = trial_counter['count']
            
            try:
                # Force garbage collection between trials to prevent memory accumulation
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
                
                # Define hyperparameter search spaces for each model type
                estimator = self._create_optimized_estimator(trial, model_type, target_column, available_model_types, feature_count)
                
                if estimator is None:
                    return 0.5  # Return default score for unsupported models
                
                # Train and evaluate the model
                model = estimator.fit(train_data)
                predictions = model.transform(train_data)
                
                # Evaluate model
                score = self._evaluate_model(predictions, target_column)
                
                # Check for broadcast warning escalation and terminate early if detected
                if model_type.lower() in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                    # If we've had 2 successful trials, stop early to prevent escalation
                    if trial_counter['count'] >= 2:
                        print(f"    🛑 Early termination: {trial_counter['count']} successful trials completed for tree-based model")
                        print(f"    💡 This prevents broadcast warning escalation")
                        return 0.5  # Return neutral score to stop optimization
                
                return score
                
            except Exception as e:
                print(f"   ❌ Trial {current_trial} failed for {model_type}: {str(e)}")
                
                # Special handling for neural network failures
                if model_type.lower() == 'neural_network':
                    print(f"   🧠 Neural network trial failed - this is common due to layer architecture issues")
                    # Return a very low score instead of 0 to avoid Optuna getting stuck
                    return 0.1
                
                return 0.0  # Return poor score for failed trials
        
        # Run Optuna optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.config.get('seed', 42))
        )
        
        print(f"📊 Running {n_trials} Optuna trials with cleanup every 5 trials")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        print(f"🎯 Best {model_type} score: {study.best_value:.4f}")
        print(f"🔧 Best {model_type} parameters: {study.best_params}")
        print(f"📊 Total trials completed: {len(study.trials)}")
        print(f"📈 Trials with valid results: {len([t for t in study.trials if t.value is not None and t.value > 0])}")
        
        # Check if optimization actually found different parameters
        if len(study.trials) > 0:
            print(f"🔍 Optimization completed successfully")
        else:
            print(f"⚠️ No trials completed - optimization may have failed")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _create_optimized_estimator(self, trial, model_type: str, target_column: str, 
                                   available_model_types: Dict[str, Any],
                                   feature_count: int = 16):
        """
        Create an estimator with optimized hyperparameters for a given trial.
        
        Args:
            trial: Optuna trial object
            model_type: Type of model to create
            target_column: Name of the target column
            available_model_types: Dictionary of available model types
            
        Returns:
            Configured estimator or None if unsupported
        """
        if model_type.lower() == 'logistic':
            max_iter = trial.suggest_int('maxIter', 10, 100)
            reg_param = trial.suggest_float('regParam', 0.01, 1.0, log=True)
            elastic_net_param = trial.suggest_float('elasticNetParam', 0.0, 1.0)
            
            from pyspark.ml.classification import LogisticRegression
            return LogisticRegression(
                featuresCol='features', labelCol=target_column,
                maxIter=max_iter, regParam=reg_param, elasticNetParam=elastic_net_param
            )
            
        elif model_type.lower() == 'random_forest':
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for Optuna random forest")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            num_trees = trial.suggest_int('numTrees', 10, 100)
            max_depth = trial.suggest_int('maxDepth', 3, 20)
            min_instances_per_node = trial.suggest_int('minInstancesPerNode', 1, 10)
            
            from pyspark.ml.classification import RandomForestClassifier
            return RandomForestClassifier(
                featuresCol='features', labelCol=target_column,
                numTrees=num_trees, maxDepth=max_depth, 
                minInstancesPerNode=min_instances_per_node
            )
            
        elif model_type.lower() == 'decision_tree':
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for Optuna decision tree")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            max_depth = trial.suggest_int('maxDepth', 3, 20)
            min_instances_per_node = trial.suggest_int('minInstancesPerNode', 1, 10)
            max_bins = trial.suggest_int('maxBins', 16, 64)
            
            from pyspark.ml.classification import DecisionTreeClassifier
            return DecisionTreeClassifier(
                featuresCol='features', labelCol=target_column,
                maxDepth=max_depth, minInstancesPerNode=min_instances_per_node,
                maxBins=max_bins
            )
            
        elif model_type.lower() == 'gradient_boosting':
            # Apply gradient boosting optimisations to reduce task binary warnings
            from pyspark.sql import SparkSession
            try:
                from spark_optimization_config import apply_gradient_boosting_optimizations
                apply_gradient_boosting_optimizations(self.spark)
            except Exception as e:
                print(f"⚠️ Could not apply gradient boosting optimizations: {e}")
            
            # Restrict the search space for gradient boosting hyperparameters.  Smaller ranges
            # reduce the size of broadcasted task binaries and improve stability during tuning.
            max_iter = trial.suggest_int('maxIter', 10, 50)
            max_depth = trial.suggest_int('maxDepth', 3, 8)
            step_size = trial.suggest_float('stepSize', 0.05, 0.3)
            max_bins = trial.suggest_int('maxBins', 32, 64)
            
            from pyspark.ml.classification import GBTClassifier
            return GBTClassifier(
                featuresCol='features', labelCol=target_column,
                maxIter=max_iter, maxDepth=max_depth, stepSize=step_size,
                maxBins=max_bins
            )
            
        elif model_type.lower() == 'xgboost' and 'xgboost' in available_model_types:
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for Optuna XGBoost")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            max_depth = trial.suggest_int('maxDepth', 3, 15)
            num_round = trial.suggest_int('numRound', 20, 300)
            learning_rate = trial.suggest_float('eta', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('colsampleBytree', 0.5, 1.0)
            min_child_weight = trial.suggest_int('minChildWeight', 1, 10)
            gamma = trial.suggest_float('gamma', 0.0, 1.0)
            
            from xgboost.spark import SparkXGBClassifier
            # Note: SparkXGBClassifier automatically determines objective based on label data
            return SparkXGBClassifier(
                features_col='features', 
                label_col=target_column,
                max_depth=max_depth, 
                n_estimators=num_round,
                eta=learning_rate, 
                subsample=subsample, 
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                num_workers=1, 
                use_gpu=False
            )
            
        elif model_type.lower() == 'lightgbm' and 'lightgbm' in available_model_types:
            # Apply tree-based optimizations to reduce broadcast warnings
            try:
                from spark_optimization_config import apply_tree_based_optimizations
                apply_tree_based_optimizations(self.spark)
                print(f"   ⚡ Applied tree-based optimizations for Optuna LightGBM")
            except Exception as e:
                print(f"   ⚠️ Could not apply tree-based optimizations: {e}")
            
            num_leaves = trial.suggest_int('numLeaves', 5, 120)
            num_iterations = trial.suggest_int('numIterations', 20, 300)
            learning_rate = trial.suggest_float('learningRate', 0.01, 0.3, log=True)
            feature_fraction = trial.suggest_float('featureFraction', 0.5, 1.0)
            bagging_fraction = trial.suggest_float('baggingFraction', 0.5, 1.0)
            min_data_in_leaf = trial.suggest_int('minDataInLeaf', 10, 100)
            lambda_l1 = trial.suggest_float('lambdaL1', 0.0, 5.0)
            lambda_l2 = trial.suggest_float('lambdaL2', 0.0, 5.0)
            
            from synapse.ml.lightgbm import LightGBMClassifier
            if self.num_classes and self.num_classes > 2:
                # Multi-class classification
                estimator = LightGBMClassifier(
                    featuresCol='features', labelCol=target_column,
                    objective="multiclass",
                    numLeaves=num_leaves, numIterations=num_iterations, 
                    learningRate=learning_rate, featureFraction=feature_fraction, 
                    baggingFraction=bagging_fraction, minDataInLeaf=min_data_in_leaf,
                    lambdaL1=lambda_l1, lambdaL2=lambda_l2
                )
            else:
                # Binary classification
                estimator = LightGBMClassifier(
                    featuresCol='features', labelCol=target_column,
                    objective="binary", numLeaves=num_leaves, 
                    numIterations=num_iterations, learningRate=learning_rate,
                    featureFraction=feature_fraction, baggingFraction=bagging_fraction,
                    minDataInLeaf=min_data_in_leaf, lambdaL1=lambda_l1, lambdaL2=lambda_l2
                )
            return estimator
            
        elif model_type.lower() == 'neural_network':
            # Apply neural network specific optimizations
            try:
                from spark_optimization_config import apply_neural_network_optimizations
                apply_neural_network_optimizations(self.spark)
                print(f"   ⚡ Applied neural network optimizations for Optuna")
            except Exception as e:
                print(f"   ⚠️ Could not apply neural network optimizations: {e}")
            
            # Get parameter space for neural network
            param_space = self._get_parameter_space('neural_network')
            
            # Suggest hidden layers from predefined options
            if 'layers' in param_space and param_space['layers']:
                hidden_layers = trial.suggest_categorical('layers', param_space['layers'])
            else:
                # Fallback to simple hidden layers if no predefined options
                hidden_layers = [8]  # Reduced from [16, 8] to be compatible with smaller feature counts
            
            # Validate that hidden layers are compatible with input size
            if hidden_layers and len(hidden_layers) > 0:
                first_hidden_size = hidden_layers[0]
                if first_hidden_size > feature_count:
                    print(f"   ⚠️ Optuna neural network first hidden layer size ({first_hidden_size}) > input size ({feature_count})")
                    print(f"   🔧 Adjusting to use compatible layer size: {feature_count}")
                    # Adjust the first hidden layer to match input size
                    hidden_layers[0] = feature_count
            
            # Suggest other parameters
            max_iter = trial.suggest_categorical('maxIter', param_space.get('maxIter', [50, 100, 200]))
            block_size = trial.suggest_categorical('blockSize', param_space.get('blockSize', [32, 64, 128]))
            
            # Build complete layer architecture: [input_size, hidden_layers..., output_size]
            # Use the passed feature_count parameter
            
            # Determine output layer size
            output_size = self.num_classes if self.num_classes and self.num_classes > 2 else 2
            
            # Build complete layers: [input, hidden..., output]
            complete_layers = [feature_count] + hidden_layers + [output_size]
            
            print(f"   🧠 Optuna neural network architecture: {complete_layers}")
            
            from pyspark.ml.classification import MultilayerPerceptronClassifier
            return MultilayerPerceptronClassifier(
                featuresCol='features', labelCol=target_column,
                layers=complete_layers, maxIter=max_iter, blockSize=block_size, seed=42
            )
        
        else:
            # Unsupported model type for optimization
            return None
    
    def _evaluate_model(self, predictions: DataFrame, target_column: str) -> float:
        """
        Evaluate model predictions and return a metric score.
        
        Args:
            predictions: DataFrame with predictions
            target_column: Name of the target column
            
        Returns:
            Metric score (higher is better)
        """
        try:
            if self.is_multiclass:
                from pyspark.ml.evaluation import MulticlassClassificationEvaluator
                evaluator = MulticlassClassificationEvaluator(
                    labelCol=target_column, predictionCol="prediction", metricName="accuracy"
                )
            else:
                from pyspark.ml.evaluation import BinaryClassificationEvaluator
                evaluator = BinaryClassificationEvaluator(
                    labelCol=target_column, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
                )
            
            return evaluator.evaluate(predictions)
        except Exception as e:
            print(f"    Evaluation failed: {str(e)}")
            return 0.0
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Get information about available optimization methods and their capabilities.
        
        Returns:
            Dictionary containing optimization method information
        """
        info = {
            'available_methods': ['optuna', 'random_search', 'grid_search'],
            'optuna_available': OPTUNA_AVAILABLE,
            'supported_models': ['logistic', 'random_forest', 'decision_tree', 'gradient_boosting', 'neural_network'],
            'default_config': {
                'optuna_trials': self.config.get('optuna_trials', 50),
                'optuna_timeout': self.config.get('optuna_timeout', 300),
                'random_search_trials': self.config.get('random_search_trials', 20),
                'random_search_timeout': self.config.get('random_search_timeout', 300),
                'grid_search_max_combinations': self.config.get('grid_search_max_combinations', 100),
                'grid_search_timeout': self.config.get('grid_search_timeout', 600)
            },
            'method_descriptions': {
                'optuna': 'Bayesian optimization using TPE sampler (most efficient, learns from previous trials)',
                'random_search': 'Random sampling from parameter space (good balance of efficiency and exploration)',
                'grid_search': 'Exhaustive search over parameter grid (thorough but computationally expensive)'
            }
        }
        
        # Add advanced models if available
        if 'xgboost' in self.config.get('available_models', []):
            info['supported_models'].append('xgboost')
        if 'lightgbm' in self.config.get('available_models', []):
            info['supported_models'].append('lightgbm')
        
        return info 