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
        optimization_method = self.config.get('optimization_method', 'optuna')
        
        if optimization_method == 'optuna':
            return self._optimize_with_optuna(
                train_data, target_column, model_type, feature_count, available_model_types
            )
        elif optimization_method == 'random_search':
            return self._optimize_with_random_search(
                train_data, target_column, model_type, feature_count, available_model_types
            )
        elif optimization_method == 'grid_search':
            return self._optimize_with_grid_search(
                train_data, target_column, model_type, feature_count, available_model_types
            )
        else:
            print(f"⚠️ Optimization method '{optimization_method}' not implemented. Using default parameters.")
            return {}

    def _get_parameter_space(self, model_type: str) -> Dict[str, List]:
        """
        Get parameter search space for a given model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing parameter names and their possible values
        """
        parameter_spaces = {
            'logistic': {
                'maxIter': [10, 20, 50, 100],
                'regParam': [0.01, 0.1, 0.5, 1.0],
                'elasticNetParam': [0.0, 0.25, 0.5, 0.75, 1.0]
            },
            'random_forest': {
                'numTrees': [10, 20, 50, 100],
                'maxDepth': [3, 5, 10, 15, 20],
                'minInstancesPerNode': [1, 2, 5, 10]
            },
            'decision_tree': {
                'maxDepth': [3, 5, 10, 15, 20],
                'minInstancesPerNode': [1, 2, 5, 10],
                'maxBins': [16, 32, 64]
            },
            'gradient_boosting': {
                'maxIter': [10, 20, 50, 100],
                'maxDepth': [3, 5, 8, 10],
                'stepSize': [0.01, 0.05, 0.1, 0.2, 0.3]
            },
            'xgboost': {
                'maxDepth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
                'numRound': [30, 50, 75, 100, 125, 150, 175, 200, 250, 300],
                'eta': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3],
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'colsampleBytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'minChildWeight': [1, 2, 3, 5, 7, 10],
                'gamma': [0, 0.1, 0.2, 0.3, 0.5, 1.0]
            },
            'lightgbm': {
                'numLeaves': [5, 10, 15, 20, 25, 31, 40, 50, 60, 80, 100, 120],
                'numIterations': [30, 50, 75, 100, 125, 150, 175, 200, 250, 300],
                'learningRate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3],
                'featureFraction': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'baggingFraction': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'minDataInLeaf': [10, 20, 30, 50, 100],
                'lambdaL1': [0, 0.1, 0.5, 1.0, 2.0, 5.0],
                'lambdaL2': [0, 0.1, 0.5, 1.0, 2.0, 5.0]
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
        
        start_time = time.time()
        
        # Chunk trials to reduce memory pressure and broadcasting warnings
        chunk_size = 5  # Process 5 trials at a time
        trial_chunks = [list(range(i, min(i + chunk_size, n_trials))) for i in range(0, n_trials, chunk_size)]
        
        print(f"📊 Processing {n_trials} trials in {len(trial_chunks)} chunks of {chunk_size} trials each")
        
        for chunk_idx, trial_chunk in enumerate(trial_chunks):
            print(f"🔄 Processing chunk {chunk_idx + 1}/{len(trial_chunks)} (trials {trial_chunk[0] + 1}-{trial_chunk[-1] + 1})")
            
            chunk_results = []
            
            for trial_idx in trial_chunk:
                if time.time() - start_time > timeout:
                    print(f"⏰ Random search timeout reached after {trial_idx} trials")
                    break
                    
                try:
                    # Sample random parameters
                    params = self._sample_random_parameters(model_type)
                    
                    # Create and train model
                    estimator = self._create_estimator_with_params(
                        model_type, target_column, params, available_model_types
                    )
                    
                    if estimator is None:
                        continue
                    
                    model = estimator.fit(train_data)
                    predictions = model.transform(train_data)
                    
                    # Evaluate model
                    score = self._evaluate_model(predictions, target_column)
                    chunk_results.append((params, score))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    print(f"    Trial {trial_idx + 1}/{n_trials}: Score = {score:.4f}, Params = {params}")
                    
                except Exception as e:
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
        
        print(f"🎯 Best {model_type} score: {best_score:.4f}")
        print(f"🔧 Best {model_type} parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(trial_results),
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
                                    available_model_types: Dict[str, Any]):
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
            from pyspark.ml.classification import RandomForestClassifier
            return RandomForestClassifier(
                featuresCol='features', labelCol=target_column,
                numTrees=params.get('numTrees', 20),
                maxDepth=params.get('maxDepth', 5),
                minInstancesPerNode=params.get('minInstancesPerNode', 1)
            )
            
        elif model_type.lower() == 'decision_tree':
            from pyspark.ml.classification import DecisionTreeClassifier
            return DecisionTreeClassifier(
                featuresCol='features', labelCol=target_column,
                maxDepth=params.get('maxDepth', 5),
                minInstancesPerNode=params.get('minInstancesPerNode', 1),
                maxBins=params.get('maxBins', 32)
            )
            
        elif model_type.lower() == 'gradient_boosting':
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
            
            from pyspark.ml.classification import GBTClassifier
            return GBTClassifier(
                featuresCol='features', labelCol=target_column,
                maxIter=params.get('maxIter', 20),
                maxDepth=params.get('maxDepth', 5),
                stepSize=params.get('stepSize', 0.1),
                maxBins=params.get('maxBins', 64)  # Added maxBins parameter with optimized default
            )
            
        elif model_type.lower() == 'xgboost' and 'xgboost' in available_model_types:
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
        
        # Apply Spark optimizations specifically for hyperparameter tuning
        try:
            from spark_optimization_config import optimize_for_hyperparameter_tuning
            print(f"   ⚡ Applying Spark optimizations for {model_type} hyperparameter tuning...")
            optimize_for_hyperparameter_tuning(self.spark, model_type, n_trials)
        except Exception as e:
            print(f"   ⚠️  Could not apply tuning optimizations: {e}")
        
        # Track trial count for cleanup
        trial_counter = {'count': 0}
        
        def objective(trial):
            trial_counter['count'] += 1
            current_trial = trial_counter['count']
            
            try:
                # Define hyperparameter search spaces for each model type
                estimator = self._create_optimized_estimator(trial, model_type, target_column, available_model_types)
                
                if estimator is None:
                    return 0.5  # Return default score for unsupported models
                
                # Train and evaluate the model
                model = estimator.fit(train_data)
                predictions = model.transform(train_data)
                
                # Calculate metric based on problem type
                metric = self._evaluate_model(predictions, target_column)
                
                # Enhanced cleanup every 3 trials for tree-based models, 5 for others
                cleanup_interval = 3 if model_type in ['gradient_boosting', 'random_forest', 'xgboost', 'lightgbm'] else 5
                
                if current_trial % cleanup_interval == 0:
                    try:
                        # Force garbage collection hint to Spark
                        if hasattr(self.spark, 'sparkContext'):
                            self.spark.sparkContext._jvm.System.gc()
                        
                        # Additional cleanup for tree-based models
                        if model_type in ['gradient_boosting', 'random_forest', 'xgboost', 'lightgbm']:
                            # Clear any cached DataFrames
                            train_data.unpersist()
                            predictions.unpersist()
                            import gc
                            gc.collect()
                            time.sleep(1.5)  # Longer pause for tree models
                        
                        print(f"   🧹 Enhanced cleanup after trial {current_trial} (interval: {cleanup_interval})")
                        time.sleep(1)  # Brief pause for GC
                    except:
                        pass
                
                return metric
                
            except Exception as e:
                print(f"    Trial {current_trial} failed: {str(e)}")
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
                                   available_model_types: Dict[str, Any]):
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
            
            max_iter = trial.suggest_int('maxIter', 10, 100)
            max_depth = trial.suggest_int('maxDepth', 3, 10)
            step_size = trial.suggest_float('stepSize', 0.01, 0.3)
            max_bins = trial.suggest_int('maxBins', 32, 128)  # Reduced range to minimize task binary size
            
            from pyspark.ml.classification import GBTClassifier
            return GBTClassifier(
                featuresCol='features', labelCol=target_column,
                maxIter=max_iter, maxDepth=max_depth, stepSize=step_size,
                maxBins=max_bins  # Added maxBins parameter with optimized range
            )
            
        elif model_type.lower() == 'xgboost' and 'xgboost' in available_model_types:
            max_depth = trial.suggest_int('maxDepth', 3, 15)
            num_round = trial.suggest_int('numRound', 30, 300)
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
            num_leaves = trial.suggest_int('numLeaves', 5, 120)
            num_iterations = trial.suggest_int('numIterations', 30, 300)
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
            'supported_models': ['logistic', 'random_forest', 'decision_tree', 'gradient_boosting'],
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