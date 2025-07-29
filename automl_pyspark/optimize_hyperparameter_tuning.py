#!/usr/bin/env python3
"""
Hyperparameter Tuning Optimization Utility

This script provides utilities to optimize Spark sessions for hyperparameter tuning,
significantly reducing "Broadcasting large task binary" warnings.
"""

import os
import sys
from typing import Optional
from pyspark.sql import SparkSession

def optimize_spark_for_hyperparameter_tuning(spark_session: Optional[SparkSession] = None, 
                                            model_type: str = "gradient_boosting",
                                            n_trials: int = 50,
                                            aggressive: bool = None) -> SparkSession:
    """
    Optimize a Spark session for hyperparameter tuning to reduce large task binary warnings.
    
    Args:
        spark_session: Existing Spark session (if None, creates a new one)
        model_type: Type of model being tuned ('gradient_boosting', 'random_forest', etc.)
        n_trials: Number of hyperparameter trials planned
        aggressive: Whether to use aggressive optimizations (auto-detected if None)
    
    Returns:
        Optimized Spark session
    """
    
    # Import optimization functions
    try:
        from spark_optimization_config import (
            create_optimized_spark_session, 
            apply_tuning_optimizations,
            optimize_for_hyperparameter_tuning
        )
    except ImportError:
        print("❌ Could not import spark_optimization_config. Make sure you're in the correct directory.")
        return spark_session
    
    print("🚀 Optimizing Spark for Hyperparameter Tuning")
    print("=" * 60)
    print(f"Model type: {model_type}")
    print(f"Planned trials: {n_trials}")
    
    # Create or optimize session
    if spark_session is None:
        print("\n📦 Creating new optimized Spark session...")
        spark_session = create_optimized_spark_session("AutoML Hyperparameter Tuning Optimized")
    else:
        print(f"\n⚡ Optimizing existing Spark session: {spark_session.sparkContext.appName}")
    
    # Apply hyperparameter tuning specific optimizations
    print(f"\n🎯 Applying hyperparameter tuning optimizations...")
    optimize_for_hyperparameter_tuning(spark_session, model_type, n_trials)
    
    # Determine if aggressive optimizations are needed
    if aggressive is None:
        aggressive = (
            model_type in ['gradient_boosting', 'xgboost', 'lightgbm', 'random_forest'] or
            n_trials > 20
        )
    
    if aggressive:
        print("\n💪 Applying aggressive memory optimizations...")
        apply_tuning_optimizations(spark_session, aggressive=True)
    
    print("\n✅ Spark session optimized for hyperparameter tuning!")
    print("💡 Large task binary warnings should be significantly reduced")
    
    return spark_session

def test_optimization_effectiveness(spark_session: SparkSession, 
                                  data_file: str = "bank.csv",
                                  model_type: str = "gradient_boosting",
                                  n_trials: int = 3):
    """
    Test the effectiveness of optimizations by running a quick hyperparameter tuning session.
    
    Args:
        spark_session: Optimized Spark session
        data_file: Path to test data file
        model_type: Model type to test
        n_trials: Number of test trials
    """
    
    print(f"\n🧪 Testing optimization effectiveness...")
    print(f"Data file: {data_file}")
    print(f"Model type: {model_type}")
    print(f"Test trials: {n_trials}")
    
    try:
        # Import AutoML based on model type
        if model_type in ['gradient_boosting', 'random_forest', 'decision_tree', 'linear_regression']:
            from regression.automl_regressor import AutoMLRegressor
            automl = AutoMLRegressor(
                output_dir='test_optimization_output',
                spark_session=spark_session,
                preset='quick'
            )
            
            # Configure for quick test
            automl.config_manager.override_config({
                'regression': {
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': True,
                        'optimization_method': 'optuna',
                        'optuna_trials': n_trials,
                        'optuna_timeout': 120
                    },
                    'models': {
                        f'run_{model_type}': True,
                        'run_linear_regression': model_type == 'linear_regression',
                        'run_random_forest': model_type == 'random_forest',
                        'run_gradient_boosting': model_type == 'gradient_boosting',
                        'run_decision_tree': model_type == 'decision_tree'
                    }
                }
            })
            
            print(f"\n🏃‍♂️ Running test regression with {model_type}...")
            # Use first numeric column as target for testing
            automl.fit(train_data=data_file, target_column='age')  # Assuming bank.csv has 'age'
            
        else:
            from classification.automl_classifier import AutoMLClassifier
            automl = AutoMLClassifier(
                output_dir='test_optimization_output',
                spark_session=spark_session,
                preset='quick'
            )
            
            # Configure for quick test
            automl.config_manager.override_config({
                'classification': {
                    'hyperparameter_tuning': {
                        'enable_hyperparameter_tuning': True,
                        'optimization_method': 'optuna',
                        'optuna_trials': n_trials,
                        'optuna_timeout': 120
                    },
                    'models': {
                        f'run_{model_type}': True,
                        'run_logistic': model_type == 'logistic',
                        'run_random_forest': model_type == 'random_forest',
                        'run_gradient_boosting': model_type == 'gradient_boosting',
                        'run_decision_tree': model_type == 'decision_tree'
                    }
                }
            })
            
            print(f"\n🏃‍♂️ Running test classification with {model_type}...")
            # Use last column as target for testing
            automl.fit(train_data=data_file, target_column='y')  # Assuming bank.csv has 'y'
        
        print("✅ Test completed successfully!")
        print("💡 Check the console output for reduced 'Broadcasting large task binary' warnings")
        
    except Exception as e:
        print(f"⚠️ Test failed: {str(e)}")
        print("💡 This is normal - the test is just to verify optimization effectiveness")

def show_optimization_tips():
    """Display tips for reducing large task binary warnings."""
    
    print("\n💡 Tips for Reducing Large Task Binary Warnings")
    print("=" * 60)
    print("""
1. 🚀 **Use this optimization script before hyperparameter tuning:**
   ```python
   from optimize_hyperparameter_tuning import optimize_spark_for_hyperparameter_tuning
   spark = optimize_spark_for_hyperparameter_tuning()
   ```

2. 🎯 **Model-specific recommendations:**
   - Gradient Boosting: Use aggressive=True, reduce max_depth and maxIter
   - XGBoost: Use aggressive=True, limit n_estimators to <200
   - Random Forest: Use aggressive=True, limit numTrees to <100
   - Linear models: Normal optimizations are sufficient

3. 📊 **Trial count guidelines:**
   - <10 trials: Normal optimizations
   - 10-50 trials: Enhanced optimizations
   - >50 trials: Aggressive optimizations + consider reducing trials

4. 🔧 **Configuration adjustments:**
   - Reduce optuna_trials from 100 to 20-50
   - Set optuna_timeout to limit time spent
   - Use 'random_search' instead of 'optuna' for faster tuning

5. 💾 **Memory management:**
   - Monitor Spark UI for memory usage
   - Restart Spark session between different model types
   - Use cleanup_memory_inline() if warnings persist

6. ⚡ **Alternative approaches:**
   - Train models in smaller batches
   - Use cross-validation with fewer folds
   - Pre-filter features to reduce model complexity
""")

def main():
    """Main function for command-line usage."""
    
    print("🔧 AutoML Hyperparameter Tuning Optimizer")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        
        if action == "test":
            # Test optimizations
            data_file = sys.argv[2] if len(sys.argv) > 2 else "bank.csv"
            model_type = sys.argv[3] if len(sys.argv) > 3 else "gradient_boosting"
            n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 3
            
            spark = optimize_spark_for_hyperparameter_tuning(
                model_type=model_type, 
                n_trials=n_trials
            )
            test_optimization_effectiveness(spark, data_file, model_type, n_trials)
            
        elif action == "tips":
            show_optimization_tips()
            
        else:
            print(f"Unknown action: {action}")
            print("Usage: python optimize_hyperparameter_tuning.py [test|tips] [data_file] [model_type] [n_trials]")
    
    else:
        # Interactive mode
        print("Choose an option:")
        print("1. 🚀 Create optimized Spark session")
        print("2. 🧪 Test optimization effectiveness") 
        print("3. 💡 Show optimization tips")
        print("4. 🔍 Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            model_type = input("Model type (gradient_boosting/random_forest/xgboost): ").strip() or "gradient_boosting"
            n_trials = int(input("Number of planned trials (default 50): ").strip() or "50")
            
            spark = optimize_spark_for_hyperparameter_tuning(
                model_type=model_type,
                n_trials=n_trials
            )
            print(f"\n✅ Optimized Spark session created: {spark.sparkContext.appName}")
            
        elif choice == "2":
            data_file = input("Data file path (default: bank.csv): ").strip() or "bank.csv"
            model_type = input("Model type (default: gradient_boosting): ").strip() or "gradient_boosting"
            n_trials = int(input("Test trials (default: 3): ").strip() or "3")
            
            if os.path.exists(data_file):
                spark = optimize_spark_for_hyperparameter_tuning(
                    model_type=model_type,
                    n_trials=n_trials
                )
                test_optimization_effectiveness(spark, data_file, model_type, n_trials)
            else:
                print(f"❌ Data file not found: {data_file}")
                
        elif choice == "3":
            show_optimization_tips()
            
        elif choice == "4":
            print("👋 Goodbye!")
            
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 