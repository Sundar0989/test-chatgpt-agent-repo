"""
Model Validator

Class responsible for model validation and metrics calculation.
This class encapsulates the validation functionality from the original modules.
"""

import os
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from pyspark.sql import SparkSession, DataFrame

# Import functions from original modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import using flexible import strategy for both package and direct execution
try:
    from automl_pyspark.metrics_calculator import calculate_metrics
    from automl_pyspark.validation_and_plots import model_validation, draw_ks_plot, draw_roc_plot, draw_confusion_matrix
except ImportError:
    # Fallback to direct imports (for direct script execution)
    import sys
    import os
    
    # Add parent directory to path if not already there
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from metrics_calculator import calculate_metrics
    from validation_and_plots import model_validation, draw_ks_plot, draw_roc_plot, draw_confusion_matrix


class ModelValidator:
    """
    Model validator class that handles model validation and performance evaluation.
    
    This class provides functionality for:
    - Model validation on multiple datasets
    - Performance metrics calculation
    - Visualization generation
    - Organized output file management
    """
    
    def __init__(self, spark_session: SparkSession, output_dir: str, 
                 user_id: str, model_literal: str):
        """
        Initialize the model validator.
        
        Args:
            spark_session: PySpark SparkSession
            output_dir: Output directory for results
            user_id: User identifier
            model_literal: Model literal/tag
        """
        self.spark = spark_session
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def validate_model(self, model: Any, datasets: List[DataFrame], 
                      dataset_names: List[str], target_column: str, 
                      model_type: str) -> Dict[str, float]:
        """
        Validate a model on multiple datasets using comprehensive validation.
        
        Args:
            model: Trained model to validate
            datasets: List of datasets to validate on
            dataset_names: Names of the datasets
            target_column: Name of the target column
            model_type: Type of model being validated
            
        Returns:
            Dictionary containing metrics for each dataset
        """
        print(f"Validating {model_type} model on {len(datasets)} datasets...")
        
        # Create model output directory
        model_output_dir = os.path.join(self.output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Validate on each dataset using the original validation function
        all_metrics = []
        for dataset, dataset_name in zip(datasets, dataset_names):
            if dataset is not None:
                print(f"Validating on {dataset_name} dataset...")
                
                # Use the new package-compatible model_validation function
                metrics = model_validation(
                    model_output_dir, dataset, target_column, 
                    model, model_type, dataset_name
                )
                
                # Handle case where model_validation returns None due to failure
                if metrics is not None:
                    all_metrics.extend(metrics)
                else:
                    # Add default metrics for failed validation
                    print(f"⚠️ Model validation failed for {dataset_name}, using default metrics")
                    all_metrics.extend([0.0, 0.0, 0.0])
                
            else:
                # Add default metrics for missing dataset
                all_metrics.extend([0.0, 0.0, 0.0])
        
        # Create metrics dictionary
        metrics_dict = self._create_metrics_dict(all_metrics, dataset_names)
        
        # Save metrics
        self._save_metrics(metrics_dict, model_type, model_output_dir)
        
        # Generate plots using the original functions
        self._generate_plots(model_type, model_output_dir)
        
        print(f"{model_type} model validation completed.")
        return metrics_dict
    
    def _create_metrics_dict(self, all_metrics: List[float], dataset_names: List[str]) -> Dict[str, float]:
        """
        Create metrics dictionary from metrics list.
        
        Args:
            all_metrics: List of metrics [roc, accuracy, ks] for each dataset
            dataset_names: Names of the datasets
            
        Returns:
            Dictionary with metrics organized by dataset
        """
        metrics_dict = {}
        
        # Create metrics dictionary based on available datasets
        for i, dataset_name in enumerate(dataset_names):
            if i * 3 + 2 < len(all_metrics):
                metrics_dict[f'roc_{dataset_name}'] = all_metrics[i * 3]
                metrics_dict[f'accuracy_{dataset_name}'] = all_metrics[i * 3 + 1]
                metrics_dict[f'ks_{dataset_name}'] = all_metrics[i * 3 + 2]
            # Don't populate missing datasets with 0 values
        
        return metrics_dict
    
    def _save_metrics(self, metrics: Dict[str, float], model_type: str, model_output_dir: str):
        """
        Save metrics to file in model-specific directory.
        
        Args:
            metrics: Dictionary of metrics
            model_type: Type of model
            model_output_dir: Model-specific output directory
        """
        try:
            # Convert to list format expected by original code
            metrics_list = []
            for dataset in ['train', 'valid', 'test', 'oot1', 'oot2']:
                for metric in ['roc', 'accuracy', 'ks']:
                    key = f'{metric}_{dataset}'
                    # Only include metrics that exist, don't populate with 0.0
                    if key in metrics:
                        metrics_list.append(metrics[key])
                    else:
                        # Skip missing datasets instead of adding 0.0
                        continue
            
            # Save using joblib (as in original code)
            import joblib
            output_file = os.path.join(model_output_dir, f'{model_type}_metrics.z')
            joblib.dump(metrics_list, output_file)
            
            # Also save as JSON for easier reading
            import json
            json_file = os.path.join(model_output_dir, f'{model_type}_metrics.json')
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Metrics saved to {output_file} and {json_file}")
            
        except Exception as e:
            print(f"Error saving metrics for {model_type}: {str(e)}")
    
    def _generate_plots(self, model_type: str, model_output_dir: str):
        """
        Generate visualization plots for a model.
        
        Args:
            model_type: Type of model
            model_output_dir: Model-specific output directory
        """
        try:
            # Generate KS plot using the new package-compatible function
            draw_ks_plot(model_output_dir, model_type)
            
            # Create a summary report
            self._create_validation_summary(model_type, model_output_dir)
            
        except Exception as e:
            print(f"Error generating plots for {model_type}: {str(e)}")
    
    def _create_validation_summary(self, model_type: str, model_output_dir: str):
        """
        Create a validation summary report.
        
        Args:
            model_type: Type of model
            model_output_dir: Model-specific output directory
        """
        try:
            # Read metrics
            import joblib
            metrics_file = os.path.join(model_output_dir, f'{model_type}_metrics.z')
            if os.path.exists(metrics_file):
                metrics_list = joblib.load(metrics_file)
                
                # Create summary
                summary_file = os.path.join(model_output_dir, f'{model_type}_validation_summary.txt')
                with open(summary_file, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"VALIDATION SUMMARY - {model_type.upper()}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    datasets = ['train', 'valid', 'test', 'oot1', 'oot2']
                    for i, dataset in enumerate(datasets):
                        if i * 3 + 2 < len(metrics_list):
                            f.write(f"{dataset.upper()} DATASET:\n")
                            f.write(f"  ROC: {metrics_list[i * 3]:.4f}\n")
                            f.write(f"  Accuracy: {metrics_list[i * 3 + 1]:.4f}\n")
                            f.write(f"  KS: {metrics_list[i * 3 + 2]:.4f}\n\n")
                    
                    f.write("=" * 60 + "\n")
                    f.write("Generated files:\n")
                    f.write("- " + f"{model_type}_metrics.z (joblib format)\n")
                    f.write("- " + f"{model_type}_metrics.json (JSON format)\n")
                    f.write("- " + f"decile_table_*.xlsx (for each dataset)\n")
                    f.write("- " + f"ks_plot_{model_type}.png\n")
                    f.write("=" * 60 + "\n")
                
                print(f"Validation summary saved to {summary_file}")
                
        except Exception as e:
            print(f"Error creating validation summary for {model_type}: {str(e)}")
    

    
    def get_validation_summary(self, model_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Get a summary of validation results for all models.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            
        Returns:
            DataFrame with validation summary
        """
        # Create DataFrame from model_metrics without predefined columns
        data = []
        for model_type, metrics in model_metrics.items():
            row = {'model_type': model_type}
            for key, value in metrics.items():
                row[key] = value
            data.append(row)

        df = pd.DataFrame(data)
        return df
    
    def compare_models(self, model_metrics: Dict[str, Dict[str, float]], 
                      metric: str = 'ks', dataset: str = 'test') -> pd.DataFrame:
        """
        Compare models based on a specific metric.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            metric: Metric to compare ('roc', 'accuracy', 'ks')
            dataset: Dataset to compare on ('train', 'valid', 'test', 'oot1', 'oot2')
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_type, metrics in model_metrics.items():
            metric_key = f'{metric}_{dataset}'
            value = metrics.get(metric_key, 0.0)
            comparison_data.append({
                'model_type': model_type,
                'metric': metric,
                'dataset': dataset,
                'value': value
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('value', ascending=False)
        
        return comparison_df
    
    def _safe_evaluate_metric(self, evaluator, predictions: DataFrame, metric_name: str, max_retries: int = 3) -> float:
        """
        Safely evaluate a metric with Py4J error handling and retries.
        
        Args:
            evaluator: Spark ML evaluator object
            predictions: DataFrame with predictions
            metric_name: Name of the metric to evaluate
            max_retries: Maximum number of retry attempts
            
        Returns:
            Metric value or 0.0 if all attempts fail
        """
        for attempt in range(max_retries):
            try:
                # Force garbage collection before evaluation
                if hasattr(self.spark, '_jvm'):
                    self.spark._jvm.System.gc()
                
                time.sleep(0.5)  # Brief pause to allow JVM recovery
                
                metric_value = evaluator.evaluate(predictions, {evaluator.metricName: metric_name})
                return metric_value
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for Py4J errors
                if any(py4j_error in error_msg for py4j_error in [
                    'py4jnetworkerror', 'answer from java side is empty', 
                    'error while sending or receiving', 'connection refused'
                ]):
                    print(f"Py4J error during {metric_name} evaluation (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        print("Attempting retry with resource cleanup...")
                        # Cleanup and brief pause
                        predictions.unpersist()
                        time.sleep(2)
                        predictions.cache()
                    else:
                        print(f"All retry attempts failed for {metric_name} evaluation. Returning 0.0")
                        return 0.0
                else:
                    # Non-Py4J error, don't retry
                    print(f"Non-Py4J error during {metric_name} evaluation: {e}")
                    return 0.0
        
        return 0.0
    
    def validate_model_multiclass(self, model, datasets: List[DataFrame], 
                                dataset_names: List[str], target_column: str, 
                                model_type: str, output_dir: str) -> Dict[str, float]:
        """
        Validate a model on multiple datasets for multiclass classification.
        Returns a dict of metrics for each dataset.
        
        Args:
            model: Trained model to validate
            datasets: List of datasets to validate on
            dataset_names: Names of the datasets
            target_column: Name of the target column
            model_type: Type of model being validated
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing metrics for each dataset
        """
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        
        metrics_dict = {}
        
        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        for dataset, name in zip(datasets, dataset_names):
            if dataset is not None:
                print(f"Validating {model_type} on {name} dataset...")
                
                predictions = model.transform(dataset)
                predictions.cache()
                
                # Calculate multiclass metrics with Py4J error handling
                evaluator = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction")
                accuracy = self._safe_evaluate_metric(evaluator, predictions, "accuracy")
                f1 = self._safe_evaluate_metric(evaluator, predictions, "f1")
                weighted_precision = self._safe_evaluate_metric(evaluator, predictions, "weightedPrecision")
                weighted_recall = self._safe_evaluate_metric(evaluator, predictions, "weightedRecall")
                
                metrics_dict[f'accuracy_{name}'] = accuracy
                metrics_dict[f'f1_{name}'] = f1
                metrics_dict[f'precision_{name}'] = weighted_precision
                metrics_dict[f'recall_{name}'] = weighted_recall
                
                # Save predictions for analysis
                predictions_path = os.path.join(model_output_dir, f'predictions_{name}.parquet')
                predictions.select(target_column, 'prediction', 'probability').write.mode('overwrite').parquet(predictions_path)
                
                predictions.unpersist()
            # Don't populate missing datasets with 0 values
        
        # Save multiclass metrics with model-specific naming
        metrics_file = os.path.join(model_output_dir, f'{model_type}_multiclass_metrics.json')
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Create multiclass validation summary
        self._create_multiclass_validation_summary(model_type, model_output_dir, metrics_dict)
        
        print(f"Multiclass validation completed for {model_type}")
        return metrics_dict
    
    def _create_multiclass_validation_summary(self, model_type: str, model_output_dir: str, metrics_dict: Dict[str, float]):
        """
        Create a validation summary report for multiclass classification.
        
        Args:
            model_type: Type of model
            model_output_dir: Model-specific output directory
            metrics_dict: Dictionary of metrics
        """
        try:
            summary_file = os.path.join(model_output_dir, f'{model_type}_multiclass_validation_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write(f"MULTICLASS VALIDATION SUMMARY - {model_type.upper()}\n")
                f.write("=" * 60 + "\n\n")
                
                datasets = ['train', 'valid', 'test', 'oot1', 'oot2']
                for dataset in datasets:
                    f.write(f"{dataset.upper()} DATASET:\n")
                    f.write(f"  Accuracy: {metrics_dict.get(f'accuracy_{dataset}', 0.0):.4f}\n")
                    f.write(f"  F1 Score: {metrics_dict.get(f'f1_{dataset}', 0.0):.4f}\n")
                    f.write(f"  Precision: {metrics_dict.get(f'precision_{dataset}', 0.0):.4f}\n")
                    f.write(f"  Recall: {metrics_dict.get(f'recall_{dataset}', 0.0):.4f}\n\n")
                
                f.write("=" * 60 + "\n")
                f.write("Generated files:\n")
                f.write("- " + f"{model_type}_multiclass_metrics.json\n")
                f.write("- " + f"predictions_*.parquet (for each dataset)\n")
                f.write("- " + f"{model_type}_multiclass_validation_summary.txt\n")
                f.write("=" * 60 + "\n")
            
            print(f"Multiclass validation summary saved to {summary_file}")
            
        except Exception as e:
            print(f"Error creating multiclass validation summary for {model_type}: {str(e)}") 