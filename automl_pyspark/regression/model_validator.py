"""
Regression Model Validator

Validates and evaluates regression models using comprehensive metrics and visualizations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as spark_abs, when, isnan, isnull
import pandas as pd
from scipy import stats


class RegressionModelValidator:
    """Validates regression models with comprehensive metrics and visualizations."""
    
    def __init__(self, spark_session: SparkSession, output_dir: str, user_id: str, model_literal: str):
        self.spark = spark_session
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
        
        # Create plots directory
        self.plots_dir = os.path.join(output_dir, 'validation_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def validate_model(self, model, datasets: List[DataFrame], dataset_names: List[str], 
                      target_column: str, model_type: str, output_dir: str) -> Dict[str, Any]:
        """Validate regression model with comprehensive metrics and plots."""
        
        print(f"🔍 Validating {model_type} regression model...")
        
        all_metrics = {}
        plot_files = []
        
        for dataset, name in zip(datasets, dataset_names):
            if dataset is not None:
                print(f"   📊 Evaluating on {name} dataset...")
                
                # Get predictions
                predictions = model.transform(dataset)
                
                # Clean predictions: Remove rows with null/NaN values in target or prediction columns
                print(f"      🔍 Cleaning predictions for {name} dataset...")
                from pyspark.sql.functions import isnull, isnan, col
                
                original_count = predictions.count()
                
                # Filter out null/NaN values in target and prediction columns
                clean_predictions = predictions.filter(
                    ~isnull(col(target_column)) & 
                    ~isnan(col(target_column)) & 
                    ~isnull(col("prediction")) & 
                    ~isnan(col("prediction"))
                )
                
                clean_count = clean_predictions.count()
                filtered_count = original_count - clean_count
                
                if filtered_count > 0:
                    print(f"      📊 Filtered out {filtered_count} rows with null/NaN values ({original_count} → {clean_count})")
                else:
                    print(f"      ✅ No null/NaN values found ({clean_count} rows)")
                
                if clean_count == 0:
                    print(f"      ⚠️ No valid predictions remaining for {name} dataset - skipping metrics")
                    continue
                
                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(
                    clean_predictions, target_column, name
                )
                all_metrics.update(metrics)
                
                # Generate validation plots
                if name in ['train', 'valid', 'test'] and clean_count > 0:  # Only plot for main datasets with data
                    plots = self._generate_regression_plots(
                        clean_predictions, target_column, model_type, name
                    )
                    plot_files.extend(plots)
        
        # Save metrics
        self._save_metrics(all_metrics, model_type)
        
        # Add plot information to metrics
        all_metrics['validation_plots'] = plot_files
        
        print(f"   ✅ Validation completed - {len(all_metrics)} metrics calculated")
        return all_metrics
    
    def _calculate_comprehensive_metrics(self, predictions: DataFrame, 
                                       target_column: str, dataset_name: str) -> Dict[str, Any]:
        """Calculate essential regression metrics that are displayed in Streamlit."""
        
        metrics = {}
        
        # Standard Spark ML evaluators - only the essential ones
        evaluators = {
            'rmse': RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse"),
            'mae': RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae"),
            'r2': RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2"),
            'mse': RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mse")
        }
        
        # Calculate standard metrics
        for metric_name, evaluator in evaluators.items():
            try:
                value = evaluator.evaluate(predictions)
                metrics[f'{dataset_name}_{metric_name}'] = value
            except Exception as e:
                print(f"   ⚠️ Could not calculate {metric_name}: {str(e)}")
                metrics[f'{dataset_name}_{metric_name}'] = None
        
        # Calculate only essential additional metrics
        try:
            # Convert to Pandas for additional calculations
            pred_pandas = predictions.select(target_column, "prediction").toPandas()
            actual = pred_pandas[target_column].values
            predicted = pred_pandas["prediction"].values
            
            # Remove any NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) > 0:
                # Mean Absolute Percentage Error (MAPE)
                mape = np.mean(np.abs((actual_clean - predicted_clean) / 
                                    np.where(actual_clean != 0, actual_clean, 1e-8))) * 100
                metrics[f'{dataset_name}_mape'] = mape
                
                # Explained Variance Score
                explained_var = 1 - np.var(actual_clean - predicted_clean) / np.var(actual_clean)
                metrics[f'{dataset_name}_explained_variance'] = explained_var
                
        except Exception as e:
            print(f"   ⚠️ Could not calculate additional metrics: {str(e)}")
        
        return metrics
    
    def _generate_regression_plots(self, predictions: DataFrame, target_column: str, 
                                 model_type: str, dataset_name: str) -> List[str]:
        """Generate comprehensive regression validation plots."""
        
        plot_files = []
        
        try:
            # Convert to pandas for plotting
            pred_pandas = predictions.select(target_column, "prediction").toPandas()
            actual = pred_pandas[target_column].values
            predicted = pred_pandas["prediction"].values
            
            # Remove NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) < 10:  # Skip plotting if too few points
                return plot_files
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Actual vs Predicted Plot
            plot_file = self._create_actual_vs_predicted_plot(
                actual_clean, predicted_clean, model_type, dataset_name
            )
            if plot_file:
                plot_files.append(plot_file)
            
            # 2. Residuals Plot
            plot_file = self._create_residuals_plot(
                actual_clean, predicted_clean, model_type, dataset_name
            )
            if plot_file:
                plot_files.append(plot_file)
            
            # 3. Residuals Distribution Plot
            plot_file = self._create_residuals_distribution_plot(
                actual_clean, predicted_clean, model_type, dataset_name
            )
            if plot_file:
                plot_files.append(plot_file)
            
            # 4. QQ Plot for Residuals
            plot_file = self._create_qq_plot(
                actual_clean, predicted_clean, model_type, dataset_name
            )
            if plot_file:
                plot_files.append(plot_file)
            
            # 5. Prediction Error Distribution
            plot_file = self._create_error_distribution_plot(
                actual_clean, predicted_clean, model_type, dataset_name
            )
            if plot_file:
                plot_files.append(plot_file)
            
        except Exception as e:
            print(f"   ⚠️ Could not generate plots: {str(e)}")
        
        return plot_files
    
    def _create_actual_vs_predicted_plot(self, actual, predicted, model_type, dataset_name):
        """Create actual vs predicted scatter plot."""
        try:
            plt.figure(figsize=(10, 8))
            
            # Scatter plot
            plt.scatter(actual, predicted, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Calculate R²
            r2 = np.corrcoef(actual, predicted)[0, 1] ** 2
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{model_type} - Actual vs Predicted ({dataset_name})\nR² = {r2:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(actual, predicted, 1)
            p = np.poly1d(z)
            plt.plot(actual, p(actual), "b--", alpha=0.8, label=f'Trend (slope={z[0]:.3f})')
            plt.legend()
            
            filename = f'{model_type}_{dataset_name}_actual_vs_predicted.png'
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"   ⚠️ Could not create actual vs predicted plot: {str(e)}")
            plt.close()
            return None
    
    def _create_residuals_plot(self, actual, predicted, model_type, dataset_name):
        """Create residuals scatter plot."""
        try:
            residuals = actual - predicted
            
            plt.figure(figsize=(10, 8))
            plt.scatter(predicted, residuals, alpha=0.6, s=30)
            plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals (Actual - Predicted)')
            plt.title(f'{model_type} - Residuals Plot ({dataset_name})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add trend line to check for patterns
            z = np.polyfit(predicted, residuals, 1)
            p = np.poly1d(z)
            plt.plot(predicted, p(predicted), "b--", alpha=0.8, 
                    label=f'Trend (slope={z[0]:.6f})')
            plt.legend()
            
            filename = f'{model_type}_{dataset_name}_residuals.png'
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"   ⚠️ Could not create residuals plot: {str(e)}")
            plt.close()
            return None
    
    def _create_residuals_distribution_plot(self, actual, predicted, model_type, dataset_name):
        """Create residuals distribution histogram."""
        try:
            residuals = actual - predicted
            
            plt.figure(figsize=(12, 8))
            
            # Histogram
            plt.subplot(2, 2, 1)
            plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Residuals Distribution')
            plt.grid(True, alpha=0.3)
            
            # Box plot
            plt.subplot(2, 2, 2)
            plt.boxplot(residuals)
            plt.ylabel('Residuals')
            plt.title('Residuals Box Plot')
            plt.grid(True, alpha=0.3)
            
            # Statistics text
            plt.subplot(2, 2, 3)
            plt.text(0.1, 0.8, f'Mean: {np.mean(residuals):.4f}', transform=plt.gca().transAxes)
            plt.text(0.1, 0.7, f'Std: {np.std(residuals):.4f}', transform=plt.gca().transAxes)
            plt.text(0.1, 0.6, f'Skewness: {stats.skew(residuals):.4f}', transform=plt.gca().transAxes)
            plt.text(0.1, 0.5, f'Kurtosis: {stats.kurtosis(residuals):.4f}', transform=plt.gca().transAxes)
            plt.text(0.1, 0.4, f'Min: {np.min(residuals):.4f}', transform=plt.gca().transAxes)
            plt.text(0.1, 0.3, f'Max: {np.max(residuals):.4f}', transform=plt.gca().transAxes)
            plt.axis('off')
            plt.title('Residuals Statistics')
            
            # Density plot
            plt.subplot(2, 2, 4)
            plt.hist(residuals, bins=30, density=True, alpha=0.7, label='Residuals')
            
            # Overlay normal distribution
            mu, sigma = stats.norm.fit(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                    label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
            plt.xlabel('Residuals')
            plt.ylabel('Density')
            plt.title('Residuals vs Normal Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'{model_type} - Residuals Analysis ({dataset_name})')
            plt.tight_layout()
            
            filename = f'{model_type}_{dataset_name}_residuals_distribution.png'
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"   ⚠️ Could not create residuals distribution plot: {str(e)}")
            plt.close()
            return None
    
    def _create_qq_plot(self, actual, predicted, model_type, dataset_name):
        """Create Q-Q plot for residuals normality check."""
        try:
            residuals = actual - predicted
            
            plt.figure(figsize=(10, 8))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title(f'{model_type} - Q-Q Plot of Residuals ({dataset_name})')
            plt.grid(True, alpha=0.3)
            
            # Add R² for the Q-Q plot
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)
            qq_r2 = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
            plt.text(0.05, 0.95, f'Q-Q R² = {qq_r2:.4f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            filename = f'{model_type}_{dataset_name}_qq_plot.png'
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"   ⚠️ Could not create Q-Q plot: {str(e)}")
            plt.close()
            return None
    
    def _create_error_distribution_plot(self, actual, predicted, model_type, dataset_name):
        """Create prediction error distribution plot."""
        try:
            errors = np.abs(actual - predicted)
            relative_errors = errors / np.where(np.abs(actual) > 1e-8, np.abs(actual), 1e-8) * 100
            
            plt.figure(figsize=(15, 10))
            
            # Absolute errors
            plt.subplot(2, 3, 1)
            plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.title('Absolute Error Distribution')
            plt.grid(True, alpha=0.3)
            
            # Relative errors
            plt.subplot(2, 3, 2)
            plt.hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Relative Error (%)')
            plt.ylabel('Frequency')
            plt.title('Relative Error Distribution')
            plt.grid(True, alpha=0.3)
            
            # Error vs Actual
            plt.subplot(2, 3, 3)
            plt.scatter(actual, errors, alpha=0.6, s=20)
            plt.xlabel('Actual Values')
            plt.ylabel('Absolute Error')
            plt.title('Error vs Actual Values')
            plt.grid(True, alpha=0.3)
            
            # Error vs Predicted
            plt.subplot(2, 3, 4)
            plt.scatter(predicted, errors, alpha=0.6, s=20)
            plt.xlabel('Predicted Values')
            plt.ylabel('Absolute Error')
            plt.title('Error vs Predicted Values')
            plt.grid(True, alpha=0.3)
            
            # Cumulative error distribution
            plt.subplot(2, 3, 5)
            sorted_errors = np.sort(errors)
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            plt.plot(sorted_errors, cumulative)
            plt.xlabel('Absolute Error')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Error Distribution')
            plt.grid(True, alpha=0.3)
            
            # Error statistics
            plt.subplot(2, 3, 6)
            stats_text = f'''Error Statistics:
Mean Abs Error: {np.mean(errors):.4f}
Median Abs Error: {np.median(errors):.4f}
Max Error: {np.max(errors):.4f}
90th Percentile: {np.percentile(errors, 90):.4f}
95th Percentile: {np.percentile(errors, 95):.4f}
99th Percentile: {np.percentile(errors, 99):.4f}

Mean Rel Error: {np.mean(relative_errors):.2f}%
Median Rel Error: {np.median(relative_errors):.2f}%'''
            
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            
            plt.suptitle(f'{model_type} - Prediction Error Analysis ({dataset_name})')
            plt.tight_layout()
            
            filename = f'{model_type}_{dataset_name}_error_analysis.png'
            filepath = os.path.join(self.plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"   ⚠️ Could not create error distribution plot: {str(e)}")
            plt.close()
            return None
    
    def _save_metrics(self, metrics: Dict[str, Any], model_type: str):
        """Save metrics to JSON file."""
        try:
            filename = f'{model_type}_regression_metrics.json'
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print(f"   💾 Metrics saved to {filename}")
            
        except Exception as e:
            print(f"   ⚠️ Could not save metrics: {str(e)}")
    
    def get_available_metrics(self) -> List[str]:
        """Get list of all available regression metrics."""
        return [
            'rmse', 'mae', 'r2', 'mse', 'mape', 'explained_variance',
            'max_error', 'median_ae', 'msle', 'residual_mean', 'residual_std',
            'residual_skew', 'residual_kurtosis', 'prediction_mean', 'prediction_std',
            'actual_mean', 'actual_std', 'sample_count', 'null_predictions'
        ] 