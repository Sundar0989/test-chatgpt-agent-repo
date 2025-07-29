"""
Model Selector

Class responsible for selecting the best model based on performance metrics.
This class encapsulates the model selection functionality from the original modules.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import glob

# Import functions from original modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelSelector:
    """
    Model selector class that handles model selection and comparison.
    
    This class provides functionality for:
    - Model selection based on performance criteria
    - Model comparison and ranking
    - Champion/Challenger model identification
    """
    
    def __init__(self, output_dir: str, user_id: str, model_literal: str):
        """
        Initialize the model selector.
        
        Args:
            output_dir: Output directory for results
            user_id: User identifier
            model_literal: Model literal/tag
        """
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def select_best_model(self, model_metrics: Dict[str, Dict[str, float]], 
                         selection_criteria: str = 'ks', 
                         dataset_to_use: str = 'train', dataset_names: List[str] = None) -> Dict[str, Any]:
        """
        Select the best model based on specified criteria.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            selection_criteria: Criteria for selection ('ks', 'roc', 'accuracy')
            dataset_to_use: Dataset to use for selection ('train', 'valid', 'test', 'oot1', 'oot2')
            
        Returns:
            Dictionary containing selection results
        """
        print(f"Selecting best model based on {selection_criteria} on {dataset_to_use} dataset...")
        
        # Create DataFrame from metrics
        df = self._create_metrics_dataframe(model_metrics)
        
        # Apply selection criteria
        best_model_info = self._apply_selection_criteria(
            df, selection_criteria, dataset_to_use, dataset_names
        )
        
        # Save results using the DataFrame with selection labels
        self._save_selection_results(best_model_info['all_results'], best_model_info)
        
        print(f"Best model selected: {best_model_info['model_type']}")
        return best_model_info
    
    def _create_metrics_dataframe(self, model_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create a DataFrame from model metrics.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            
        Returns:
            DataFrame with model metrics
        """
        
        # Create DataFrame from model_metrics without predefined columns
        data = []
        for model_type, metrics in model_metrics.items():
            row = {'model_type': model_type}
            # Safely add all metric key-value pairs
            if metrics is not None:
                for key, value in metrics.items():
                    row[key] = value
            data.append(row)

        df = pd.DataFrame(data)
        
        # Debug: Print the created DataFrame
        print(f"Debug: Created DataFrame with shape: {df.shape}")
        print(f"Debug: DataFrame columns: {list(df.columns)}")
        print(f"Debug: DataFrame head:\n{df.head()}")
        
        return df
    
    def _apply_selection_criteria(self, df: pd.DataFrame, 
                                selection_criteria: str, 
                                dataset_to_use: str, 
                                dataset_names: List[str] = None) -> Dict[str, Any]:
        """
        Apply selection criteria to rank models.
        
        Args:
            df: DataFrame with model metrics
            selection_criteria: Criteria for selection
            dataset_to_use: Dataset to use for selection
            dataset_names: List of dataset names for stability calculation
            
        Returns:
            Dictionary with best model information
        """
        
        # Get the column to sort by
        column_to_sort = f"{selection_criteria}_{dataset_to_use}"
        
        if column_to_sort not in df.columns:
            print(f"Warning: Column {column_to_sort} not found in metrics DataFrame")
            print(f"Available columns: {list(df.columns)}")
            
            # Try to find an alternative metric for the same dataset
            available_metrics_for_dataset = [col for col in df.columns if col.endswith(f'_{dataset_to_use}')]
            
            if available_metrics_for_dataset:
                # Prioritize accuracy for multiclass, then f1, then others
                metric_priority = ['accuracy', 'f1', 'precision', 'recall', 'roc', 'ks']
                
                for priority_metric in metric_priority:
                    fallback_column = f"{priority_metric}_{dataset_to_use}"
                    if fallback_column in available_metrics_for_dataset:
                        column_to_sort = fallback_column
                        print(f"Using fallback metric: {column_to_sort}")
                        break
                else:
                    # Use the first available metric for the dataset
                    column_to_sort = available_metrics_for_dataset[0]
                    print(f"Using first available metric: {column_to_sort}")
            else:
                raise ValueError(f"No metrics found for dataset '{dataset_to_use}'. Available columns: {list(df.columns)}")
        
        # Extract the actual metric being used (in case we fell back to a different metric)
        actual_metric = column_to_sort.replace(f'_{dataset_to_use}', '')
        
        # Set checker value based on criteria
        checker_value = 0.03
        if actual_metric == 'ks':
            checker_value = checker_value * 100
        
        # Calculate stability score (similar to original code)
        df['counter'] = 0
        if dataset_names is not None:
            for dataset in dataset_names:
                col_name = f"{selection_criteria}_{dataset}"
                if col_name in df.columns:
                    df['counter'] += (np.abs(df[column_to_sort] - df[col_name]) > checker_value).astype(int)
        else:
            # If no dataset_names provided, set counter to 0 (no stability penalty)
            df['counter'] = 0
        
        # Sort by stability and performance
        df = df.sort_values(['counter', column_to_sort], ascending=[True, False]).reset_index(drop=True)
        
        # Add selection labels
        df['selected_model'] = ''
        df.loc[0, 'selected_model'] = 'Champion'
        if len(df) > 1:
            df.loc[1, 'selected_model'] = 'Challenger'
        
        # Get best model info
        best_model_info = {
            'model_type': df.loc[0, 'model_type'],
            'selection_criteria': selection_criteria,
            'dataset_used': dataset_to_use,
            'performance_score': df.loc[0, column_to_sort],
            'stability_score': df.loc[0, 'counter'],
            'rank': 1,
            'all_results': df
        }
        
        return best_model_info
    
    def _save_selection_results(self, df: pd.DataFrame, best_model_info: Dict[str, Any]):
        """
        Save model selection results.
        
        Args:
            df: DataFrame with all results
            best_model_info: Information about the best model
        """
        try:
            # Debug: Print available keys in best_model_info
            print(f"Debug: best_model_info keys: {list(best_model_info.keys())}")
            print(f"Debug: DataFrame columns: {list(df.columns)}")
            
            # Save to Excel
            output_file = os.path.join(self.output_dir, 'model_selection_results.xlsx')
            df.to_excel(output_file, index=False)
            
            # Save summary
            summary_file = os.path.join(self.output_dir, 'model_selection_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("MODEL SELECTION RESULTS\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Best Model: {best_model_info['model_type']}\n")
                f.write(f"Selection Criteria: {best_model_info['selection_criteria']}\n")
                f.write(f"Dataset Used: {best_model_info['dataset_used']}\n")
                f.write(f"Performance Score: {best_model_info['performance_score']:.4f}\n")
                f.write(f"Stability Score: {best_model_info['stability_score']}\n")
                f.write(f"Rank: {best_model_info['rank']}\n\n")
                
                f.write("All Models Ranking:\n")
                f.write("-" * 30 + "\n")
                for idx, row in df.iterrows():
                    f.write(f"{idx + 1}. {row['model_type']} - {row['selected_model']}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            print(f"Selection results saved to {output_file}")
            print(f"Selection summary saved to {summary_file}")
            
        except Exception as e:
            print(f"Error saving selection results: {str(e)}")
            # Print more detailed error information
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
    
    def get_model_ranking(self, model_metrics: Dict[str, Dict[str, float]], 
                         metric: str = 'ks', dataset: str = 'test') -> pd.DataFrame:
        """
        Get ranking of models based on a specific metric.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            metric: Metric to rank by ('roc', 'accuracy', 'ks')
            dataset: Dataset to rank on ('train', 'valid', 'test', 'oot1', 'oot2')
            
        Returns:
            DataFrame with model ranking
        """
        df = self._create_metrics_dataframe(model_metrics)
        
        # Normalize model type names
        df['model_type'] = df['model_type'].str.replace('_', '')
        
        # Get the column to sort by
        column_to_sort = f"{metric}_{dataset}"
        
        if column_to_sort not in df.columns:
            raise ValueError(f"Column {column_to_sort} not found in metrics DataFrame")
        
        # Sort by performance
        df_sorted = df.sort_values(column_to_sort, ascending=False).reset_index(drop=True)
        df_sorted['rank'] = df_sorted.index + 1
        
        return df_sorted[['rank', 'model_type', column_to_sort]]
    
    def compare_models_detailed(self, model_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Get detailed comparison of all models across all metrics and datasets.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            
        Returns:
            DataFrame with detailed comparison
        """
        df = self._create_metrics_dataframe(model_metrics)
        
        # Normalize model type names
        df['model_type'] = df['model_type'].str.replace('_', '')
        
        # Calculate summary statistics
        metrics_columns = [col for col in df.columns if col != 'model_type']
        
        summary_stats = []
        for metric in ['roc', 'accuracy', 'ks']:
            metric_cols = [col for col in metrics_columns if col.startswith(metric)]
            if metric_cols:
                df[f'{metric}_mean'] = df[metric_cols].mean(axis=1)
                df[f'{metric}_std'] = df[metric_cols].std(axis=1)
                df[f'{metric}_min'] = df[metric_cols].min(axis=1)
                df[f'{metric}_max'] = df[metric_cols].max(axis=1)
        
        return df
    
    def get_stability_analysis(self, model_metrics: Dict[str, Dict[str, float]], 
                              metric: str = 'ks') -> pd.DataFrame:
        """
        Analyze model stability across different datasets.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            metric: Metric to analyze ('roc', 'accuracy', 'ks')
            
        Returns:
            DataFrame with stability analysis
        """
        df = self._create_metrics_dataframe(model_metrics)
        
        # Normalize model type names
        df['model_type'] = df['model_type'].str.replace('_', '')
        
        # Get metric columns for the specified metric
        metric_cols = [col for col in df.columns if col.startswith(metric)]
        
        if not metric_cols:
            raise ValueError(f"No columns found for metric: {metric}")
        
        # Calculate stability metrics
        stability_data = []
        for _, row in df.iterrows():
            values = [row[col] for col in metric_cols if row[col] > 0]
            if values:
                stability_data.append({
                    'model_type': row['model_type'],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                })
        
        stability_df = pd.DataFrame(stability_data)
        stability_df = stability_df.sort_values('cv', ascending=True)  # Lower CV = more stable
        
        return stability_df 