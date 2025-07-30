"""
Regression Model Selector

Selects the best regression model based on performance metrics with Champion/Challenger methodology
and stability analysis across multiple datasets.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class RegressionModelSelector:
    """
    Regression model selector with Champion/Challenger methodology and stability analysis.
    
    Implements sophisticated model selection that considers both performance and stability
    across multiple datasets (train, valid, test, oot1, oot2).
    """
    
    def __init__(self, output_dir: str, user_id: str, model_literal: str):
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def select_best_model(self, model_metrics: Dict[str, Dict[str, float]], 
                         selection_criteria: str = 'rmse',
                         dataset_to_use: str = 'valid',
                         dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Select the best model using Champion/Challenger methodology with stability analysis.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            selection_criteria: Criteria for selection ('rmse', 'mae', 'r2')
            dataset_to_use: Primary dataset to use for selection ('train', 'valid', 'test', 'oot1', 'oot2')
            dataset_names: List of all dataset names for stability calculation
            
        Returns:
            Dictionary containing Champion/Challenger selection results
        """
        print(f"🏆 Selecting Champion/Challenger models based on {selection_criteria} on {dataset_to_use} dataset...")
        
        if not model_metrics:
            raise ValueError("No model metrics provided for selection")
        
        # Create DataFrame from metrics
        df = self._create_metrics_dataframe(model_metrics)
        
        # Apply Champion/Challenger selection with stability analysis
        best_model_info = self._apply_champion_challenger_selection(
            df, selection_criteria, dataset_to_use, dataset_names
        )
        
        # Save results
        self._save_champion_challenger_results(best_model_info['all_results'], best_model_info)
        
        champion_model = best_model_info['model_type']
        champion_score = best_model_info['performance_score']
        stability_score = best_model_info['stability_score']
        
        print(f"🥇 Champion model selected: {champion_model} ({selection_criteria}: {champion_score:.4f}, stability: {stability_score})")
        
        return best_model_info
    
    def _create_metrics_dataframe(self, model_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a pandas DataFrame from model metrics dictionary."""
        
        # Collect all data
        data = []
        for model_type, metrics in model_metrics.items():
            row = {'model_type': model_type}
            for metric_name, value in metrics.items():
                row[metric_name] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Fill NaN values with appropriate defaults
        for col in df.columns:
            if col != 'model_type':
                if 'rmse' in col.lower() or 'mae' in col.lower():
                    df[col] = df[col].fillna(float('inf'))  # High values for error metrics
                elif 'r2' in col.lower():
                    df[col] = df[col].fillna(0.0)  # Low values for R²
                else:
                    df[col] = df[col].fillna(0.0)
        
        return df
    
    def _apply_champion_challenger_selection(self, df: pd.DataFrame,
                                           selection_criteria: str,
                                           dataset_to_use: str,
                                           dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply Champion/Challenger selection with stability analysis.
        
        Args:
            df: DataFrame with model metrics
            selection_criteria: Criteria for selection ('rmse', 'mae', 'r2')
            dataset_to_use: Primary dataset for selection
            dataset_names: List of dataset names for stability calculation
            
        Returns:
            Dictionary with Champion/Challenger selection results
        """
        
        # Get the primary column to sort by (format: dataset_metric)
        primary_column = f"{dataset_to_use}_{selection_criteria}"
        
        if primary_column not in df.columns:
            print(f"⚠️ Column {primary_column} not found in metrics DataFrame")
            print(f"Available columns: {list(df.columns)}")
            
            # Try to find fallback metric for the same dataset
            available_metrics_for_dataset = [col for col in df.columns if col.startswith(f'{dataset_to_use}_')]
            
            if available_metrics_for_dataset:
                # Prioritize rmse, then mae, then r2 for regression
                metric_priority = ['rmse', 'mae', 'r2']
                
                for priority_metric in metric_priority:
                    fallback_column = f"{dataset_to_use}_{priority_metric}"
                    if fallback_column in available_metrics_for_dataset:
                        primary_column = fallback_column
                        selection_criteria = priority_metric
                        print(f"✅ Using fallback metric: {primary_column}")
                        break
                else:
                    # Use the first available metric for the dataset
                    primary_column = available_metrics_for_dataset[0]
                    selection_criteria = primary_column.replace(f'{dataset_to_use}_', '')
                    print(f"✅ Using first available metric: {primary_column}")
            else:
                raise ValueError(f"No metrics found for dataset '{dataset_to_use}'. Available columns: {list(df.columns)}")
        
        # Determine if lower or higher is better
        ascending = True  # For RMSE, MAE (lower is better)
        if selection_criteria.lower() == 'r2':
            ascending = False  # For R² (higher is better)
        
        # Set stability checker value based on criteria
        if selection_criteria.lower() == 'rmse':
            checker_value = 0.1  # 0.1 RMSE units
        elif selection_criteria.lower() == 'mae':
            checker_value = 0.05  # 0.05 MAE units
        elif selection_criteria.lower() == 'r2':
            checker_value = 0.02  # 0.02 R² units
        else:
            checker_value = 0.05  # Default threshold
        
        print(f"📊 Using stability threshold: {checker_value} for {selection_criteria}")
        
        # Calculate stability score (counter)
        df['stability_score'] = 0
        
        if dataset_names is not None and len(dataset_names) > 1:
            print(f"🔍 Calculating stability across datasets: {dataset_names}")
            
            for dataset in dataset_names:
                if dataset != dataset_to_use:  # Don't compare dataset with itself
                    comparison_column = f"{dataset}_{selection_criteria}"
                    if comparison_column in df.columns:
                        # Count models where performance differs significantly
                        deviation = np.abs(df[primary_column] - df[comparison_column])
                        df['stability_score'] += (deviation > checker_value).astype(int)
                        
                        print(f"   📈 Comparing {primary_column} vs {comparison_column}")
                    else:
                        print(f"   ⚠️ Comparison column {comparison_column} not found")
        else:
            print("📊 No stability analysis (insufficient datasets)")
        
        # Sort by stability first (lower is better), then by performance
        df = df.sort_values(['stability_score', primary_column], ascending=[True, ascending]).reset_index(drop=True)
        
        # Assign Champion/Challenger labels
        df['selected_model'] = ''
        df.loc[0, 'selected_model'] = 'Champion'
        if len(df) > 1:
            df.loc[1, 'selected_model'] = 'Challenger'
        
        # Add ranking
        df['rank'] = df.index + 1
        
        # Get Champion model info
        champion_info = {
            'model_type': df.loc[0, 'model_type'],
            'selection_criteria': selection_criteria,
            'dataset_used': dataset_to_use,
            'performance_score': df.loc[0, primary_column],
            'stability_score': df.loc[0, 'stability_score'],
            'rank': 1,
            'role': 'Champion',
            'all_results': df
        }
        
        # Log selection results
        self._log_champion_challenger_selection(df, selection_criteria, champion_info)
        
        return champion_info
    
    def _log_champion_challenger_selection(self, df: pd.DataFrame, selection_criteria: str, champion_info: Dict[str, Any]):
        """Log detailed Champion/Challenger selection results."""
        
        print(f"\n🏆 CHAMPION/CHALLENGER SELECTION RESULTS:")
        print(f"   📊 Selection metric: {selection_criteria.upper()}")
        print(f"   📈 {'Lower is better' if selection_criteria in ['rmse', 'mae'] else 'Higher is better'}")
        print("")
        
        for idx, row in df.iterrows():
            if idx < 2:  # Only show Champion and Challenger
                role = row['selected_model']
                icon = "🥇" if role == "Champion" else "🥈"
                stability_status = "stable" if row['stability_score'] == 0 else f"stability issues: {int(row['stability_score'])}"
                
                print(f"   {icon} {role}: {row['model_type']}")
                
                # Get the performance column name (format: dataset_metric)
                performance_col = f"{champion_info['dataset_used']}_{selection_criteria}"
                performance_value = row[performance_col]
                
                print(f"      📊 {selection_criteria.upper()}: {performance_value:.4f}")
                print(f"      🎯 Stability: {stability_status}")
                print(f"      🏅 Rank: #{int(row['rank'])}")
                print("")
    
    def _save_champion_challenger_results(self, df: pd.DataFrame, champion_info: Dict[str, Any]):
        """Save Champion/Challenger selection results to files."""
        
        try:
            # Save detailed results as Excel
            results_file = os.path.join(self.output_dir, 'champion_challenger_selection.xlsx')
            df.to_excel(results_file, index=False)
            
            # Save summary as text
            summary_file = os.path.join(self.output_dir, 'champion_challenger_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("🏆 CHAMPION/CHALLENGER MODEL SELECTION RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Champion Model: {champion_info['model_type']}\n")
                f.write(f"Selection Criteria: {champion_info['selection_criteria']}\n")
                f.write(f"Primary Dataset: {champion_info['dataset_used']}\n")
                f.write(f"Performance Score: {champion_info['performance_score']:.4f}\n")
                f.write(f"Stability Score: {champion_info['stability_score']}\n\n")
                
                f.write("Champion/Challenger Ranking:\n")
                f.write("-" * 50 + "\n")
                f.write("Rank | Role       | Model Type        | Performance | Stability\n")
                f.write("-----|------------|-------------------|-------------|----------\n")
                
                for idx, row in df.iterrows():
                    rank = int(row['rank'])
                    role = row['selected_model'] or f"Rank {rank}"
                    model_type = row['model_type']
                    performance = row[f"{champion_info['dataset_used']}_{champion_info['selection_criteria']}"]
                    stability = int(row['stability_score'])
                    
                    f.write(f"{rank:4d} | {role:10s} | {model_type:17s} | {performance:11.4f} | {stability:9d}\n")
                
                f.write("\nDetailed Performance Metrics:\n")
                f.write("-" * 40 + "\n")
                
                for idx, row in df.iterrows():
                    f.write(f"\n{idx + 1}. {row['model_type']} ({row['selected_model'] or 'Participant'}):\n")
                    
                    # Show key metrics
                    metrics_to_show = ['rmse', 'mae', 'r2']
                    for metric in metrics_to_show:
                        for dataset in ['train', 'valid', 'test', 'oot1', 'oot2']:
                            col_name = f"{metric}_{dataset}"
                            if col_name in df.columns and not pd.isna(row[col_name]):
                                f.write(f"   {metric.upper()} ({dataset}): {row[col_name]:.4f}\n")
                    f.write(f"   Stability Score: {int(row['stability_score'])}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("INTERPRETATION GUIDE:\n")
                f.write("• Champion: Best performing model with highest stability\n")
                f.write("• Challenger: Second best model for comparison\n") 
                f.write("• Stability Score: Number of datasets with significant performance deviation\n")
                f.write("• Lower stability scores indicate more consistent performance\n")
                f.write("=" * 80 + "\n")
            
            # Save as JSON for programmatic access
            json_file = os.path.join(self.output_dir, 'champion_challenger_results.json')
            json_data = {
                'champion_info': champion_info,
                'all_models': df.to_dict('records')
            }
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"💾 Champion/Challenger results saved to: {results_file}")
            print(f"💾 Summary saved to: {summary_file}")
            print(f"💾 JSON data saved to: {json_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving Champion/Challenger results: {e}")
    
    def get_stability_analysis(self, model_metrics: Dict[str, Dict[str, float]], 
                              metric: str = 'rmse') -> pd.DataFrame:
        """
        Analyze model stability across different datasets.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            metric: Metric to analyze ('rmse', 'mae', 'r2')
            
        Returns:
            DataFrame with stability analysis
        """
        df = self._create_metrics_dataframe(model_metrics)
        
        # Get metric columns for the specified metric
        metric_cols = [col for col in df.columns if col.startswith(metric) and col != 'model_type']
        
        if not metric_cols:
            raise ValueError(f"No columns found for metric: {metric}")
        
        # Calculate stability metrics
        stability_data = []
        for _, row in df.iterrows():
            values = [row[col] for col in metric_cols if not pd.isna(row[col]) and row[col] != float('inf')]
            if values and len(values) > 1:
                stability_data.append({
                    'model_type': row['model_type'],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values),
                    'datasets_count': len(values)
                })
        
        stability_df = pd.DataFrame(stability_data)
        if not stability_df.empty:
            stability_df = stability_df.sort_values('cv', ascending=True)  # Lower CV = more stable
        
        return stability_df
    
    def get_model_ranking(self, model_metrics: Dict[str, Dict[str, float]], 
                         metric: str = 'rmse',
                         dataset: str = 'valid') -> List[Dict[str, Any]]:
        """
        Get ranking of models based on specified metric with stability consideration.
        
        Args:
            model_metrics: Dictionary containing metrics for all models
            metric: Metric to rank by ('rmse', 'mae', 'r2')
            dataset: Primary dataset to rank on
            
        Returns:
            List of models ranked by performance and stability
        """
        df = self._create_metrics_dataframe(model_metrics)
        
        # Apply Champion/Challenger selection to get proper ranking
        selection_result = self._apply_champion_challenger_selection(df, metric, dataset)
        ranked_df = selection_result['all_results']
        
        ranking = []
        for _, row in ranked_df.iterrows():
            ranking.append({
                'rank': int(row['rank']),
                'model_type': row['model_type'],
                'role': row['selected_model'] or f"Rank {int(row['rank'])}",
                'performance': row[f"{metric}_{dataset}"],
                'stability_score': int(row['stability_score']),
                'metric': metric,
                'dataset': dataset
            })
        
        return ranking 