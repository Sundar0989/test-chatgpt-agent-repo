"""
Clustering Model Selector

Selects the best clustering model based on unsupervised metrics.
"""

from typing import Dict, Any


class ClusteringModelSelector:
    """Selects best clustering model."""
    
    def __init__(self, output_dir: str, user_id: str, model_literal: str, evaluation_method: str = 'silhouette'):
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
        self.evaluation_method = evaluation_method
    
    def select_best_model(self, trained_models: Dict[str, Any], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Select best clustering model based on specified evaluation method."""
        
        best_model_type = None
        best_score = -1 if self.evaluation_method == 'silhouette' else -float('inf')
        best_metrics = {}
        
        print(f"📊 Comparing {len(trained_models)} trained models using {self.evaluation_method}...")
        
        for model_type, model_data in trained_models.items():
            metrics = model_data['metrics']
            current_score = self._extract_evaluation_score(metrics)
            
            print(f"   {model_type}: {self.evaluation_method} = {current_score:.4f}")
            
            # For davies_bouldin, lower is better; for others, higher is better
            is_better = current_score < best_score if self.evaluation_method == 'davies_bouldin' else current_score > best_score
            
            if is_better:
                best_score = current_score
                best_model_type = model_type
                best_metrics = metrics
        
        if best_model_type is None:
            raise RuntimeError(f"No valid clustering models found. All models failed to produce valid {self.evaluation_method} scores.")
        
        print(f"   🏆 Winner: {best_model_type} ({self.evaluation_method}: {best_score:.4f})")
        
        return {
            'best_model_type': best_model_type,
            'best_metric': best_score,
            'metrics': best_metrics
        }
    
    def _extract_evaluation_score(self, metrics: Dict[str, Any]) -> float:
        """
        Extract evaluation score from different metrics formats.
        
        Handles:
        1. Single dataset validation: metrics['{evaluation_method}_score']
        2. Multi-dataset validation: metrics['datasets']['train']['{evaluation_method}_score'] (prioritize training)
        3. Cross-validation: metrics['mean_{evaluation_method}_score']
        """
        
        score_key = f'{self.evaluation_method}_score'
        mean_score_key = f'mean_{self.evaluation_method}_score'
        
        # Case 1: Direct score (old format or single dataset)
        if score_key in metrics:
            return float(metrics[score_key])
        
        # Case 2: Multi-dataset validation
        if 'datasets' in metrics:
            datasets = metrics['datasets']
            
            # Priority order: validation > train > test (validation set gives best unbiased estimate)
            for dataset_name in ['validation', 'train', 'test']:
                if dataset_name in datasets and score_key in datasets[dataset_name]:
                    score = datasets[dataset_name][score_key]
                    if score is not None:
                        return float(score)
            
            # Fallback: take first available dataset with score
            for dataset_name, dataset_metrics in datasets.items():
                if score_key in dataset_metrics and dataset_metrics[score_key] is not None:
                    return float(dataset_metrics[score_key])
        
        # Case 3: Cross-validation
        if mean_score_key in metrics:
            return float(metrics[mean_score_key])
        
        # Case 4: Check if it's nested under validation_type
        if 'validation_type' in metrics:
            if metrics['validation_type'] == 'cross_validation' and mean_score_key in metrics:
                return float(metrics[mean_score_key])
        
        # Fallback: return worst possible score
        worst_score = -1.0 if self.evaluation_method == 'silhouette' else -float('inf')
        print(f"   ⚠️ No {self.evaluation_method} score found in metrics: {list(metrics.keys())}")
        return worst_score 