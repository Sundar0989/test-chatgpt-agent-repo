"""
Clustering Model Selector

Selects the best clustering model based on unsupervised metrics.
"""

from typing import Dict, Any


class ClusteringModelSelector:
    """Selects best clustering model."""
    
    def __init__(self, output_dir: str, user_id: str, model_literal: str):
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
    
    def select_best_model(self, trained_models: Dict[str, Any], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Select best clustering model based on silhouette score."""
        
        best_model_type = None
        best_silhouette = -1
        best_metrics = {}
        
        print(f"📊 Comparing {len(trained_models)} trained models...")
        
        for model_type, model_data in trained_models.items():
            metrics = model_data['metrics']
            current_silhouette = self._extract_silhouette_score(metrics)
            
            print(f"   {model_type}: Silhouette = {current_silhouette:.4f}")
            
            if current_silhouette > best_silhouette:
                best_silhouette = current_silhouette
                best_model_type = model_type
                best_metrics = metrics
        
        if best_model_type is None:
            raise RuntimeError("No valid clustering models found. All models failed to produce valid silhouette scores.")
        
        print(f"   🏆 Winner: {best_model_type} (Silhouette: {best_silhouette:.4f})")
        
        return {
            'best_model_type': best_model_type,
            'best_metric': best_silhouette,
            'metrics': best_metrics
        }
    
    def _extract_silhouette_score(self, metrics: Dict[str, Any]) -> float:
        """
        Extract silhouette score from different metrics formats.
        
        Handles:
        1. Single dataset validation: metrics['silhouette_score']
        2. Multi-dataset validation: metrics['datasets']['train']['silhouette_score'] (prioritize training)
        3. Cross-validation: metrics['mean_silhouette_score']
        """
        
        # Case 1: Direct silhouette score (old format or single dataset)
        if 'silhouette_score' in metrics:
            return float(metrics['silhouette_score'])
        
        # Case 2: Multi-dataset validation
        if 'datasets' in metrics:
            datasets = metrics['datasets']
            
            # Priority order: validation > train > test (validation set gives best unbiased estimate)
            for dataset_name in ['validation', 'train', 'test']:
                if dataset_name in datasets and 'silhouette_score' in datasets[dataset_name]:
                    score = datasets[dataset_name]['silhouette_score']
                    if score is not None:
                        return float(score)
            
            # Fallback: take first available dataset with silhouette score
            for dataset_name, dataset_metrics in datasets.items():
                if 'silhouette_score' in dataset_metrics and dataset_metrics['silhouette_score'] is not None:
                    return float(dataset_metrics['silhouette_score'])
        
        # Case 3: Cross-validation
        if 'mean_silhouette_score' in metrics:
            return float(metrics['mean_silhouette_score'])
        
        # Case 4: Check if it's nested under validation_type
        if 'validation_type' in metrics:
            if metrics['validation_type'] == 'cross_validation' and 'mean_silhouette_score' in metrics:
                return float(metrics['mean_silhouette_score'])
        
        # Fallback: return -1 if no silhouette score found
        print(f"   ⚠️ No silhouette score found in metrics: {list(metrics.keys())}")
        return -1.0 