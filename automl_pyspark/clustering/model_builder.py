"""
Clustering Model Builder

Contains clustering algorithms for unsupervised learning.
"""

from typing import Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.clustering import KMeans, BisectingKMeans


class ClusteringModelBuilder:
    """Model builder for clustering algorithms."""
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        
        self.model_types = {
            'kmeans': KMeans,
            'bisecting_kmeans': BisectingKMeans
        }
        
        print(f"✅ ClusteringModelBuilder initialized with {len(self.model_types)} model types")
    
    def build_model(self, train_data: DataFrame, features_col: str, 
                   model_type: str, k_range: List[int] = None, **params) -> Any:
        """Build clustering models for all k values and return best with comprehensive results."""
        
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported clustering model: {model_type}")
        
        # Use provided k_range or default to [3]
        if k_range is None or len(k_range) == 0:
            k_range = [3]
        
        print(f"Building {model_type} clustering model with k_range={k_range}...")
        
        # Import required libraries
        from pyspark.ml.evaluation import ClusteringEvaluator
        import numpy as np
        
        evaluator = ClusteringEvaluator(featuresCol=features_col, predictionCol="prediction")
        
        best_model = None
        best_k = None
        best_score = -1.0
        all_results = []
        
        # Test all k values and find the best one
        for k in k_range:
            print(f"   🔍 Testing k={k}...")
            
            try:
                # Build model for this k
                if model_type == 'kmeans':
                    clusterer = KMeans(featuresCol=features_col, predictionCol="prediction", k=k, seed=42, **params)
                elif model_type == 'bisecting_kmeans':
                    clusterer = BisectingKMeans(featuresCol=features_col, predictionCol="prediction", k=k, seed=42, **params)
                else:
                    raise ValueError(f"Model type {model_type} not implemented yet")
                
                # Fit the model ONCE
                model = clusterer.fit(train_data)
                
                # Make predictions ONCE
                predictions = model.transform(train_data)
                
                # Evaluate with multiple metrics ONCE
                silhouette_score = evaluator.evaluate(predictions)
                
                # Calculate additional metrics for elbow analysis
                try:
                    # Convert to numpy for sklearn metrics (more comprehensive)
                    features_list = []
                    labels_list = []
                    
                    for row in predictions.select("features", "prediction").collect():
                        features_list.append(row.features.toArray())
                        labels_list.append(row.prediction)
                    
                    features_array = np.array(features_list)
                    labels_array = np.array(labels_list)
                    
                    # Calculate comprehensive metrics
                    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
                    
                    # Calculate inertia (WCSS) manually
                    inertia = 0.0
                    for cluster_id in np.unique(labels_array):
                        cluster_points = features_array[labels_array == cluster_id]
                        if len(cluster_points) > 0:
                            centroid = np.mean(cluster_points, axis=0)
                            inertia += np.sum((cluster_points - centroid) ** 2)
                    
                    calinski_score = calinski_harabasz_score(features_array, labels_array)
                    davies_bouldin_score_val = davies_bouldin_score(features_array, labels_array)
                    
                except Exception as e:
                    print(f"      ⚠️ Could not calculate additional metrics: {str(e)}")
                    inertia = None
                    calinski_score = None
                    davies_bouldin_score_val = None
                
                # Store comprehensive results
                result = {
                    'k': k,
                    'model': model,
                    'predictions': predictions,
                    'silhouette_score': silhouette_score,
                    'inertia': inertia,
                    'calinski_score': calinski_score,
                    'davies_bouldin_score': davies_bouldin_score_val,
                    'features_array': features_array if 'features_array' in locals() else None,
                    'labels_array': labels_array if 'labels_array' in locals() else None
                }
                all_results.append(result)
                
                print(f"      Silhouette Score: {silhouette_score:.4f}")
                if inertia is not None:
                    print(f"      Inertia (WCSS): {inertia:.2f}")
                
                # Check if this is the best model so far
                if silhouette_score > best_score:
                    best_model = model
                    best_k = k
                    best_score = silhouette_score
                    
            except Exception as e:
                print(f"      ❌ Error with k={k}: {str(e)}")
                continue
        
        if best_model is None:
            raise RuntimeError(f"Failed to build any {model_type} models for k_range={k_range}")
        
        print(f"✅ Best {model_type} model: k={best_k} (Silhouette: {best_score:.4f})")
        print(f"   📊 Tested {len(all_results)} different k values")
        
        # Add comprehensive metadata to the model
        best_model._automl_metadata = {
            'best_k': best_k,
            'best_score': best_score,
            'all_results': all_results,
            'k_range_tested': k_range,
            'model_type': model_type
        }
        
        return best_model
    
    def validate_model_type(self, model_type: str) -> bool:
        """Validate if a model type is supported."""
        return model_type in self.model_types 