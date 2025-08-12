"""
Clustering Model Builder

Contains clustering algorithms for unsupervised learning.
"""

from typing import Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture

# Optional imports for advanced clustering
try:
    from pyspark.ml.clustering import LDA
    LDA_AVAILABLE = True
except ImportError:
    LDA_AVAILABLE = False

# For DBSCAN implementation (since PySpark doesn't have native DBSCAN)
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ClusteringModelBuilder:
    """Model builder for clustering algorithms."""
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        
        self.model_types = {
            'kmeans': KMeans,
            'bisecting_kmeans': BisectingKMeans,
            'gaussian_mixture': GaussianMixture
        }
        
        # Add DBSCAN if sklearn is available
        if SKLEARN_AVAILABLE:
            self.model_types['dbscan'] = 'sklearn_dbscan'  # Special marker for sklearn implementation
        
        print(f"✅ ClusteringModelBuilder initialized with {len(self.model_types)} model types")
    
    def build_model(self, train_data: DataFrame, features_col: str, 
                   model_type: str, k_range: List[int] = None, use_elbow_method: bool = True, **params) -> Any:
        """Build clustering models for all k values and return best with comprehensive results."""
        
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported clustering model: {model_type}")
        
        # Use provided k_range or default to [3]
        if k_range is None or len(k_range) == 0:
            k_range = [3]
        
        print(f"Building {model_type} clustering model with k_range={k_range}...")
        print(f"🎯 Elbow method enabled: {use_elbow_method}")
        
        # Import required libraries
        from pyspark.ml.evaluation import ClusteringEvaluator
        import numpy as np
        
        evaluator = ClusteringEvaluator(featuresCol=features_col, predictionCol="prediction")
        
        best_model = None
        best_k = None
        best_score = -1.0
        all_results = []
        
        # Handle DBSCAN differently (doesn't use k parameter)
        if model_type == 'dbscan':
            return self._build_dbscan_model(train_data, features_col, **params)
        
        # Test all k values and find the best one
        for k in k_range:
            print(f"   🔍 Testing k={k}...")
            
            try:
                # Build model for this k
                if model_type == 'kmeans':
                    clusterer = KMeans(featuresCol=features_col, predictionCol="prediction", k=k, seed=42, **params)
                elif model_type == 'bisecting_kmeans':
                    clusterer = BisectingKMeans(featuresCol=features_col, predictionCol="prediction", k=k, seed=42, **params)
                elif model_type == 'gaussian_mixture':
                    clusterer = GaussianMixture(featuresCol=features_col, predictionCol="prediction", k=k, seed=42, **params)
                elif model_type == 'dbscan':
                    # DBSCAN doesn't use k parameter, it uses eps and minPts
                    eps = params.get('eps', 0.5)
                    minPts = params.get('minPts', 5)
                    clusterer = self._create_dbscan_model(eps, minPts)
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
        
        # Apply elbow method if enabled
        if use_elbow_method and len(all_results) > 2:
            elbow_k = self._find_elbow_point(all_results)
            if elbow_k is not None:
                print(f"🎯 Elbow method suggests optimal k={elbow_k}")
                # Find the model with elbow k
                for result in all_results:
                    if result['k'] == elbow_k:
                        best_model = result['model']
                        best_k = elbow_k
                        best_score = result['silhouette_score']
                        print(f"✅ Using elbow method recommendation: k={elbow_k} (Silhouette: {best_score:.4f})")
                        break
            else:
                print(f"🎯 Elbow method could not determine optimal k, using best silhouette score")
        
        print(f"✅ Best {model_type} model: k={best_k} (Silhouette: {best_score:.4f})")
        print(f"   📊 Tested {len(all_results)} different k values")
        
        # Add comprehensive metadata to the model
        best_model._automl_metadata = {
            'best_k': best_k,
            'best_score': best_score,
            'all_results': all_results,
            'k_range_tested': k_range,
            'model_type': model_type,
            'elbow_method_used': use_elbow_method
        }
        
        return best_model
    
    def _find_elbow_point(self, all_results: List[dict]) -> int:
        """
        Find the elbow point using the elbow method.
        
        Args:
            all_results: List of results with 'k' and 'inertia' keys
            
        Returns:
            Optimal k value based on elbow analysis
        """
        try:
            # Extract k values and inertia scores
            k_values = [result['k'] for result in all_results if result['inertia'] is not None]
            inertia_values = [result['inertia'] for result in all_results if result['inertia'] is not None]
            
            if len(k_values) < 3:
                print("   ⚠️ Need at least 3 k values for elbow analysis")
                return None
            
            # Calculate the second derivative (rate of change of rate of change)
            # The elbow point is where the second derivative is maximum
            first_derivative = []
            for i in range(1, len(inertia_values)):
                first_derivative.append(inertia_values[i] - inertia_values[i-1])
            
            second_derivative = []
            for i in range(1, len(first_derivative)):
                second_derivative.append(first_derivative[i] - first_derivative[i-1])
            
            if len(second_derivative) == 0:
                print("   ⚠️ Cannot calculate second derivative for elbow analysis")
                return None
            
            # Find the k value corresponding to the maximum second derivative
            max_second_derivative_idx = second_derivative.index(max(second_derivative))
            elbow_k = k_values[max_second_derivative_idx + 1]  # +1 because we lost one point in each derivative
            
            print(f"   📊 Elbow analysis: k={elbow_k} (max second derivative)")
            return elbow_k
            
        except Exception as e:
            print(f"   ⚠️ Error in elbow analysis: {str(e)}")
            return None
    
    def validate_model_type(self, model_type: str) -> bool:
        """Validate if a model type is supported."""
        return model_type in self.model_types
    
    def _create_dbscan_model(self, eps: float, minPts: int):
        """
        Create a DBSCAN model wrapper for PySpark compatibility.

        The returned wrapper scales features using scikit‑learn's StandardScaler,
        fits a DBSCAN model on the scaled data, and provides a ``transform``
        method that attaches cluster labels to the original Spark DataFrame.
        To avoid referencing undefined ``spark`` attributes on the wrapper,
        the outer SparkSession is captured via closure.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("DBSCAN requires sklearn. Install with: pip install scikit-learn")

        # Import numpy for array operations
        import numpy as np

        # Capture Spark session from the outer class.  This allows the wrapper to
        # create DataFrames without accessing undefined attributes on itself.
        spark_session = self.spark

        class DBSCANWrapper:
            def __init__(self, eps_val: float, min_pts: int):
                self.eps = eps_val
                self.minPts = min_pts
                self.model = None
                self.scaler = StandardScaler()

            def fit(self, data: DataFrame):
                """
                Fit the DBSCAN model on the provided DataFrame.  The feature
                vectors are collected to the driver, scaled and the sklearn
                DBSCAN model is trained.  Returns ``self`` so that transform
                can be called subsequently.
                """
                # Extract features from the DataFrame
                features_list = [row.features.toArray() for row in data.select("features").collect()]
                features_array = np.array(features_list)
                # Scale features
                features_scaled = self.scaler.fit_transform(features_array)
                # Fit DBSCAN
                self.model = DBSCAN(eps=self.eps, min_samples=self.minPts)
                self.model.fit(features_scaled)
                return self

            def transform(self, data: DataFrame) -> DataFrame:
                """
                Assign cluster labels to each row in the provided DataFrame.
                The function scales the input features using the scaler
                learned during ``fit`` and predicts cluster labels using
                the fitted DBSCAN model.  The labels are then joined back
                to the original DataFrame by attaching a monotonically
                increasing index to both the features and the predictions.
                """
                from pyspark.sql.functions import monotonically_increasing_id
                from pyspark.sql.types import IntegerType
                # Extract and scale features
                features_list = [row.features.toArray() for row in data.select("features").collect()]
                features_array = np.array(features_list)
                features_scaled = self.scaler.transform(features_array)
                # Predict cluster assignments (-1 represents noise)
                predictions = self.model.fit_predict(features_scaled)
                # Create a DataFrame for predictions with row indices
                pred_rows = [(int(pred),) for pred in predictions]
                pred_df = spark_session.createDataFrame(pred_rows, ["prediction"])
                pred_df = pred_df.withColumn("row_index", monotonically_increasing_id())
                # Attach row indices to original data
                data_with_index = data.withColumn("row_index", monotonically_increasing_id())
                # Join predictions back to the original data using the row index and drop it afterwards
                result = data_with_index.join(pred_df, on="row_index").drop("row_index")
                return result

        return DBSCANWrapper(eps, minPts)
    
    def _build_dbscan_model(self, train_data: DataFrame, features_col: str, **params) -> Any:
        """Build DBSCAN model with parameter optimization."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("DBSCAN requires sklearn. Install with: pip install scikit-learn")
        
        print("🔍 Building DBSCAN model with parameter optimization...")
        
        # Get data characteristics to better tune parameters
        data_count = train_data.count()
        print(f"📊 Dataset size: {data_count} samples")
        
        # Adaptive parameter selection based on data size
        if data_count < 50:
            # Small dataset - use more conservative parameters
            eps_values = [0.1, 0.2, 0.3, 0.5]
            minPts_values = [2, 3, 4]
        elif data_count < 200:
            # Medium dataset
            eps_values = [0.1, 0.2, 0.3, 0.5, 0.7]
            minPts_values = [3, 4, 5, 7]
        else:
            # Large dataset - use default parameters
            eps_values = params.get('eps_values', [0.1, 0.3, 0.5, 0.7, 1.0])
            minPts_values = params.get('minPts_values', [3, 5, 7, 10])
        
        # Ensure minPts doesn't exceed data size
        max_minPts = min(max(minPts_values), max(3, data_count // 10))
        minPts_values = [p for p in minPts_values if p <= max_minPts]
        
        print(f"🔧 Adaptive parameters: eps_values={eps_values}, minPts_values={minPts_values}")
        
        best_model = None
        best_score = -1.0
        best_params = {}
        all_results = []
        valid_models = 0
        
        # Test different parameter combinations
        for eps in eps_values:
            for minPts in minPts_values:
                print(f"   🔍 Testing DBSCAN with eps={eps}, minPts={minPts}...")
                
                try:
                    # Create and fit DBSCAN model
                    dbscan_model = self._create_dbscan_model(eps, minPts)
                    fitted_model = dbscan_model.fit(train_data)
                    predictions = fitted_model.transform(train_data)
                    
                    # Calculate silhouette score with better error handling
                    silhouette_score = self._calculate_dbscan_silhouette(predictions)
                    
                    # Count clusters (excluding noise points with label -1)
                    unique_clusters = predictions.select("prediction").distinct().collect()
                    n_clusters = len([c for c in unique_clusters if c.prediction != -1])
                    
                    # Count noise points
                    noise_count = predictions.filter(predictions.prediction == -1).count()
                    noise_percentage = (noise_count / data_count) * 100
                    
                    print(f"      Silhouette Score: {silhouette_score:.4f}, Clusters: {n_clusters}, Noise: {noise_percentage:.1f}%")
                    
                    # Store results
                    result = {
                        'eps': eps,
                        'minPts': minPts,
                        'model': fitted_model,
                        'predictions': predictions,
                        'silhouette_score': silhouette_score,
                        'n_clusters': n_clusters,
                        'noise_percentage': noise_percentage
                    }
                    all_results.append(result)
                    
                    # Check if this is a valid model (at least 2 clusters, reasonable noise)
                    is_valid = (n_clusters >= 2 and 
                               silhouette_score > -1.0 and 
                               noise_percentage < 80.0)  # Less than 80% noise
                    
                    if is_valid:
                        valid_models += 1
                        # Check if this is the best model
                        if silhouette_score > best_score:
                            best_model = fitted_model
                            best_score = silhouette_score
                            best_params = {'eps': eps, 'minPts': minPts}
                        
                except Exception as e:
                    print(f"      ❌ Error with eps={eps}, minPts={minPts}: {str(e)}")
                    continue
        
        # If no valid models found, try more conservative parameters
        if best_model is None:
            print("⚠️ No valid DBSCAN model found, trying conservative parameters...")
            conservative_eps = [0.05, 0.1, 0.15, 0.2]
            conservative_minPts = [2, 3]
            
            for eps in conservative_eps:
                for minPts in conservative_minPts:
                    try:
                        dbscan_model = self._create_dbscan_model(eps, minPts)
                        fitted_model = dbscan_model.fit(train_data)
                        predictions = fitted_model.transform(train_data)
                        
                        silhouette_score = self._calculate_dbscan_silhouette(predictions)
                        unique_clusters = predictions.select("prediction").distinct().collect()
                        n_clusters = len([c for c in unique_clusters if c.prediction != -1])
                        
                        if n_clusters >= 2 and silhouette_score > -1.0:
                            best_model = fitted_model
                            best_score = silhouette_score
                            best_params = {'eps': eps, 'minPts': minPts}
                            print(f"✅ Found conservative model: eps={eps}, minPts={minPts} (Silhouette: {silhouette_score:.4f})")
                            break
                    except Exception as e:
                        continue
                if best_model is not None:
                    break
        
        # Final fallback
        if best_model is None:
            print("⚠️ DBSCAN failed to find meaningful clusters, using fallback model")
            # Create a simple model that assigns all points to one cluster
            best_model = self._create_fallback_model(train_data)
            best_params = {'method': 'fallback'}
            best_score = 0.0
        
        # Add metadata to the model
        best_model._automl_metadata = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'model_type': 'dbscan',
            'elbow_method_used': False,  # DBSCAN doesn't use elbow method
            'valid_models_found': valid_models,
            'total_combinations_tested': len(eps_values) * len(minPts_values)
        }
        
        print(f"✅ Best DBSCAN model: eps={best_params.get('eps', 'N/A')}, minPts={best_params.get('minPts', 'N/A')} (Silhouette: {best_score:.4f})")
        return best_model
    
    def _create_fallback_model(self, train_data: DataFrame):
        """Create a fallback model when DBSCAN fails."""
        from pyspark.sql.functions import lit
        
        class FallbackModel:
            def __init__(self):
                self.fitted = True
            
            def transform(self, data: DataFrame) -> DataFrame:
                # Assign all points to cluster 0
                return data.withColumn("prediction", lit(0))
        
        return FallbackModel()
    
    def _calculate_dbscan_silhouette(self, predictions: DataFrame) -> float:
        """Calculate silhouette score for DBSCAN predictions with improved error handling."""
        try:
            from sklearn.metrics import silhouette_score
            import numpy as np
            
            # Extract features and predictions
            features_list = []
            labels_list = []
            
            for row in predictions.select("features", "prediction").collect():
                features_list.append(row.features.toArray())
                labels_list.append(row.prediction)
            
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)
            
            # Filter out noise points (label -1) for silhouette calculation
            valid_mask = labels_array != -1
            if np.sum(valid_mask) < 2:
                return -1.0  # Invalid clustering
            
            valid_features = features_array[valid_mask]
            valid_labels = labels_array[valid_mask]
            
            # Additional validation
            unique_labels = np.unique(valid_labels)
            if len(unique_labels) < 2:
                return -1.0  # Need at least 2 clusters
            
            # Check if we have enough samples per cluster
            min_samples_per_cluster = min([np.sum(valid_labels == label) for label in unique_labels])
            if min_samples_per_cluster < 2:
                return -1.0  # Need at least 2 samples per cluster
            
            # Check for NaN or infinite values
            if np.any(np.isnan(valid_features)) or np.any(np.isinf(valid_features)):
                return -1.0  # Invalid features
            
            # Calculate silhouette score with error handling
            try:
                score = silhouette_score(valid_features, valid_labels)
                # Ensure the score is within valid range [-1, 1]
                if np.isnan(score) or np.isinf(score):
                    return -1.0
                return max(-1.0, min(1.0, score))  # Clamp to valid range
            except Exception as silhouette_error:
                print(f"   ⚠️ Silhouette calculation failed: {str(silhouette_error)}")
                return -1.0
            
        except Exception as e:
            print(f"   ⚠️ Error calculating DBSCAN silhouette: {str(e)}")
            return -1.0 