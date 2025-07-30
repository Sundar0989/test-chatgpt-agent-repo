"""
Clustering Model Validator

Validates and evaluates clustering models using comprehensive metrics and visualizations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.sql.functions import col
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class ClusteringModelValidator:
    """Validates clustering models with comprehensive metrics and visualizations."""
    
    def __init__(self, spark_session: SparkSession, output_dir: str, user_id: str, model_literal: str):
        self.spark = spark_session
        self.output_dir = output_dir
        self.user_id = user_id
        self.model_literal = model_literal
        
        # Create plots directory
        self.plots_dir = os.path.join(output_dir, 'validation_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def validate_model(self, model, dataset: DataFrame, model_type: str, output_dir: str, 
                      k_range: List[int] = None) -> Dict[str, Any]:
        """Validate clustering model with comprehensive metrics and plots."""
        
        print(f"🔍 Validating {model_type} clustering model...")
        
        # Check if model has precomputed results (from efficient building)
        if hasattr(model, '_automl_metadata') and model._automl_metadata.get('all_results'):
            print(f"   📊 Using precomputed results from model building (no re-fitting)")
            
            # Use the best model's predictions (already computed)
            best_k = model._automl_metadata['best_k']
            all_results = model._automl_metadata['all_results']
            
            # Find the best result for metrics calculation
            best_result = None
            for result in all_results:
                if result['k'] == best_k:
                    best_result = result
                    break
            
            if best_result and 'predictions' in best_result:
                predictions = best_result['predictions']
            else:
                # Fallback: compute predictions if not stored
                predictions = model.transform(dataset)
            
            # Calculate comprehensive metrics using precomputed data
            metrics = self._calculate_comprehensive_metrics(predictions, model_type)
            
            # Add precomputed metrics to results
            if best_result:
                metrics.update({
                    'silhouette_score': best_result.get('silhouette_score'),
                    'inertia': best_result.get('inertia'),
                    'calinski_score': best_result.get('calinski_score'),
                    'davies_bouldin_score': best_result.get('davies_bouldin_score')
                })
            
            # Generate clustering plots using precomputed predictions
            plot_files = self._generate_clustering_plots(predictions, model_type, k_range)
            
            # Generate elbow plot using precomputed results (NO RE-FITTING!)
            elbow_plot = self._generate_elbow_plot_from_results(all_results, model_type)
            if elbow_plot:
                plot_files.append(elbow_plot)
                
        else:
            print(f"   ⚠️ No precomputed results found - using standard validation")
            
            # Fallback to original method if no precomputed results
            predictions = model.transform(dataset)
            metrics = self._calculate_comprehensive_metrics(predictions, model_type)
            plot_files = self._generate_clustering_plots(predictions, model_type, k_range)
            
            # Generate elbow plot if k_range provided (this will re-fit models)
            if k_range and len(k_range) > 1:
                elbow_plot = self._generate_elbow_plot(dataset, model_type, k_range)
                if elbow_plot:
                    plot_files.append(elbow_plot)
        
        # Save metrics
        self._save_metrics(metrics, model_type)
        
        # Add plot information to metrics
        metrics['validation_plots'] = plot_files
        
        print(f"   ✅ Validation completed - {len(metrics)} metrics calculated")
        return metrics
    
    def validate_model_multiple_datasets(self, model, datasets: List[DataFrame], dataset_names: List[str], 
                                       model_type: str, output_dir: str, k_range: List[int] = None) -> Dict[str, Any]:
        """
        Validate clustering model on multiple datasets (train/validation/test).
        
        Args:
            model: Trained clustering model
            datasets: List of DataFrames to validate on
            dataset_names: Names corresponding to each dataset (e.g., ['train', 'validation', 'test'])
            model_type: Type of clustering model
            output_dir: Output directory for plots and metrics
            k_range: Range of k values tested
            
        Returns:
            Dictionary containing metrics for each dataset
        """
        print(f"🔍 Validating {model_type} clustering model on {len(datasets)} datasets...")
        
        all_metrics = {}
        all_plot_files = []
        
        for i, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names)):
            print(f"   📊 Validating on {dataset_name} dataset...")
            
            # Generate predictions for this dataset
            predictions = model.transform(dataset)
            
            # Calculate metrics for this dataset
            dataset_metrics = self._calculate_comprehensive_metrics(predictions, model_type)
            
            # Add dataset identifier to metrics
            dataset_metrics['dataset'] = dataset_name
            dataset_metrics['sample_count'] = dataset.count()
            
            # Store metrics for this dataset
            all_metrics[dataset_name] = dataset_metrics
            
            # Generate plots for each dataset (with dataset name in plot titles/filenames)
            dataset_plot_files = self._generate_clustering_plots_for_dataset(
                predictions, model_type, dataset_name, k_range
            )
            all_plot_files.extend(dataset_plot_files)
        
        # Generate elbow plot if we have precomputed results from training
        if hasattr(model, '_automl_metadata') and model._automl_metadata.get('all_results'):
            print(f"   📈 Generating elbow plot from training results...")
            all_results = model._automl_metadata['all_results']
            elbow_plot = self._generate_elbow_plot_from_results(all_results, model_type)
            if elbow_plot:
                all_plot_files.append(elbow_plot)
        elif k_range and len(k_range) > 1:
            # Generate elbow plot using training dataset (first dataset)
            print(f"   📈 Generating elbow plot using training dataset...")
            elbow_plot = self._generate_elbow_plot(datasets[0], model_type, k_range)
            if elbow_plot:
                all_plot_files.append(elbow_plot)
        
        # Calculate comparison metrics across datasets
        comparison_metrics = self._calculate_cross_dataset_comparison(all_metrics, dataset_names)
        
        # Combine all metrics
        final_metrics = {
            'datasets': all_metrics,
            'comparison': comparison_metrics,
            'plot_files': all_plot_files,
            'model_metadata': {
                'model_type': model_type,
                'datasets_validated': dataset_names,
                'total_datasets': len(datasets)
            }
        }
        
        # Add model metadata if available
        if hasattr(model, '_automl_metadata'):
            metadata = model._automl_metadata
            final_metrics['model_metadata'].update({
                'best_k': metadata.get('best_k'),
                'best_score': metadata.get('best_score'),
                'k_range_tested': metadata.get('k_range_tested')
            })
        
        # Save comprehensive metrics
        self._save_multiple_dataset_metrics(final_metrics, model_type)
        
        # COMPATIBILITY: Also save in the standard format expected by the reporting system
        self._save_standard_metrics_for_compatibility(final_metrics, model_type)
        
        print(f"✅ Multi-dataset validation completed for {model_type}")
        return final_metrics
    
    def _calculate_comprehensive_metrics(self, predictions: DataFrame, model_type: str) -> Dict[str, Any]:
        """Calculate comprehensive clustering metrics."""
        
        metrics = {}
        
        try:
            # Silhouette Score
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)
            metrics['silhouette_score'] = silhouette
            
            # Get number of clusters
            num_clusters = self._get_num_clusters(predictions)
            metrics['num_clusters'] = num_clusters
            
            # Convert to pandas for additional metrics
            pred_pandas = predictions.select("features", "prediction").toPandas()
            
            if len(pred_pandas) > 0:
                # Extract features (assuming VectorUDT can be converted)
                try:
                    # Convert Spark vectors to numpy arrays
                    features_list = []
                    for row in predictions.select("features").collect():
                        features_list.append(row.features.toArray())
                    features_array = np.array(features_list)
                    
                    cluster_labels = pred_pandas["prediction"].values
                    
                    # Calculate additional metrics using sklearn
                    if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
                        
                        # Calinski-Harabasz Index (higher is better)
                        calinski_harabasz = calinski_harabasz_score(features_array, cluster_labels)
                        metrics['calinski_harabasz_score'] = calinski_harabasz
                        
                        # Davies-Bouldin Index (lower is better)
                        davies_bouldin = davies_bouldin_score(features_array, cluster_labels)
                        metrics['davies_bouldin_score'] = davies_bouldin
                        
                        # Inertia (WCSS - Within-Cluster Sum of Squares)
                        inertia = self._calculate_inertia(features_array, cluster_labels)
                        metrics['inertia'] = inertia
                        metrics['wcss'] = inertia  # Alias for WCSS
                        
                        # Cluster statistics
                        cluster_sizes = np.bincount(cluster_labels.astype(int))
                        metrics['cluster_sizes'] = cluster_sizes.tolist()
                        metrics['min_cluster_size'] = int(np.min(cluster_sizes))
                        metrics['max_cluster_size'] = int(np.max(cluster_sizes))
                        metrics['mean_cluster_size'] = float(np.mean(cluster_sizes))
                        metrics['cluster_size_std'] = float(np.std(cluster_sizes))
                        
                        # Cluster balance (coefficient of variation of cluster sizes)
                        cluster_balance = np.std(cluster_sizes) / np.mean(cluster_sizes)
                        metrics['cluster_balance'] = cluster_balance
                        
                        # Inter-cluster distances
                        cluster_centers = []
                        for i in range(num_clusters):
                            cluster_mask = cluster_labels == i
                            if np.any(cluster_mask):
                                center = np.mean(features_array[cluster_mask], axis=0)
                                cluster_centers.append(center)
                        
                        if len(cluster_centers) > 1:
                            cluster_centers = np.array(cluster_centers)
                            # Calculate pairwise distances between cluster centers
                            from scipy.spatial.distance import pdist
                            inter_cluster_distances = pdist(cluster_centers)
                            metrics['min_inter_cluster_distance'] = float(np.min(inter_cluster_distances))
                            metrics['max_inter_cluster_distance'] = float(np.max(inter_cluster_distances))
                            metrics['mean_inter_cluster_distance'] = float(np.mean(inter_cluster_distances))
                            
                        # Sample count
                        metrics['sample_count'] = len(features_array)
                        
                except Exception as e:
                    print(f"   ⚠️ Could not calculate advanced metrics: {str(e)}")
            
        except Exception as e:
            print(f"   ⚠️ Could not calculate basic metrics: {str(e)}")
            metrics['silhouette_score'] = -1.0
            metrics['num_clusters'] = 0
        
        return metrics
    
    def _get_num_clusters(self, predictions: DataFrame) -> int:
        """Get the number of clusters from predictions."""
        try:
            distinct_clusters = predictions.select("prediction").distinct().count()
            return distinct_clusters
        except:
            return 0
    
    def _calculate_inertia(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        try:
            inertia = 0.0
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_points = features[cluster_mask]
                
                if len(cluster_points) > 0:
                    cluster_center = np.mean(cluster_points, axis=0)
                    # Sum of squared distances from points to cluster center
                    distances_sq = np.sum((cluster_points - cluster_center) ** 2, axis=1)
                    inertia += np.sum(distances_sq)
            
            return inertia
        except:
            return float('inf')
    
    def _generate_clustering_plots(self, predictions: DataFrame, model_type: str, 
                                 k_range: List[int] = None) -> List[str]:
        """Generate comprehensive clustering validation plots."""
        
        plot_files = []
        
        print(f"   🎨 Generating comprehensive clustering plots...")
        print(f"      📂 Plots directory: {self.plots_dir}")
        
        # Ensure plots directory exists
        os.makedirs(self.plots_dir, exist_ok=True)
        
        try:
            # Convert to pandas and numpy for plotting
            print(f"      📊 Converting predictions to pandas...")
            pred_pandas = predictions.select("features", "prediction").toPandas()
            print(f"      📊 Pandas data shape: {pred_pandas.shape}")
            
            if len(pred_pandas) == 0:
                print(f"      ⚠️ No data to plot - empty predictions")
                return plot_files
            
            # Extract features
            features_list = []
            for row in predictions.select("features").collect():
                features_list.append(row.features.toArray())
            features_array = np.array(features_list)
            
            cluster_labels = pred_pandas["prediction"].values
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Cluster Visualization (2D PCA)
            print(f"      📈 Creating PCA visualization plot...")
            plot_file = self._create_cluster_visualization_plot(
                features_array, cluster_labels, model_type, method='PCA'
            )
            if plot_file:
                plot_files.append(plot_file)
                print(f"      ✅ PCA plot created: {plot_file}")
            else:
                print(f"      ❌ Failed to create PCA plot")
            
            # 2. Cluster Visualization (2D t-SNE) - if not too many points
            if len(features_array) <= 1000:  # t-SNE is slow for large datasets
                plot_file = self._create_cluster_visualization_plot(
                    features_array, cluster_labels, model_type, method='t-SNE'
                )
                if plot_file:
                    plot_files.append(plot_file)
            
            # 3. Silhouette Analysis Plot
            plot_file = self._create_silhouette_plot(features_array, cluster_labels, model_type)
            if plot_file:
                plot_files.append(plot_file)
            
            # 4. Cluster Statistics Plot
            plot_file = self._create_cluster_statistics_plot(
                features_array, cluster_labels, model_type
            )
            if plot_file:
                plot_files.append(plot_file)
            
            # 5. Cluster Centers Heatmap (if features are not too many)
            if features_array.shape[1] <= 50:  # Only for reasonable number of features
                plot_file = self._create_cluster_centers_heatmap(
                    features_array, cluster_labels, model_type
                )
                if plot_file:
                    plot_files.append(plot_file)
            
        except Exception as e:
            print(f"   ⚠️ Could not generate plots: {str(e)}")
            import traceback
            print(f"   🐛 Plot generation error traceback:")
            traceback.print_exc()
        
        print(f"   📈 Generated {len(plot_files)} comprehensive plots")
        return plot_files
    
    def _generate_elbow_plot(self, dataset: DataFrame, model_type: str, 
                           k_range: List[int]) -> str:
        """Generate elbow plot to determine optimal number of clusters."""
        
        print(f"   📊 Generating elbow plot for k={min(k_range)} to k={max(k_range)}...")
        
        try:
            inertias = []
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            
            # Convert dataset features to numpy for sklearn metrics
            features_list = []
            for row in dataset.select("features").collect():
                features_list.append(row.features.toArray())
            features_array = np.array(features_list)
            
            for k in k_range:
                try:
                    print(f"     Testing k={k}...")
                    
                    # Fit KMeans for this k
                    if model_type == 'kmeans':
                        kmeans = KMeans(featuresCol="features", k=k, seed=42)
                    elif model_type == 'bisecting_kmeans':
                        kmeans = BisectingKMeans(featuresCol="features", k=k, seed=42)
                    else:
                        kmeans = KMeans(featuresCol="features", k=k, seed=42)
                    
                    model_k = kmeans.fit(dataset)
                    predictions_k = model_k.transform(dataset)
                    
                    # Get cluster labels
                    labels_k = np.array([row.prediction for row in predictions_k.select("prediction").collect()])
                    
                    # Calculate inertia (WCSS)
                    inertia = self._calculate_inertia(features_array, labels_k)
                    inertias.append(inertia)
                    
                    # Calculate silhouette score
                    if k > 1:
                        evaluator = ClusteringEvaluator()
                        silhouette = evaluator.evaluate(predictions_k)
                        silhouette_scores.append(silhouette)
                        
                        # Additional sklearn metrics
                        if len(np.unique(labels_k)) > 1:
                            calinski = calinski_harabasz_score(features_array, labels_k)
                            calinski_scores.append(calinski)
                            
                            davies_bouldin = davies_bouldin_score(features_array, labels_k)
                            davies_bouldin_scores.append(davies_bouldin)
                        else:
                            calinski_scores.append(0)
                            davies_bouldin_scores.append(float('inf'))
                    else:
                        silhouette_scores.append(-1)
                        calinski_scores.append(0)
                        davies_bouldin_scores.append(float('inf'))
                        
                except Exception as e:
                    print(f"     ⚠️ Error with k={k}: {str(e)}")
                    inertias.append(float('inf'))
                    silhouette_scores.append(-1)
                    calinski_scores.append(0)
                    davies_bouldin_scores.append(float('inf'))
            
            # Create elbow plot
            plt.figure(figsize=(16, 12))
            
            # 1. Elbow plot (Inertia/WCSS)
            plt.subplot(2, 2, 1)
            plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
            plt.title('Elbow Method for Optimal k')
            plt.grid(True, alpha=0.3)
            
            # Find elbow point using the rate of change
            if len(inertias) > 2:
                # Calculate second derivative to find elbow
                deltas = np.diff(inertias)
                second_deltas = np.diff(deltas)
                if len(second_deltas) > 0:
                    elbow_idx = np.argmax(second_deltas) + 1
                    optimal_k = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[0]
                    plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.8, 
                               label=f'Elbow at k={optimal_k}')
                    plt.legend()
            
            # 2. Silhouette scores
            plt.subplot(2, 2, 2)
            plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score vs Number of Clusters')
            plt.grid(True, alpha=0.3)
            
            # Mark best silhouette score
            if silhouette_scores:
                best_silhouette_idx = np.argmax(silhouette_scores)
                best_silhouette_k = k_range[best_silhouette_idx]
                plt.axvline(x=best_silhouette_k, color='g', linestyle='--', alpha=0.8,
                           label=f'Best at k={best_silhouette_k}')
                plt.legend()
            
            # 3. Calinski-Harabasz Index
            plt.subplot(2, 2, 3)
            plt.plot(k_range, calinski_scores, 'mo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Calinski-Harabasz Index')
            plt.title('Calinski-Harabasz Index vs Number of Clusters (Higher is Better)')
            plt.grid(True, alpha=0.3)
            
            # Mark best Calinski-Harabasz score
            if calinski_scores:
                best_calinski_idx = np.argmax(calinski_scores)
                best_calinski_k = k_range[best_calinski_idx]
                plt.axvline(x=best_calinski_k, color='m', linestyle='--', alpha=0.8,
                           label=f'Best at k={best_calinski_k}')
                plt.legend()
            
            # 4. Davies-Bouldin Index
            plt.subplot(2, 2, 4)
            plt.plot(k_range, davies_bouldin_scores, 'co-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Davies-Bouldin Index')
            plt.title('Davies-Bouldin Index vs Number of Clusters (Lower is Better)')
            plt.grid(True, alpha=0.3)
            
            # Mark best Davies-Bouldin score (lowest)
            if davies_bouldin_scores:
                valid_scores = [s for s in davies_bouldin_scores if s != float('inf')]
                if valid_scores:
                    best_db_idx = davies_bouldin_scores.index(min(valid_scores))
                    best_db_k = k_range[best_db_idx]
                    plt.axvline(x=best_db_k, color='c', linestyle='--', alpha=0.8,
                               label=f'Best at k={best_db_k}')
                    plt.legend()
            
            plt.suptitle(f'{model_type} - Clustering Validation Metrics', fontsize=16)
            plt.tight_layout()
            
            # Add recommendation text
            recommendation_text = self._generate_k_recommendation(
                k_range, inertias, silhouette_scores, calinski_scores, davies_bouldin_scores
            )
            
            plt.figtext(0.02, 0.02, recommendation_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            filename = f'{model_type}_elbow_plot.png'
            filepath = os.path.join(self.plots_dir, filename)
            print(f"      💾 Saving elbow plot: {filename}")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save elbow analysis data
            elbow_data = {
                'k_range': k_range,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'recommendation': recommendation_text
            }
            
            elbow_file = os.path.join(self.output_dir, f'{model_type}_elbow_analysis.json')
            with open(elbow_file, 'w') as f:
                json.dump(elbow_data, f, indent=2)
            
            # Verify plot was saved and return full path
            if os.path.exists(filepath):
                print(f"      ✅ Elbow plot saved successfully: {filepath}")
                return filepath
            else:
                print(f"      ❌ Failed to save elbow plot: {filepath}")
                return None
            
        except Exception as e:
            print(f"   ⚠️ Could not create elbow plot: {str(e)}")
            plt.close()
            return None
    
    def _generate_k_recommendation(self, k_range, inertias, silhouette_scores, 
                                 calinski_scores, davies_bouldin_scores):
        """Generate recommendation for optimal k based on multiple metrics."""
        
        recommendations = []
        
        # Elbow method recommendation
        if len(inertias) > 2:
            deltas = np.diff(inertias)
            second_deltas = np.diff(deltas)
            if len(second_deltas) > 0:
                elbow_idx = np.argmax(second_deltas) + 1
                if elbow_idx < len(k_range):
                    recommendations.append(f"Elbow method suggests k={k_range[elbow_idx]}")
        
        # Silhouette method recommendation
        if silhouette_scores and max(silhouette_scores) > 0:
            best_silhouette_idx = np.argmax(silhouette_scores)
            recommendations.append(f"Silhouette analysis suggests k={k_range[best_silhouette_idx]}")
        
        # Calinski-Harabasz recommendation
        if calinski_scores and max(calinski_scores) > 0:
            best_calinski_idx = np.argmax(calinski_scores)
            recommendations.append(f"Calinski-Harabasz index suggests k={k_range[best_calinski_idx]}")
        
        # Davies-Bouldin recommendation
        valid_db_scores = [s for s in davies_bouldin_scores if s != float('inf')]
        if valid_db_scores:
            best_db_idx = davies_bouldin_scores.index(min(valid_db_scores))
            recommendations.append(f"Davies-Bouldin index suggests k={k_range[best_db_idx]}")
        
        if recommendations:
            return "Recommendations:\n" + "\n".join(recommendations)
        else:
            return "Unable to provide clear recommendation. Consider domain knowledge."
    
    def _create_cluster_visualization_plot(self, features, labels, model_type, method='PCA'):
        """Create 2D cluster visualization using PCA or t-SNE."""
        try:
            plt.figure(figsize=(12, 10))
            
            # Reduce dimensions to 2D
            if method == 'PCA':
                reducer = PCA(n_components=2, random_state=42)
                features_2d = reducer.fit_transform(features)
                variance_explained = reducer.explained_variance_ratio_
                title_suffix = f"(PC1: {variance_explained[0]:.1%}, PC2: {variance_explained[1]:.1%})"
            elif method == 't-SNE':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
                features_2d = reducer.fit_transform(features)
                title_suffix = ""
            else:
                return None
            
            # Create scatter plot
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[colors[i]], label=f'Cluster {int(label)}', 
                          alpha=0.7, s=50)
            
            plt.xlabel(f'{method} Component 1')
            plt.ylabel(f'{method} Component 2')
            plt.title(f'{model_type} - Cluster Visualization using {method} {title_suffix}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Add cluster centers if not too many clusters
            if len(unique_labels) <= 10:
                for label in unique_labels:
                    mask = labels == label
                    if np.any(mask):
                        center_x = np.mean(features_2d[mask, 0])
                        center_y = np.mean(features_2d[mask, 1])
                        plt.scatter(center_x, center_y, c='black', marker='x', s=200, linewidths=3)
            
            filename = f'{model_type}_cluster_visualization_{method.lower()}.png'
            filepath = os.path.join(self.plots_dir, filename)
            print(f"      💾 Saving {method} plot: {filename}")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Verify plot was saved and return full path
            if os.path.exists(filepath):
                print(f"      ✅ {method} plot saved successfully: {filepath}")
                return filepath
            else:
                print(f"      ❌ Failed to save {method} plot: {filepath}")
                return None
            
        except Exception as e:
            print(f"   ⚠️ Could not create {method} visualization: {str(e)}")
            plt.close()
            return None
    
    def _create_silhouette_plot(self, features, labels, model_type):
        """Create silhouette analysis plot."""
        try:
            from sklearn.metrics import silhouette_samples, silhouette_score
            
            # Calculate silhouette scores
            sample_silhouette_values = silhouette_samples(features, labels)
            silhouette_avg = silhouette_score(features, labels)
            
            plt.figure(figsize=(12, 8))
            
            y_lower = 10
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                # Aggregate silhouette scores for samples belonging to cluster
                cluster_silhouette_values = sample_silhouette_values[labels == label]
                cluster_silhouette_values.sort()
                
                size_cluster = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster
                
                plt.fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_silhouette_values,
                                facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
                
                # Label the silhouette plots with their cluster numbers at the middle
                plt.text(-0.05, y_lower + 0.5 * size_cluster, str(int(label)))
                y_lower = y_upper + 10
            
            plt.xlabel('Silhouette Coefficient Values')
            plt.ylabel('Cluster Labels')
            plt.title(f'{model_type} - Silhouette Analysis (Average Score: {silhouette_avg:.3f})')
            
            # Vertical line for average silhouette score
            plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
                       label=f'Average Score: {silhouette_avg:.3f}')
            plt.legend()
            
            filename = f'{model_type}_silhouette_analysis.png'
            filepath = os.path.join(self.plots_dir, filename)
            print(f"      💾 Saving silhouette plot: {filename}")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Verify plot was saved and return full path
            if os.path.exists(filepath):
                print(f"      ✅ Silhouette plot saved successfully: {filepath}")
                return filepath
            else:
                print(f"      ❌ Failed to save silhouette plot: {filepath}")
                return None
            
        except Exception as e:
            print(f"   ⚠️ Could not create silhouette plot: {str(e)}")
            plt.close()
            return None
    
    def _create_cluster_statistics_plot(self, features, labels, model_type):
        """Create cluster statistics visualization."""
        try:
            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels)
            
            plt.figure(figsize=(15, 10))
            
            # 1. Cluster sizes
            plt.subplot(2, 3, 1)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            plt.bar(range(num_clusters), cluster_sizes, color=plt.cm.Set3(np.linspace(0, 1, num_clusters)))
            plt.xlabel('Cluster')
            plt.ylabel('Number of Points')
            plt.title('Cluster Sizes')
            plt.xticks(range(num_clusters), [f'C{int(label)}' for label in unique_labels])
            
            # 2. Cluster densities (average distance to centroid)
            plt.subplot(2, 3, 2)
            cluster_densities = []
            for label in unique_labels:
                mask = labels == label
                cluster_points = features[mask]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    avg_distance = np.mean(distances)
                    cluster_densities.append(avg_distance)
                else:
                    cluster_densities.append(0)
            
            plt.bar(range(num_clusters), cluster_densities, color=plt.cm.Set3(np.linspace(0, 1, num_clusters)))
            plt.xlabel('Cluster')
            plt.ylabel('Average Distance to Centroid')
            plt.title('Cluster Density')
            plt.xticks(range(num_clusters), [f'C{int(label)}' for label in unique_labels])
            
            # 3. Feature means by cluster (if not too many features)
            if features.shape[1] <= 10:
                plt.subplot(2, 3, 3)
                cluster_means = []
                for label in unique_labels:
                    mask = labels == label
                    cluster_points = features[mask]
                    if len(cluster_points) > 0:
                        cluster_means.append(np.mean(cluster_points, axis=0))
                    else:
                        cluster_means.append(np.zeros(features.shape[1]))
                
                cluster_means = np.array(cluster_means)
                for i in range(features.shape[1]):
                    plt.plot(range(num_clusters), cluster_means[:, i], 'o-', label=f'Feature {i}')
                
                plt.xlabel('Cluster')
                plt.ylabel('Feature Mean')
                plt.title('Feature Means by Cluster')
                plt.xticks(range(num_clusters), [f'C{int(label)}' for label in unique_labels])
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 4. Cluster separation (distances between centroids)
            plt.subplot(2, 3, 4)
            centroids = []
            for label in unique_labels:
                mask = labels == label
                cluster_points = features[mask]
                if len(cluster_points) > 0:
                    centroids.append(np.mean(cluster_points, axis=0))
            
            if len(centroids) > 1:
                from scipy.spatial.distance import pdist, squareform
                centroids = np.array(centroids)
                distances = squareform(pdist(centroids))
                
                # Create heatmap
                im = plt.imshow(distances, cmap='viridis')
                plt.colorbar(im)
                plt.xlabel('Cluster')
                plt.ylabel('Cluster')
                plt.title('Inter-Cluster Distances')
                plt.xticks(range(num_clusters), [f'C{int(label)}' for label in unique_labels])
                plt.yticks(range(num_clusters), [f'C{int(label)}' for label in unique_labels])
                
                # Add text annotations
                for i in range(num_clusters):
                    for j in range(num_clusters):
                        plt.text(j, i, f'{distances[i, j]:.2f}', 
                               ha="center", va="center", color="white")
            
            # 5. Summary statistics
            plt.subplot(2, 3, 5)
            stats_text = f'''Clustering Statistics:
Number of Clusters: {num_clusters}
Total Points: {len(labels)}
Avg Cluster Size: {np.mean(cluster_sizes):.1f}
Min Cluster Size: {min(cluster_sizes)}
Max Cluster Size: {max(cluster_sizes)}
Cluster Size Std: {np.std(cluster_sizes):.1f}
'''
            
            if cluster_densities:
                stats_text += f'''
Avg Cluster Density: {np.mean(cluster_densities):.3f}
Min Cluster Density: {min(cluster_densities):.3f}
Max Cluster Density: {max(cluster_densities):.3f}'''
            
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            
            plt.suptitle(f'{model_type} - Cluster Statistics', fontsize=16)
            plt.tight_layout()
            
            filename = f'{model_type}_cluster_statistics.png'
            filepath = os.path.join(self.plots_dir, filename)
            print(f"      💾 Saving cluster statistics plot: {filename}")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Verify plot was saved and return full path
            if os.path.exists(filepath):
                print(f"      ✅ Cluster statistics plot saved successfully: {filepath}")
                return filepath
            else:
                print(f"      ❌ Failed to save cluster statistics plot: {filepath}")
                return None
            
        except Exception as e:
            print(f"   ⚠️ Could not create cluster statistics plot: {str(e)}")
            plt.close()
            return None
    
    def _create_cluster_centers_heatmap(self, features, labels, model_type):
        """Create heatmap of cluster centers."""
        try:
            unique_labels = np.unique(labels)
            centroids = []
            
            for label in unique_labels:
                mask = labels == label
                cluster_points = features[mask]
                if len(cluster_points) > 0:
                    centroids.append(np.mean(cluster_points, axis=0))
                else:
                    centroids.append(np.zeros(features.shape[1]))
            
            centroids = np.array(centroids)
            
            plt.figure(figsize=(12, 8))
            
            # Create heatmap
            sns.heatmap(centroids, 
                       annot=True if features.shape[1] <= 20 else False,
                       fmt='.2f',
                       cmap='RdBu_r',
                       center=0,
                       xticklabels=[f'Feature {i}' for i in range(features.shape[1])],
                       yticklabels=[f'Cluster {int(label)}' for label in unique_labels])
            
            plt.title(f'{model_type} - Cluster Centers Heatmap')
            plt.xlabel('Features')
            plt.ylabel('Clusters')
            
            filename = f'{model_type}_cluster_centers_heatmap.png'
            filepath = os.path.join(self.plots_dir, filename)
            print(f"      💾 Saving cluster centers heatmap: {filename}")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Verify plot was saved and return full path
            if os.path.exists(filepath):
                print(f"      ✅ Cluster centers heatmap saved successfully: {filepath}")
                return filepath
            else:
                print(f"      ❌ Failed to save cluster centers heatmap: {filepath}")
                return None
            
        except Exception as e:
            print(f"   ⚠️ Could not create cluster centers heatmap: {str(e)}")
            plt.close()
            return None
    
    def _save_metrics(self, metrics: Dict[str, Any], model_type: str):
        """Save metrics to JSON file."""
        try:
            filename = f'{model_type}_clustering_metrics.json'
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            print(f"   💾 Metrics saved to {filename}")
            
        except Exception as e:
            print(f"   ⚠️ Could not save metrics: {str(e)}")
    
    def get_available_metrics(self) -> List[str]:
        """Get list of all available clustering metrics."""
        return [
            'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
            'inertia', 'wcss', 'num_clusters', 'cluster_sizes', 'min_cluster_size',
            'max_cluster_size', 'mean_cluster_size', 'cluster_size_std', 'cluster_balance',
            'min_inter_cluster_distance', 'max_inter_cluster_distance', 
            'mean_inter_cluster_distance', 'sample_count'
        ]
    
    def _generate_elbow_plot_from_results(self, all_results: List[Dict], model_type: str) -> str:
        """Generate elbow plot using precomputed results (no model re-fitting)."""
        
        print(f"   📊 Generating elbow plot from precomputed results...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import json
            import os
            
            # Extract data from precomputed results
            k_values = []
            inertias = []
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            
            for result in sorted(all_results, key=lambda x: x['k']):
                k_values.append(result['k'])
                inertias.append(result.get('inertia', 0))
                silhouette_scores.append(result.get('silhouette_score', 0))
                calinski_scores.append(result.get('calinski_score', 0))
                davies_bouldin_scores.append(result.get('davies_bouldin_score', 0))
            
            if not k_values:
                print("   ⚠️ No results to plot")
                return None
            
            # Create the plot
            plt.figure(figsize=(15, 10))
            
            # 1. Inertia (WCSS) - Elbow Method
            plt.subplot(2, 2, 1)
            plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia (WCSS)')
            plt.title('Elbow Method: Inertia vs Number of Clusters')
            plt.grid(True, alpha=0.3)
            
            # Find elbow point using the rate of change
            if len(inertias) > 2:
                deltas = np.diff(inertias)
                second_deltas = np.diff(deltas)
                if len(second_deltas) > 0:
                    elbow_idx = np.argmax(second_deltas) + 1
                    optimal_k = k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[0]
                    plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.8, 
                               label=f'Elbow at k={optimal_k}')
                    plt.legend()
            
            # 2. Silhouette scores
            plt.subplot(2, 2, 2)
            plt.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score vs Number of Clusters')
            plt.grid(True, alpha=0.3)
            
            # Mark best silhouette score
            if silhouette_scores:
                best_silhouette_idx = np.argmax(silhouette_scores)
                best_silhouette_k = k_values[best_silhouette_idx]
                plt.axvline(x=best_silhouette_k, color='g', linestyle='--', alpha=0.8,
                           label=f'Best at k={best_silhouette_k}')
                plt.legend()
            
            # 3. Calinski-Harabasz Index
            plt.subplot(2, 2, 3)
            plt.plot(k_values, calinski_scores, 'mo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Calinski-Harabasz Index')
            plt.title('Calinski-Harabasz Index vs Number of Clusters (Higher is Better)')
            plt.grid(True, alpha=0.3)
            
            # Mark best Calinski-Harabasz score (higher is better)
            if calinski_scores and any(score > 0 for score in calinski_scores):
                valid_scores = [(i, score) for i, score in enumerate(calinski_scores) if score > 0]
                if valid_scores:
                    best_ch_idx, _ = max(valid_scores, key=lambda x: x[1])
                    best_ch_k = k_values[best_ch_idx]
                    plt.axvline(x=best_ch_k, color='m', linestyle='--', alpha=0.8,
                               label=f'Best at k={best_ch_k}')
                    plt.legend()
            
            # 4. Davies-Bouldin Index (lower is better)
            plt.subplot(2, 2, 4)
            plt.plot(k_values, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Davies-Bouldin Index')
            plt.title('Davies-Bouldin Index vs Number of Clusters (Lower is Better)')
            plt.grid(True, alpha=0.3)
            
            # Mark best Davies-Bouldin score
            if davies_bouldin_scores and any(score > 0 for score in davies_bouldin_scores):
                valid_scores = [(i, score) for i, score in enumerate(davies_bouldin_scores) if score > 0]
                if valid_scores:
                    best_db_idx, _ = min(valid_scores, key=lambda x: x[1])
                    best_db_k = k_values[best_db_idx]
                    plt.axvline(x=best_db_k, color='r', linestyle='--', alpha=0.8,
                               label=f'Best at k={best_db_k}')
                    plt.legend()
            
            # Add overall title and recommendation
            best_k_by_silhouette = k_values[np.argmax(silhouette_scores)] if silhouette_scores else k_values[0]
            
            plt.suptitle(f'{model_type.upper()} Clustering: Optimal K Analysis\n'
                        f'Recommended k={best_k_by_silhouette} (based on silhouette score)', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the plot
            filename = f'{model_type}_elbow_plot_efficient.png'
            filepath = os.path.join(self.output_dir, filename)
            print(f"      💾 Saving efficient elbow plot: {filename}")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save elbow analysis data
            elbow_data = {
                'k_range': k_values,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'recommended_k': best_k_by_silhouette,
                'method': 'precomputed_efficient'
            }
            
            elbow_file = os.path.join(self.output_dir, f'{model_type}_elbow_analysis_efficient.json')
            with open(elbow_file, 'w') as f:
                json.dump(elbow_data, f, indent=2)
            
            # Verify plot was saved and return full path
            if os.path.exists(filepath):
                print(f"      ✅ Efficient elbow plot saved successfully: {filepath}")
                return filepath
            else:
                print(f"      ❌ Failed to save efficient elbow plot: {filepath}")
                return None
            
        except Exception as e:
            print(f"   ⚠️ Could not create efficient elbow plot: {str(e)}")
            plt.close()
            return None 
    
    def _generate_clustering_plots_for_dataset(self, predictions: DataFrame, model_type: str, 
                                             dataset_name: str, k_range: List[int] = None) -> List[str]:
        """Generate clustering plots for a specific dataset."""
        
        plot_files = []
        
        print(f"   🎨 Generating plots for {dataset_name} dataset...")
        print(f"      📂 Plots directory: {self.plots_dir}")
        
        # Ensure plots directory exists
        os.makedirs(self.plots_dir, exist_ok=True)
        print(f"      ✅ Plots directory confirmed: {os.path.exists(self.plots_dir)}")
        
        try:
            # Convert to pandas for plotting
            print(f"      📊 Converting predictions to pandas...")
            pandas_data = predictions.select('features', 'prediction').toPandas()
            print(f"      📊 Pandas data shape: {pandas_data.shape}")
            
            # Extract features from vector column
            print(f"      🔢 Extracting features array...")
            features_array = np.array([row.features.toArray() for row in predictions.select('features').collect()])
            print(f"      🔢 Features array shape: {features_array.shape}")
            
            # 1. Cluster scatter plot with PCA
            if features_array.shape[1] >= 2:
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features_array)
                
                plt.figure(figsize=(10, 8))
                clusters = pandas_data['prediction'].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))
                
                for i, cluster in enumerate(sorted(clusters)):
                    cluster_mask = pandas_data['prediction'] == cluster
                    plt.scatter(features_2d[cluster_mask, 0], features_2d[cluster_mask, 1], 
                              c=[colors[i]], label=f'Cluster {cluster}', alpha=0.6, s=50)
                
                plt.title(f'{model_type.title()} Clustering Results - {dataset_name.title()} Dataset\n'
                         f'PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_filename = f"{model_type}_{dataset_name}_clustering_scatter.png"
                plot_path = os.path.join(self.plots_dir, plot_filename)
                print(f"      💾 Saving scatter plot: {plot_filename}")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Verify plot was saved
                if os.path.exists(plot_path):
                    plot_files.append(plot_path)
                    print(f"      ✅ Scatter plot saved successfully: {plot_path}")
                else:
                    print(f"      ❌ Failed to save scatter plot: {plot_path}")
                
            # 2. Cluster distribution plot
            plt.figure(figsize=(10, 6))
            cluster_counts = pandas_data['prediction'].value_counts().sort_index()
            
            plt.bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='skyblue', edgecolor='navy')
            plt.title(f'Cluster Distribution - {dataset_name.title()} Dataset\n{model_type.title()} Model')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Points')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(cluster_counts.values):
                plt.text(cluster_counts.index[i], v + max(cluster_counts.values) * 0.01, str(v), 
                        ha='center', va='bottom')
            
            # Save plot
            plot_filename = f"{model_type}_{dataset_name}_distribution.png"
            plot_path = os.path.join(self.plots_dir, plot_filename)
            print(f"      💾 Saving distribution plot: {plot_filename}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Verify plot was saved
            if os.path.exists(plot_path):
                plot_files.append(plot_path)
                print(f"      ✅ Distribution plot saved successfully: {plot_path}")
            else:
                print(f"      ❌ Failed to save distribution plot: {plot_path}")
            
        except Exception as e:
            print(f"   ⚠️ Error generating plots for {dataset_name}: {str(e)}")
            import traceback
            print(f"   🐛 Full error traceback:")
            traceback.print_exc()
        
        print(f"   📈 Generated {len(plot_files)} plots for {dataset_name}")
        return plot_files
    
    def _calculate_cross_dataset_comparison(self, all_metrics: Dict[str, Dict], dataset_names: List[str]) -> Dict[str, Any]:
        """Calculate comparison metrics across different datasets."""
        
        comparison_metrics = {}
        
        try:
            # Compare silhouette scores across datasets
            silhouette_scores = {}
            for dataset_name in dataset_names:
                if 'silhouette_score' in all_metrics[dataset_name]:
                    silhouette_scores[dataset_name] = all_metrics[dataset_name]['silhouette_score']
            
            if silhouette_scores:
                comparison_metrics['silhouette_comparison'] = silhouette_scores
                comparison_metrics['best_silhouette_dataset'] = max(silhouette_scores, key=silhouette_scores.get)
                comparison_metrics['silhouette_std'] = np.std(list(silhouette_scores.values()))
                
            # Compare cluster counts across datasets
            cluster_counts = {}
            for dataset_name in dataset_names:
                if 'n_clusters' in all_metrics[dataset_name]:
                    cluster_counts[dataset_name] = all_metrics[dataset_name]['n_clusters']
            
            if cluster_counts:
                comparison_metrics['cluster_count_comparison'] = cluster_counts
                comparison_metrics['consistent_cluster_count'] = len(set(cluster_counts.values())) == 1
                
            # Compare sample sizes
            sample_sizes = {}
            for dataset_name in dataset_names:
                if 'sample_count' in all_metrics[dataset_name]:
                    sample_sizes[dataset_name] = all_metrics[dataset_name]['sample_count']
            
            if sample_sizes:
                comparison_metrics['sample_size_comparison'] = sample_sizes
                comparison_metrics['total_samples'] = sum(sample_sizes.values())
                
        except Exception as e:
            print(f"   ⚠️ Error calculating cross-dataset comparison: {str(e)}")
            comparison_metrics['error'] = str(e)
        
        return comparison_metrics
    
    def _save_multiple_dataset_metrics(self, metrics: Dict[str, Any], model_type: str):
        """Save metrics from multiple dataset validation."""
        
        try:
            # Save comprehensive metrics to file
            metrics_file = os.path.join(self.output_dir, f"{model_type}_multi_dataset_validation_metrics.json")
            
            with open(metrics_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.float64, np.float32)):
                        return float(obj)
                    else:
                        return obj
                
                json_metrics = convert_numpy(metrics)
                json.dump(json_metrics, f, indent=2)
            
            print(f"   📄 Multi-dataset metrics saved to: {metrics_file}")
            
            # Create summary report
            summary_file = os.path.join(self.output_dir, f"{model_type}_dataset_validation_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Multi-Dataset Clustering Validation Summary\n")
                f.write(f"============================================\n\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Datasets Validated: {', '.join(metrics['model_metadata']['datasets_validated'])}\n")
                f.write(f"Total Datasets: {metrics['model_metadata']['total_datasets']}\n\n")
                
                # Write dataset-specific metrics
                for dataset_name, dataset_metrics in metrics['datasets'].items():
                    f.write(f"{dataset_name.upper()} DATASET METRICS:\n")
                    f.write(f"  Sample Count: {dataset_metrics.get('sample_count', 'N/A')}\n")
                    if 'silhouette_score' in dataset_metrics:
                        f.write(f"  Silhouette Score: {dataset_metrics['silhouette_score']:.4f}\n")
                    else:
                        f.write(f"  Silhouette Score: N/A\n")
                    f.write(f"  Number of Clusters: {dataset_metrics.get('num_clusters', 'N/A')}\n")
                    if 'inertia' in dataset_metrics:
                        f.write(f"  Inertia: {dataset_metrics['inertia']:.4f}\n")
                    else:
                        f.write(f"  Inertia: N/A\n")
                    f.write("\n")
                
                # Write comparison metrics
                if 'comparison' in metrics:
                    f.write("CROSS-DATASET COMPARISON:\n")
                    comp = metrics['comparison']
                    if 'best_silhouette_dataset' in comp:
                        f.write(f"  Best Silhouette Dataset: {comp['best_silhouette_dataset']}\n")
                    if 'consistent_cluster_count' in comp:
                        f.write(f"  Consistent Cluster Count: {comp['consistent_cluster_count']}\n")
                    if 'total_samples' in comp:
                        f.write(f"  Total Samples: {comp['total_samples']:,}\n")
            
            print(f"   📄 Summary report saved to: {summary_file}")
            
        except Exception as e:
            print(f"   ⚠️ Error saving multi-dataset metrics: {str(e)}")
    
    def _save_standard_metrics_for_compatibility(self, multi_dataset_metrics: Dict[str, Any], model_type: str):
        """
        Save metrics in the standard format expected by the reporting system.
        
        This ensures compatibility with existing score generators and performance overview reports.
        """
        try:
            # Extract the best dataset metrics for the standard format
            # Priority: validation > test > train (validation gives most unbiased estimate)
            standard_metrics = {}
            datasets = multi_dataset_metrics.get('datasets', {})
            
            if 'validation' in datasets:
                standard_metrics = datasets['validation'].copy()
                source_dataset = 'validation'
            elif 'test' in datasets:
                standard_metrics = datasets['test'].copy()
                source_dataset = 'test'
            elif 'train' in datasets:
                standard_metrics = datasets['train'].copy()
                source_dataset = 'train'
            else:
                # Fallback: use first available dataset
                if datasets:
                    first_key = list(datasets.keys())[0]
                    standard_metrics = datasets[first_key].copy()
                    source_dataset = first_key
                else:
                    print(f"   ⚠️ No dataset metrics found for {model_type}")
                    return
            
            # Add validation plots from the multi-dataset results
            if 'plot_files' in multi_dataset_metrics:
                standard_metrics['validation_plots'] = multi_dataset_metrics['plot_files']
            
            # Add metadata about the source
            standard_metrics['metric_source'] = f'{source_dataset}_dataset'
            standard_metrics['multi_dataset_available'] = True
            standard_metrics['validation_type'] = 'multi_dataset'
            
            # Add comparison summary
            if 'comparison' in multi_dataset_metrics:
                standard_metrics['dataset_comparison'] = multi_dataset_metrics['comparison']
            
            # Save using the standard method (creates {model_type}_clustering_metrics.json)
            self._save_metrics(standard_metrics, model_type)
            
            print(f"   📄 Standard metrics saved for compatibility (source: {source_dataset} dataset)")
            
        except Exception as e:
            print(f"   ⚠️ Error saving standard compatibility metrics: {str(e)}")