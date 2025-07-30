"""
Data Balancing Module

This module contains all data balancing and sampling techniques including:
- Class distribution analysis
- Imbalance detection
- Oversampling methods
- SMOTE (Synthetic Minority Oversampling Technique)
- Other sampling techniques

This separation allows for easier maintenance and testing of sampling algorithms.
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, monotonically_increasing_id

class DataBalancer:
    """
    Class responsible for data balancing and sampling techniques.
    
    This class provides various methods to handle imbalanced datasets:
    - Distribution analysis and imbalance detection
    - Simple oversampling with replacement
    - SMOTE (Synthetic Minority Oversampling Technique)
    - Support for custom sampling ratios and parameters
    """
    
    def __init__(self, spark_session: SparkSession):
        """
        Initialize the DataBalancer.
        
        Args:
            spark_session: PySpark SparkSession
        """
        self.spark = spark_session
    
    def analyze_class_distribution(self, data: DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyze class distribution to detect imbalanced data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Dictionary containing class distribution statistics
        """
        print(f"Analyzing class distribution for {target_column}...")
        
        # Get class counts
        class_counts = data.groupBy(target_column).count().collect()
        total_count = data.count()
        
        distribution = {}
        for row in class_counts:
            class_value = row[target_column]
            count = row['count']
            percentage = count / total_count
            distribution[str(class_value)] = {
                'count': count,
                'percentage': percentage
            }
        
        # Find minority and majority classes
        sorted_classes = sorted(distribution.items(), key=lambda x: x[1]['count'])
        minority_class = sorted_classes[0]
        majority_class = sorted_classes[-1]
        
        print(f"Class distribution:")
        for class_val, stats in sorted_classes:
            print(f"  Class {class_val}: {stats['count']} samples ({stats['percentage']:.2%})")
        
        return {
            'distribution': distribution,
            'minority_class': minority_class[0],
            'minority_count': minority_class[1]['count'],
            'minority_percentage': minority_class[1]['percentage'],
            'majority_class': majority_class[0],
            'majority_count': majority_class[1]['count'],
            'majority_percentage': majority_class[1]['percentage'],
            'total_count': total_count,
            'num_classes': len(distribution)
        }
    
    def detect_imbalance(self, class_stats: Dict[str, Any], threshold: float = 0.05) -> bool:
        """
        Detect if dataset has imbalanced classes.
        For multiclass: checks if ANY class is below threshold (not just smallest).
        
        Args:
            class_stats: Class distribution statistics from analyze_class_distribution
            threshold: Minimum percentage for a class to be considered balanced
            
        Returns:
            True if imbalanced, False otherwise
        """
        # Check if ANY class is below threshold (improved multiclass support)
        imbalanced_classes = []
        for class_value, stats in class_stats['distribution'].items():
            if stats['percentage'] < threshold:
                imbalanced_classes.append((class_value, stats['percentage']))
        
        is_imbalanced = len(imbalanced_classes) > 0
        
        if is_imbalanced:
            print(f"⚠️  Imbalanced dataset detected: {len(imbalanced_classes)} class(es) below {threshold:.1%} threshold")
            for class_val, percentage in imbalanced_classes:
                print(f"    - Class {class_val}: {percentage:.2%} (< {threshold:.1%})")
        else:
            minority_percentage = class_stats['minority_percentage']
            print(f"✅ Balanced dataset: all classes >= {threshold:.1%} (smallest: {minority_percentage:.2%})")
        
        return is_imbalanced
    
    def upsample_minority_classes(self, data: DataFrame, target_column: str, 
                                class_stats: Dict[str, Any], method: str = 'oversample',
                                max_ratio: float = 0.3, seed: int = 12345, 
                                config: Optional[Dict[str, Any]] = None) -> DataFrame:
        """
        Upsample minority classes to improve balance.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            class_stats: Class distribution statistics
            method: Upsampling method ('oversample' or 'smote')
            max_ratio: Maximum ratio to balance minority class to
            seed: Random seed
            config: Configuration dictionary with optional parameters like 'smote_k_neighbors'
            
        Returns:
            Upsampled DataFrame
        """
        print(f"Upsampling minority classes using {method} method...")
        
        if method == 'oversample':
            return self._oversample_data(data, target_column, class_stats, max_ratio, seed)
        elif method == 'smote':
            k_neighbors = config.get('smote_k_neighbors', 5) if config else 5
            return self._smote_data(data, target_column, class_stats, max_ratio, seed, k_neighbors)
        else:
            print(f"Unknown upsampling method: {method}, returning original data")
            return data
    
    def _oversample_data(self, data: DataFrame, target_column: str, 
                       class_stats: Dict[str, Any], max_ratio: float, 
                       seed: int) -> DataFrame:
        """
        Oversample minority classes using random sampling with replacement.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            class_stats: Class distribution statistics
            max_ratio: Maximum ratio to balance minority class to
            seed: Random seed
            
        Returns:
            Oversampled DataFrame
        """
        majority_count = class_stats['majority_count']
        target_minority_count = int(majority_count * max_ratio)
        
        print(f"Target minority class size: {target_minority_count} samples ({max_ratio:.1%} of majority class)")
        
        # Create list to store DataFrames for union
        dataframes_to_union = []
        
        # Add original data
        dataframes_to_union.append(data)
        
        # For each minority class that needs upsampling
        for class_value, stats in class_stats['distribution'].items():
            current_count = stats['count']
            current_percentage = stats['percentage']
            
            # Check if this class needs upsampling
            if current_percentage < max_ratio and current_count < target_minority_count:
                needed_samples = target_minority_count - current_count
                
                if needed_samples > 0:
                    print(f"  Upsampling class {class_value}: {current_count} → {target_minority_count} (+{needed_samples} samples)")
                    
                    # Filter data for this class
                    class_data = data.filter(col(target_column) == class_value)
                    
                    # Calculate sample fraction for oversampling
                    # We want to sample with replacement to get needed_samples additional samples
                    sample_fraction = needed_samples / current_count
                    
                    # Sample with replacement
                    oversampled_class = class_data.sample(withReplacement=True, 
                                                        fraction=sample_fraction, 
                                                        seed=seed)
                    
                    dataframes_to_union.append(oversampled_class)
        
        # Union all DataFrames
        if len(dataframes_to_union) > 1:
            result = dataframes_to_union[0]
            for df in dataframes_to_union[1:]:
                result = result.union(df)
            
            # Shuffle the result
            result = result.orderBy(col(target_column).desc(), monotonically_increasing_id())
            
            print(f"Upsampling completed. New dataset size: {result.count()} samples")
            return result
        else:
            print("No upsampling needed.")
            return data
    
    def _smote_data(self, data: DataFrame, target_column: str, 
                   class_stats: Dict[str, Any], max_ratio: float, 
                   seed: int, k_neighbors: int = 5) -> DataFrame:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique) to balance minority classes.
        
        SMOTE generates synthetic samples by interpolating between minority class samples
        and their k-nearest neighbors in the feature space.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            class_stats: Class distribution statistics
            max_ratio: Maximum ratio to balance minority class to
            seed: Random seed
            k_neighbors: Number of nearest neighbors to consider (default: 5)
            
        Returns:
            DataFrame with synthetic minority class samples added
        """
        print("🧬 Applying SMOTE (Synthetic Minority Oversampling Technique)...")
        
        majority_count = class_stats['majority_count']
        target_minority_count = int(majority_count * max_ratio)
        
        print(f"Target minority class size: {target_minority_count} samples ({max_ratio:.1%} of majority class)")
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in data.columns if col != target_column]
        
        # List to store DataFrames for union
        dataframes_to_union = [data]  # Start with original data
        
        # Process each minority class that needs upsampling
        for class_value, stats in class_stats['distribution'].items():
            current_count = stats['count']
            current_percentage = stats['percentage']
            
            # Check if this class needs upsampling
            if current_percentage < max_ratio and current_count < target_minority_count:
                needed_samples = target_minority_count - current_count
                
                if needed_samples > 0 and current_count >= k_neighbors:
                    print(f"  🧬 SMOTE for class {class_value}: {current_count} → {target_minority_count} (+{needed_samples} synthetic samples)")
                    
                    # Generate synthetic samples for this class
                    synthetic_samples = self._generate_smote_samples(
                        data, target_column, class_value, 
                        feature_cols, needed_samples, k_neighbors, seed
                    )
                    
                    if synthetic_samples.count() > 0:
                        dataframes_to_union.append(synthetic_samples)
                        
                elif current_count < k_neighbors:
                    print(f"  ⚠️ Class {class_value} has only {current_count} samples, less than k_neighbors={k_neighbors}")
                    print(f"     Falling back to oversampling for this class...")
                    
                    # Fall back to simple oversampling for classes with too few samples
                    class_data = data.filter(col(target_column) == class_value)
                    sample_fraction = needed_samples / current_count
                    oversampled_class = class_data.sample(withReplacement=True, 
                                                        fraction=sample_fraction, 
                                                        seed=seed)
                    dataframes_to_union.append(oversampled_class)
        
        # Union all DataFrames
        if len(dataframes_to_union) > 1:
            result = dataframes_to_union[0]
            for df in dataframes_to_union[1:]:
                result = result.union(df)
            
            # Shuffle the result
            result = result.orderBy(col(target_column).desc(), monotonically_increasing_id())
            
            print(f"🧬 SMOTE completed. New dataset size: {result.count()} samples")
            return result
        else:
            print("No SMOTE upsampling needed.")
            return data
    
    def _generate_smote_samples(self, data: DataFrame, target_column: str, 
                              class_value: str, feature_cols: List[str], 
                              num_samples: int, k_neighbors: int, seed: int) -> DataFrame:
        """
        Generate synthetic samples for a specific class using SMOTE algorithm.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            class_value: Class value to generate samples for
            feature_cols: List of feature column names
            num_samples: Number of synthetic samples to generate
            k_neighbors: Number of nearest neighbors to consider
            seed: Random seed
            
        Returns:
            DataFrame containing synthetic samples
        """
        # Filter data for the specific class
        class_data = data.filter(col(target_column) == class_value).cache()
        class_count = class_data.count()
        
        if class_count == 0:
            return data.limit(0)  # Return empty DataFrame with same schema
        
        # Collect class data to driver for neighbor calculations
        # Note: This assumes minority class data fits in memory, which is usually reasonable
        class_rows = class_data.collect()
        
        # Extract feature vectors from individual columns
        feature_vectors = []
        for row in class_rows:
            # Get feature values from individual columns
            feature_values = [row[col] for col in feature_cols]
            feature_vectors.append(np.array(feature_values))
        
        feature_vectors = np.array(feature_vectors)
        
        print(f"    Generating {num_samples} synthetic samples using {len(feature_vectors)} base samples")
        
        # Generate synthetic samples
        synthetic_rows = []
        random.seed(seed)
        np.random.seed(seed)
        
        for _ in range(num_samples):
            # Randomly select a base sample
            base_idx = random.randint(0, len(feature_vectors) - 1)
            base_sample = feature_vectors[base_idx]
            
            # Find k nearest neighbors (excluding the base sample itself)
            distances = []
            for i, sample in enumerate(feature_vectors):
                if i != base_idx:
                    # Euclidean distance
                    dist = np.sqrt(np.sum((base_sample - sample) ** 2))
                    distances.append((dist, i))
            
            # Sort by distance and take k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:min(k_neighbors, len(distances))]
            
            if k_nearest:
                # Randomly select one of the k nearest neighbors
                _, neighbor_idx = random.choice(k_nearest)
                neighbor_sample = feature_vectors[neighbor_idx]
                
                # Generate synthetic sample by interpolating
                # new_sample = base_sample + rand(0,1) * (neighbor_sample - base_sample)
                alpha = random.random()  # Random value between 0 and 1
                synthetic_sample = base_sample + alpha * (neighbor_sample - base_sample)
                
                # Create a row for the synthetic sample
                row_dict = {}
                for i, col_name in enumerate(feature_cols):
                    # Get the original data type for this column
                    field = data.schema[col_name]
                    value = synthetic_sample[i]
                    
                    # Convert to appropriate type based on original schema
                    if field.dataType.simpleString() in ['int', 'integer']:
                        row_dict[col_name] = int(round(value))
                    elif field.dataType.simpleString() in ['bigint', 'long']:
                        row_dict[col_name] = int(round(value))
                    elif field.dataType.simpleString() in ['double', 'float']:
                        row_dict[col_name] = float(value)
                    else:
                        # Default to float for other numeric types
                        row_dict[col_name] = float(value)
                
                # Set target column value with appropriate type
                target_field = data.schema[target_column]
                if target_field.dataType.simpleString() in ['int', 'integer', 'bigint', 'long']:
                    row_dict[target_column] = int(class_value)
                else:
                    row_dict[target_column] = class_value
                
                synthetic_rows.append(row_dict)
        
        # Create DataFrame from synthetic samples
        if synthetic_rows:
            # Use the original data's schema to ensure consistent data types
            original_schema = data.schema
            synthetic_df = self.spark.createDataFrame(synthetic_rows, schema=original_schema)
            return synthetic_df
        else:
            return data.limit(0)  # Return empty DataFrame with same schema
    
    def get_balancing_info(self) -> Dict[str, Any]:
        """
        Get information about available balancing methods and their capabilities.
        
        Returns:
            Dictionary containing balancing method information
        """
        return {
            'available_methods': ['oversample', 'smote'],
            'method_descriptions': {
                'oversample': 'Random sampling with replacement - simple and fast',
                'smote': 'Synthetic Minority Oversampling Technique - generates synthetic samples using k-nearest neighbors'
            },
            'default_config': {
                'smote_k_neighbors': 5,
                'imbalance_threshold': 0.05,
                'max_balance_ratio': 0.05
            },
            'limitations': {
                'smote': 'Requires minority class to have at least k_neighbors samples, falls back to oversampling otherwise'
            }
        } 