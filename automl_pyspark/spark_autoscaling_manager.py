"""
Spark Autoscaling Manager

This module provides dynamic Spark cluster scaling capabilities based on:
- Data size and complexity
- Processing requirements
- Available system resources
- Performance targets

Features:
- Automatic executor scaling based on data size
- Dynamic memory allocation
- Partition optimization
- Resource monitoring and adjustment
"""

import os
import time
import psutil
from typing import Dict, Any, Optional, Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, lit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkAutoscalingManager:
    """
    Manages dynamic Spark cluster scaling based on data characteristics and processing needs.
    """
    
    def __init__(self, spark_session: SparkSession, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the autoscaling manager.
        
        Args:
            spark_session: Active Spark session
            config: Configuration dictionary for autoscaling behavior
        """
        self.spark = spark_session
        
        # Get default config and merge with provided config
        default_config = self._get_default_config()
        if config:
            # Deep merge configurations
            self.config = self._deep_merge_configs(default_config, config)
        else:
            self.config = default_config
            
        self.initial_config = self._capture_current_config()
        self.scaling_history = []
        
        # Performance monitoring
        self.performance_metrics = {
            'processing_times': [],
            'memory_usage': [],
            'executor_counts': [],
            'data_sizes': []
        }
        
        logger.info("🚀 Spark Autoscaling Manager initialized")
    
    def _deep_merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge user configuration with default configuration."""
        merged_config = default_config.copy()
        
        for key, value in user_config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged_config[key] = self._deep_merge_configs(merged_config[key], value)
            else:
                # Override with user value
                merged_config[key] = value
        
        return merged_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default autoscaling configuration."""
        return {
            # Scaling thresholds
            'min_executors': 1,
            'max_executors': 20,
            'target_executor_memory': '4g',
            'target_driver_memory': '8g',
            
            # Data size thresholds (in MB)
            'small_dataset_threshold': 100,      # < 100 MB
            'medium_dataset_threshold': 1000,    # 100 MB - 1 GB
            'large_dataset_threshold': 10000,   # 1 GB - 10 GB
            'huge_dataset_threshold': 100000,   # > 10 GB
            
            # Scaling factors
            'executor_scaling_factor': 0.1,     # 10% of data size in GB
            'memory_scaling_factor': 2.0,       # 2x data size for memory
            'partition_scaling_factor': 0.5,    # 50% of data size in GB
            
            # Performance targets
            'target_processing_time_minutes': 30,
            'memory_efficiency_threshold': 0.8,  # 80% memory utilization target
            
            # Auto-adjustment settings
            'enable_auto_adjustment': True,
            'adjustment_interval_minutes': 5,
            'performance_history_window': 10,    # Number of runs to consider
            
            # Resource constraints
            'max_memory_gb': 64,
            'max_cores': psutil.cpu_count(),
            'available_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    def _capture_current_config(self) -> Dict[str, Any]:
        """Capture current Spark configuration for comparison."""
        try:
            return {
                'executor_instances': self.spark.conf.get('spark.executor.instances', '1'),
                'executor_memory': self.spark.conf.get('spark.executor.memory', '1g'),
                'driver_memory': self.spark.conf.get('spark.driver.memory', '1g'),
                'sql_shuffle_partitions': self.spark.conf.get('spark.sql.shuffle.partitions', '200'),
                'default_parallelism': self.spark.conf.get('spark.default.parallelism', '200')
            }
        except Exception as e:
            logger.warning(f"Could not capture current config: {e}")
            return {}
    
    def analyze_data_size(self, data: DataFrame) -> Dict[str, Any]:
        """
        Analyze data size and characteristics for scaling decisions.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with size analysis
        """
        try:
            logger.info("📊 Analyzing data size and characteristics...")
            
            # Get row count
            row_count = data.count()
            
            # Estimate data size in MB
            sample_size = min(1000, row_count)
            sample_data = data.limit(sample_size)
            
            # Get schema info
            schema_info = []
            total_estimated_size = 0
            
            for field in data.schema.fields:
                field_name = field.name
                field_type = field.dataType.typeName()
                
                # Estimate field size based on type
                if field_type in ['string', 'binary']:
                    # Sample string lengths
                    if field_type == 'string':
                        max_length = sample_data.select(
                            col(field_name).cast("string")
                        ).rdd.map(lambda x: len(str(x[0]))).max()
                        avg_length = sample_data.select(
                            col(field_name).cast("string")
                        ).rdd.map(lambda x: len(str(x[0]))).mean()
                        field_size = (avg_length + max_length) / 2
                    else:
                        field_size = 64  # Default binary size
                elif field_type in ['double', 'long', 'bigint']:
                    field_size = 8
                elif field_type in ['integer', 'int']:
                    field_size = 4
                elif field_type in ['boolean']:
                    field_size = 1
                else:
                    field_size = 16  # Default for other types
                
                schema_info.append({
                    'name': field_name,
                    'type': field_type,
                    'estimated_size_bytes': field_size
                })
                total_estimated_size += field_size
            
            # Calculate total estimated size
            total_size_mb = (total_estimated_size * row_count) / (1024 * 1024)
            
            # Determine dataset category
            if total_size_mb < self.config['small_dataset_threshold']:
                category = 'small'
            elif total_size_mb < self.config['medium_dataset_threshold']:
                category = 'medium'
            elif total_size_mb < self.config['large_dataset_threshold']:
                category = 'large'
            else:
                category = 'huge'
            
            analysis = {
                'row_count': row_count,
                'column_count': len(data.columns),
                'total_size_mb': total_size_mb,
                'category': category,
                'schema_info': schema_info,
                'estimated_memory_gb': total_size_mb / 1024,
                'complexity_score': self._calculate_complexity_score(data, schema_info)
            }
            
            logger.info(f"📊 Data analysis complete: {row_count:,} rows, {total_size_mb:.2f} MB, {category} dataset")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing data size: {e}")
            return {
                'row_count': 0,
                'column_count': 0,
                'total_size_mb': 0,
                'category': 'unknown',
                'error': str(e)
            }
    
    def _calculate_complexity_score(self, data: DataFrame, schema_info: List[Dict]) -> float:
        """Calculate complexity score based on data characteristics."""
        try:
            # Factors that increase complexity
            complexity_factors = {
                'string_columns': sum(1 for info in schema_info if info['type'] == 'string'),
                'high_cardinality': 0,
                'missing_values': 0,
                'data_types': len(set(info['type'] for info in schema_info))
            }
            
            # Check for high cardinality columns
            for info in schema_info:
                if info['type'] == 'string':
                    try:
                        distinct_count = data.select(info['name']).distinct().count()
                        if distinct_count > 1000:  # High cardinality threshold
                            complexity_factors['high_cardinality'] += 1
                    except:
                        pass
            
            # Calculate complexity score (0-1, higher = more complex)
            score = (
                complexity_factors['string_columns'] * 0.1 +
                complexity_factors['high_cardinality'] * 0.2 +
                complexity_factors['data_types'] * 0.1
            )
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Could not calculate complexity score: {e}")
            return 0.5
    
    def calculate_optimal_configuration(self, data_analysis: Dict[str, Any], 
                                     processing_type: str = 'general') -> Dict[str, Any]:
        """
        Calculate optimal Spark configuration based on data analysis.
        
        Args:
            data_analysis: Result from analyze_data_size()
            processing_type: Type of processing ('general', 'ml', 'etl', 'query')
            
        Returns:
            Dictionary with optimal configuration
        """
        try:
            logger.info(f"🎯 Calculating optimal configuration for {processing_type} processing...")
            
            # Base configuration
            optimal_config = {}
            
            # Calculate executor count based on data size
            data_size_gb = data_analysis['total_size_mb'] / 1024
            complexity_score = data_analysis.get('complexity_score', 0.5)
            
            # Base executor calculation
            base_executors = max(
                self.config['min_executors'],
                min(
                    self.config['max_executors'],
                    int(data_size_gb * self.config['executor_scaling_factor'] * (1 + complexity_score))
                )
            )
            
            # Adjust for processing type
            if processing_type == 'ml':
                base_executors = int(base_executors * 1.5)  # ML needs more resources
            elif processing_type == 'etl':
                base_executors = int(base_executors * 0.8)  # ETL can be more efficient
            elif processing_type == 'query':
                base_executors = int(base_executors * 0.6)  # Queries are usually lighter
            
            optimal_config['executor_instances'] = base_executors
            
            # Calculate memory requirements
            base_memory_gb = max(2, int(data_size_gb * self.config['memory_scaling_factor']))
            
            # Adjust memory based on complexity
            if complexity_score > 0.7:
                base_memory_gb = int(base_memory_gb * 1.3)  # Complex data needs more memory
            
            # Ensure memory is within bounds
            executor_memory_gb = min(
                base_memory_gb // base_executors,
                int(self.config['available_memory_gb'] * 0.8 // base_executors)
            )
            
            optimal_config['executor_memory'] = f"{max(1, executor_memory_gb)}g"
            
            # Calculate driver memory
            driver_memory_gb = max(
                4,
                min(
                    int(data_size_gb * 0.5),
                    int(self.config['available_memory_gb'] * 0.3)
                )
            )
            optimal_config['driver_memory'] = f"{driver_memory_gb}g"
            
            # Calculate partition count
            optimal_partitions = max(
                10,
                min(
                    int(data_size_gb * self.config['partition_scaling_factor'] * 100),
                    1000  # Cap at 1000 partitions
                )
            )
            optimal_config['sql_shuffle_partitions'] = optimal_partitions
            optimal_config['default_parallelism'] = optimal_partitions
            
            # Additional ML-specific optimizations
            if processing_type == 'ml':
                optimal_config.update({
                    'spark.sql.adaptive.enabled': 'true',
                    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
                    'spark.sql.adaptive.advisoryPartitionSizeInBytes': '128MB',
                    'spark.ml.optimization.vectorizedReader.enabled': 'true'
                })
            
            logger.info(f"✅ Optimal configuration calculated: {base_executors} executors, "
                       f"{executor_memory_gb}g per executor, {driver_memory_gb}g driver")
            
            return optimal_config
            
        except Exception as e:
            logger.error(f"❌ Error calculating optimal configuration: {e}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration if optimal calculation fails."""
        return {
            'executor_instances': 2,
            'executor_memory': '2g',
            'driver_memory': '4g',
            'sql_shuffle_partitions': 200,
            'default_parallelism': 200
        }
    
    def apply_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Apply the calculated configuration to the Spark session.
        
        Args:
            config: Configuration dictionary to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🔧 Applying Spark configuration...")
            
            applied_count = 0
            failed_count = 0
            
            # Filter out configurations that can't be modified at runtime
            runtime_configs = {}
            for key, value in config.items():
                if key in ['sql_shuffle_partitions', 'default_parallelism']:
                    runtime_configs[key] = value
                # Skip executor_instances, executor_memory, driver_memory as they can't be modified at runtime
            
            for key, value in runtime_configs.items():
                try:
                    if key == 'sql_shuffle_partitions':
                        self.spark.conf.set('spark.sql.shuffle.partitions', str(value))
                    elif key == 'default_parallelism':
                        self.spark.conf.set('spark.default.parallelism', str(value))
                    else:
                        # Handle other configuration keys
                        self.spark.conf.set(key, str(value))
                    
                    applied_count += 1
                    logger.debug(f"   ✅ {key}: {value}")
                    
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"   ⚠️ Could not set {key}: {e}")
            
            # Record the configuration change
            self.scaling_history.append({
                'timestamp': time.time(),
                'config': config,
                'applied_count': applied_count,
                'failed_count': failed_count,
                'runtime_applicable': len(runtime_configs),
                'total_configs': len(config)
            })
            
            logger.info(f"✅ Configuration applied: {applied_count} successful, {failed_count} failed")
            logger.info(f"   📊 Runtime configs: {len(runtime_configs)}/{len(config)} can be applied")
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"❌ Error applying configuration: {e}")
            return False
    
    def auto_scale_for_data(self, data: DataFrame, processing_type: str = 'general') -> Dict[str, Any]:
        """
        Automatically scale the Spark cluster for the given data.
        
        Args:
            data: DataFrame to process
            processing_type: Type of processing to perform
            
        Returns:
            Dictionary with scaling results
        """
        try:
            logger.info(f"🚀 Auto-scaling Spark cluster for {processing_type} processing...")
            
            # Analyze data
            data_analysis = self.analyze_data_size(data)
            
            # Calculate optimal configuration
            optimal_config = self.calculate_optimal_configuration(data_analysis, processing_type)
            
            # Apply configuration
            success = self.apply_configuration(optimal_config)
            
            # Return results
            results = {
                'success': success,
                'data_analysis': data_analysis,
                'applied_config': optimal_config,
                'scaling_recommendations': self._generate_scaling_recommendations(data_analysis, optimal_config),
                'note': 'Some configurations (executor_instances, executor_memory, driver_memory) cannot be modified at runtime and require session restart'
            }
            
            if success:
                logger.info("✅ Auto-scaling completed successfully")
                logger.info("💡 Note: Some configurations require session restart to take effect")
            else:
                logger.warning("⚠️ Auto-scaling completed with some failures")
                logger.info("💡 Note: Some configurations require session restart to take effect")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Auto-scaling failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_config': self._get_fallback_config()
            }
    
    def _generate_scaling_recommendations(self, data_analysis: Dict[str, Any], 
                                       config: Dict[str, Any]) -> List[str]:
        """Generate scaling recommendations based on data analysis."""
        recommendations = []
        
        data_size_gb = data_analysis['total_size_mb'] / 1024
        category = data_analysis['category']
        
        if category == 'huge':
            recommendations.append("Consider using Spark on Kubernetes for better resource management")
            recommendations.append("Monitor memory usage closely - may need to increase driver memory")
            recommendations.append("Consider data partitioning strategies for better performance")
        
        elif category == 'large':
            recommendations.append("Current configuration should handle this dataset efficiently")
            recommendations.append("Monitor executor memory usage during processing")
        
        elif category == 'medium':
            recommendations.append("Dataset size is optimal for current cluster configuration")
            recommendations.append("Consider reducing executor count for cost optimization")
        
        elif category == 'small':
            recommendations.append("Dataset is small - consider reducing resources for cost savings")
            recommendations.append("Single executor may be sufficient")
        
        # Complexity-based recommendations
        complexity_score = data_analysis.get('complexity_score', 0.5)
        if complexity_score > 0.7:
            recommendations.append("High complexity detected - consider increasing memory allocation")
            recommendations.append("Monitor processing times - may need more executors")
        
        return recommendations
    
    def monitor_performance(self, start_time: float, data_size_mb: float) -> Dict[str, Any]:
        """
        Monitor performance metrics and suggest adjustments.
        
        Args:
            start_time: Start time of processing
            data_size_mb: Size of data being processed
            
        Returns:
            Dictionary with performance metrics and suggestions
        """
        try:
            end_time = time.time()
            processing_time_minutes = (end_time - start_time) / 60
            
            # Record performance metrics
            self.performance_metrics['processing_times'].append(processing_time_minutes)
            self.performance_metrics['data_sizes'].append(data_size_mb)
            
            # Keep only recent history
            max_history = self.config['performance_history_window']
            if len(self.performance_metrics['processing_times']) > max_history:
                self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-max_history:]
                self.performance_metrics['data_sizes'] = self.performance_metrics['data_sizes'][-max_history:]
            
            # Calculate performance statistics
            avg_processing_time = sum(self.performance_metrics['processing_times']) / len(self.performance_metrics['processing_times'])
            avg_data_size = sum(self.performance_metrics['data_sizes']) / len(self.performance_metrics['data_sizes'])
            
            # Performance analysis
            performance_analysis = {
                'current_processing_time': processing_time_minutes,
                'average_processing_time': avg_processing_time,
                'data_size_mb': data_size_mb,
                'average_data_size': avg_data_size,
                'performance_trend': self._analyze_performance_trend(),
                'recommendations': []
            }
            
            # Generate recommendations
            target_time = self.config['target_processing_time_minutes']
            if processing_time_minutes > target_time * 1.5:
                performance_analysis['recommendations'].append(
                    f"Processing time ({processing_time_minutes:.1f} min) exceeds target ({target_time} min). "
                    "Consider increasing executor count or memory."
                )
            
            if processing_time_minutes < target_time * 0.5:
                performance_analysis['recommendations'].append(
                    f"Processing time ({processing_time_minutes:.1f} min) is well below target ({target_time} min). "
                    "Consider reducing resources for cost optimization."
                )
            
            logger.info(f"📊 Performance monitoring: {processing_time_minutes:.1f} min for {data_size_mb:.1f} MB")
            return performance_analysis
            
        except Exception as e:
            logger.error(f"❌ Error monitoring performance: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend over recent runs."""
        if len(self.performance_metrics['processing_times']) < 3:
            return "insufficient_data"
        
        recent_times = self.performance_metrics['processing_times'][-3:]
        if recent_times[-1] < recent_times[0]:
            return "improving"
        elif recent_times[-1] > recent_times[0]:
            return "degrading"
        else:
            return "stable"
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of all scaling operations."""
        return {
            'total_scaling_operations': len(self.scaling_history),
            'recent_configurations': self.scaling_history[-5:] if self.scaling_history else [],
            'performance_metrics': self.performance_metrics,
            'current_config': self._capture_current_config(),
            'initial_config': self.initial_config
        }
    
    def reset_to_initial_config(self) -> bool:
        """Reset Spark configuration to initial values."""
        try:
            logger.info("🔄 Resetting to initial Spark configuration...")
            return self.apply_configuration(self.initial_config)
        except Exception as e:
            logger.error(f"❌ Error resetting configuration: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources and reset configuration."""
        try:
            logger.info("🧹 Cleaning up autoscaling manager...")
            
            # Reset to initial configuration
            self.reset_to_initial_config()
            
            # Clear performance metrics
            self.performance_metrics = {
                'processing_times': [],
                'memory_usage': [],
                'executor_counts': [],
                'data_sizes': []
            }
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def create_autoscaling_manager(spark_session: SparkSession, 
                             config: Optional[Dict[str, Any]] = None) -> SparkAutoscalingManager:
    """
    Factory function to create an autoscaling manager.
    
    Args:
        spark_session: Active Spark session
        config: Optional configuration overrides
        
    Returns:
        Configured SparkAutoscalingManager instance
    """
    return SparkAutoscalingManager(spark_session, config)


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Spark Autoscaling Manager")
    print("=" * 50)
    
    # This would typically be used within a Spark application
    print("💡 Import and use within your Spark application:")
    print("   from spark_autoscaling_manager import create_autoscaling_manager")
    print("   manager = create_autoscaling_manager(spark)")
    print("   results = manager.auto_scale_for_data(your_dataframe, 'ml')")
