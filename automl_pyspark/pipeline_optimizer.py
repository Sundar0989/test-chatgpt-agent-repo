"""
Pipeline Optimizer

This module integrates Spark autoscaling and temporary table management to optimize
the entire AutoML pipeline. It provides a unified interface for managing performance
and resources throughout the pipeline execution.

Features:
- Integrated autoscaling and temp table management
- Pipeline stage optimization
- Performance monitoring and adjustment
- Automatic resource management
- Pipeline execution optimization
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, count

# Import our optimization modules
try:
    from .spark_autoscaling_manager import SparkAutoscalingManager, create_autoscaling_manager
    from .temp_table_manager import TempTableManager, create_temp_table_manager
except ImportError:
    # Fallback for direct execution
    from spark_autoscaling_manager import SparkAutoscalingManager, create_autoscaling_manager
    from temp_table_manager import TempTableManager, create_temp_table_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOptimizer:
    """
    Integrates autoscaling and temporary table management for optimal pipeline performance.
    """
    
    def __init__(self, spark_session: SparkSession, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline optimizer.
        
        Args:
            spark_session: Active Spark session
            config: Configuration dictionary for optimization behavior
        """
        self.spark = spark_session
        self.config = config or self._get_default_config()
        
        # Initialize optimization managers
        self.autoscaling_manager = create_autoscaling_manager(spark_session, self.config.get('autoscaling', {}))
        self.temp_table_manager = create_temp_table_manager(spark_session, self.config.get('temp_tables', {}))
        
        # Pipeline tracking
        self.pipeline_stages = []
        self.stage_performance = {}
        self.optimization_history = []
        
        # Performance monitoring
        self.start_time = time.time()
        self.total_optimizations = 0
        
        logger.info("🚀 Pipeline Optimizer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for pipeline optimization."""
        return {
            # General optimization settings
            'enable_autoscaling': True,
            'enable_temp_tables': True,
            'auto_optimize': True,
            'performance_threshold': 0.8,  # 80% performance target
            
            # Autoscaling configuration
            'autoscaling': {
                'min_executors': 2,
                'max_executors': 20,
                'target_processing_time_minutes': 30,
                'enable_auto_adjustment': True
            },
            
            # Temporary table configuration
            'temp_tables': {
                'enable_caching': True,
                'auto_cleanup': True,
                'max_table_age_hours': 12,
                'optimize_storage': True
            },
            
            # Pipeline optimization
            'pipeline': {
                'stage_optimization': True,
                'memory_management': True,
                'performance_monitoring': True,
                'auto_cleanup': True
            },
            
            # Monitoring and reporting
            'monitoring': {
                'track_performance': True,
                'performance_history_size': 50,
                'enable_metrics_collection': True,
                'report_interval_minutes': 10
            }
        }
    
    def optimize_pipeline_stage(self, stage_name: str, 
                              data: DataFrame,
                              stage_type: str = 'general',
                              optimization_target: str = 'balanced') -> Dict[str, Any]:
        """
        Optimize a specific pipeline stage with autoscaling and temp table management.
        
        Args:
            stage_name: Name of the pipeline stage
            data: Data to process in this stage
            stage_type: Type of stage ('data_loading', 'preprocessing', 'feature_selection', 
                       'model_training', 'evaluation', 'prediction')
            optimization_target: Optimization focus ('speed', 'memory', 'balanced', 'quality')
            
        Returns:
            Dictionary with optimization results and optimized data reference
        """
        try:
            stage_start_time = time.time()
            logger.info(f"🎯 Optimizing pipeline stage: {stage_name} ({stage_type})")
            
            # Stage 1: Autoscaling optimization
            scaling_results = self._optimize_scaling_for_stage(data, stage_type, optimization_target)
            
            # Stage 2: Create optimized temporary table
            temp_table_results = self._optimize_temp_table_for_stage(
                data, stage_name, stage_type, optimization_target
            )
            
            # Stage 3: Apply stage-specific optimizations
            stage_optimizations = self._apply_stage_specific_optimizations(
                stage_name, stage_type, optimization_target
            )
            
            # Record stage performance
            stage_time = time.time() - stage_start_time
            self._record_stage_performance(stage_name, stage_type, stage_time, scaling_results, temp_table_results)
            
            # Return optimization results
            results = {
                'stage_name': stage_name,
                'stage_type': stage_type,
                'optimization_target': optimization_target,
                'execution_time': stage_time,
                'scaling_results': scaling_results,
                'temp_table_results': temp_table_results,
                'stage_optimizations': stage_optimizations,
                'optimized_data_reference': temp_table_results.get('table_name'),
                'performance_metrics': self._calculate_stage_metrics(stage_name)
            }
            
            # Add to pipeline tracking
            self.pipeline_stages.append({
                'name': stage_name,
                'type': stage_type,
                'timestamp': time.time(),
                'results': results
            })
            
            logger.info(f"✅ Stage {stage_name} optimized in {stage_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error optimizing stage {stage_name}: {e}")
            return {
                'stage_name': stage_name,
                'error': str(e),
                'success': False
            }
    
    def _optimize_scaling_for_stage(self, data: DataFrame, stage_type: str, 
                                  optimization_target: str) -> Dict[str, Any]:
        """Optimize Spark scaling for the current stage."""
        try:
            if not self.config['enable_autoscaling']:
                return {'enabled': False, 'message': 'Autoscaling disabled'}
            
            # Determine processing type based on stage type
            processing_type_map = {
                'data_loading': 'etl',
                'preprocessing': 'etl',
                'feature_selection': 'ml',
                'model_training': 'ml',
                'evaluation': 'query',
                'prediction': 'ml'
            }
            
            processing_type = processing_type_map.get(stage_type, 'general')
            
            # Auto-scale for the data
            scaling_results = self.autoscaling_manager.auto_scale_for_data(data, processing_type)
            
            # Apply additional stage-specific scaling if needed
            if optimization_target == 'speed':
                scaling_results['additional_optimizations'] = self._apply_speed_optimizations()
            elif optimization_target == 'memory':
                scaling_results['additional_optimizations'] = self._apply_memory_optimizations()
            
            return scaling_results
            
        except Exception as e:
            logger.warning(f"Scaling optimization failed: {e}")
            return {'error': str(e), 'enabled': False}
    
    def _optimize_temp_table_for_stage(self, data: DataFrame, stage_name: str, 
                                     stage_type: str, optimization_target: str) -> Dict[str, Any]:
        """Optimize temporary table creation for the current stage."""
        try:
            if not self.config['enable_temp_tables']:
                return {'enabled': False, 'message': 'Temp tables disabled'}
            
            # Determine optimization strategy based on stage and target
            if optimization_target == 'speed':
                optimize_for = 'query'
            elif optimization_target == 'memory':
                optimize_for = 'storage'
            elif optimization_target == 'quality':
                optimize_for = 'ml'
            else:
                optimize_for = 'balanced'
            
            # Create optimized temporary table
            table_name = self.temp_table_manager.create_temp_table(
                data=data,
                table_name=stage_name,
                stage=stage_type,
                optimize_for=optimize_for
            )
            
            # Apply additional table optimizations
            table_optimizations = self._apply_table_optimizations(table_name, stage_type, optimization_target)
            
            return {
                'table_name': table_name,
                'optimize_for': optimize_for,
                'stage': stage_type,
                'table_optimizations': table_optimizations,
                'success': True
            }
            
        except Exception as e:
            logger.warning(f"Temp table optimization failed: {e}")
            return {'error': str(e), 'enabled': False}
    
    def _apply_stage_specific_optimizations(self, stage_name: str, stage_type: str, 
                                          optimization_target: str) -> Dict[str, Any]:
        """Apply stage-specific optimizations."""
        try:
            optimizations = {}
            
            if stage_type == 'data_loading':
                optimizations.update(self._optimize_data_loading())
            elif stage_type == 'preprocessing':
                optimizations.update(self._optimize_preprocessing())
            elif stage_type == 'feature_selection':
                optimizations.update(self._optimize_feature_selection())
            elif stage_type == 'model_training':
                optimizations.update(self._optimize_model_training())
            elif stage_type == 'evaluation':
                optimizations.update(self._optimize_evaluation())
            elif stage_type == 'prediction':
                optimizations.update(self._optimize_prediction())
            
            # Apply target-specific optimizations
            if optimization_target == 'speed':
                optimizations.update(self._apply_speed_optimizations())
            elif optimization_target == 'memory':
                optimizations.update(self._apply_memory_optimizations())
            elif optimization_target == 'quality':
                optimizations.update(self._apply_quality_optimizations())
            
            return optimizations
            
        except Exception as e:
            logger.warning(f"Stage-specific optimizations failed: {e}")
            return {'error': str(e)}
    
    def _optimize_data_loading(self) -> Dict[str, Any]:
        """Optimize data loading stage."""
        return {
            'partition_strategy': 'adaptive',
            'caching_strategy': 'aggressive',
            'broadcast_threshold': '5m'
        }
    
    def _optimize_preprocessing(self) -> Dict[str, Any]:
        """Optimize preprocessing stage."""
        return {
            'memory_management': 'efficient',
            'partition_optimization': 'balanced',
            'caching_strategy': 'moderate'
        }
    
    def _optimize_feature_selection(self) -> Dict[str, Any]:
        """Optimize feature selection stage."""
        return {
            'ml_optimizations': 'enabled',
            'memory_allocation': 'generous',
            'parallel_processing': 'maximized'
        }
    
    def _optimize_model_training(self) -> Dict[str, Any]:
        """Optimize model training stage."""
        return {
            'ml_optimizations': 'enabled',
            'memory_allocation': 'generous',
            'executor_optimization': 'ml_focused'
        }
    
    def _optimize_evaluation(self) -> Dict[str, Any]:
        """Optimize evaluation stage."""
        return {
            'query_optimization': 'enabled',
            'caching_strategy': 'aggressive',
            'partition_strategy': 'query_focused'
        }
    
    def _optimize_prediction(self) -> Dict[str, Any]:
        """Optimize prediction stage."""
        return {
            'ml_optimizations': 'enabled',
            'memory_allocation': 'balanced',
            'caching_strategy': 'moderate'
        }
    
    def _apply_speed_optimizations(self) -> Dict[str, Any]:
        """Apply speed-focused optimizations."""
        return {
            'executor_count': 'increased',
            'memory_allocation': 'generous',
            'caching_strategy': 'aggressive',
            'partition_strategy': 'speed_optimized'
        }
    
    def _apply_memory_optimizations(self) -> Dict[str, Any]:
        """Apply memory-focused optimizations."""
        return {
            'executor_count': 'reduced',
            'memory_allocation': 'conservative',
            'caching_strategy': 'minimal',
            'partition_strategy': 'memory_efficient'
        }
    
    def _apply_quality_optimizations(self) -> Dict[str, Any]:
        """Apply quality-focused optimizations."""
        return {
            'executor_count': 'balanced',
            'memory_allocation': 'adequate',
            'caching_strategy': 'moderate',
            'partition_strategy': 'quality_focused'
        }
    
    def _apply_table_optimizations(self, table_name: str, stage_type: str, 
                                 optimization_target: str) -> Dict[str, Any]:
        """Apply additional table optimizations."""
        try:
            optimizations = {}
            
            # Stage-specific table optimizations
            if stage_type == 'model_training':
                # Optimize for ML training
                self.spark.sql(f"ANALYZE TABLE {table_name} COMPUTE STATISTICS")
                optimizations['ml_optimizations'] = 'enabled'
            
            elif stage_type == 'evaluation':
                # Optimize for queries
                self.spark.sql(f"ANALYZE TABLE {table_name} COMPUTE STATISTICS FOR ALL COLUMNS")
                optimizations['query_optimizations'] = 'enabled'
            
            # Target-specific optimizations
            if optimization_target == 'speed':
                optimizations['caching'] = 'enabled'
                optimizations['partition_optimization'] = 'speed_focused'
            
            elif optimization_target == 'memory':
                optimizations['storage_optimization'] = 'enabled'
                optimizations['compression'] = 'enabled'
            
            return optimizations
            
        except Exception as e:
            logger.warning(f"Table optimizations failed: {e}")
            return {'error': str(e)}
    
    def _record_stage_performance(self, stage_name: str, stage_type: str, 
                                execution_time: float, scaling_results: Dict, 
                                temp_table_results: Dict):
        """Record performance metrics for the stage."""
        try:
            self.stage_performance[stage_name] = {
                'type': stage_type,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'scaling_results': scaling_results,
                'temp_table_results': temp_table_results,
                'success': True
            }
            
            # Keep only recent history
            max_history = self.config['monitoring']['performance_history_size']
            if len(self.stage_performance) > max_history:
                # Remove oldest entries
                oldest_stages = sorted(self.stage_performance.keys(), 
                                     key=lambda x: self.stage_performance[x]['timestamp'])[:-max_history]
                for stage in oldest_stages:
                    del self.stage_performance[stage]
            
        except Exception as e:
            logger.warning(f"Could not record stage performance: {e}")
    
    def _calculate_stage_metrics(self, stage_name: str) -> Dict[str, Any]:
        """Calculate performance metrics for a stage."""
        try:
            if stage_name not in self.stage_performance:
                return {}
            
            stage_data = self.stage_performance[stage_name]
            
            # Calculate basic metrics
            metrics = {
                'execution_time': stage_data['execution_time'],
                'success': stage_data['success'],
                'timestamp': stage_data['timestamp']
            }
            
            # Add scaling metrics if available
            if 'scaling_results' in stage_data:
                scaling = stage_data['scaling_results']
                if 'data_analysis' in scaling:
                    metrics['data_size_mb'] = scaling['data_analysis'].get('total_size_mb', 0)
                    metrics['data_category'] = scaling['data_analysis'].get('category', 'unknown')
            
            # Add temp table metrics if available
            if 'temp_table_results' in stage_data:
                temp_table = stage_data['temp_table_results']
                if 'table_name' in temp_table:
                    table_info = self.temp_table_manager.get_table_info(temp_table['table_name'])
                    if 'error' not in table_info:
                        metrics['table_size_mb'] = table_info.get('estimated_size_mb', 0)
                        metrics['table_rows'] = table_info.get('row_count', 0)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not calculate stage metrics: {e}")
            return {}
    
    def execute_optimized_pipeline(self, pipeline_stages: List[Dict[str, Any]], 
                                 optimization_target: str = 'balanced') -> Dict[str, Any]:
        """
        Execute a complete pipeline with optimization at each stage.
        
        Args:
            pipeline_stages: List of pipeline stage definitions
            optimization_target: Overall optimization target
            
        Returns:
            Dictionary with pipeline execution results
        """
        try:
            pipeline_start_time = time.time()
            logger.info(f"🚀 Executing optimized pipeline with {len(pipeline_stages)} stages")
            
            pipeline_results = {
                'stages': [],
                'total_execution_time': 0,
                'optimization_summary': {},
                'performance_summary': {},
                'success': True
            }
            
            # Execute each stage with optimization
            for i, stage_def in enumerate(pipeline_stages):
                try:
                    logger.info(f"📋 Executing stage {i+1}/{len(pipeline_stages)}: {stage_def['name']}")
                    
                    # Get data for this stage
                    data = stage_def.get('data')
                    if data is None:
                        logger.warning(f"No data provided for stage {stage_def['name']}")
                        continue
                    
                    # Optimize the stage
                    stage_results = self.optimize_pipeline_stage(
                        stage_name=stage_def['name'],
                        data=data,
                        stage_type=stage_def.get('type', 'general'),
                        optimization_target=optimization_target
                    )
                    
                    # Add to pipeline results
                    pipeline_results['stages'].append(stage_results)
                    
                    # Check for errors
                    if 'error' in stage_results:
                        pipeline_results['success'] = False
                        logger.error(f"Stage {stage_def['name']} failed: {stage_results['error']}")
                        break
                    
                    # Update data reference for next stage if available
                    if 'optimized_data_reference' in stage_results:
                        stage_def['data'] = self.temp_table_manager.query_temp_table(
                            stage_results['optimized_data_reference'],
                            f"SELECT * FROM {stage_results['optimized_data_reference']}"
                        )
                    
                except Exception as e:
                    error_msg = f"Stage {stage_def['name']} failed: {e}"
                    logger.error(error_msg)
                    pipeline_results['success'] = False
                    pipeline_results['stages'].append({
                        'stage_name': stage_def['name'],
                        'error': error_msg,
                        'success': False
                    })
                    break
            
            # Calculate pipeline summary
            pipeline_results['total_execution_time'] = time.time() - pipeline_start_time
            pipeline_results['optimization_summary'] = self._generate_optimization_summary()
            pipeline_results['performance_summary'] = self._generate_performance_summary()
            
            # Record pipeline execution
            self.optimization_history.append({
                'timestamp': time.time(),
                'pipeline_results': pipeline_results,
                'optimization_target': optimization_target
            })
            
            logger.info(f"✅ Pipeline completed in {pipeline_results['total_execution_time']:.2f}s")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stages': [],
                'total_execution_time': 0
            }
    
    def _generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate summary of all optimizations applied."""
        try:
            summary = {
                'total_stages': len(self.pipeline_stages),
                'successful_stages': sum(1 for stage in self.pipeline_stages if stage.get('success', False)),
                'autoscaling_summary': self.autoscaling_manager.get_scaling_summary(),
                'temp_table_summary': self.temp_table_manager.get_performance_summary(),
                'total_optimizations': self.total_optimizations
            }
            
            return summary
            
        except Exception as e:
            logger.warning(f"Could not generate optimization summary: {e}")
            return {'error': str(e)}
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary for the pipeline."""
        try:
            if not self.stage_performance:
                return {}
            
            # Calculate performance statistics
            execution_times = [stage['execution_time'] for stage in self.stage_performance.values()]
            total_time = sum(execution_times)
            avg_time = total_time / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            # Performance analysis
            performance_summary = {
                'total_execution_time': total_time,
                'average_stage_time': avg_time,
                'fastest_stage_time': min_time,
                'slowest_stage_time': max_time,
                'total_stages': len(self.stage_performance),
                'performance_trend': self._analyze_performance_trend(),
                'bottlenecks': self._identify_bottlenecks()
            }
            
            return performance_summary
            
        except Exception as e:
            logger.warning(f"Could not generate performance summary: {e}")
            return {'error': str(e)}
    
    def _analyze_performance_trend(self) -> str:
        """Analyze overall performance trend."""
        try:
            if len(self.stage_performance) < 3:
                return "insufficient_data"
            
            # Get recent stages
            recent_stages = sorted(self.stage_performance.items(), 
                                 key=lambda x: x[1]['timestamp'])[-3:]
            
            recent_times = [stage[1]['execution_time'] for stage in recent_stages]
            
            if recent_times[-1] < recent_times[0]:
                return "improving"
            elif recent_times[-1] > recent_times[0]:
                return "degrading"
            else:
                return "stable"
                
        except Exception as e:
            logger.warning(f"Could not analyze performance trend: {e}")
            return "unknown"
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks in the pipeline."""
        try:
            bottlenecks = []
            
            if not self.stage_performance:
                return bottlenecks
            
            # Find slowest stages
            slow_stages = sorted(self.stage_performance.items(), 
                               key=lambda x: x[1]['execution_time'], reverse=True)[:3]
            
            avg_time = sum(stage[1]['execution_time'] for stage in self.stage_performance.values()) / len(self.stage_performance)
            
            for stage_name, stage_data in slow_stages:
                if stage_data['execution_time'] > avg_time * 2:  # 2x slower than average
                    bottlenecks.append(f"{stage_name}: {stage_data['execution_time']:.2f}s (2x slower than average)")
            
            return bottlenecks
            
        except Exception as e:
            logger.warning(f"Could not identify bottlenecks: {e}")
            return []
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        try:
            report = {
                'pipeline_summary': {
                    'total_stages_executed': len(self.pipeline_stages),
                    'total_optimizations': self.total_optimizations,
                    'pipeline_start_time': self.start_time,
                    'current_time': time.time(),
                    'uptime_minutes': (time.time() - self.start_time) / 60
                },
                'autoscaling_report': self.autoscaling_manager.get_scaling_summary(),
                'temp_table_report': self.temp_table_manager.get_performance_summary(),
                'performance_report': self._generate_performance_summary(),
                'optimization_history': self.optimization_history[-10:] if self.optimization_history else [],
                'recommendations': self._generate_optimization_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating optimization report: {e}")
            return {'error': str(e)}
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current performance."""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if self.stage_performance:
                avg_time = sum(stage['execution_time'] for stage in self.stage_performance.values()) / len(self.stage_performance)
                
                if avg_time > 60:  # Average stage takes more than 1 minute
                    recommendations.append("Consider increasing executor count for faster processing")
                    recommendations.append("Review data partitioning strategy for better parallelism")
                
                if avg_time < 5:  # Average stage takes less than 5 seconds
                    recommendations.append("Consider reducing resources for cost optimization")
                    recommendations.append("Current configuration may be over-provisioned")
            
            # Autoscaling recommendations
            scaling_summary = self.autoscaling_manager.get_scaling_summary()
            if scaling_summary.get('total_scaling_operations', 0) > 10:
                recommendations.append("High number of scaling operations - consider pre-allocating resources")
            
            # Temp table recommendations
            temp_table_summary = self.temp_table_manager.get_performance_summary()
            if temp_table_summary.get('storage_efficiency', 0) < 0.5:
                recommendations.append("Low storage efficiency - consider enabling more aggressive caching")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Could not generate recommendations: {e}")
            return []
    
    def cleanup(self):
        """Clean up all resources and reset optimizations."""
        try:
            logger.info("🧹 Cleaning up pipeline optimizer...")
            
            # Clean up managers
            if hasattr(self, 'autoscaling_manager'):
                self.autoscaling_manager.cleanup()
            
            if hasattr(self, 'temp_table_manager'):
                self.temp_table_manager.cleanup()
            
            # Clear tracking
            self.pipeline_stages.clear()
            self.stage_performance.clear()
            self.optimization_history.clear()
            
            logger.info("✅ Pipeline optimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def create_pipeline_optimizer(spark_session: SparkSession, 
                            config: Optional[Dict[str, Any]] = None) -> PipelineOptimizer:
    """
    Factory function to create a pipeline optimizer.
    
    Args:
        spark_session: Active Spark session
        config: Optional configuration overrides
        
    Returns:
        Configured PipelineOptimizer instance
    """
    return PipelineOptimizer(spark_session, config)


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Pipeline Optimizer")
    print("=" * 50)
    
    # This would typically be used within a Spark application
    print("💡 Import and use within your Spark application:")
    print("   from pipeline_optimizer import create_pipeline_optimizer")
    print("   optimizer = create_pipeline_optimizer(spark)")
    print("   results = optimizer.execute_optimized_pipeline(pipeline_stages, 'balanced')")
    print("   report = optimizer.get_optimization_report()")
