"""
Temporary Table Manager

This module provides efficient temporary table management for the AutoML pipeline.
Instead of working with original data repeatedly, it creates optimized temporary tables
at each stage of the pipeline for better performance and memory management.

Features:
- Automatic temporary table creation at pipeline stages
- Memory-efficient data storage
- Automatic cleanup and garbage collection
- Performance monitoring and optimization
- Support for different storage strategies
"""

import os
import time
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, count
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TempTableManager:
    """
    Manages temporary tables throughout the AutoML pipeline for optimal performance.
    """
    
    def __init__(self, spark_session: SparkSession, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the temporary table manager.
        
        Args:
            spark_session: Active Spark session
            config: Configuration dictionary for temp table behavior
        """
        self.spark = spark_session
        
        # Get default config and merge with provided config
        default_config = self._get_default_config()
        if config:
            # Deep merge configurations
            self.config = self._deep_merge_configs(default_config, config)
        else:
            self.config = default_config
            
        self.temp_tables = {}
        self.table_metadata = {}
        self.performance_metrics = {}
        
        # Generate unique session ID for temp table naming
        self.session_id = str(uuid.uuid4())[:8]
        
        logger.info(f"🗄️ Temp Table Manager initialized with session ID: {self.session_id}")
    
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
        """Get default configuration for temporary table management."""
        return {
            # Table naming and organization
            'table_prefix': 'temp_automl',
            'use_session_id': True,
            'auto_cleanup': True,
            'cleanup_interval_minutes': 30,
            
            # Storage optimization
            'enable_caching': True,
            'cache_strategy': 'auto',  # 'auto', 'always', 'never'
            'partition_strategy': 'auto',  # 'auto', 'hash', 'range', 'none'
            'compression': 'snappy',
            
            # Performance settings
            'batch_size': 10000,
            'max_memory_fraction': 0.8,
            'enable_broadcast_join': True,
            'broadcast_threshold_mb': 10,
            
            # Monitoring
            'track_performance': True,
            'performance_history_size': 20,
            'enable_metrics_collection': True,
            
            # Cleanup policies
            'max_table_age_hours': 24,
            'max_total_tables': 50,
            'cleanup_batch_size': 10
        }
    
    def create_temp_table(self, data: DataFrame, 
                         table_name: str,
                         stage: str = 'general',
                         optimize_for: str = 'query',
                         force_recreate: bool = False) -> str:
        """
        Create a temporary table from DataFrame with optimization.
        
        Args:
            data: DataFrame to convert to temp table
            table_name: Name for the temporary table
            stage: Pipeline stage ('raw', 'cleaned', 'feature_selected', 'scaled', 'split', 'final')
            optimize_for: Optimization target ('query', 'ml', 'etl', 'storage')
            force_recreate: Force recreation even if table exists
            
        Returns:
            Full temporary table name
        """
        try:
            # Generate full table name
            full_table_name = self._generate_table_name(table_name, stage)
            
            # Check if table already exists
            if not force_recreate and self._table_exists(full_table_name):
                logger.info(f"📋 Temp table {full_table_name} already exists, skipping creation")
                return full_table_name
            
            logger.info(f"🗄️ Creating temp table: {full_table_name} for stage: {stage}")
            
            # Optimize DataFrame before creating table
            optimized_data = self._optimize_dataframe(data, optimize_for, stage)
            
            # Create temporary table
            start_time = time.time()
            
            # Use createOrReplaceTempView for better performance
            optimized_data.createOrReplaceTempView(full_table_name)
            
            creation_time = time.time() - start_time
            
            # Register table metadata
            self._register_table(full_table_name, data, stage, optimize_for, creation_time)
            
            # Apply optimizations
            self._apply_table_optimizations(full_table_name, optimize_for)
            
            logger.info(f"✅ Temp table {full_table_name} created in {creation_time:.2f}s")
            return full_table_name
            
        except Exception as e:
            logger.error(f"❌ Error creating temp table {table_name}: {e}")
            raise
    
    def _generate_table_name(self, base_name: str, stage: str) -> str:
        """Generate unique temporary table name."""
        prefix = self.config['table_prefix']
        session_part = f"_{self.session_id}" if self.config['use_session_id'] else ""
        stage_part = f"_{stage}" if stage != 'general' else ""
        
        # Clean base name (remove special characters)
        clean_base = "".join(c for c in base_name if c.isalnum() or c == '_')
        
        return f"{prefix}{session_part}_{clean_base}{stage_part}"
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if temporary table exists."""
        try:
            # Try to query the table
            test_df = self.spark.sql(f"SELECT 1 FROM {table_name} LIMIT 1")
            test_df.count()
            return True
        except Exception:
            return False
    
    def _optimize_dataframe(self, data: DataFrame, optimize_for: str, stage: str) -> DataFrame:
        """Optimize DataFrame based on intended use."""
        try:
            optimized_data = data
            
            # Apply stage-specific optimizations
            if stage == 'raw':
                # Raw data - minimal optimization
                pass
            elif stage == 'cleaned':
                # Cleaned data - basic optimization
                optimized_data = self._apply_basic_optimizations(data)
            elif stage == 'feature_selected':
                # Feature selected data - ML optimization
                optimized_data = self._apply_ml_optimizations(data)
            elif stage == 'scaled':
                # Scaled data - query optimization
                optimized_data = self._apply_query_optimizations(data)
            elif stage == 'split':
                # Split data - storage optimization
                optimized_data = self._apply_storage_optimizations(data)
            else:
                # Default optimization
                optimized_data = self._apply_basic_optimizations(data)
            
            # Apply use-specific optimizations
            if optimize_for == 'ml':
                optimized_data = self._apply_ml_optimizations(optimized_data)
            elif optimize_for == 'query':
                optimized_data = self._apply_query_optimizations(optimized_data)
            elif optimize_for == 'etl':
                optimized_data = self._apply_etl_optimizations(optimized_data)
            elif optimize_for == 'storage':
                optimized_data = self._apply_storage_optimizations(optimized_data)
            
            return optimized_data
            
        except Exception as e:
            logger.warning(f"Could not optimize DataFrame: {e}")
            return data
    
    def _apply_basic_optimizations(self, data: DataFrame) -> DataFrame:
        """Apply basic DataFrame optimizations."""
        try:
            # Repartition if needed
            num_partitions = data.rdd.getNumPartitions()
            if num_partitions > 100:
                target_partitions = max(10, int(num_partitions * 0.5))
                data = data.repartition(target_partitions)
            
            # Cache if enabled
            if self.config['enable_caching']:
                data = data.cache()
            
            return data
        except Exception as e:
            logger.warning(f"Basic optimization failed: {e}")
            return data
    
    def _apply_ml_optimizations(self, data: DataFrame) -> DataFrame:
        """Apply ML-specific optimizations."""
        try:
            # Ensure data is cached for ML operations
            if not data.is_cached:
                data = data.cache()
            
            # Optimize partitions for ML
            num_partitions = data.rdd.getNumPartitions()
            if num_partitions < 10:
                data = data.repartition(10)
            elif num_partitions > 200:
                data = data.repartition(200)
            
            return data
        except Exception as e:
            logger.warning(f"ML optimization failed: {e}")
            return data
    
    def _apply_query_optimizations(self, data: DataFrame) -> DataFrame:
        """Apply query-specific optimizations."""
        try:
            # Optimize for fast queries
            data = data.cache()
            
            # Reduce partitions for faster queries
            num_partitions = data.rdd.getNumPartitions()
            if num_partitions > 50:
                data = data.coalesce(50)
            
            return data
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return data
    
    def _apply_etl_optimizations(self, data: DataFrame) -> DataFrame:
        """Apply ETL-specific optimizations."""
        try:
            # Optimize for ETL operations
            num_partitions = data.rdd.getNumPartitions()
            
            # ETL benefits from more partitions for parallelism
            if num_partitions < 20:
                data = data.repartition(20)
            
            return data
        except Exception as e:
            logger.warning(f"ETL optimization failed: {e}")
            return data
    
    def _apply_storage_optimizations(self, data: DataFrame) -> DataFrame:
        """Apply storage-specific optimizations."""
        try:
            # Optimize for storage efficiency
            num_partitions = data.rdd.getNumPartitions()
            
            # Fewer, larger partitions for storage
            if num_partitions > 20:
                data = data.coalesce(20)
            
            return data
        except Exception as e:
            logger.warning(f"Storage optimization failed: {e}")
            return data
    
    def _register_table(self, table_name: str, data: DataFrame, stage: str, 
                       optimize_for: str, creation_time: float):
        """Register table metadata for tracking."""
        try:
            # Get table statistics
            row_count = data.count()
            column_count = len(data.columns)
            
            # Estimate size
            sample_size = min(1000, row_count)
            sample_data = data.limit(sample_size)
            
            # Calculate estimated size in MB
            estimated_size_mb = self._estimate_table_size(sample_data, row_count)
            
            # Register table
            self.temp_tables[table_name] = {
                'stage': stage,
                'optimize_for': optimize_for,
                'row_count': row_count,
                'column_count': column_count,
                'estimated_size_mb': estimated_size_mb,
                'creation_time': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'is_cached': data.is_cached
            }
            
            # Add to stage tracking
            if stage not in self.table_metadata:
                self.table_metadata[stage] = []
            self.table_metadata[stage].append(table_name)
            
            logger.debug(f"📋 Registered table {table_name}: {row_count:,} rows, {estimated_size_mb:.2f} MB")
            
        except Exception as e:
            logger.warning(f"Could not register table metadata: {e}")
    
    def _estimate_table_size(self, sample_data: DataFrame, total_rows: int) -> float:
        """Estimate table size in MB based on sample."""
        try:
            total_size_bytes = 0
            
            for field in sample_data.schema.fields:
                field_type = field.dataType.typeName()
                
                # Estimate field size based on type
                if field_type in ['string', 'binary']:
                    field_size = 64  # Average string/binary size
                elif field_type in ['double', 'long', 'bigint']:
                    field_size = 8
                elif field_type in ['integer', 'int']:
                    field_size = 4
                elif field_type in ['boolean']:
                    field_size = 1
                else:
                    field_size = 16  # Default for other types
                
                total_size_bytes += field_size
            
            # Calculate total size
            total_size_mb = (total_size_bytes * total_rows) / (1024 * 1024)
            return total_size_mb
            
        except Exception as e:
            logger.warning(f"Could not estimate table size: {e}")
            return 0.0
    
    def _apply_table_optimizations(self, table_name: str, optimize_for: str):
        """Apply additional table-level optimizations."""
        try:
            if optimize_for == 'ml':
                # ML-specific table optimizations
                self.spark.sql(f"ANALYZE TABLE {table_name} COMPUTE STATISTICS")
                
            elif optimize_for == 'query':
                # Query-specific optimizations
                self.spark.sql(f"ANALYZE TABLE {table_name} COMPUTE STATISTICS FOR ALL COLUMNS")
                
        except Exception as e:
            logger.debug(f"Table optimization failed for {table_name}: {e}")
    
    def get_temp_table(self, table_name: str, stage: str = None) -> Optional[str]:
        """
        Get temporary table name, optionally filtering by stage.
        
        Args:
            table_name: Base table name
            stage: Optional stage filter
            
        Returns:
            Full table name if found, None otherwise
        """
        try:
            # Generate possible table names
            if stage:
                possible_names = [self._generate_table_name(table_name, stage)]
            else:
                # Check all stages
                possible_names = []
                for stage_name in self.table_metadata.keys():
                    possible_names.append(self._generate_table_name(table_name, stage_name))
            
            # Find existing table
            for name in possible_names:
                if name in self.temp_tables:
                    # Update access metadata
                    self.temp_tables[name]['last_accessed'] = time.time()
                    self.temp_tables[name]['access_count'] += 1
                    return name
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting temp table {table_name}: {e}")
            return None
    
    def query_temp_table(self, table_name: str, query: str, stage: str = None) -> DataFrame:
        """
        Query a temporary table with automatic table resolution.
        
        Args:
            table_name: Base table name
            query: SQL query to execute
            stage: Optional stage filter
            
        Returns:
            DataFrame result
        """
        try:
            # Get full table name
            full_table_name = self.get_temp_table(table_name, stage)
            if not full_table_name:
                raise ValueError(f"Temp table {table_name} not found")
            
            # Replace table name in query
            modified_query = query.replace(table_name, full_table_name)
            
            logger.debug(f"🔍 Executing query on {full_table_name}: {modified_query}")
            
            # Execute query
            result = self.spark.sql(modified_query)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error querying temp table {table_name}: {e}")
            raise
    
    def get_table_info(self, table_name: str = None) -> Dict[str, Any]:
        """
        Get information about temporary tables.
        
        Args:
            table_name: Specific table name, or None for all tables
            
        Returns:
            Dictionary with table information
        """
        try:
            if table_name:
                if table_name in self.temp_tables:
                    return {
                        'table_name': table_name,
                        **self.temp_tables[table_name]
                    }
                else:
                    return {'error': f'Table {table_name} not found'}
            
            # Return all tables grouped by stage
            result = {
                'total_tables': len(self.temp_tables),
                'total_size_mb': sum(t['estimated_size_mb'] for t in self.temp_tables.values()),
                'stages': {}
            }
            
            for stage, tables in self.table_metadata.items():
                stage_info = {
                    'table_count': len(tables),
                    'tables': []
                }
                
                for table_name in tables:
                    if table_name in self.temp_tables:
                        table_info = self.temp_tables[table_name].copy()
                        table_info['table_name'] = table_name
                        stage_info['tables'].append(table_info)
                
                result['stages'][stage] = stage_info
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting table info: {e}")
            return {'error': str(e)}
    
    def cleanup_old_tables(self, max_age_hours: int = None, batch_size: int = None) -> Dict[str, Any]:
        """
        Clean up old temporary tables.
        
        Args:
            max_age_hours: Maximum age in hours (uses config default if None)
            batch_size: Batch size for cleanup (uses config default if None)
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            max_age = max_age_hours or self.config['max_table_age_hours']
            batch_size = batch_size or self.config['cleanup_batch_size']
            
            current_time = time.time()
            max_age_seconds = max_age * 3600
            
            # Find old tables
            old_tables = []
            for table_name, metadata in self.temp_tables.items():
                age_seconds = current_time - metadata['creation_time']
                if age_seconds > max_age_seconds:
                    old_tables.append((table_name, age_seconds))
            
            # Sort by age (oldest first)
            old_tables.sort(key=lambda x: x[1], reverse=True)
            
            # Clean up in batches
            cleaned_count = 0
            failed_count = 0
            
            for i in range(0, len(old_tables), batch_size):
                batch = old_tables[i:i + batch_size]
                
                for table_name, age_seconds in batch:
                    try:
                        success = self._drop_temp_table(table_name)
                        if success:
                            cleaned_count += 1
                            logger.debug(f"🧹 Cleaned up old table: {table_name} (age: {age_seconds/3600:.1f}h)")
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to cleanup table {table_name}: {e}")
            
            result = {
                'cleaned_count': cleaned_count,
                'failed_count': failed_count,
                'total_old_tables': len(old_tables),
                'max_age_hours': max_age
            }
            
            logger.info(f"🧹 Cleanup completed: {cleaned_count} tables cleaned, {failed_count} failed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            return {'error': str(e)}
    
    def _drop_temp_table(self, table_name: str) -> bool:
        """Drop a temporary table."""
        try:
            # Drop the temp view
            self.spark.catalog.dropTempView(table_name)
            
            # Remove from tracking
            if table_name in self.temp_tables:
                del self.temp_tables[table_name]
            
            # Remove from stage tracking
            for stage, tables in self.table_metadata.items():
                if table_name in tables:
                    tables.remove(table_name)
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not drop table {table_name}: {e}")
            return False
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage for all temporary tables.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("🔧 Optimizing temporary table storage...")
            
            optimization_results = {
                'tables_optimized': 0,
                'storage_saved_mb': 0,
                'errors': []
            }
            
            for table_name, metadata in self.temp_tables.items():
                try:
                    # Apply storage optimizations
                    if metadata['stage'] == 'storage':
                        # Already optimized for storage
                        continue
                    
                    # Get current table
                    current_df = self.spark.table(table_name)
                    
                    # Apply storage optimizations
                    optimized_df = self._apply_storage_optimizations(current_df)
                    
                    # Replace table if optimization was beneficial
                    if optimized_df.rdd.getNumPartitions() < current_df.rdd.getNumPartitions():
                        optimized_df.createOrReplaceTempView(table_name)
                        optimization_results['tables_optimized'] += 1
                        
                        # Update metadata
                        old_size = metadata['estimated_size_mb']
                        new_size = self._estimate_table_size(optimized_df, metadata['row_count'])
                        size_saved = old_size - new_size
                        optimization_results['storage_saved_mb'] += max(0, size_saved)
                        
                        # Update metadata
                        metadata['estimated_size_mb'] = new_size
                        
                        logger.debug(f"✅ Optimized {table_name}: saved {size_saved:.2f} MB")
                
                except Exception as e:
                    error_msg = f"Error optimizing {table_name}: {e}"
                    optimization_results['errors'].append(error_msg)
                    logger.warning(error_msg)
            
            logger.info(f"✅ Storage optimization completed: {optimization_results['tables_optimized']} tables optimized")
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Error during storage optimization: {e}")
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all temporary tables."""
        try:
            summary = {
                'total_tables': len(self.temp_tables),
                'total_size_mb': sum(t['estimated_size_mb'] for t in self.temp_tables.values()),
                'cached_tables': sum(1 for t in self.temp_tables.values() if t['is_cached']),
                'stage_distribution': {},
                'access_patterns': {},
                'storage_efficiency': 0.0
            }
            
            # Stage distribution
            for stage, tables in self.table_metadata.items():
                summary['stage_distribution'][stage] = len(tables)
            
            # Access patterns
            if self.temp_tables:
                total_accesses = sum(t['access_count'] for t in self.temp_tables.values())
                avg_accesses = total_accesses / len(self.temp_tables)
                summary['access_patterns'] = {
                    'total_accesses': total_accesses,
                    'average_accesses_per_table': avg_accesses,
                    'most_accessed_table': max(self.temp_tables.items(), key=lambda x: x[1]['access_count'])[0]
                }
            
            # Storage efficiency (ratio of cached to total tables)
            if summary['total_tables'] > 0:
                summary['storage_efficiency'] = summary['cached_tables'] / summary['total_tables']
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up all temporary tables and resources."""
        try:
            logger.info("🧹 Cleaning up all temporary tables...")
            
            # Clean up all tables
            cleanup_results = self.cleanup_old_tables(max_age_hours=0)  # Force cleanup of all tables
            
            # Clear tracking
            self.temp_tables.clear()
            self.table_metadata.clear()
            
            logger.info(f"✅ Cleanup completed: {cleanup_results.get('cleaned_count', 0)} tables removed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def create_temp_table_manager(spark_session: SparkSession, 
                            config: Optional[Dict[str, Any]] = None) -> TempTableManager:
    """
    Factory function to create a temporary table manager.
    
    Args:
        spark_session: Active Spark session
        config: Optional configuration overrides
        
    Returns:
        Configured TempTableManager instance
    """
    return TempTableManager(spark_session, config)


# Example usage and testing
if __name__ == "__main__":
    print("🗄️ Temporary Table Manager")
    print("=" * 50)
    
    # This would typically be used within a Spark application
    print("💡 Import and use within your Spark application:")
    print("   from temp_table_manager import create_temp_table_manager")
    print("   manager = create_temp_table_manager(spark)")
    print("   table_name = manager.create_temp_table(data, 'features', 'feature_selected', 'ml')")
    print("   result = manager.query_temp_table('features', 'SELECT * FROM features LIMIT 10')")
