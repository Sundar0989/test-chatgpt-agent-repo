"""
AutoML Optimization Example

This script demonstrates how to use the new optimization features:
1. Spark Autoscaling Manager
2. Temporary Table Manager  
3. Pipeline Optimizer

The example shows how to optimize a complete AutoML pipeline with:
- Automatic cluster scaling based on data size
- Temporary table management for better performance
- Integrated pipeline optimization
"""

import os
import sys
import time
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, rand, when

# Import our optimization modules
try:
    from spark_autoscaling_manager import create_autoscaling_manager
    from temp_table_manager import create_temp_table_manager
    from pipeline_optimizer import create_pipeline_optimizer
except ImportError:
    print("❌ Could not import optimization modules. Make sure they are in the same directory.")
    sys.exit(1)


def create_sample_data(spark: SparkSession, num_rows: int = 10000, num_features: int = 50) -> DataFrame:
    """Create sample data for demonstration purposes."""
    print(f"📊 Creating sample dataset: {num_rows:,} rows, {num_features} features")
    
    # Create feature columns
    feature_cols = []
    for i in range(num_features):
        feature_cols.append(f"feature_{i:02d}")
    
    # Create sample data with some patterns
    data = []
    for i in range(num_rows):
        row = []
        for j in range(num_features):
            # Create some correlation with target
            if j < 10:  # First 10 features have correlation with target
                value = (i % 100) / 100.0 + (j * 0.1) + (0.2 * (i % 10) / 10.0)
            else:  # Other features are mostly random
                value = (i * j) % 200 / 100.0 - 1.0
            row.append(float(value))
        
        # Create target (binary classification)
        target = 1 if sum(row[:10]) > 5 else 0
        row.append(target)
        
        data.append(row)
    
    # Create DataFrame
    columns = feature_cols + ['target']
    df = spark.createDataFrame(data, columns)
    
    print(f"✅ Sample dataset created: {df.count():,} rows, {len(df.columns)} columns")
    return df


def demonstrate_autoscaling(spark: SparkSession, data: DataFrame):
    """Demonstrate Spark autoscaling capabilities."""
    print("\n" + "="*60)
    print("🚀 SPARK AUTOSCALING DEMONSTRATION")
    print("="*60)
    
    # Create autoscaling manager
    autoscaling_config = {
        'min_executors': 1,
        'max_executors': 10,
        'target_processing_time_minutes': 5,
        'enable_auto_adjustment': True
    }
    
    autoscaling_manager = create_autoscaling_manager(spark, autoscaling_config)
    
    # Demonstrate autoscaling for different processing types
    processing_types = ['etl', 'ml', 'query']
    
    for processing_type in processing_types:
        print(f"\n📊 Testing autoscaling for {processing_type.upper()} processing...")
        
        # Auto-scale for the data
        scaling_results = autoscaling_manager.auto_scale_for_data(data, processing_type)
        
        if scaling_results['success']:
            print(f"✅ Autoscaling successful for {processing_type}")
            print(f"   📈 Data analysis: {scaling_results['data_analysis']['category']} dataset")
            print(f"   🔧 Applied config: {scaling_results['applied_config']}")
            
            # Show recommendations
            if 'scaling_recommendations' in scaling_results:
                print(f"   💡 Recommendations:")
                for rec in scaling_results['scaling_recommendations']:
                    print(f"      • {rec}")
        else:
            print(f"❌ Autoscaling failed for {processing_type}: {scaling_results.get('error', 'Unknown error')}")
    
    # Get scaling summary
    scaling_summary = autoscaling_manager.get_scaling_summary()
    print(f"\n📋 Autoscaling Summary:")
    print(f"   Total scaling operations: {scaling_summary['total_scaling_operations']}")
    print(f"   Recent configurations: {len(scaling_summary['recent_configurations'])}")
    
    return autoscaling_manager


def demonstrate_temp_tables(spark: SparkSession, data: DataFrame):
    """Demonstrate temporary table management capabilities."""
    print("\n" + "="*60)
    print("🗄️ TEMPORARY TABLE MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create temp table manager
    temp_table_config = {
        'enable_caching': True,
        'auto_cleanup': True,
        'max_table_age_hours': 2,
        'optimize_storage': True
    }
    
    temp_table_manager = create_temp_table_manager(spark, temp_table_config)
    
    # Demonstrate creating temp tables for different stages
    stages = [
        ('raw_data', 'raw', 'storage'),
        ('cleaned_data', 'cleaned', 'etl'),
        ('feature_selected', 'feature_selection', 'ml'),
        ('scaled_data', 'scaled', 'query'),
        ('split_data', 'split', 'storage')
    ]
    
    created_tables = []
    
    for table_name, stage, optimize_for in stages:
        print(f"\n🗄️ Creating temp table: {table_name} ({stage}, {optimize_for})")
        
        # Create temp table
        full_table_name = temp_table_manager.create_temp_table(
            data=data,
            table_name=table_name,
            stage=stage,
            optimize_for=optimize_for
        )
        
        created_tables.append(full_table_name)
        print(f"   ✅ Created: {full_table_name}")
        
        # Show table info
        table_info = temp_table_manager.get_table_info(full_table_name)
        if 'error' not in table_info:
            print(f"   📊 Size: {table_info.get('estimated_size_mb', 0):.2f} MB")
            print(f"   📈 Rows: {table_info.get('row_count', 0):,}")
    
    # Demonstrate querying temp tables
    print(f"\n🔍 Demonstrating temp table queries...")
    
    for table_name in created_tables[:2]:  # Test first 2 tables
        try:
            # Query the temp table
            result = temp_table_manager.query_temp_table(
                table_name.split('_')[-1],  # Extract base name
                f"SELECT COUNT(*) as count, AVG(feature_00) as avg_feature FROM {table_name}"
            )
            
            count_result = result.collect()[0]
            print(f"   📊 {table_name}: {count_result['count']:,} rows, avg feature_00: {count_result['avg_feature']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Query failed for {table_name}: {e}")
    
    # Get performance summary
    performance_summary = temp_table_manager.get_performance_summary()
    print(f"\n📋 Temp Table Performance Summary:")
    print(f"   Total tables: {performance_summary['total_tables']}")
    print(f"   Total size: {performance_summary['total_size_mb']:.2f} MB")
    print(f"   Cached tables: {performance_summary['cached_tables']}")
    print(f"   Storage efficiency: {performance_summary['storage_efficiency']:.2%}")
    
    return temp_table_manager


def demonstrate_pipeline_optimization(spark: SparkSession, data: DataFrame):
    """Demonstrate integrated pipeline optimization."""
    print("\n" + "="*60)
    print("🎯 PIPELINE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create pipeline optimizer
    pipeline_config = {
        'enable_autoscaling': True,
        'enable_temp_tables': True,
        'auto_optimize': True,
        'optimization_target': 'balanced'
    }
    
    pipeline_optimizer = create_pipeline_optimizer(spark, pipeline_config)
    
    # Define pipeline stages
    pipeline_stages = [
        {
            'name': 'data_loading',
            'type': 'data_loading',
            'data': data
        },
        {
            'name': 'preprocessing',
            'type': 'preprocessing',
            'data': data  # Will be updated by optimizer
        },
        {
            'name': 'feature_selection',
            'type': 'feature_selection',
            'data': data  # Will be updated by optimizer
        },
        {
            'name': 'model_training',
            'type': 'model_training',
            'data': data  # Will be updated by optimizer
        },
        {
            'name': 'evaluation',
            'type': 'evaluation',
            'data': data  # Will be updated by optimizer
        }
    ]
    
    print(f"🚀 Executing optimized pipeline with {len(pipeline_stages)} stages...")
    
    # Execute optimized pipeline
    pipeline_results = pipeline_optimizer.execute_optimized_pipeline(
        pipeline_stages, 
        optimization_target='balanced'
    )
    
    if pipeline_results['success']:
        print(f"✅ Pipeline completed successfully!")
        print(f"   📊 Total execution time: {pipeline_results['total_execution_time']:.2f}s")
        print(f"   📋 Stages executed: {len(pipeline_results['stages'])}")
        
        # Show stage details
        for i, stage_result in enumerate(pipeline_results['stages']):
            print(f"   📋 Stage {i+1}: {stage_result['stage_name']}")
            print(f"      ⏱️  Time: {stage_result['execution_time']:.2f}s")
            print(f"      🎯 Type: {stage_result['stage_type']}")
            
            if 'optimized_data_reference' in stage_result:
                print(f"      🗄️  Temp table: {stage_result['optimized_data_reference']}")
        
        # Show optimization summary
        if 'optimization_summary' in pipeline_results:
            opt_summary = pipeline_results['optimization_summary']
            print(f"\n📊 Optimization Summary:")
            print(f"   Total stages: {opt_summary.get('total_stages', 0)}")
            print(f"   Successful stages: {opt_summary.get('successful_stages', 0)}")
            print(f"   Total optimizations: {opt_summary.get('total_optimizations', 0)}")
        
        # Show performance summary
        if 'performance_summary' in pipeline_results:
            perf_summary = pipeline_results['performance_summary']
            print(f"\n📈 Performance Summary:")
            print(f"   Total execution time: {perf_summary.get('total_execution_time', 0):.2f}s")
            print(f"   Average stage time: {perf_summary.get('average_stage_time', 0):.2f}s")
            print(f"   Performance trend: {perf_summary.get('performance_trend', 'unknown')}")
            
            bottlenecks = perf_summary.get('bottlenecks', [])
            if bottlenecks:
                print(f"   🚨 Bottlenecks detected:")
                for bottleneck in bottlenecks:
                    print(f"      • {bottleneck}")
    
    else:
        print(f"❌ Pipeline failed: {pipeline_results.get('error', 'Unknown error')}")
    
    # Get comprehensive optimization report
    print(f"\n📋 Generating comprehensive optimization report...")
    optimization_report = pipeline_optimizer.get_optimization_report()
    
    if 'error' not in optimization_report:
        report = optimization_report
        print(f"✅ Optimization Report Generated:")
        print(f"   Pipeline uptime: {report['pipeline_summary']['uptime_minutes']:.1f} minutes")
        print(f"   Total optimizations: {report['pipeline_summary']['total_optimizations']}")
        
        # Show recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"   💡 Recommendations:")
            for rec in recommendations[:5]:  # Show first 5
                print(f"      • {rec}")
    
    return pipeline_optimizer


def demonstrate_environment_switching(spark: SparkSession, data: DataFrame):
    """Demonstrate how optimization settings change with environment."""
    print("\n" + "="*60)
    print("🌍 ENVIRONMENT-BASED OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Test different optimization targets
    optimization_targets = ['speed', 'memory', 'balanced', 'quality']
    
    for target in optimization_targets:
        print(f"\n🎯 Testing {target.upper()} optimization target...")
        
        # Create optimizer with specific target
        config = {
            'optimization_target': target,
            'autoscaling': {
                'enable_autoscaling': True,
                'min_executors': 2 if target == 'memory' else 4,
                'max_executors': 8 if target == 'memory' else 15
            },
            'temp_tables': {
                'enable_caching': target != 'memory',
                'max_table_age_hours': 2 if target == 'memory' else 12
            }
        }
        
        optimizer = create_pipeline_optimizer(spark, config)
        
        # Test single stage optimization
        stage_result = optimizer.optimize_pipeline_stage(
            stage_name=f'test_{target}',
            data=data,
            stage_type='feature_selection',
            optimization_target=target
        )
        
        if 'error' not in stage_result:
            print(f"   ✅ {target} optimization successful")
            print(f"   ⏱️  Execution time: {stage_result['execution_time']:.2f}s")
            
            # Show specific optimizations
            if 'stage_optimizations' in stage_result:
                optimizations = stage_result['stage_optimizations']
                print(f"   🔧 Applied optimizations: {len(optimizations)}")
        else:
            print(f"   ❌ {target} optimization failed: {stage_result['error']}")
        
        # Cleanup
        optimizer.cleanup()
    
    print(f"\n✅ Environment-based optimization demonstration completed")


def main():
    """Main demonstration function."""
    print("🚀 AutoML Optimization Features Demonstration")
    print("=" * 80)
    
    # Create Spark session
    print("🔧 Creating Spark session...")
    spark = SparkSession.builder \
        .appName("AutoML Optimization Demo") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    try:
        # Create sample data
        data = create_sample_data(spark, num_rows=5000, num_features=30)
        
        # Demonstrate individual components
        autoscaling_manager = demonstrate_autoscaling(spark, data)
        temp_table_manager = demonstrate_temp_tables(spark, data)
        pipeline_optimizer = demonstrate_pipeline_optimization(spark, data)
        
        # Demonstrate environment switching
        demonstrate_environment_switching(spark, data)
        
        print("\n" + "="*80)
        print("🎉 OPTIMIZATION DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n💡 Key Benefits Demonstrated:")
        print("   🚀 Automatic Spark cluster scaling based on data size")
        print("   🗄️  Efficient temporary table management")
        print("   🎯 Integrated pipeline optimization")
        print("   🌍 Environment-specific optimization strategies")
        print("   📊 Performance monitoring and recommendations")
        
        print("\n🔧 Usage in Your AutoML Pipeline:")
        print("   1. Import the optimization modules")
        print("   2. Create a pipeline optimizer")
        print("   3. Define your pipeline stages")
        print("   4. Execute with automatic optimization")
        print("   5. Monitor performance and get recommendations")
        
        # Cleanup
        print("\n🧹 Cleaning up resources...")
        autoscaling_manager.cleanup()
        temp_table_manager.cleanup()
        pipeline_optimizer.cleanup()
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop Spark session
        print("🛑 Stopping Spark session...")
        spark.stop()


if __name__ == "__main__":
    main()
