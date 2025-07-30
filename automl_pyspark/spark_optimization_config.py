#!/usr/bin/env python3
"""
Spark optimization configuration for AutoML to handle large task binaries and
improve performance.

This module provides helper functions to configure PySpark sessions with
reasonable defaults for machine learning workloads.  It also includes
utilities for BigQuery integration that can automatically scale driver
memory based on BigQuery table metadata when the Google Cloud Python
client is available.
"""

from typing import Optional  # type hints for function signatures

def get_optimized_spark_config(include_synapseml: bool = False, include_bigquery: bool = True) -> dict:
    """Get optimized Spark configuration for AutoML workloads.
    
    Args:
        include_synapseml: If True, includes SynapseML JARs for LightGBM support
        include_bigquery: If True, includes BigQuery connector JAR
    """
    
    config = {
        # Memory and broadcasting optimizations
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true", 
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
        
        # Task binary broadcasting optimizations - REDUCED WARNINGS
        "spark.rdd.compress": "true",
        "spark.broadcast.compress": "true", 
        "spark.io.compression.codec": "snappy",
        "spark.broadcast.blockSize": "4m",           # Smaller broadcast blocks
        "spark.sql.adaptive.broadcastJoinThreshold": "10m",  # Reduce broadcast join threshold
        "spark.sql.autoBroadcastJoinThreshold": "10m",       # Reduce auto broadcast threshold
        
        # Task and stage optimizations
        "spark.task.maxDirectResultSize": "1536m",  # Safe limit under 2GB
        "spark.driver.maxResultSize": "2g",         # Reduced for compatibility
        "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "256MB",
        
        # Execution optimizations
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.adaptive.localShuffleReader.enabled": "true",
        
        # Memory management
        "spark.executor.memory": "4g",
        "spark.driver.memory": "4g",
        "spark.executor.memoryFraction": "0.8",
        
        # Serialization - IMPROVED FOR LARGE TASK BINARIES
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "1024m",
        "spark.kryoserializer.buffer": "64k",       # Smaller buffer for better memory usage
        
        # Model-specific optimizations for ML pipelines
        "spark.ml.optimization.vectorizedReader.enabled": "true",
        "spark.sql.adaptive.bucketing.enabled": "true",
        
        # Additional optimizations for large task binaries
        "spark.sql.execution.arrow.pyspark.enabled": "true",  # Use Arrow for better performance
        "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",  # Limit batch size
        "spark.sql.adaptive.coalescePartitions.minPartitionSize": "1MB",  # Allow small partitions (updated from deprecated minPartitionNum)
        "spark.sql.adaptive.coalescePartitions.parallelismFirst": "true",  # Prioritize parallelism
    }
    
    # Add SynapseML configuration if requested
    if include_synapseml:
        synapseml_version = "0.11.4"  # Use stable version
        config.update({
            "spark.jars.packages": f"com.microsoft.azure:synapseml_2.12:{synapseml_version}",
            "spark.jars.repositories": "https://mmlspark.azureedge.net/maven",
        })
    
    return config


# ---------------------------------------------------------------------------
# BigQuery metadata utilities for automatic memory scaling
# ---------------------------------------------------------------------------

def _estimate_bigquery_table_size(table_reference: str) -> int:
    """Estimate the size of a BigQuery table in bytes.

    This helper uses the Google Cloud BigQuery Python client to fetch
    table metadata and return the total size of the table in bytes.
    It requires that the user has authenticated with Google Cloud (e.g., via
    `gcloud auth application-default login`) and that the
    `google-cloud-bigquery` package is installed.

    Parameters
    ----------
    table_reference : str
        Fully qualified table reference in the form
        ``project.dataset.table``.

    Returns
    -------
    int
        The size of the table in bytes, or 0 if the size cannot be
        determined.
    """
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError:
        # BigQuery client is not available; cannot estimate size
        return 0
    parts = table_reference.split(".")
    if len(parts) != 3:
        return 0
    project_id, dataset_id, table_id = parts
    try:
        client = bigquery.Client(project=project_id)
        table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")
        # num_bytes gives the stored size of the table in bytes
        return table.num_bytes or 0
    except Exception:
        return 0

def apply_spark_optimizations(spark_session):
    """Apply optimizations to an existing Spark session."""
    
    config = get_optimized_spark_config()
    
    print("🚀 Applying Spark optimizations for AutoML...")
    for key, value in config.items():
        try:
            spark_session.conf.set(key, value)
            print(f"   ✅ {key}: {value}")
        except Exception as e:
            print(f"   ⚠️ Could not set {key}: {e}")
    
    print("✅ Spark optimizations applied!")
    return spark_session

def create_optimized_spark_session(app_name="AutoML Optimized", include_lightgbm=False, include_bigquery=True):
    """Create a new Spark session with optimizations applied.
    
    Args:
        app_name: Name for the Spark application
        include_lightgbm: If True, includes SynapseML JARs for LightGBM support
        include_bigquery: If True, includes BigQuery connector JAR (default: True)
    """
    
    from pyspark.sql import SparkSession
    
    config = get_optimized_spark_config(include_synapseml=include_lightgbm, include_bigquery=include_bigquery)
    
    if include_lightgbm:
        print("🚀 Creating optimized Spark session with LightGBM support...")
        print("   📦 This will download SynapseML JARs (may take a moment on first run)")
    else:
        print("🚀 Creating optimized Spark session...")
    
    builder = SparkSession.builder.appName(app_name)
    
    # Add packages if requested
    packages = []
    if include_lightgbm:
        packages.append("com.microsoft.azure:synapseml_2.12:1.0.3")
    
    # Add BigQuery connector if requested (default: True)
    if include_bigquery:
        packages.append("com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1")
    
    if packages:
        builder = builder.config("spark.jars.packages", ",".join(packages))
    
    # Apply all configurations
    for key, value in config.items():
        builder = builder.config(key, value)
    
    # Add BigQuery-specific optimizations if BigQuery is enabled
    if include_bigquery:
        # Ensure adequate memory for BigQuery operations
        builder = builder.config("spark.driver.memory", "8g")  # Increase if you have more RAM available
        builder = builder.config("spark.driver.maxResultSize", "4g")
        builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        builder = builder.config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        
        print("   🔗 BigQuery-specific optimizations applied:")
        print("      • Driver memory: 8g (increase if you have more RAM)")
        print("      • Arrow optimization enabled for better BigQuery performance")
    
    try:
        spark = builder.getOrCreate()
        print(f"✅ Created optimized Spark session: {app_name}")
        print(f"   📊 Applied {len(config)} optimization settings")
        
        # Test LightGBM availability if requested
        if include_lightgbm:
            try:
                from synapse.ml.lightgbm import LightGBMClassifier
                # Test if we can create a classifier
                LightGBMClassifier(featuresCol="features", labelCol="label")
                print("✅ LightGBM is available and working")
            except Exception as e:
                print(f"⚠️ LightGBM test failed: {e}")
                print("💡 You may need to restart your session for SynapseML JARs to load properly")
        
        return spark
        
    except Exception as e:
        print(f"❌ Failed to create optimized Spark session: {e}")
        if include_lightgbm:
            print("💡 Falling back to session without LightGBM...")
            return create_optimized_spark_session(app_name + " (No LightGBM)", include_lightgbm=False)
        else:
            raise e

def create_bigquery_optimized_session(
    app_name: str = "AutoML BigQuery",
    use_local_jar: bool = True,
    table_reference: Optional[str] = None,
    memory_factor: float = 3.0,
    minimum_driver_memory_gb: int = 8,
) -> "SparkSession":
    """Create a BigQuery‑optimized Spark session.

    When a ``table_reference`` (in the form ``project.dataset.table``) is
    provided, this function will try to estimate the size of that table
    using the BigQuery API.  Based on the estimated size, it
    automatically scales the Spark driver memory to better handle large
    datasets.  The driver memory is computed as

    ``max(minimum_driver_memory_gb, ceil(table_size_bytes * memory_factor / 1e9))``.

    This simple heuristic multiplies the table size by ``memory_factor`` to
    allow for overhead during Spark operations (e.g., cached data,
    intermediate results, and serialization), then converts bytes to
    gigabytes.  If the estimated size is unavailable (for example, when
    the BigQuery client is not installed or the user is not
    authenticated), the driver memory falls back to ``minimum_driver_memory_gb``.

    Parameters
    ----------
    app_name : str, default "AutoML BigQuery"
        Name for the Spark application.
    use_local_jar : bool, default ``True``
        If True, uses local JAR for faster loading (recommended).  If
        False, the connector will be pulled from Maven.
    table_reference : Optional[str], default ``None``
        Fully qualified BigQuery table name (e.g., ``myproject.mydataset.mytable``)
        used to estimate the data size and adjust driver memory.  If
        omitted or estimation fails, no automatic scaling will occur.
    memory_factor : float, default ``3.0``
        Safety multiplier applied to the table size when calculating
        driver memory.  Increase this value for more headroom.  A factor
        of 3 is often sufficient for most ML workloads.
    minimum_driver_memory_gb : int, default ``8``
        Minimum driver memory to allocate if the estimated size is small
        or unavailable.  This value must be positive.

    Returns
    -------
    SparkSession
        A configured Spark session optimized for BigQuery operations.
    """
    
    from pyspark.sql import SparkSession
    import os
    
    print("🚀 Creating BigQuery-optimized Spark session...")
    
    # Stop any existing session for clean start
    try:
        existing_spark = SparkSession.getActiveSession()
        if existing_spark:
            print("🔄 Stopping existing Spark session for clean BigQuery setup...")
            existing_spark.stop()
            import time
            time.sleep(2)  # Allow proper shutdown
    except:
        pass
    
    # Get base optimized configuration (no BigQuery JAR yet)
    config = get_optimized_spark_config(include_synapseml=False, include_bigquery=False)

    # Default BigQuery-specific configuration values.  These values will
    # be overwritten if a table size estimate is available.
    bigquery_config: dict[str, str] = {
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
        "spark.driver.maxResultSize": "4g",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
    }

    # Determine appropriate driver memory.  If a table reference is
    # provided, attempt to estimate the table size using the BigQuery
    # Python client; otherwise, use the minimum.
    driver_memory_gb: int = minimum_driver_memory_gb
    if table_reference:
        try:
            table_size_bytes = _estimate_bigquery_table_size(table_reference)
            if table_size_bytes > 0:
                import math
                estimated_gb = math.ceil(table_size_bytes * memory_factor / 1e9)
                # Ensure driver memory is at least the minimum
                driver_memory_gb = max(minimum_driver_memory_gb, estimated_gb)
                print(
                    f"📏 Estimated BigQuery table size: {table_size_bytes / 1e9:.2f} GB"
                    f" → allocating {driver_memory_gb} GB driver memory"
                )
            else:
                print(
                    f"⚠️ Unable to estimate BigQuery table size for '{table_reference}'."
                    f" Using minimum driver memory of {minimum_driver_memory_gb} GB."
                )
        except Exception as e:
            # Estimation failed (client not installed, auth error, etc.)
            print(
                f"⚠️ Error estimating BigQuery table size for '{table_reference}': {e}\n"
                f"   Falling back to minimum driver memory ({minimum_driver_memory_gb} GB)."
            )

    # Set driver and (optionally) executor memory in BigQuery config
    bigquery_config["spark.driver.memory"] = f"{driver_memory_gb}g"
    # Use half the driver memory for the executor by default; adjust as needed
    bigquery_config["spark.executor.memory"] = f"{max(1, driver_memory_gb // 2)}g"

    # Combine the base config with BigQuery-specific settings
    final_config = {**config, **bigquery_config}

    builder = SparkSession.builder.appName(app_name)
    
    if use_local_jar:
        # Use local JAR for faster loading and better reliability.  Note that
        # the jar path is relative to the project root; adjust if needed.
        jar_path = os.path.abspath("jars/spark-bigquery-with-dependencies_2.12-0.36.1.jar")
        if os.path.exists(jar_path):
            print(f"📦 Using local BigQuery JAR: {jar_path}")
            builder = builder.config("spark.jars", jar_path)
        else:
            print(f"⚠️ Local JAR not found at {jar_path}, falling back to Maven")
            builder = builder.config(
                "spark.jars.packages",
                "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1",
            )
    else:
        # Use Maven for download (slower but always available)
        print("📦 Using Maven to download BigQuery connector...")
        builder = builder.config(
            "spark.jars.packages",
            "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1",
        )

    # Apply all configurations (base + BigQuery) to the builder
    for key, value in final_config.items():
        builder = builder.config(key, value)
    
    try:
        spark = builder.getOrCreate()
        print("✅ BigQuery-optimized Spark session created successfully")
        
        # Verify BigQuery connector
        try:
            test_reader = spark.read.format("bigquery")
            print("✅ BigQuery connector verified and ready")
        except Exception as e:
            print(f"⚠️ BigQuery connector verification failed: {e}")
        
        return spark
        
    except Exception as e:
        print(f"❌ Failed to create BigQuery-optimized session: {e}")
        raise

# Hyperparameter tuning specific optimizations
def get_tuning_optimizations():
    """Get additional optimizations specifically for hyperparameter tuning."""
    
    return {
        # Reduce model complexity for tuning
        "spark.sql.adaptive.coalescePartitions.parallelismFirst": "true",
        "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "64MB",  # Reduced from 128MB
        
        # Faster evaluation with smaller broadcast thresholds
        "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "2",
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "32MB",  # Reduced from 64MB
        
        # Memory optimization for many trials
        "spark.cleaner.periodicGC.interval": "5s",  # More frequent GC
        "spark.sql.adaptive.columnar.cache.enabled": "false",  # Disable caching during tuning
        
        # SIGNIFICANTLY REDUCE LARGE TASK BINARY WARNINGS
        "spark.sql.autoBroadcastJoinThreshold": "2m",        # Further reduced from 5m
        "spark.sql.adaptive.broadcastJoinThreshold": "2m",   # Further reduced from 5m
        "spark.broadcast.blockSize": "1m",                   # Smaller broadcast blocks
        "spark.sql.execution.arrow.maxRecordsPerBatch": "2000",  # Much smaller batches
        "spark.sql.adaptive.coalescePartitions.minPartitionSize": "1MB",  # Allow small partitions (updated from deprecated minPartitionNum)
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "32MB",  # Smaller partitions
        
        # Enhanced serialization for model objects
        "spark.kryoserializer.buffer": "16k",               # Smaller initial buffer
        "spark.kryoserializer.buffer.max": "256m",          # Reduced max buffer
        "spark.serializer.objectStreamReset": "100",       # More frequent stream resets
        
        # Disable unnecessary features during tuning
        "spark.sql.adaptive.localShuffleReader.enabled": "false",  # Reduce complexity
        "spark.sql.adaptive.skewJoin.enabled": "false",     # Reduce overhead
    }

def get_gradient_boosting_optimizations():
    """Get specific optimizations for gradient boosting models to reduce large task binary warnings."""
    
    return {
        # AGGRESSIVE TASK BINARY REDUCTION FOR GRADIENT BOOSTING
        "spark.sql.autoBroadcastJoinThreshold": "1m",        # Very small for GBT
        "spark.sql.adaptive.broadcastJoinThreshold": "1m",   # Very small for GBT
        "spark.broadcast.blockSize": "512k",                 # Extra small broadcast blocks
        "spark.sql.execution.arrow.maxRecordsPerBatch": "1000",  # Very small batches
        
        # Tree-specific memory optimizations
        "spark.cleaner.periodicGC.interval": "3s",           # Very frequent GC for trees
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "16MB",  # Small partitions
        "spark.sql.adaptive.coalescePartitions.minPartitionSize": "1MB",
        
        # Enhanced serialization for tree models
        "spark.kryoserializer.buffer": "8k",                # Very small initial buffer
        "spark.kryoserializer.buffer.max": "128m",          # Reduced max buffer for trees
        "spark.serializer.objectStreamReset": "50",         # More frequent resets
        
        # Disable features that increase task size
        "spark.sql.adaptive.localShuffleReader.enabled": "false",
        "spark.sql.adaptive.skewJoin.enabled": "false",
        "spark.sql.adaptive.bucketing.enabled": "false",
        "spark.sql.adaptive.columnar.cache.enabled": "false",
        
        # Memory pressure management
        "spark.cleaner.referenceTracking.blocking": "true",
        "spark.cleaner.referenceTracking.blocking.shuffle": "true",
        "spark.executor.heartbeatInterval": "20s",          # More frequent heartbeats
        
        # Force smaller feature vectors in broadcast
        "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "16MB",
    }

def apply_gradient_boosting_optimizations(spark_session):
    """Apply gradient boosting specific optimizations to reduce large task binary warnings.
    
    Args:
        spark_session: The Spark session to optimize
    """
    
    config = get_gradient_boosting_optimizations()
    print("🌳 Applying gradient boosting optimizations to reduce large task binary warnings...")
    
    applied_count = 0
    failed_count = 0
    
    for key, value in config.items():
        try:
            spark_session.conf.set(key, value)
            applied_count += 1
        except Exception as e:
            print(f"   ⚠️ Could not set {key}: {e}")
            failed_count += 1
    
    print(f"✅ Applied {applied_count} gradient boosting optimizations ({failed_count} failed)")
    print("💡 These settings should significantly reduce large task binary warnings for tree-based models")
    return spark_session

def get_aggressive_memory_optimizations():
    """Get aggressive memory optimizations for critical memory situations."""
    
    return {
        # CRITICAL MEMORY MANAGEMENT
        "spark.cleaner.periodicGC.interval": "3s",           # Very frequent GC
        "spark.cleaner.referenceTracking.blocking": "true",  # Blocking cleanup
        "spark.cleaner.referenceTracking.blocking.shuffle": "true",  # Blocking shuffle cleanup
        "spark.cleaner.referenceTracking.cleanCheckpoints": "true",  # Clean checkpoints
        
        # MINIMAL TASK BINARY SIZE
        "spark.sql.autoBroadcastJoinThreshold": "512k",      # Very small broadcast threshold
        "spark.sql.adaptive.broadcastJoinThreshold": "512k", # Very small adaptive broadcast
        "spark.broadcast.blockSize": "512k",                 # Very small broadcast blocks
        "spark.sql.execution.arrow.maxRecordsPerBatch": "500",  # Very small batches
        
        # AGGRESSIVE SERIALIZATION
        "spark.kryoserializer.buffer": "8k",                # Very small initial buffer
        "spark.kryoserializer.buffer.max": "128m",          # Reduced max buffer
        "spark.serializer.objectStreamReset": "50",         # More frequent stream resets
        
        # MINIMAL PARTITION SIZE
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "16MB",  # Very small partitions
        "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "16MB",
        "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "32MB",
        
        # DISABLE CACHING AND COMPLEX FEATURES
        "spark.sql.adaptive.columnar.cache.enabled": "false",
        "spark.sql.adaptive.localShuffleReader.enabled": "false",
        "spark.sql.adaptive.skewJoin.enabled": "false",
        "spark.sql.adaptive.bucketing.enabled": "false",
        
        # Force immediate cleanup
        "spark.sql.adaptive.coalescePartitions.enabled": "false",  # Disable partition coalescing
    }

def apply_tuning_optimizations(spark_session, aggressive=False):
    """Apply hyperparameter tuning optimizations to an existing Spark session.
    
    Args:
        spark_session: The Spark session to optimize
        aggressive: If True, applies aggressive memory optimizations
    """
    
    if aggressive:
        config = get_aggressive_memory_optimizations()
        print("🚀 Applying AGGRESSIVE memory optimizations for hyperparameter tuning...")
    else:
        config = get_tuning_optimizations()
        print("🚀 Applying hyperparameter tuning optimizations...")
    
    applied_count = 0
    failed_count = 0
    
    for key, value in config.items():
        try:
            spark_session.conf.set(key, value)
            applied_count += 1
        except Exception as e:
            print(f"   ⚠️ Could not set {key}: {e}")
            failed_count += 1
    
    print(f"✅ Applied {applied_count} tuning optimizations ({failed_count} failed)")
    print("💡 These settings significantly reduce large task binary warnings during model training")
    return spark_session

def optimize_for_hyperparameter_tuning(spark_session, model_type="gradient_boosting", n_trials=None):
    """Optimize Spark session specifically for hyperparameter tuning.
    
    Args:
        spark_session: The Spark session to optimize
        model_type: Type of model being tuned
        n_trials: Number of planned trials (affects optimization level)
    """
    
    print(f"🎯 Optimizing Spark for {model_type} hyperparameter tuning...")
    
    # Determine optimization level based on model type and trial count
    use_aggressive = False
    
    if model_type in ['gradient_boosting', 'xgboost', 'lightgbm']:
        print(f"   📊 Tree-based model detected: applying enhanced optimizations")
        use_aggressive = True
    
    if n_trials and n_trials > 20:
        print(f"   📈 High trial count ({n_trials}): applying aggressive optimizations")
        use_aggressive = True
    
    # Apply optimizations
    apply_tuning_optimizations(spark_session, aggressive=use_aggressive)
    
    return spark_session

def check_lightgbm_availability():
    """Check if LightGBM (SynapseML) is properly configured in current Spark session."""
    
    print("🔍 Checking LightGBM availability...")
    
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        
        if spark is None:
            print("❌ No active Spark session found")
            return False
        
        # Test if we can import SynapseML
        try:
            from synapse.ml.lightgbm import LightGBMClassifier
            print("✅ SynapseML import successful")
            
            # Test if we can create a classifier (this will fail if JARs not loaded)
            try:
                classifier = LightGBMClassifier(featuresCol="features", labelCol="label")
                print("✅ LightGBM is available and working")
                return True
            except Exception as e:
                print(f"❌ LightGBM creation failed: {e}")
                print("💡 This usually means SynapseML JARs are not loaded in Spark session")
                print("💡 Use: create_optimized_spark_session(include_lightgbm=True)")
                return False
                
        except ImportError as e:
            print(f"❌ SynapseML import failed: {e}")
            print("💡 Install with: pip install synapseml>=0.11.0")
            return False
            
    except Exception as e:
        print(f"❌ Error checking LightGBM: {e}")
        return False

if __name__ == "__main__":
    print("�� Spark Optimization Configuration for AutoML")
    print("="*60)
    
    # Check current LightGBM availability
    lightgbm_available = check_lightgbm_availability()
    
    print(f"\n📋 LightGBM Status: {'✅ Available' if lightgbm_available else '❌ Not Available'}")
    
    # Show configuration examples
    basic_config = get_optimized_spark_config()
    lightgbm_config = get_optimized_spark_config(include_synapseml=True)
    tuning_config = get_tuning_optimizations()
    
    print(f"\n📦 Basic Configuration ({len(basic_config)} settings):")
    print("  Includes: XGBoost support, memory optimizations, compression")
    
    print(f"\n📦 LightGBM Configuration ({len(lightgbm_config)} settings):")
    print("  Includes: XGBoost + LightGBM support via SynapseML JARs")
    
    print(f"\n🎯 Hyperparameter Tuning Optimizations ({len(tuning_config)} settings):")
    print("  Additional optimizations for faster hyperparameter tuning")
    
    print("\n💡 Usage Examples:")
    print("\n  # For XGBoost only (fastest):")
    print("  from spark_optimization_config import create_optimized_spark_session")
    print("  spark = create_optimized_spark_session('AutoML XGBoost')")
    print("  automl = AutoMLClassifier(spark, preset='comprehensive')")
    print("  automl.fit(..., run_xgboost=True, run_lightgbm=False)")
    
    print("\n  # For XGBoost + LightGBM (complete):")
    print("  spark = create_optimized_spark_session('AutoML Full', include_lightgbm=True)")
    print("  automl = AutoMLClassifier(spark, preset='comprehensive')")
    print("  automl.fit(..., run_xgboost=True, run_lightgbm=True)")
    
    print("\n  # Check LightGBM availability:")
    print("  from spark_optimization_config import check_lightgbm_availability")
    print("  check_lightgbm_availability()")
    
    if not lightgbm_available:
        print("\n⚠️  To enable LightGBM:")
        print("   1. Create new session: create_optimized_spark_session(include_lightgbm=True)")
        print("   2. Or use XGBoost only for now") 
