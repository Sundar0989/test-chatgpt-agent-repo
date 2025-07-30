
import sys
import os
import json
import traceback
import re
from datetime import datetime

# Add the automl_pyspark directory to Python path
automl_dir = os.path.dirname(os.path.abspath(__file__))
if automl_dir not in sys.path:
    sys.path.insert(0, automl_dir)

# Also add parent directory for automl_pyspark imports
parent_dir = os.path.dirname(automl_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def load_oot_datasets(config, data_manager=None, oot_bigquery_options=None):
    """Helper function to load OOT datasets based on configuration."""
    oot1_data = None
    oot2_data = None
    
    # Handle OOT1 - check for BigQuery table first, then file
    if config.get('oot1_bigquery_table'):
        log_message('{job_id}', f"📅 Loading OOT1 data from BigQuery: {{config['oot1_bigquery_table']}}")
        if data_manager and oot_bigquery_options:
            oot1_data, _ = data_manager.load_data(config['oot1_bigquery_table'], 'bigquery', **oot_bigquery_options)
        else:
            oot1_data = config['oot1_bigquery_table']  # Pass as string for backward compatibility
    elif config.get('oot1_file'):
        log_message('{job_id}', f"📅 Loading OOT1 data from file: {{config['oot1_file']}}")
        if data_manager:
            oot1_data, _ = data_manager.load_data(config['oot1_file'], 'existing')
        else:
            oot1_data = config['oot1_file']  # Pass as string for backward compatibility
    elif config.get('oot1_config'):
        oot1_config = config['oot1_config']
        if oot1_config.get('source_type') == 'bigquery':
            log_message('{job_id}', f"📅 Loading OOT1 data from BigQuery: {{oot1_config['data_source']}}")
            if data_manager:
                bigquery_options = oot1_config.get('options', {{}})
                oot1_data, _ = data_manager.load_data(oot1_config['data_source'], 'bigquery', **bigquery_options)
            else:
                oot1_data = oot1_config['data_source']  # Pass as string for backward compatibility
    
    # Handle OOT2 - check for BigQuery table first, then file
    if config.get('oot2_bigquery_table'):
        log_message('{job_id}', f"📅 Loading OOT2 data from BigQuery: {{config['oot2_bigquery_table']}}")
        if data_manager and oot_bigquery_options:
            oot2_data, _ = data_manager.load_data(config['oot2_bigquery_table'], 'bigquery', **oot_bigquery_options)
        else:
            oot2_data = config['oot2_bigquery_table']  # Pass as string for backward compatibility
    elif config.get('oot2_file'):
        log_message('{job_id}', f"📅 Loading OOT2 data from file: {{config['oot2_file']}}")
        if data_manager:
            oot2_data, _ = data_manager.load_data(config['oot2_file'], 'existing')
        else:
            oot2_data = config['oot2_file']  # Pass as string for backward compatibility
    elif config.get('oot2_config'):
        oot2_config = config['oot2_config']
        if oot2_config.get('source_type') == 'bigquery':
            log_message('{job_id}', f"📅 Loading OOT2 data from BigQuery: {{oot2_config['data_source']}}")
            if data_manager:
                bigquery_options = oot2_config.get('options', {{}})
                oot2_data, _ = data_manager.load_data(oot2_config['data_source'], 'bigquery', **bigquery_options)
            else:
                oot2_data = oot2_config['data_source']  # Pass as string for backward compatibility
    
    return oot1_data, oot2_data


def log_message(job_id, message):
    """Log a message to the job log file."""
    log_file = os.path.join('automl_jobs', f"{job_id}_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def update_progress(job_id, current_step, total_steps, current_task):
    """Update progress for the job."""
    progress_file = os.path.join('automl_jobs', f"{job_id}_progress.json")
    progress_data = {
        'current_step': current_step,
        'total_steps': total_steps,
        'current_task': current_task,
        'progress_percentage': round((current_step / total_steps) * 100, 1),
        'timestamp': datetime.now().isoformat()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def detect_progress_from_logs(job_id, task_type):
    """Detect progress based on log patterns."""
    log_file = os.path.join('automl_jobs', f"{job_id}_log.txt")
    if not os.path.exists(log_file):
        return 0, "Initializing..."
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Define progress patterns for different task types
        if task_type == 'classification':
            progress_patterns = [
                (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing"]),
                (2, "Feature Selection", ["Feature Selection", "feature selection", "selecting features"]),
                (3, "Data Splitting and Scaling", ["Data Splitting", "scaling", "split", "train/valid"]),
                (4, "Preparing Out-of-Time Datasets", ["Out-of-Time", "OOT", "oot"]),
                (5, "Model Building and Validation", ["Model Building", "training", "XGBoost", "Random Forest", "Logistic", "Gradient"]),
                (6, "Model Selection", ["Model Selection", "selecting best", "best model"]),
                (7, "Generating Scoring Code", ["Generating Scoring", "scoring code", "save model"]),
                (8, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully"])
            ]
        elif task_type == 'regression':
            progress_patterns = [
                (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing"]),
                (2, "Feature Selection", ["Feature Selection", "feature selection", "selecting features"]),
                (3, "Data Splitting and Scaling", ["Data Splitting", "scaling", "split", "train/valid"]),
                (4, "Preparing Out-of-Time Datasets", ["Out-of-Time", "OOT", "oot"]),
                (5, "Model Building and Validation", ["Model Building", "training", "Linear Regression", "Random Forest", "Gradient"]),
                (6, "Model Selection", ["Model Selection", "selecting best", "best model"]),
                (7, "Generating Scoring Code", ["Generating Scoring", "scoring code", "save model"]),
                (8, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully"])
            ]
        else:  # clustering
            progress_patterns = [
                (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing"]),
                (2, "Feature Scaling", ["Feature Scaling", "scaling", "normalize"]),
                (3, "Cluster Analysis", ["Cluster Analysis", "KMeans", "Bisecting", "clustering"]),
                (4, "Model Validation", ["Model Validation", "silhouette", "validation"]),
                (5, "Model Selection", ["Model Selection", "selecting best", "best model"]),
                (6, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully"])
            ]
        
        # Check patterns in reverse order (latest progress first)
        for step, description, patterns in reversed(progress_patterns):
            for pattern in patterns:
                if pattern.lower() in log_content.lower():
                    return step, description
        
        return 0, "Initializing..."
    
    except Exception as e:
        return 0, f"Error reading logs: {str(e)}"

try:
    # Load job configuration
    with open('automl_jobs/job_0001_automl_user_automl_model_1753845996.json') as f:
        config = json.load(f)
    
    task_type = config.get('task_type', 'classification')
    task_type_title = task_type.title()
    log_message('job_0001_automl_user_automl_model_1753845996', f"📋 Task type: {task_type}")
    
    # Define task steps based on task type
    if task_type == 'classification':
        total_steps = 8
    elif task_type == 'regression':
        total_steps = 8
    else:  # clustering
        total_steps = 6
    
    # Update initial progress
    update_progress('job_0001_automl_user_automl_model_1753845996', 0, total_steps, "Initializing AutoML Pipeline...")
    log_message('job_0001_automl_user_automl_model_1753845996', "🔧 Initializing AutoML Pipeline...")
    
    # Import appropriate AutoML class based on task type
    log_message('job_0001_automl_user_automl_model_1753845996', "📦 Importing AutoML classes...")
    if task_type == 'classification':
        from classification.automl_classifier import AutoMLClassifier
        automl_class = AutoMLClassifier
    elif task_type == 'regression':
        from regression.automl_regressor import AutoMLRegressor
        automl_class = AutoMLRegressor
    elif task_type == 'clustering':
        from clustering.automl_clusterer import AutoMLClusterer
        automl_class = AutoMLClusterer
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Create optimized Spark session with BigQuery support for BigQuery jobs
    enhanced_data_config = config.get('enhanced_data_config')
    if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery':
        log_message('job_0001_automl_user_automl_model_1753845996', "🔗 Preparing BigQuery-optimized Spark session...")
        
        # Use the EXACT proven working configuration from Jupyter
        log_message('job_0001_automl_user_automl_model_1753845996', "📦 Using proven BigQuery configuration (same as Jupyter working version)")
        
        try:
            from pyspark.sql import SparkSession
            
            # Stop any existing session cleanly
            try:
                existing_spark = SparkSession.getActiveSession()
                if existing_spark:
                    log_message('job_0001_automl_user_automl_model_1753845996', "🔄 Stopping existing Spark session for clean BigQuery setup...")
                    existing_spark.stop()
                    import time
                    time.sleep(2)  # Allow proper shutdown
            except:
                pass
            
            # Create session with EXACT proven working configuration + Large Dataset Optimizations
            optimized_spark = SparkSession.builder                 .appName(f"AutoML {task_type_title} BigQuery Job")                 .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1")                 .config("spark.driver.memory", "64g")                 .config("spark.driver.maxResultSize", "8g")                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")                 .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")                 .config("spark.executor.memory", "8g")                 .config("spark.executor.memoryFraction", "0.8")                 .config("spark.executor.memoryStorageFraction", "0.3")                 .config("spark.sql.adaptive.enabled", "true")                 .config("spark.sql.adaptive.coalescePartitions.enabled", "true")                 .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")                 .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")                 .config("spark.kryo.registrationRequired", "false")                 .config("spark.sql.execution.arrow.maxRecordsPerBatch", "5000")                 .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap")                 .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")                 .getOrCreate()
            
            log_message('job_0001_automl_user_automl_model_1753845996', "✅ BigQuery session created with proven working configuration")
            
            # Verify BigQuery connector
            try:
                test_reader = optimized_spark.read.format("bigquery")
                log_message('job_0001_automl_user_automl_model_1753845996', "✅ BigQuery connector verified and ready")
            except Exception as e:
                log_message('job_0001_automl_user_automl_model_1753845996', f"⚠️ BigQuery connector verification warning: {e}")
            
        except Exception as e:
            log_message('job_0001_automl_user_automl_model_1753845996', f"❌ Failed to create BigQuery session: {e}")
            raise
    
    # Initialize AutoML
    update_progress('job_0001_automl_user_automl_model_1753845996', 1, total_steps, "Initializing AutoML Class...")
    log_message('job_0001_automl_user_automl_model_1753845996', "🏗️ Initializing AutoML Class...")
    
    # Pass pre-created BigQuery session if available
    if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery' and 'optimized_spark' in locals():
        automl = automl_class(
            output_dir=config['output_dir'],
            config_path=config.get('config_path', 'config.yaml'),
            environment=config.get('environment', 'production'),
            preset=config.get('preset', ''),
            spark_session=optimized_spark  # Pass the BigQuery session directly
        )
        log_message('job_0001_automl_user_automl_model_1753845996', "✅ AutoML initialized with BigQuery-optimized Spark session")
    else:
        automl = automl_class(
            output_dir=config['output_dir'],
            config_path=config.get('config_path', 'config.yaml'),
            environment=config.get('environment', 'production'),
            preset=config.get('preset', '')
        )
    
    # For BigQuery jobs, the connector was already verified during session creation
    # Skip redundant test to avoid LiveListenerBus errors
    
    # Prepare fit parameters based on task type
    fit_params = config.get('model_params', {})
    fit_params.update(config.get('data_params', {}))
    fit_params.update(config.get('advanced_params', {}))
    
    # Handle enhanced data input (BigQuery, uploaded files, etc.)
    enhanced_data_config = config.get('enhanced_data_config')
    
    if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery':
        # Load data directly from BigQuery
        log_message('job_0001_automl_user_automl_model_1753845996', f"🔗 Loading data from BigQuery: {enhanced_data_config['data_source']}")
        
        # Import DataInputManager for BigQuery support
        from data_input_manager import DataInputManager
        
        # Initialize data input manager (will be updated after Spark session verification)
        data_manager = DataInputManager(
            spark=automl.spark,
            output_dir=config['output_dir'],
            user_id=config['user_id']
        )
        
        # BigQuery connector should be ready with local JAR approach
        log_message('job_0001_automl_user_automl_model_1753845996', "🔍 BigQuery connector ready - proceeding with data loading...")
        
        # Load data from BigQuery
        bigquery_options = enhanced_data_config.get('options', {})
        train_data, metadata = data_manager.load_data(
            enhanced_data_config['data_source'],
            'bigquery',
            **bigquery_options
        )
        
        log_message('job_0001_automl_user_automl_model_1753845996', f"✅ BigQuery data loaded: {metadata['row_count']} rows × {metadata['column_count']} columns")
        
        # Run the job with DataFrame directly
        if task_type in ['classification', 'regression']:
            log_message('job_0001_automl_user_automl_model_1753845996', f"🎯 Starting {task_type} training with BigQuery data...")
            
            # Handle OOT datasets if provided
            # Create OOT-specific options (same WHERE clause and column selection, but no sampling/limiting)
            oot_bigquery_options = {}
            if enhanced_data_config.get('options'):
                # Copy all options except row limiting and sampling
                for key, value in enhanced_data_config['options'].items():
                    if key not in ['row_limit', 'sample_percent']:
                        oot_bigquery_options[key] = value
                
                log_message('job_0001_automl_user_automl_model_1753845996', f"📋 OOT datasets will use same WHERE clause and column selection as training data")
                if oot_bigquery_options.get('where_clause'):
                    log_message('job_0001_automl_user_automl_model_1753845996', f"   🔍 WHERE clause: {oot_bigquery_options['where_clause']}")
                if oot_bigquery_options.get('select_columns'):
                    log_message('job_0001_automl_user_automl_model_1753845996', f"   📝 Column selection: {oot_bigquery_options['select_columns']}")
            
            # Load OOT datasets using helper function
            oot1_data, oot2_data = load_oot_datasets(config, data_manager, oot_bigquery_options)
            
            automl.fit(
                train_data=train_data,
                target_column=config['target_column'],
                oot1_data=oot1_data,
                oot2_data=oot2_data,
                **fit_params
            )
        else:  # clustering
            log_message('job_0001_automl_user_automl_model_1753845996', "🎯 Starting clustering analysis with BigQuery data...")
            automl.fit(
                train_data=train_data,
                **fit_params
            )
    
    elif enhanced_data_config and enhanced_data_config.get('source_type') == 'upload':
        # Handle uploaded files with enhanced options
        log_message('job_0001_automl_user_automl_model_1753845996', f"📁 Loading uploaded file: {config['data_file']}")
        
        # Import DataInputManager for enhanced file loading
        from data_input_manager import DataInputManager
        
        # Initialize data input manager
        data_manager = DataInputManager(
            spark=automl.spark,
            output_dir=config['output_dir'],
            user_id=config['user_id']
        )
        
        # Load data with enhanced options
        upload_options = enhanced_data_config.get('options', {})
        train_data, metadata = data_manager.load_data(
            config['data_file'],
            'upload',
            **upload_options
        )
        
        log_message('job_0001_automl_user_automl_model_1753845996', f"✅ Enhanced file data loaded: {metadata['row_count']} rows × {metadata['column_count']} columns")
        
        # Run the job with DataFrame directly
        if task_type in ['classification', 'regression']:
            log_message('job_0001_automl_user_automl_model_1753845996', f"🎯 Starting {task_type} training with enhanced file data...")
            
            # Handle OOT datasets if provided
            # Load OOT datasets using helper function
            oot1_data, oot2_data = load_oot_datasets(config, data_manager)
            
            automl.fit(
                train_data=train_data,
                target_column=config['target_column'],
                oot1_data=oot1_data,
                oot2_data=oot2_data,
                **fit_params
            )
        else:  # clustering
            log_message('job_0001_automl_user_automl_model_1753845996', "🎯 Starting clustering analysis with enhanced file data...")
            automl.fit(
                train_data=train_data,
                **fit_params
            )
    
    else:
        # Standard file loading (backward compatibility)
        log_message('job_0001_automl_user_automl_model_1753845996', f"📁 Loading data from file: {config['data_file']}")
        
        # Run the job with file path (existing behavior)
        if task_type in ['classification', 'regression']:
            log_message('job_0001_automl_user_automl_model_1753845996', f"🎯 Starting {task_type} training...")
            
            # Handle OOT datasets if provided
            # Create OOT-specific options for BigQuery datasets (same WHERE clause and column selection, but no sampling/limiting)
            oot_bigquery_options = {}
            if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery' and enhanced_data_config.get('options'):
                # Copy all options except row limiting and sampling
                for key, value in enhanced_data_config['options'].items():
                    if key not in ['row_limit', 'sample_percent']:
                        oot_bigquery_options[key] = value
                
                log_message('job_0001_automl_user_automl_model_1753845996', f"📋 OOT BigQuery datasets will use same WHERE clause and column selection as training data")
                if oot_bigquery_options.get('where_clause'):
                    log_message('job_0001_automl_user_automl_model_1753845996', f"   🔍 WHERE clause: {oot_bigquery_options['where_clause']}")
                if oot_bigquery_options.get('select_columns'):
                    log_message('job_0001_automl_user_automl_model_1753845996', f"   📝 Column selection: {oot_bigquery_options['select_columns']}")
            
            # Load OOT datasets using helper function
            oot1_data, oot2_data = load_oot_datasets(config, None, oot_bigquery_options)
            
            automl.fit(
                train_data=config['data_file'],
                target_column=config['target_column'],
                oot1_data=oot1_data,
                oot2_data=oot2_data,
                **fit_params
            )
        else:  # clustering
            log_message('job_0001_automl_user_automl_model_1753845996', "🎯 Starting clustering analysis...")
            automl.fit(
                train_data=config['data_file'],
                **fit_params
            )
    
    # Final progress update
    update_progress('job_0001_automl_user_automl_model_1753845996', total_steps, total_steps, "Training completed successfully!")
    log_message('job_0001_automl_user_automl_model_1753845996', "🎉 Training completed successfully!")
    
    # Write success status
    with open('automl_jobs/job_0001_automl_user_automl_model_1753845996_status.txt', 'w') as f:
        f.write("COMPLETED")

except Exception as e:
    # Log the full traceback
    error_details = traceback.format_exc()
    log_message('job_0001_automl_user_automl_model_1753845996', f"❌ Error occurred: {str(e)}")
    log_message('job_0001_automl_user_automl_model_1753845996', f"Full traceback:\n{error_details}")
    
    # Detect progress from logs if possible
    try:
        from pathlib import Path
        progress_step, progress_task = detect_progress_from_logs('job_0001_automl_user_automl_model_1753845996', 'classification')
        if progress_step > 0:
            update_progress('job_0001_automl_user_automl_model_1753845996', progress_step, 8, f"Failed during: {progress_task}")
        else:
            update_progress('job_0001_automl_user_automl_model_1753845996', 0, 8, "Failed during initialization")
    except:
        update_progress('job_0001_automl_user_automl_model_1753845996', 0, 8, "Failed during initialization")
    
    # Write error status
    with open('automl_jobs/job_0001_automl_user_automl_model_1753845996_status.txt', 'w') as f:
        f.write("FAILED")
    
    # Write detailed error log
    with open('automl_jobs/job_0001_automl_user_automl_model_1753845996_error.log', 'w') as f:
        f.write(f"Error: {str(e)}\n\n")
        f.write(f"Full traceback:\n{error_details}")
    
    sys.exit(1)
