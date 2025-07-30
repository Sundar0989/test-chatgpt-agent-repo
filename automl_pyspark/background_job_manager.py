"""
Background Job Manager for AutoML PySpark

This module handles background job execution independently from the Streamlit UI,
preventing UI freezing and enabling real-time log streaming.
"""

import os
import json
import time
import subprocess
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional
import signal
import sys

class BackgroundJobManager:
    """Manages background AutoML job execution."""
    
    def __init__(self, jobs_dir: str = "automl_jobs"):
        self.jobs_dir = jobs_dir
        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.job_logs: Dict[str, queue.Queue] = {}
        self.job_threads: Dict[str, threading.Thread] = {}
    
    def _clean_config_for_json(self, config: Dict) -> Dict:
        """Clean configuration to remove non-JSON-serializable objects."""
        import copy
        
        def clean_value(value):
            """Recursively clean a value to make it JSON serializable."""
            if value is None:
                return None
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            elif hasattr(value, 'name') and hasattr(value, 'size'):
                # This is likely an UploadedFile object
                return {
                    'type': 'uploaded_file',
                    'name': value.name,
                    'size': value.size,
                    'file_type': getattr(value, 'type', 'unknown')
                }
            elif hasattr(value, '__dict__'):
                # Handle other objects with attributes
                try:
                    return str(value)
                except:
                    return f"<{type(value).__name__} object>"
            else:
                # Try to copy normally
                try:
                    return copy.deepcopy(value)
                except:
                    return str(value)
        
        return clean_value(config)
        
    def start_job(self, job_id: str, config: Dict) -> bool:
        """Start a job in the background."""
        try:
            # Create jobs directory if it doesn't exist
            os.makedirs(self.jobs_dir, exist_ok=True)
            
            # Save job configuration first (clean non-serializable objects)
            config_file = os.path.join(self.jobs_dir, f"{job_id}.json")
            clean_config = self._clean_config_for_json(config)
            with open(config_file, 'w') as f:
                json.dump(clean_config, f, indent=2)
            
            # Create job script using the cleaned config
            script_content = self._create_job_script(job_id, clean_config)
            script_file = os.path.join(self.jobs_dir, f"{job_id}_script.py")
            
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Create log queue for this job
            self.job_logs[job_id] = queue.Queue()
            
            # Start job in background thread
            thread = threading.Thread(
                target=self._run_job_background,
                args=(job_id, script_file),
                daemon=True
            )
            thread.start()
            self.job_threads[job_id] = thread
            
            # Update status to Running (not Submitted)
            self._update_job_status(job_id, "Running")
            self._log_job_message(job_id, f"🚀 Job {job_id} started at {datetime.now().isoformat()}")
            
            return True
            
        except Exception as e:
            # Log detailed error information
            import traceback
            error_msg = str(e)
            traceback_msg = traceback.format_exc()
            self._log_job_message(job_id, f"❌ Failed to start job: {error_msg}")
            self._log_job_message(job_id, f"🔍 Full traceback: {traceback_msg}")
            self._update_job_status(job_id, "Failed")
            return False
    
    def _create_job_script(self, job_id: str, config: Dict) -> str:
        """Create the job execution script."""
        jobs_dir = self.jobs_dir
        job_config_file = os.path.join(self.jobs_dir, f"{job_id}.json")
        job_status_file = os.path.join(self.jobs_dir, f"{job_id}_status.txt")
        job_error_file = os.path.join(self.jobs_dir, f"{job_id}_error.log")
        
        # Helper function to load OOT datasets
        oot_loading_helper = '''
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
'''
        
        script_template = '''
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

{oot_loading_helper}

def log_message(job_id, message):
    """Log a message to the job log file."""
    log_file = os.path.join('{jobs_dir}', f"{{job_id}}_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{{timestamp}}] {{message}}\\n")

def update_progress(job_id, current_step, total_steps, current_task):
    """Update progress for the job."""
    progress_file = os.path.join('{jobs_dir}', f"{{job_id}}_progress.json")
    progress_data = {{
        'current_step': current_step,
        'total_steps': total_steps,
        'current_task': current_task,
        'progress_percentage': round((current_step / total_steps) * 100, 1),
        'timestamp': datetime.now().isoformat()
    }}
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def detect_progress_from_logs(job_id, task_type):
    """Detect progress based on log patterns."""
    log_file = os.path.join('{jobs_dir}', f"{{job_id}}_log.txt")
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
        return 0, f"Error reading logs: {{str(e)}}"

try:
    # Load job configuration
    with open('{job_config_file}') as f:
        config = json.load(f)
    
    task_type = config.get('task_type', 'classification')
    task_type_title = task_type.title()
    log_message('{job_id}', f"📋 Task type: {{task_type}}")
    
    # Define task steps based on task type
    if task_type == 'classification':
        total_steps = 8
    elif task_type == 'regression':
        total_steps = 8
    else:  # clustering
        total_steps = 6
    
    # Update initial progress
    update_progress('{job_id}', 0, total_steps, "Initializing AutoML Pipeline...")
    log_message('{job_id}', "🔧 Initializing AutoML Pipeline...")
    
    # Import appropriate AutoML class based on task type
    log_message('{job_id}', "📦 Importing AutoML classes...")
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
        raise ValueError(f"Unsupported task type: {{task_type}}")
    
    # Create optimized Spark session with BigQuery support for BigQuery jobs
    enhanced_data_config = config.get('enhanced_data_config')
    if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery':
        log_message('{job_id}', "🔗 Preparing BigQuery-optimized Spark session...")
        
        # Use the EXACT proven working configuration from Jupyter
        log_message('{job_id}', "📦 Using proven BigQuery configuration (same as Jupyter working version)")
        
        try:
            from pyspark.sql import SparkSession
            
            # Stop any existing session cleanly
            try:
                existing_spark = SparkSession.getActiveSession()
                if existing_spark:
                    log_message('{job_id}', "🔄 Stopping existing Spark session for clean BigQuery setup...")
                    existing_spark.stop()
                    import time
                    time.sleep(2)  # Allow proper shutdown
            except:
                pass
            
            # Create session with EXACT proven working configuration + Large Dataset Optimizations
            optimized_spark = SparkSession.builder \
                .appName(f"AutoML {{task_type_title}} BigQuery Job") \
                .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1") \
                .config("spark.driver.memory", "64g") \
                .config("spark.driver.maxResultSize", "8g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
                .config("spark.executor.memory", "8g") \
                .config("spark.executor.memoryFraction", "0.8") \
                .config("spark.executor.memoryStorageFraction", "0.3") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.kryo.registrationRequired", "false") \
                .config("spark.sql.execution.arrow.maxRecordsPerBatch", "5000") \
                .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap") \
                .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
                .getOrCreate()
            
            log_message('{job_id}', "✅ BigQuery session created with proven working configuration")
            
            # Verify BigQuery connector
            try:
                test_reader = optimized_spark.read.format("bigquery")
                log_message('{job_id}', "✅ BigQuery connector verified and ready")
            except Exception as e:
                log_message('{job_id}', f"⚠️ BigQuery connector verification warning: {{e}}")
            
        except Exception as e:
            log_message('{job_id}', f"❌ Failed to create BigQuery session: {{e}}")
            raise
    
    # Initialize AutoML
    update_progress('{job_id}', 1, total_steps, "Initializing AutoML Class...")
    log_message('{job_id}', "🏗️ Initializing AutoML Class...")
    
    # Pass pre-created BigQuery session if available
    if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery' and 'optimized_spark' in locals():
        automl = automl_class(
            output_dir=config['output_dir'],
            config_path=config.get('config_path', 'config.yaml'),
            environment=config.get('environment', 'production'),
            preset=config.get('preset', ''),
            spark_session=optimized_spark  # Pass the BigQuery session directly
        )
        log_message('{job_id}', "✅ AutoML initialized with BigQuery-optimized Spark session")
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
    fit_params = config.get('model_params', {{}})
    fit_params.update(config.get('data_params', {{}}))
    fit_params.update(config.get('advanced_params', {{}}))
    
    # Handle enhanced data input (BigQuery, uploaded files, etc.)
    enhanced_data_config = config.get('enhanced_data_config')
    
    if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery':
        # Load data directly from BigQuery
        log_message('{job_id}', f"🔗 Loading data from BigQuery: {{enhanced_data_config['data_source']}}")
        
        # Import DataInputManager for BigQuery support
        from data_input_manager import DataInputManager
        
        # Initialize data input manager (will be updated after Spark session verification)
        data_manager = DataInputManager(
            spark=automl.spark,
            output_dir=config['output_dir'],
            user_id=config['user_id']
        )
        
        # BigQuery connector should be ready with local JAR approach
        log_message('{job_id}', "🔍 BigQuery connector ready - proceeding with data loading...")
        
        # Load data from BigQuery
        bigquery_options = enhanced_data_config.get('options', {{}})
        train_data, metadata = data_manager.load_data(
            enhanced_data_config['data_source'],
            'bigquery',
            **bigquery_options
        )
        
        log_message('{job_id}', f"✅ BigQuery data loaded: {{metadata['row_count']}} rows × {{metadata['column_count']}} columns")
        
        # Run the job with DataFrame directly
        if task_type in ['classification', 'regression']:
            log_message('{job_id}', f"🎯 Starting {{task_type}} training with BigQuery data...")
            
            # Handle OOT datasets if provided
            # Create OOT-specific options (same WHERE clause and column selection, but no sampling/limiting)
            oot_bigquery_options = {{}}
            if enhanced_data_config.get('options'):
                # Copy all options except row limiting and sampling
                for key, value in enhanced_data_config['options'].items():
                    if key not in ['row_limit', 'sample_percent']:
                        oot_bigquery_options[key] = value
                
                log_message('{job_id}', f"📋 OOT datasets will use same WHERE clause and column selection as training data")
                if oot_bigquery_options.get('where_clause'):
                    log_message('{job_id}', f"   🔍 WHERE clause: {{oot_bigquery_options['where_clause']}}")
                if oot_bigquery_options.get('select_columns'):
                    log_message('{job_id}', f"   📝 Column selection: {{oot_bigquery_options['select_columns']}}")
            
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
            log_message('{job_id}', "🎯 Starting clustering analysis with BigQuery data...")
            automl.fit(
                train_data=train_data,
                **fit_params
            )
    
    elif enhanced_data_config and enhanced_data_config.get('source_type') == 'upload':
        # Handle uploaded files with enhanced options
        log_message('{job_id}', f"📁 Loading uploaded file: {{config['data_file']}}")
        
        # Import DataInputManager for enhanced file loading
        from data_input_manager import DataInputManager
        
        # Initialize data input manager
        data_manager = DataInputManager(
            spark=automl.spark,
            output_dir=config['output_dir'],
            user_id=config['user_id']
        )
        
        # Load data with enhanced options
        upload_options = enhanced_data_config.get('options', {{}})
        train_data, metadata = data_manager.load_data(
            config['data_file'],
            'upload',
            **upload_options
        )
        
        log_message('{job_id}', f"✅ Enhanced file data loaded: {{metadata['row_count']}} rows × {{metadata['column_count']}} columns")
        
        # Run the job with DataFrame directly
        if task_type in ['classification', 'regression']:
            log_message('{job_id}', f"🎯 Starting {{task_type}} training with enhanced file data...")
            
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
            log_message('{job_id}', "🎯 Starting clustering analysis with enhanced file data...")
            automl.fit(
                train_data=train_data,
                **fit_params
            )
    
    else:
        # Standard file loading (backward compatibility)
        log_message('{job_id}', f"📁 Loading data from file: {{config['data_file']}}")
        
        # Run the job with file path (existing behavior)
        if task_type in ['classification', 'regression']:
            log_message('{job_id}', f"🎯 Starting {{task_type}} training...")
            
            # Handle OOT datasets if provided
            # Create OOT-specific options for BigQuery datasets (same WHERE clause and column selection, but no sampling/limiting)
            oot_bigquery_options = {{}}
            if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery' and enhanced_data_config.get('options'):
                # Copy all options except row limiting and sampling
                for key, value in enhanced_data_config['options'].items():
                    if key not in ['row_limit', 'sample_percent']:
                        oot_bigquery_options[key] = value
                
                log_message('{job_id}', f"📋 OOT BigQuery datasets will use same WHERE clause and column selection as training data")
                if oot_bigquery_options.get('where_clause'):
                    log_message('{job_id}', f"   🔍 WHERE clause: {{oot_bigquery_options['where_clause']}}")
                if oot_bigquery_options.get('select_columns'):
                    log_message('{job_id}', f"   📝 Column selection: {{oot_bigquery_options['select_columns']}}")
            
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
            log_message('{job_id}', "🎯 Starting clustering analysis...")
            automl.fit(
                train_data=config['data_file'],
                **fit_params
            )
    
    # Final progress update
    update_progress('{job_id}', total_steps, total_steps, "Training completed successfully!")
    log_message('{job_id}', "🎉 Training completed successfully!")
    
    # Write success status
    with open('{job_status_file}', 'w') as f:
        f.write("COMPLETED")

except Exception as e:
    # Log the full traceback
    error_details = traceback.format_exc()
    log_message('{job_id}', f"❌ Error occurred: {{str(e)}}")
    log_message('{job_id}', f"Full traceback:\\n{{error_details}}")
    
    # Detect progress from logs if possible
    try:
        from pathlib import Path
        progress_step, progress_task = detect_progress_from_logs('{job_id}', '{task_type}')
        if progress_step > 0:
            update_progress('{job_id}', progress_step, 8, f"Failed during: {{progress_task}}")
        else:
            update_progress('{job_id}', 0, 8, "Failed during initialization")
    except:
        update_progress('{job_id}', 0, 8, "Failed during initialization")
    
    # Write error status
    with open('{job_status_file}', 'w') as f:
        f.write("FAILED")
    
    # Write detailed error log
    with open('{job_error_file}', 'w') as f:
        f.write(f"Error: {{str(e)}}\\n\\n")
        f.write(f"Full traceback:\\n{{error_details}}")
    
    sys.exit(1)
'''
        
        # Extract task type for template
        task_type = config.get('task_type', 'classification')
        task_type_title = task_type.title()
        
        return script_template.format(
            job_id=job_id,
            jobs_dir=jobs_dir,
            job_config_file=job_config_file,
            job_status_file=job_status_file,
            job_error_file=job_error_file,
            task_type=task_type,
            task_type_title=task_type_title,
            oot_loading_helper=oot_loading_helper
        )

    def _run_job_background(self, job_id: str, script_file: str):
        """Run job in background thread."""
        try:
            # Run the script
            process = subprocess.Popen(
                [sys.executable, script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.running_jobs[job_id] = process
            
            # Stream output in real-time
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        self._log_job_message(job_id, line.strip())
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                self._log_job_message(job_id, "✅ Job process completed successfully")
            else:
                self._log_job_message(job_id, f"❌ Job process failed with return code: {return_code}")
                
        except Exception as e:
            self._log_job_message(job_id, f"❌ Background job execution error: {str(e)}")
        finally:
            # Clean up
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            if job_id in self.job_threads:
                del self.job_threads[job_id]
    
    def stop_job(self, job_id: str) -> bool:
        """Stop a running job."""
        if job_id in self.running_jobs:
            try:
                process = self.running_jobs[job_id]
                process.terminate()
                self._log_job_message(job_id, "🛑 Job stopped by user")
                self._update_job_status(job_id, "Stopped")
                return True
            except Exception as e:
                self._log_job_message(job_id, f"❌ Failed to stop job: {str(e)}")
                return False
        return False
    
    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        status_file = os.path.join(self.jobs_dir, f"{job_id}_status.txt")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return f.read().strip()
        return "Submitted"
    
    def get_job_logs(self, job_id: str, max_lines: int = 100) -> List[str]:
        """Get recent job logs."""
        log_file = os.path.join(self.jobs_dir, f"{job_id}_log.txt")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return lines[-max_lines:] if len(lines) > max_lines else lines
            except Exception:
                return []
        return []
    
    def get_job_progress(self, job_id: str) -> Dict:
        """Get job progress information."""
        # First try to get progress from file
        progress_file = os.path.join(self.jobs_dir, f"{job_id}_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # If job is running, try to detect real progress from logs
                if progress_data.get('current_task') != 'Completed':
                    # Get task type from job config
                    job_config_file = os.path.join(self.jobs_dir, f"{job_id}.json")
                    if os.path.exists(job_config_file):
                        try:
                            with open(job_config_file, 'r') as f:
                                config = json.load(f)
                            task_type = config.get('task_type', 'classification')
                            
                            # Detect progress from logs
                            current_step, current_task = self._detect_progress_from_logs(job_id, task_type)
                            
                            # Update progress if we found more recent activity
                            if current_step > progress_data.get('current_step', 0):
                                progress_data['current_step'] = current_step
                                progress_data['current_task'] = current_task
                                
                                # Calculate progress percentage with special handling for hyperparameter tuning
                                if "Hyperparameter Tuning" in current_task:
                                    # Extract tuning progress if available
                                    import re
                                    tuning_progress = re.search(r'(\d+)%', current_task)
                                    if tuning_progress:
                                        tuning_pct = int(tuning_progress.group(1))
                                        # Base progress is step 5 (Model Building), add tuning progress
                                        base_progress = (5 / progress_data.get('total_steps', 8)) * 100
                                        tuning_adjustment = (tuning_pct / 100) * 0.2  # Tuning is ~20% of total time
                                        progress_data['progress_percentage'] = round(base_progress + tuning_adjustment, 1)
                                    else:
                                        progress_data['progress_percentage'] = round((current_step / progress_data.get('total_steps', 8)) * 100, 1)
                                else:
                                    progress_data['progress_percentage'] = round((current_step / progress_data.get('total_steps', 8)) * 100, 1)
                                
                                progress_data['timestamp'] = datetime.now().isoformat()
                                
                                # Save updated progress
                                with open(progress_file, 'w') as f:
                                    json.dump(progress_data, f, indent=2)
                        except:
                            pass
                
                return progress_data
            except:
                pass
        
        # Fallback to log-based detection
        job_config_file = os.path.join(self.jobs_dir, f"{job_id}.json")
        if os.path.exists(job_config_file):
            try:
                with open(job_config_file, 'r') as f:
                    config = json.load(f)
                task_type = config.get('task_type', 'classification')
                current_step, current_task = self._detect_progress_from_logs(job_id, task_type)
                
                total_steps = 8 if task_type in ['classification', 'regression'] else 6
                return {
                    'current_step': current_step,
                    'total_steps': total_steps,
                    'current_task': current_task,
                    'progress_percentage': round((current_step / total_steps) * 100, 1),
                    'timestamp': datetime.now().isoformat()
                }
            except:
                pass
        
        return {
            'current_step': 0,
            'total_steps': 8,
            'current_task': 'Initializing...',
            'progress_percentage': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_progress_from_logs(self, job_id: str, task_type: str) -> tuple:
        """Detect progress based on log patterns."""
        log_file = os.path.join(self.jobs_dir, f"{job_id}_log.txt")
        if not os.path.exists(log_file):
            return 0, "Initializing..."
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Define progress patterns for different task types
            if task_type == 'classification':
                progress_patterns = [
                    (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing", "1. Data Preprocessing", "Loading and processing data"]),
                    (2, "Feature Selection", ["Feature Selection", "feature selection", "selecting features", "2. Feature Selection"]),
                    (3, "Data Splitting and Scaling", ["Data Splitting", "scaling", "split", "train/valid", "3. Data Splitting"]),
                    (4, "Preparing Out-of-Time Datasets", ["Out-of-Time", "OOT", "oot", "4. Preparing Out-of-Time"]),
                    (5, "Model Building and Validation", ["Model Building", "training", "XGBoost", "Random Forest", "Logistic", "Gradient", "5. Model Building", "Building models", "Training", "Building clustering models"]),
                    (6, "Model Selection", ["Model Selection", "selecting best", "best model", "6. Model Selection", "Selecting best", "Selecting best clustering model"]),
                    (7, "Generating Scoring Code", ["Generating Scoring", "scoring code", "save model", "7. Generating Scoring", "Generating scoring scripts"]),
                    (8, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully", "8. Save model", "Saving model", "Complete model saved"])
                ]
            elif task_type == 'regression':
                progress_patterns = [
                    (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing", "1. Data Preprocessing", "Loading and processing data"]),
                    (2, "Feature Selection", ["Feature Selection", "feature selection", "selecting features", "2. Feature Selection"]),
                    (3, "Data Splitting and Scaling", ["Data Splitting", "scaling", "split", "train/valid", "3. Data Splitting"]),
                    (4, "Preparing Out-of-Time Datasets", ["Out-of-Time", "OOT", "oot", "4. Preparing Out-of-Time"]),
                    (5, "Model Building and Validation", ["Model Building", "training", "Linear Regression", "Random Forest", "Gradient", "5. Model Building", "Building models", "Training"]),
                    (6, "Model Selection", ["Model Selection", "selecting best", "best model", "6. Model Selection", "Selecting best"]),
                    (7, "Generating Scoring Code", ["Generating Scoring", "scoring code", "save model", "7. Generating Scoring", "Generating scoring scripts"]),
                    (8, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully", "8. Save model", "Saving model", "Complete model saved"])
                ]
            else:  # clustering
                progress_patterns = [
                    (1, "Data Preprocessing", ["Data Preprocessing", "preprocessing", "Preprocessing", "1. Data Preprocessing", "Loading and processing data", "Loading and processing data"]),
                    (2, "Feature Scaling", ["Feature Scaling", "scaling", "normalize", "2. Feature Scaling", "Feature selection"]),
                    (3, "Clustering Analysis", ["Clustering Analysis", "K-Means", "DBSCAN", "clustering", "3. Clustering Analysis", "Building clustering models", "Training clustering models"]),
                    (4, "Model Building and Validation", ["Model Building", "training", "cluster", "4. Model Building", "Building models", "Training"]),
                    (5, "Model Selection", ["Model Selection", "selecting best", "best model", "5. Model Selection", "Selecting best clustering model", "Selecting best"]),
                    (6, "Saving Model Configuration", ["Saving Model", "model saved", "completed successfully", "6. Save model", "Saving model", "Complete clustering model saved"])
                ]
            
            # Check for completion first
            if "completed successfully" in log_content.lower() or "job completed successfully" in log_content.lower():
                return len(progress_patterns), "Completed"
            
            # Check for hyperparameter tuning progress
            if "best trial:" in log_content.lower() and "100%" in log_content.lower():
                # If hyperparameter tuning is complete, we're likely in model building/validation phase
                if task_type in ['classification', 'regression']:
                    return 5, "Model Building and Validation (Hyperparameter Tuning Complete)"
                else:
                    return 4, "Model Building and Validation (Hyperparameter Tuning Complete)"
            
            # Check for active hyperparameter tuning
            if "best trial:" in log_content.lower() and any(f"{i}%" in log_content for i in range(10, 100, 10)):
                # Extract progress percentage from tuning logs
                import re
                progress_matches = re.findall(r'(\d+)%', log_content)
                if progress_matches:
                    latest_progress = max(int(p) for p in progress_matches)
                    if task_type in ['classification', 'regression']:
                        return 5, f"Model Building and Validation (Hyperparameter Tuning: {latest_progress}%)"
                    else:
                        return 4, f"Model Building and Validation (Hyperparameter Tuning: {latest_progress}%)"
            
            # Find the highest completed step
            current_step = 0
            current_task = "Initializing..."
            
            for step, task, patterns in progress_patterns:
                if any(pattern.lower() in log_content.lower() for pattern in patterns):
                    current_step = step
                    current_task = task
            
            return current_step, current_task
            
        except Exception:
            return 0, "Initializing..."
    
    def _update_job_status(self, job_id: str, status: str):
        """Update job status."""
        status_file = os.path.join(self.jobs_dir, f"{job_id}_status.txt")
        with open(status_file, 'w') as f:
            f.write(status)
    
    def _log_job_message(self, job_id: str, message: str):
        """Log a message for the job."""
        log_file = os.path.join(self.jobs_dir, f"{job_id}_log.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        
        # Also add to queue for real-time access
        if job_id in self.job_logs:
            try:
                self.job_logs[job_id].put_nowait(message)
            except queue.Full:
                pass  # Queue is full, skip
    
    def get_running_jobs(self) -> List[str]:
        """Get list of currently running job IDs."""
        return list(self.running_jobs.keys())
    
    def cleanup_completed_jobs(self):
        """Clean up completed job references."""
        completed_jobs = []
        for job_id in list(self.running_jobs.keys()):
            if self.get_job_status(job_id) in ['Completed', 'Failed', 'Stopped']:
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            if job_id in self.job_threads:
                del self.job_threads[job_id]
            if job_id in self.job_logs:
                del self.job_logs[job_id]

# Global job manager instance
job_manager = BackgroundJobManager() 