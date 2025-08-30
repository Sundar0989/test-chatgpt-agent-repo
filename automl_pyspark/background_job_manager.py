"""
Background Job Manager for AutoML PySpark

This module handles background job execution independently from the Streamlit UI,
preventing UI freezing and enabling real-time log streaming.

Supports three execution modes:
1. Local background threads (default)
2. External message queue (Pub/Sub or Cloud Tasks)
3. Dataproc Serverless (Google Cloud managed Spark)
"""

import os
import json
import time
import subprocess
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Union
import signal
import sys

class BackgroundJobManager:
    """
    Manages background AutoML job execution.  By default jobs are
    executed in local background threads.  Optionally, jobs can be
    dispatched to an external message queue (e.g. Google Cloud Pub/Sub
    or Cloud Tasks) or to Dataproc Serverless for managed Spark execution.
    
    Execution modes:
    - Local: Set no environment variables (default)
    - Queue: Set USE_GCP_QUEUE=true
    - Dataproc: Set ENABLE_DATAPROC_SERVERLESS=true
    """

    def __init__(self, jobs_dir: str = "automl_jobs", use_gcp_queue: Union[bool, None] = None):
        self.jobs_dir = jobs_dir
        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.job_logs: Dict[str, queue.Queue] = {}
        self.job_threads: Dict[str, threading.Thread] = {}
        
        # Determine execution mode from environment variables
        env_flag = os.getenv("USE_GCP_QUEUE", "false").lower() in ("1", "true", "yes")
        self.use_gcp_queue = use_gcp_queue if use_gcp_queue is not None else env_flag
        
        # Check if Dataproc Serverless is enabled
        self.use_dataproc_serverless = os.getenv("ENABLE_DATAPROC_SERVERLESS", "false").lower() in ("1", "true", "yes")
        
        # Initialize Dataproc Serverless manager if enabled
        self.dataproc_manager = None
        if self.use_dataproc_serverless:
            try:
                from dataproc_serverless_manager import DataprocServerlessManager
                self.dataproc_manager = DataprocServerlessManager()
                print("✅ Dataproc Serverless Manager initialized")
            except ImportError as e:
                print(f"⚠️ Dataproc Serverless Manager not available: {e}")
                self.use_dataproc_serverless = False
            except Exception as e:
                print(f"❌ Failed to initialize Dataproc Serverless Manager: {e}")
                self.use_dataproc_serverless = False
        
        # Log current execution mode
        if self.use_dataproc_serverless:
            print("🚀 Execution mode: Dataproc Serverless")
        elif self.use_gcp_queue:
            print("📤 Execution mode: External Queue (Pub/Sub/Cloud Tasks)")
        else:
            print("🏠 Execution mode: Local Background Threads")
    
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

            if self.use_gcp_queue:
                # When using an external queue, publish the job configuration
                # instead of launching it locally.  The worker environment
                # should retrieve the config file from jobs_dir and execute
                # the job.  Publishing is wrapped in a try/except to avoid
                # crashing the UI if credentials are not configured.  Logs
                # will reflect whether publishing was attempted.
                try:
                    self._publish_job_to_queue(job_id, clean_config)
                    self._update_job_status(job_id, "Submitted")
                    self._log_job_message(job_id, f"📤 Job {job_id} published to message queue at {datetime.now().isoformat()}")
                except Exception as e:
                    self._log_job_message(job_id, f"❌ Failed to publish job to queue: {e}")
                    self._update_job_status(job_id, "Failed")
                    return False
            elif self.use_dataproc_serverless:
                # When using Dataproc Serverless, submit the job to Dataproc
                try:
                    # Prepare data files for Dataproc submission
                    data_files = self._extract_data_files_from_config(clean_config)
                    
                    # Submit to Dataproc Serverless
                    batch_id = self.dataproc_manager.submit_spark_job(
                        job_config=clean_config,
                        job_id=job_id,
                        data_files=data_files
                    )
                    
                    self._update_job_status(job_id, "Submitted")
                    self._log_job_message(job_id, f"🚀 Job {job_id} submitted to Dataproc Serverless (Batch ID: {batch_id}) at {datetime.now().isoformat()}")
                    
                    # Store batch ID for tracking
                    self._store_dataproc_job_info(job_id, batch_id, clean_config)
                    
                except Exception as e:
                    self._log_job_message(job_id, f"❌ Failed to submit job to Dataproc Serverless: {e}")
                    self._update_job_status(job_id, "Failed")
                    return False
            else:
                # Start job in background thread locally
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
            # Log error and update status
            self._log_job_message(job_id, f"❌ Failed to start job: {e}")
            self._update_job_status(job_id, "Failed")
            return False

    def _publish_job_to_queue(self, job_id: str, config: Dict) -> None:
        """
        Publish a job configuration to an external message queue.  The
        method supports two queue backends: Google Cloud Pub/Sub and
        Cloud Tasks.  When the environment variable ``USE_GCP_TASKS``
        is set to ``true`` (case-insensitive), the job will be
        dispatched via Cloud Tasks.  Otherwise Pub/Sub is used by
        default.  Required environment variables for each mode are
        documented below.

        **Pub/Sub Mode (default)**
            Set ``GCP_PUBSUB_TOPIC`` to the full topic path (e.g.
            ``projects/my-project/topics/my-topic``).  The
            ``google-cloud-pubsub`` library must be installed.

        **Cloud Tasks Mode**
            Set ``USE_GCP_TASKS=true`` and provide the following
            variables:

            * ``GCP_TASKS_PROJECT`` – project ID where the queue
              resides.
            * ``GCP_TASKS_LOCATION`` – location/region of the queue.
            * ``GCP_TASKS_QUEUE`` – name of the queue.
            * ``CLOUD_RUN_BASE_URL`` – base URL of the Cloud Run
              service that will process tasks.
            * ``SERVICE_ACCOUNT_EMAIL`` – (optional) service account
              email used for the OIDC token.  If omitted the email is
              inferred from the service account key.

        The job configuration is JSON serialised and included in the
        message or HTTP task body along with the job ID.
        """
        try:
            import json
            # Determine whether to use Cloud Tasks
            use_tasks = os.getenv('USE_GCP_TASKS', 'false').lower() in ('1', 'true', 'yes')
            if use_tasks:
                # Defer to Cloud Tasks helper.  Import lazily to avoid
                # dependency if unused.
                try:
                    from automl_pyspark.gcp_helpers import create_http_task  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "Cloud Tasks dispatch requested but google-cloud-tasks is not installed. "
                        "Install via 'pip install google-cloud-tasks' and set USE_GCP_TASKS=false to fall back to Pub/Sub."
                    ) from e
                project = os.getenv('GCP_TASKS_PROJECT')
                location = os.getenv('GCP_TASKS_LOCATION')
                queue = os.getenv('GCP_TASKS_QUEUE')
                base_url = os.getenv('CLOUD_RUN_BASE_URL')
                if not all([project, location, queue, base_url]):
                    raise RuntimeError(
                        "GCP_TASKS_PROJECT, GCP_TASKS_LOCATION, GCP_TASKS_QUEUE and CLOUD_RUN_BASE_URL must be set "
                        "when USE_GCP_TASKS is true."
                    )
                # Determine the target path from config.  For clustering,
                # classification and regression jobs, we post to a generic
                # '/run-job' endpoint.  Downstream services can inspect the
                # payload to decide how to handle the job.
                target_path = os.getenv('GCP_TASKS_TARGET_PATH', '/run-job')
                service_account_email = os.getenv('SERVICE_ACCOUNT_EMAIL')
                payload = {'job_id': job_id, 'config': config}
                # Create HTTP task
                response = create_http_task(
                    project=project,
                    location=location,
                    queue=queue,
                    target_path=target_path,
                    json_payload=payload,
                    task_id=job_id,
                    service_account_email=service_account_email or None
                )
                # We don't wait for result; Cloud Tasks returns immediately
                return
            else:
                # Use Pub/Sub
                try:
                    from google.cloud import pubsub_v1  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "google-cloud-pubsub library is not installed or configured. "
                        "Install via 'pip install google-cloud-pubsub' or set USE_GCP_TASKS=true to use Cloud Tasks."
                    ) from e
                topic_path = os.getenv('GCP_PUBSUB_TOPIC')
                if not topic_path:
                    raise RuntimeError("GCP_PUBSUB_TOPIC environment variable must be set when using Pub/Sub dispatch.")
                publisher = pubsub_v1.PublisherClient()
                message_bytes = json.dumps({'job_id': job_id, 'config': config}).encode('utf-8')
                future = publisher.publish(topic_path, message_bytes, job_id=job_id)
                future.result(timeout=10)
                
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
        if oot_bigquery_options:
            # Use direct Spark-BigQuery connector for optimal performance
            try:
                # Always use table option for better compatibility
                oot1_data = optimized_spark.read.format("bigquery").option("table", config['oot1_bigquery_table']).option("viewsEnabled", "true").load()
                
                # Apply custom options after loading if provided
                if oot_bigquery_options:
                    log_message('{job_id}', f"🔧 Applying custom options to OOT1 data...")
                    
                    # Apply WHERE clause if specified
                    if oot_bigquery_options.get('where_clause'):
                        where_clause = oot_bigquery_options['where_clause']
                        oot1_data = oot1_data.filter(where_clause)
                    
                    # Apply column selection if specified
                    if oot_bigquery_options.get('select_columns'):
                        select_columns = oot_bigquery_options['select_columns']
                        if select_columns and select_columns.strip() != '*':
                            columns = [col.strip() for col in select_columns.split(',') if col.strip()]
                            oot1_data = oot1_data.select(columns)
                
                log_message('{job_id}', f"✅ OOT1 BigQuery data loaded successfully using direct connector")
            except Exception as e:
                log_message('{job_id}', f"⚠️ OOT1 BigQuery loading failed: {{str(e)}} - will skip OOT1")
                oot1_data = None
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
            if oot_bigquery_options:
                # Use direct Spark-BigQuery connector for optimal performance
                try:
                    # Always use table option for better compatibility
                    oot1_data = optimized_spark.read.format("bigquery").option("table", oot1_config['data_source']).option("viewsEnabled", "true").load()
                    
                    # Apply custom options after loading if provided
                    if oot_bigquery_options:
                        log_message('{job_id}', f"🔧 Applying custom options to OOT1 data...")
                        
                        # Apply WHERE clause if specified
                        if oot_bigquery_options.get('where_clause'):
                            where_clause = oot_bigquery_options['where_clause']
                            oot1_data = oot1_data.filter(where_clause)
                        
                        # Apply column selection if specified
                        if oot_bigquery_options.get('select_columns'):
                            select_columns = oot_bigquery_options['select_columns']
                            if select_columns and select_columns.strip() != '*':
                                columns = [col.strip() for col in select_columns.split(',') if col.strip()]
                                oot1_data = oot1_data.select(columns)
                    
                    log_message('{job_id}', f"✅ OOT1 BigQuery data loaded successfully using direct connector")
                except Exception as e:
                    log_message('{job_id}', f"⚠️ OOT1 BigQuery loading failed: {{str(e)}} - will skip OOT1")
                    oot1_data = None
            else:
                oot1_data = oot1_config['data_source']  # Pass as string for backward compatibility
    
    # Handle OOT2 - check for BigQuery table first, then file
    if config.get('oot2_bigquery_table'):
        log_message('{job_id}', f"📅 Loading OOT2 data from BigQuery: {{config['oot2_bigquery_table']}}")
        if oot_bigquery_options:
            # Use direct Spark-BigQuery connector for optimal performance
            try:
                # Always use table option for better compatibility
                oot2_data = optimized_spark.read.format("bigquery").option("table", config['oot2_bigquery_table']).option("viewsEnabled", "true").load()
                
                # Apply custom options after loading if provided
                if oot_bigquery_options:
                    log_message('{job_id}', f"🔧 Applying custom options to OOT2 data...")
                    
                    # Apply WHERE clause if specified
                    if oot_bigquery_options.get('where_clause'):
                        where_clause = oot_bigquery_options['where_clause']
                        oot2_data = oot2_data.filter(where_clause)
                    
                    # Apply column selection if specified
                    if oot_bigquery_options.get('select_columns'):
                        select_columns = oot_bigquery_options['select_columns']
                        if select_columns and select_columns.strip() != '*':
                            columns = [col.strip() for col in select_columns.split(',') if col.strip()]
                            oot2_data = oot2_data.select(columns)
                log_message('{job_id}', f"✅ OOT2 BigQuery data loaded successfully using direct connector")
            except Exception as e:
                log_message('{job_id}', f"⚠️ OOT2 BigQuery loading failed: {{str(e)}} - will skip OOT2")
                oot2_data = None
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
            if oot_bigquery_options:
                # Use direct Spark-BigQuery connector for optimal performance
                try:
                    # Always use table option for better compatibility
                    oot2_data = optimized_spark.read.format("bigquery").option("table", oot2_config['data_source']).option("viewsEnabled", "true").load()
                    
                    # Apply custom options after loading if provided
                    if oot_bigquery_options:
                        log_message('{job_id}', f"🔧 Applying custom options to OOT2 data...")
                        
                        # Apply WHERE clause if specified
                        if oot_bigquery_options.get('where_clause'):
                            where_clause = oot_bigquery_options['where_clause']
                            oot2_data = oot2_data.filter(where_clause)
                        
                        # Apply column selection if specified
                        if oot_bigquery_options.get('select_columns'):
                            select_columns = oot_bigquery_options['select_columns']
                            if select_columns and select_columns.strip() != '*':
                                columns = [col.strip() for col in select_columns.split(',') if col.strip()]
                                oot2_data = oot2_data.select(columns)
                    log_message('{job_id}', f"✅ OOT2 BigQuery data loaded successfully using direct connector")
                except Exception as e:
                    log_message('{job_id}', f"⚠️ OOT2 BigQuery loading failed: {{str(e)}} - will skip OOT2")
                    oot2_data = None
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

# Define job_id early to prevent NameError
job_id = os.environ.get("JOB_ID", "unknown_job")

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
    
    # Create optimized Spark session for ALL jobs (BigQuery and non-BigQuery)
    enhanced_data_config = config.get('enhanced_data_config')
    is_bigquery_job = enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery'
    
    if is_bigquery_job:
        log_message('{job_id}', "🔗 Preparing BigQuery-optimized Spark session...")
        log_message('{job_id}', "📦 Using proven BigQuery configuration (same as Jupyter working version)")
    else:
        log_message('{job_id}', "🔧 Preparing optimized Spark session for standard jobs...")
        log_message('{job_id}', "📦 Using comprehensive optimization configuration")
    
    try:
        from pyspark.sql import SparkSession
        
        # Use robust Spark session creation with better RPC handling
        log_message('{job_id}', "🔧 Creating robust Spark session with RPC error handling...")
        
        # Add parent directory to Python path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            log_message('{job_id}', f"🔍 Added parent directory to Python path: {{parent_dir}}")
        
        # Also add the current directory to the path
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            log_message('{job_id}', f"🔍 Added current directory to Python path: {{current_dir}}")
        
        # Use proven BigQuery connection logic for all jobs
        log_message('{job_id}', "🔧 Using proven BigQuery connection logic for session creation...")
        
        # Stop any existing session cleanly
        try:
            existing_spark = SparkSession.getActiveSession()
            if existing_spark:
                log_message('{job_id}', "🔄 Stopping existing Spark session for clean setup...")
                existing_spark.stop()
                import time
                time.sleep(2)  # Allow proper shutdown
        except:
            pass
        
        # Prepare Maven package configurations
        packages_list = []
        repositories_list = []
        
        # Add BigQuery-specific configurations for BigQuery jobs
        if is_bigquery_job:
            # Use bundled BigQuery connector JAR for optimal performance with large datasets
            # This approach loads data directly into Spark DataFrame without memory issues
            log_message('{job_id}', "🔗 Using bundled BigQuery connector JAR for optimal performance")
            log_message('{job_id}', "📦 No Maven packages needed - using bundled JARs")
        
        # Add SynapseML packages if LightGBM is being used
        run_lightgbm = config.get('model_params', {{}}).get('run_lightgbm', False)
        if run_lightgbm:
            log_message('{job_id}', "📦 Adding SynapseML packages from Maven for LightGBM support...")
            packages_list.append("com.microsoft.azure:synapseml_2.12:0.11.4")
            repositories_list.append("https://mmlspark.azureedge.net/maven")
            log_message('{job_id}', "✅ Added SynapseML Maven package with LightGBM support")
        
        # Create Spark session using proven test configuration
        log_message('{job_id}', "🔧 Creating Spark session with proven test configuration...")
        
        spark_builder = SparkSession.builder \
            .appName(f"AutoML {{task_type_title}} Job") \
            .master("local[*]") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.driver.port", "0") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.instances", "1") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.sql.adaptive.skewJoin.enabled", "false") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "false") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "200") \
            .config("spark.network.timeout", "800s") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .config("spark.rpc.askTimeout", "800s") \
            .config("spark.rpc.lookupTimeout", "800s") \
            .config("spark.local.dir", "/tmp") \
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
            .config("spark.sql.catalogImplementation", "hive") \
            .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
            .config("spark.hadoop.fs.gs.auth.service.account.enable", "false") \
            .config("spark.hadoop.fs.gs.auth.impersonation.service.account.enable", "false") \
            .config("spark.driver.allowMultipleContexts", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.sql.adaptive.skewJoin.enabled", "false") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "false")
        
        # Add BigQuery-specific configurations if this is a BigQuery job
        if is_bigquery_job:
            spark_builder = spark_builder \
                .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
                .config("spark.hadoop.fs.gs.auth.service.account.enable", "false") \
                .config("spark.hadoop.fs.gs.auth.impersonation.service.account.enable", "false") \
                .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
                .config("spark.driver.maxResultSize", "4g") \
                .config("spark.hadoop.fs.gs.ssl.enabled", "false") \
                .config("spark.hadoop.fs.gs.ssl.truststore", "/etc/ssl/certs/java/cacerts") \
                .config("spark.hadoop.fs.gs.ssl.keystore.password", "changeit")
            log_message('{job_id}', "✅ Added BigQuery-specific configurations")
        
        # Configure JAR files for BigQuery and SynapseML
        JAR_DIR = "/app/automl_pyspark/libs"
        BIGQUERY_JAR = f"{{JAR_DIR}}/spark-bigquery-with-dependencies_2.12-0.36.1.jar"
        
        # Build list of JARs to include
        all_jars = [BIGQUERY_JAR]
        
        # Add SynapseML JARs if LightGBM is being used
        if config.get('model_params', {{}}).get('run_lightgbm', False):
            SYNAPSEML_JARS = [
                f"{{JAR_DIR}}/synapseml_2.12-0.11.4.jar",
                f"{{JAR_DIR}}/synapseml-core_2.12-0.11.4.jar",
                f"{{JAR_DIR}}/synapseml-lightgbm_2.12-0.11.4.jar",
                f"{{JAR_DIR}}/lightgbm-3.3.5.jar",
                f"{{JAR_DIR}}/spray-json_2.12-1.3.5.jar",
            ]
            all_jars.extend(SYNAPSEML_JARS)
            log_message('{job_id}', f"📦 Added SynapseML JARs for LightGBM support")
        
        # Configure Spark to use local JARs
        if all_jars:
            jar_paths = ",".join(all_jars)
            spark_builder = spark_builder \
                .config("spark.jars", jar_paths) \
                .config("spark.driver.extraClassPath", jar_paths) \
                .config("spark.executor.extraClassPath", jar_paths)
            log_message('{job_id}', f"📦 Configured {{len(all_jars)}} local JARs: {{jar_paths}}")
        
        # Add Maven packages if any (for other dependencies)
        if packages_list:
            all_packages = ",".join(packages_list)
            spark_builder = spark_builder.config("spark.jars.packages", all_packages)
            log_message('{job_id}', f"📦 Configured {{len(packages_list)}} Maven packages: {{all_packages}}")
        
        # Add SSL configurations for BigQuery (using local JARs, no Maven downloads)
        if is_bigquery_job:
            spark_builder = spark_builder \
                .config("spark.hadoop.fs.gs.ssl.enabled", "false") \
                .config("spark.hadoop.fs.gs.ssl.truststore", "/etc/ssl/certs/java/cacerts") \
                .config("spark.hadoop.fs.gs.ssl.keystore", "/etc/ssl/certs/java/cacerts") \
                .config("spark.hadoop.fs.gs.ssl.keystore.password", "changeit")
            log_message('{job_id}', "✅ Added SSL configurations for BigQuery (using local JARs)")
        
        # Create the session with error handling and fallback
        try:
            log_message('{job_id}', "🔧 Attempting to create Spark session...")
            optimized_spark = spark_builder.getOrCreate()
            
            if is_bigquery_job:
                log_message('{job_id}', "✅ BigQuery session created with proven test configuration")
            else:
                log_message('{job_id}', "✅ Spark session created with proven test configuration")
                
        except Exception as e:
            log_message('{job_id}', f"⚠️ Failed to create Spark session with full config: {{e}}")
            log_message('{job_id}', "🔄 Attempting fallback configuration...")
            
            # Fallback to minimal configuration
            fallback_builder = SparkSession.builder \
                .appName(f"AutoML {{task_type_title}} Job") \
                .master("local[1]") \
                .config("spark.driver.bindAddress", "127.0.0.1") \
                .config("spark.driver.host", "127.0.0.1") \
                .config("spark.driver.memory", "4g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.local.dir", "/tmp")
            
            try:
                optimized_spark = fallback_builder.getOrCreate()
                log_message('{job_id}', "✅ Spark session created with fallback configuration")
            except Exception as fallback_error:
                log_message('{job_id}', f"❌ Failed to create Spark session even with fallback: {{fallback_error}}")
                raise fallback_error
        
        # Wait for packages to be fully loaded and registered in JVM
        log_message('{job_id}', "⏳ Waiting for packages to be fully registered in JVM...")
        
        # For BigQuery jobs, we'll use direct Spark-BigQuery connector for optimal performance
        # This loads data directly into Spark DataFrame without memory issues
        if is_bigquery_job:
            log_message('{job_id}', "🔗 BigQuery data will be loaded using direct Spark-BigQuery connector")
            log_message('{job_id}', "✅ Direct loading provides optimal performance for large datasets")
        
        # Add a delay to allow any packages to be loaded
        import time
        time.sleep(2)  # Short delay for any remaining package loading
        
        # For LightGBM jobs, ensure packages are fully loaded before proceeding
        if config.get('model_params', {{}}).get('run_lightgbm', False):
            log_message('{job_id}', "🔍 Verifying LightGBM availability before starting data processing...")
            
            # Simple verification with fewer retries
            max_retries = 5  # Reduced retries for simplicity
            retry_count = 0
            lightgbm_ready = False
            
            while retry_count < max_retries and not lightgbm_ready:
                try:
                    # Test LightGBM availability
                    from synapse.ml.lightgbm import LightGBMClassifier
                    LightGBMClassifier(featuresCol="features", labelCol="label")
                    lightgbm_ready = True
                    log_message('{job_id}', f"✅ LightGBM is available and working (attempt {{retry_count + 1}})")
                except Exception as e:
                    retry_count += 1
                    log_message('{job_id}', f"⏳ LightGBM not ready yet (attempt {{retry_count}}/{{max_retries}}): {{e}}")
                    if retry_count < max_retries:
                        time.sleep(5)  # Reduced wait time
            
            if not lightgbm_ready:
                log_message('{job_id}', f"❌ LightGBM failed to load after {{max_retries}} attempts")
                log_message('{job_id}', "🛑 Stopping job - LightGBM is required but not available")
                raise RuntimeError("LightGBM is required but not available in JVM after multiple attempts")
            else:
                log_message('{job_id}', "🎉 LightGBM verification successful - proceeding with data processing")
        
        # For BigQuery jobs, we'll use direct Spark-BigQuery connector for optimal performance
        # This loads data directly into Spark DataFrame without memory issues
        if is_bigquery_job:
            log_message('{job_id}', "🔗 BigQuery data will be loaded using direct Spark-BigQuery connector")
            log_message('{job_id}', "✅ Direct loading provides optimal performance for large datasets")
    
    except Exception as e:
        log_message('{job_id}', f"❌ Failed to create optimized Spark session: {{e}}")
        raise
    
    # Initialize AutoML with optimized Spark session for ALL jobs
    update_progress('{job_id}', 1, total_steps, "Initializing AutoML Class...")
    log_message('{job_id}', "🏗️ Initializing AutoML Class...")
    
    # Use the optimized Spark session for ALL jobs (BigQuery and non-BigQuery)
    automl = automl_class(
        output_dir=config['output_dir'],
        config_path=config.get('config_path', 'config.yaml'),
        environment=config.get('environment', 'production'),
        preset=config.get('preset', ''),
        spark_session=optimized_spark  # Pass the optimized session for ALL jobs
    )
    
    if is_bigquery_job:
        log_message('{job_id}', "✅ AutoML initialized with simplified BigQuery configuration")
    else:
        log_message('{job_id}', "✅ AutoML initialized with simplified Spark configuration")
    
    # For BigQuery jobs, the connector was already verified during session creation
    # For all jobs, Spark optimizations were applied during session creation
    
    # Prepare fit parameters based on task type
    fit_params = config.get('model_params', {{}})
    fit_params.update(config.get('data_params', {{}}))
    fit_params.update(config.get('advanced_params', {{}}))
    
    # Handle enhanced data input (BigQuery, uploaded files, etc.)
    enhanced_data_config = config.get('enhanced_data_config')
    
    if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery':
        # Load data directly from BigQuery using Spark-BigQuery connector
        # This provides optimal performance for large datasets without memory issues
        log_message('{job_id}', f"🔗 Loading data from BigQuery: {{enhanced_data_config['data_source']}}")
        
        try:
            # Use direct Spark-BigQuery connector for optimal performance
            log_message('{job_id}', "✅ Using direct Spark-BigQuery connector for optimal performance")
            
            # Get table reference and options
            table_ref = enhanced_data_config['data_source']
            bigquery_options = enhanced_data_config.get('options', {{}})
            
            # Always use table option for better compatibility and performance
            # Custom options like WHERE clauses, column selection, etc. will be applied after loading
            log_message('{job_id}', f"📥 Loading BigQuery data directly from table: {{table_ref}}")
            train_data = optimized_spark.read.format("bigquery").option("table", table_ref).option("viewsEnabled", "true").load()
            
            # Apply custom options after loading if provided
            if bigquery_options:
                log_message('{job_id}', f"🔧 Applying custom options after loading...")
                
                # Apply WHERE clause if specified
                if bigquery_options.get('where_clause'):
                    where_clause = bigquery_options['where_clause']
                    log_message('{job_id}', f"   🔍 Applying WHERE clause: {{where_clause}}")
                    train_data = train_data.filter(where_clause)
                
                # Apply column selection if specified
                if bigquery_options.get('select_columns'):
                    select_columns = bigquery_options['select_columns']
                    if select_columns and select_columns.strip() != '*':
                        # Parse comma-separated columns
                        columns = [col.strip() for col in select_columns.split(',') if col.strip()]
                        log_message('{job_id}', f"   📝 Selecting columns: {{columns}}")
                        train_data = train_data.select(columns)
                
                # Apply row limit if specified
                if bigquery_options.get('row_limit'):
                    row_limit = bigquery_options['row_limit']
                    log_message('{job_id}', f"   📊 Limiting to {{row_limit}} rows")
                    train_data = train_data.limit(row_limit)
                
                # Apply sampling if specified
                if bigquery_options.get('sample_percent'):
                    sample_percent = bigquery_options['sample_percent'] / 100.0
                    log_message('{job_id}', f"   🎲 Applying {{sample_percent*100}}% sampling")
                    train_data = train_data.sample(False, sample_percent)
            
            # Get metadata
            row_count = train_data.count()
            col_count = len(train_data.columns)
            metadata = {{'row_count': row_count, 'column_count': col_count}}
            
            log_message('{job_id}', f"✅ BigQuery data loaded: {{row_count}} rows × {{col_count}} columns")
            log_message('{job_id}', "✅ Data loaded directly as Spark DataFrame - optimal performance!")
            
        except Exception as e:
            log_message('{job_id}', f"❌ Failed to load BigQuery data: {{str(e)}}")
            log_message('{job_id}', "🛑 Stopping job - BigQuery data loading failed")
            raise RuntimeError(f"BigQuery data loading failed: {{str(e)}}")
        
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
            
            # Load OOT datasets using direct Spark-BigQuery connector
            oot1_data, oot2_data = load_oot_datasets(config, None, oot_bigquery_options)
            
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
        
        # Format the oot_loading_helper first
        formatted_oot_helper = oot_loading_helper.format(job_id=job_id)
        
        # Format the main script template with all placeholders
        formatted_script = script_template.format(
            job_id=job_id,
            jobs_dir=jobs_dir,
            job_config_file=job_config_file,
            job_status_file=job_status_file,
            job_error_file=job_error_file,
            task_type=task_type,
            task_type_title=task_type_title,
            oot_loading_helper=formatted_oot_helper
        )
        
        return formatted_script

    def _run_job_background(self, job_id: str, script_file: str):
        """Run job in background thread."""
        try:
            # Set up environment with job_id
            env = os.environ.copy()
            env['JOB_ID'] = job_id
            
            # Run the script
            process = subprocess.Popen(
                [sys.executable, script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
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

    def _extract_data_files_from_config(self, config: Dict) -> List[str]:
        """Extract data file paths from job configuration."""
        data_files = []
        
        # Check for data source configuration
        if 'data_source' in config:
            data_source = config['data_source']
            if isinstance(data_source, str) and os.path.exists(data_source):
                data_files.append(data_source)
            elif isinstance(data_source, dict):
                # Handle flexible data input configuration
                if 'file_path' in data_source and os.path.exists(data_source['file_path']):
                    data_files.append(data_source['file_path'])
                elif 'uploaded_file' in data_source:
                    # Handle uploaded file information
                    pass  # Uploaded files are handled differently
        
        # Check for existing files configuration
        if 'existing_files' in config:
            existing_files = config['existing_files']
            if isinstance(existing_files, list):
                for file_path in existing_files:
                    if os.path.exists(file_path):
                        data_files.append(file_path)
        
        # Check for BigQuery configuration (no local files)
        if 'bigquery' in config:
            # BigQuery doesn't have local files to upload
            pass
        
        return data_files
    
    def _store_dataproc_job_info(self, job_id: str, batch_id: str, config: Dict):
        """Store Dataproc job information for tracking."""
        try:
            job_info = {
                'job_id': job_id,
                'batch_id': batch_id,
                'submission_time': datetime.now().isoformat(),
                'execution_mode': 'dataproc_serverless',
                'config': config
            }
            
            # Save to jobs directory
            info_file = os.path.join(self.jobs_dir, f"{job_id}_dataproc_info.json")
            with open(info_file, 'w') as f:
                json.dump(job_info, f, indent=2)
                
            print(f"✅ Dataproc job info stored: {info_file}")
            
        except Exception as e:
            print(f"⚠️ Could not store Dataproc job info: {e}")
    
    def get_dataproc_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a Dataproc Serverless job."""
        if not self.use_dataproc_serverless or not self.dataproc_manager:
            return None
            
        try:
            # Try to find batch ID from stored info
            info_file = os.path.join(self.jobs_dir, f"{job_id}_dataproc_info.json")
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    job_info = json.load(f)
                    batch_id = job_info.get('batch_id')
                    
                    if batch_id:
                        return self.dataproc_manager.get_job_status(batch_id)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Could not get Dataproc job status: {e}")
            return None

# Global job manager instance
job_manager = BackgroundJobManager() 