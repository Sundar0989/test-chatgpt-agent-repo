"""
Dataproc Serverless Manager for AutoML PySpark

This module provides integration with Google Cloud Dataproc Serverless
to run Spark jobs in a fully managed, autoscaling environment.

Features:
- Submit Spark jobs to Dataproc Serverless
- Automatic executor scaling (0 to thousands)
- No cluster management required
- Cost optimization (pay per job)
- Integration with existing AutoML pipeline
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from google.cloud import dataproc_v1
from google.cloud import storage
from google.oauth2 import service_account
from google.protobuf import duration_pb2
import tempfile
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataprocServerlessManager:
    """
    Manages Spark job submission to Google Cloud Dataproc Serverless.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dataproc Serverless manager.
        
        Args:
            config: Configuration dictionary for Dataproc Serverless
        """
        self.config = config or self._get_default_config()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize clients
        self._init_clients()
        
        # Performance monitoring
        self.job_history = []
        self.performance_metrics = {
            'job_durations': [],
            'executor_counts': [],
            'cost_estimates': []
        }
        
        logger.info("🚀 Dataproc Serverless Manager initialized")
    
    def _validate_config(self):
        """Validate the configuration and set defaults for missing values."""
        required_fields = ['project_id', 'region', 'temp_bucket']
        
        for field in required_fields:
            if not self.config.get(field):
                logger.warning(f"⚠️ Missing required field: {field}")
                if field == 'project_id':
                    self.config[field] = os.environ.get('GCP_PROJECT_ID') or os.environ.get('GOOGLE_CLOUD_PROJECT') or ''
                elif field == 'region':
                    self.config[field] = os.environ.get('GCP_REGION', 'us-central1')
                elif field == 'temp_bucket':
                    # Try to construct a default bucket name
                    project_id = self.config.get('project_id', '')
                    if project_id:
                        self.config[field] = f"{project_id}-automl-temp"
                    else:
                        # Use a fallback bucket name that won't cause indexing errors
                        self.config[field] = 'automl-temp-bucket-default'
        
        # Validate project ID
        if not self.config['project_id']:
            raise ValueError("GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable must be set")
        
        # Validate temp bucket - ensure it's not empty and has valid characters
        if not self.config['temp_bucket'] or not self.config['temp_bucket'].strip():
            raise ValueError("GCP_TEMP_BUCKET environment variable must be set or project_id must be available")
        
        # Ensure bucket name starts and ends with alphanumeric characters
        bucket_name = self.config['temp_bucket'].strip()
        if not bucket_name[0].isalnum() or not bucket_name[-1].isalnum():
            # Fix invalid bucket name
            if not bucket_name[0].isalnum():
                bucket_name = 'a' + bucket_name
            if not bucket_name[-1].isalnum():
                bucket_name = bucket_name + 'z'
            self.config['temp_bucket'] = bucket_name
            logger.warning(f"⚠️ Fixed invalid bucket name to: {bucket_name}")
        
        logger.info(f"✅ Configuration validated - Project: {self.config['project_id']}, Region: {self.config['region']}, Bucket: {self.config['temp_bucket']}")
    
    def _init_clients(self):
        """Initialize Google Cloud clients."""
        try:
            # Check if service account credentials are available
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and credentials_path.strip():
                try:
                    with open(credentials_path, 'r') as f:
                        service_account_info = json.load(f)
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info
                    )
                    
                    # Configure Dataproc client with regional endpoint
                    client_options = {"api_endpoint": f"{self.config.get('region', 'us-central1')}-dataproc.googleapis.com:443"}
                    self.dataproc_client = dataproc_v1.BatchControllerClient(
                        credentials=credentials,
                        client_options=client_options
                    )
                    self.storage_client = storage.Client(
                        credentials=credentials
                    )
                except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
                    logger.warning(f"⚠️ Could not load service account credentials from {credentials_path}: {e}")
                    logger.info("🔄 Falling back to default credentials (Cloud Run Workload Identity)")
                    # Use default credentials (Cloud Run Workload Identity)
                    # Configure Dataproc client with regional endpoint
                    client_options = {"api_endpoint": f"{self.config.get('region', 'us-central1')}-dataproc.googleapis.com:443"}
                    self.dataproc_client = dataproc_v1.BatchControllerClient(
                        client_options=client_options
                    )
                    self.storage_client = storage.Client()
            else:
                # Use default credentials (Cloud Run Workload Identity)
                # Configure Dataproc client with regional endpoint
                client_options = {"api_endpoint": f"{self.config.get('region', 'us-central1')}-dataproc.googleapis.com:443"}
                self.dataproc_client = dataproc_v1.BatchControllerClient(
                    client_options=client_options
                )
                self.storage_client = storage.Client()
                
            logger.info("✅ Google Cloud clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Cloud clients: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Dataproc Serverless configuration."""
        return {
            # Dataproc Serverless settings
            'project_id': os.environ.get('GCP_PROJECT_ID', ''),
            'region': os.environ.get('GCP_REGION', 'us-central1'),
            'batch_id_prefix': 'automl-spark',
            
            # Spark configuration
            'spark_version': '3.4',
            'runtime_config_version': '1.0',
            
            # Resource configuration
            'executor_count': {
                'min': 2,
                'max': 100
            },
            'executor_memory': '4g',
            'executor_cpu': '2',
            'driver_memory': '8g',
            'driver_cpu': '2',
            
            # Storage configuration
            'temp_bucket': os.environ.get('GCP_TEMP_BUCKET', 'automl-temp-bucket-default'),
            'results_bucket': os.environ.get('GCP_RESULTS_BUCKET', ''),
            
            # Job configuration
            'timeout_minutes': 60,
            'idle_timeout_minutes': 10,
            
            # Cost optimization
            'enable_autoscaling': True,
            'max_executor_count': 100,
            'min_executor_count': 2
        }
    
    def submit_spark_job(
        self,
        job_config: Dict[str, Any],
        job_id: str,
        data_files: List[str] = None,
        dependencies: List[str] = None
    ) -> str:
        """
        Submit a Spark job to Dataproc Serverless.
        
        Args:
            job_config: AutoML job configuration
            job_id: Unique job identifier
            data_files: List of data file paths to upload
            dependencies: List of dependency files to upload
            
        Returns:
            Batch ID from Dataproc Serverless
        """
        try:
            logger.info(f"🚀 Submitting Spark job {job_id} to Dataproc Serverless")
            
            # Validate configuration before proceeding
            if not self.config.get('project_id'):
                raise ValueError("project_id is not configured. Please set GCP_PROJECT_ID environment variable.")
            
            if not self.config.get('temp_bucket'):
                raise ValueError("temp_bucket is not configured. Please set GCP_TEMP_BUCKET environment variable.")
            
            # Create unique batch ID - Dataproc requires lowercase letters, digits, and hyphens only, 4-63 chars
            batch_id = self._sanitize_batch_id(job_id, self.config['batch_id_prefix'])
            
            # Upload job files to Cloud Storage
            job_files = self._upload_job_files(job_id, job_config, data_files, dependencies)
            
            # Create batch request
            batch_request = self._create_batch_request(batch_id, job_files, job_config)
            
            # Submit batch
            operation = self.dataproc_client.create_batch(
                parent=f"projects/{self.config['project_id']}/locations/{self.config['region']}",
                batch=batch_request,
                batch_id=batch_id  # Important: pass the batch_id
            )
            
            # Wait for batch creation
            batch = operation.result()
            
            logger.info(f"✅ Job {job_id} submitted successfully. Batch ID: {batch_id}")
            
            # Store job metadata
            self._store_job_metadata(job_id, batch_id, batch, job_config)
            
            return batch_id
            
        except Exception as e:
            logger.error(f"❌ Failed to submit job {job_id}: {e}")
            logger.error(f"🔍 Job config keys: {list(job_config.keys()) if job_config else 'None'}")
            logger.error(f"🔍 Data files: {data_files}")
            logger.error(f"🔍 Dependencies: {dependencies}")
            raise
    
    def _upload_job_files(
        self,
        job_id: str,
        job_config: Dict[str, Any],
        data_files: List[str] = None,
        dependencies: List[str] = None
    ) -> Dict[str, str]:
        """
        Upload job files to Cloud Storage.
        
        Returns:
            Dictionary mapping file types to Cloud Storage URIs
        """
        # Validate temp_bucket before using it
        if not self.config.get('temp_bucket') or not self.config['temp_bucket'].strip():
            raise ValueError("temp_bucket is not configured or is empty. Please set GCP_TEMP_BUCKET environment variable.")
        
        # Ensure bucket name is valid for Google Cloud Storage
        bucket_name = self.config['temp_bucket'].strip()
        if not bucket_name[0].isalnum() or not bucket_name[-1].isalnum():
            # Fix invalid bucket name
            if not bucket_name[0].isalnum():
                bucket_name = 'a' + bucket_name
            if not bucket_name[-1].isalnum():
                bucket_name = bucket_name + 'z'
            self.config['temp_bucket'] = bucket_name
            logger.warning(f"⚠️ Fixed invalid bucket name to: {bucket_name}")
        
        bucket = self.storage_client.bucket(self.config['temp_bucket'])
        job_files = {}
        
        try:
            # Upload job configuration
            config_blob = bucket.blob(f"jobs/{job_id}/job_config.json")
            config_blob.upload_from_string(
                json.dumps(job_config, indent=2),
                content_type='application/json'
            )
            job_files['config'] = f"gs://{self.config['temp_bucket']}/jobs/{job_id}/job_config.json"
            
            # Upload data files if provided
            if data_files:
                data_uris = []
                for data_file in data_files:
                    if os.path.exists(data_file):
                        blob_name = f"jobs/{job_id}/data/{os.path.basename(data_file)}"
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(data_file)
                        data_uris.append(f"gs://{self.config['temp_bucket']}/{blob_name}")
                job_files['data'] = data_uris
            
            # Upload dependencies if provided
            if dependencies:
                deps_uris = []
                for dep_file in dependencies:
                    if os.path.exists(dep_file):
                        blob_name = f"jobs/{job_id}/dependencies/{os.path.basename(dep_file)}"
                        blob = bucket.blob(blob_name)
                        blob.upload_from_filename(dep_file)
                        deps_uris.append(f"gs://{self.config['temp_bucket']}/{blob_name}")
                job_files['dependencies'] = deps_uris
            
            # Create requirements.txt for package installation
            requirements_content = """# Core AutoML dependencies
automl_pyspark
xgboost
lightgbm
optuna
plotly
scikit-learn
pandas
numpy

# Additional dependencies can be added here
# For example:
# custom-package==1.0.0
# another-package>=2.0.0
"""
            
            requirements_blob = bucket.blob(f"jobs/{job_id}/requirements.txt")
            requirements_blob.upload_from_string(requirements_content, content_type='text/plain')
            job_files['requirements'] = f"gs://{self.config['temp_bucket']}/jobs/{job_id}/requirements.txt"
            
            logger.info("📦 Created requirements.txt for package installation")
            
            # Create main job script
            job_script = self._create_job_script(job_id, job_config, job_files)
            script_blob = bucket.blob(f"jobs/{job_id}/main.py")
            script_blob.upload_from_string(job_script, content_type='text/plain')
            job_files['script'] = f"gs://{self.config['temp_bucket']}/jobs/{job_id}/main.py"
            
            logger.info(f"✅ Job files uploaded successfully for job {job_id}")
            return job_files
            
        except Exception as e:
            logger.error(f"❌ Failed to upload job files for {job_id}: {e}")
            raise
    
    def _create_job_script(
        self,
        job_id: str,
        job_config: Dict[str, Any],
        job_files: Dict[str, str]
    ) -> str:
        """Create the main Spark job script."""
        
        # Import the necessary AutoML modules
        script = f'''#!/usr/bin/env python3
"""
AutoML PySpark Job Script for Dataproc Serverless
Job ID: {job_id}
Generated: {datetime.now().isoformat()}
"""

import sys
import os
import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, expr

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main AutoML job execution."""
    try:
        # Install required packages from requirements.txt
        logger.info("📦 Installing required packages...")
        try:
            import subprocess
            import sys
            
            # Install packages from requirements.txt
            requirements_uri = "gs://{self.config.get('temp_bucket', '')}/jobs/{job_id}/requirements.txt"
            if requirements_uri.startswith('gs://'):
                # Download requirements.txt from GCS
                rdd = spark.sparkContext.wholeTextFiles(requirements_uri)
                _, requirements_content = rdd.collect()[0]
                
                # Write to local file
                with open('/tmp/requirements.txt', 'w') as f:
                    f.write(requirements_content)
                
                # Install packages
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', '/tmp/requirements.txt'])
                logger.info("✅ Packages installed successfully")
            else:
                logger.warning("⚠️ No requirements.txt found, using pre-installed packages")
        except Exception as e:
            logger.warning("⚠️ Package installation failed: " + str(e))
            logger.info("ℹ️ Continuing with pre-installed packages")
        
        logger.info("🚀 Starting AutoML job " + str("{job_id}"))
        
        # Initialize Spark session - Dataproc Serverless compatible configuration
        # NO local networking configs - these break serverless connectivity
        spark = SparkSession.builder \\
            .appName("AutoML-" + str("{job_id}")) \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \\
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \\
            .config("spark.network.timeout", "800s") \\
            .config("spark.executor.heartbeatInterval", "60s") \\
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \\
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \\
            .getOrCreate()
        
        # CRITICAL: Let Dataproc Serverless manage ALL networking configurations
        # Don't set spark.driver.host, spark.driver.bindAddress, or spark.master
        # These break serverless connectivity and cause "Connection to master failed" errors
        
        # Only remove problematic local configurations if they exist
        if spark.conf.get("spark.master", "").startswith("local"):
            spark.conf.unset("spark.master")
            logger.info("✅ Removed local master configuration")
        
        # Additional serverless-specific configurations
        spark.conf.set("spark.sql.adaptive.enabled", "true")
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
        spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
        
        # Network timeout configurations for serverless
        spark.conf.set("spark.network.timeout", "800s")
        spark.conf.set("spark.executor.heartbeatInterval", "60s")
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        
        logger.info("✅ Spark session configured for Dataproc Serverless")
        
        # IMMEDIATE JAR VERIFICATION - Check what's available right after Spark init
        logger.info("🔍 IMMEDIATE JAR verification after Spark initialization...")
        try:
            # Check spark.jars configuration
            jars_conf = spark.sparkContext.getConf().get("spark.jars")
            if jars_conf:
                logger.info("📦 JAR files in spark.jars configuration:")
                for jar in jars_conf.split(","):
                    if jar.strip():
                        logger.info("   - " + jar.strip())
            else:
                logger.warning("⚠️ spark.jars configuration is empty")
            
            # Check spark.submit.pyFiles configuration
            pyfiles_conf = spark.sparkContext.getConf().get("spark.submit.pyFiles")
            if pyfiles_conf:
                logger.info("🐍 Python files in spark.submit.pyFiles configuration:")
                for py_file in pyfiles_conf.split(","):
                    if py_file.strip():
                        logger.info("   - " + py_file.strip())
            else:
                logger.info("ℹ️ spark.submit.pyFiles configuration is empty")
                
            # Check if we can access the BigQuery format
            try:
                # This will fail if the BigQuery connector isn't loaded
                test_reader = spark.read.format("bigquery")
                logger.info("✅ BigQuery format is available (connector JAR loaded)")
            except Exception as e:
                logger.error("❌ BigQuery format is NOT available: " + str(e))
                
        except Exception as e:
            logger.error("❌ Error during JAR verification: " + str(e))
        
        # NOTE: No need for external compatibility helper functions
        # All logic is now self-contained in this script
        
        logger.info("✅ Spark session initialized")
        
        # Load job configuration
        config_uri = "{job_files.get('config', '')}"
        if config_uri.startswith('gs://'):
            # Download config from Cloud Storage - read the WHOLE file, not just first line
            rdd = spark.sparkContext.wholeTextFiles(config_uri)
            _, config_content = rdd.collect()[0]
            job_config = json.loads(config_content)
        else:
            # Use local config
            with open(config_uri, 'r') as f:
                job_config = json.load(f)
        
        logger.info("✅ Job configuration loaded: " + str(job_config.get('model_name', 'Unknown')))
        
        # Execute AutoML pipeline based on task type
        task_type = job_config.get('task_type', 'classification')
        
        if task_type == 'classification':
            # Import and use the actual automl_pyspark package
            logger.info("🔍 Running classification AutoML pipeline")
            from automl_pyspark.classification.automl_classifier import AutoMLClassifier
            automl = AutoMLClassifier(spark, job_config)
            results = automl.run()
        elif task_type == 'regression':
            # Import and use the actual automl_pyspark package
            logger.info("🔍 Running regression AutoML pipeline")
            from automl_pyspark.regression.automl_regressor import AutoMLRegressor
            automl = AutoMLRegressor(spark, job_config)
            results = automl.run()
        elif task_type == 'clustering':
            # Import and use the actual automl_pyspark package
            logger.info("🔍 Running clustering AutoML pipeline")
            from automl_pyspark.clustering.automl_clusterer import AutoMLClusterer
            automl = AutoMLClusterer(spark, job_config)
            results = automl.run()
        else:
            raise ValueError("Unsupported task type: " + str(task_type))
        
        # NOTE: No need for external module compatibility checks
        # All logic is now inlined in this script
        
        # VERIFY: Check if JAR files are properly loaded
        logger.info("🔍 Verifying JAR file availability...")
        try:
            # Check if BigQuery connector is available
            from pyspark.sql import DataFrame
            from pyspark.sql.types import StringType, IntegerType, DoubleType, StructType, StructField
            
            # Try to create a BigQuery read option (this will fail if connector isn't loaded)
            test_df = spark.read.format("bigquery") \
                .option("table", "dummy_table") \
                .load()
            logger.info("✅ BigQuery connector JAR is properly loaded and accessible")
        except Exception as e:
            if "bigquery" in str(e).lower():
                logger.error("❌ BigQuery connector JAR is NOT loaded: " + str(e))
            else:
                logger.info("ℹ️ BigQuery test failed (expected for dummy table): " + str(e))
        
        # List all loaded JARs for debugging
        try:
            jars_conf = spark.sparkContext.getConf().get("spark.jars")
            if jars_conf:
                logger.info("📦 Loaded JAR files:")
                for jar in jars_conf.split(","):
                    if jar.strip():
                        logger.info("   - " + jar.strip())
            else:
                logger.warning("⚠️ No JAR files found in spark.jars configuration")
        except Exception as e:
            logger.warning("⚠️ Could not retrieve JAR configuration: " + str(e))
        
        # Check Python files availability
        try:
            python_files_conf = spark.sparkContext.getConf().get("spark.submit.pyFiles")
            if python_files_conf:
                logger.info("🐍 Loaded Python files:")
                for py_file in python_files_conf.split(","):
                    if py_file.strip():
                        logger.info("   - " + py_file.strip())
            else:
                logger.info("ℹ️ No additional Python files loaded via spark.submit.pyFiles")
        except Exception as e:
            logger.info("ℹ️ Could not retrieve Python files configuration: " + str(e))
        
        # Execute AutoML pipeline
        logger.info("✅ AutoML pipeline completed successfully")
        
        # Save results to Cloud Storage
        results_bucket = self.config.get('results_bucket', '')
        if results_bucket:
            results_uri = "gs://" + str(results_bucket) + "/results/" + str("{job_id}") + "/"
            automl.save_results(results_uri)
            logger.info("✅ Results saved to " + str(results_uri))
        
        logger.info("🎉 AutoML job " + str("{job_id}") + " completed successfully")
        return 0
        
    except Exception as e:
        logger.error("❌ AutoML job " + str("{job_id}") + " failed: " + str(e))
        return 1
    
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    sys.exit(main())
'''
        
        return script
    
    def _create_batch_request(
        self,
        batch_id: str,
        job_files: Dict[str, str],
        job_config: Dict[str, Any]
    ) -> dataproc_v1.Batch:
        """Create the Dataproc Serverless batch request."""
        
        # Determine executor count based on data size and complexity
        executor_count = self._calculate_executor_count(job_config)
        
        batch = dataproc_v1.Batch(
            pyspark_batch=dataproc_v1.PySparkBatch(
                main_python_file_uri=job_files['script'],
                args=job_config.get('args', []),
                # Use requirements.txt for package installation
                python_file_uris=job_files.get('py_deps', []),
                jar_file_uris=job_files.get('jar_uris', [
                    # BigQuery connector - automatically available in Dataproc Serverless
                    "gs://spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.36.1.jar",
                    # Additional Spark libraries if needed
                    # "gs://spark-lib/avro/spark-avro_2.12-3.3.0.jar",
                    # "gs://spark-lib/delta/delta-core_2.12-2.4.0.jar"
                ]),
                file_uris=job_files.get('file_uris', []),
                archive_uris=job_files.get('archive_uris', [])
            ),
            runtime_config=dataproc_v1.RuntimeConfig(
                version=job_config.get('runtime_version', '2.2')
            ),
            environment_config=dataproc_v1.EnvironmentConfig(
                execution_config=dataproc_v1.ExecutionConfig(
                    # NOTE: bucket NAME, NOT 'gs://...'
                    staging_bucket=self.config['temp_bucket']
                )
            ),
            labels={
                "job-type": "automl",
                "user": job_config.get('user_id', 'unknown'),
                "model": job_config.get('model_name', 'unknown')
            }
        )
        
        return batch
    
    def _sanitize_batch_id(self, raw: str, prefix: str) -> str:
        """Sanitize batch ID to meet Dataproc Serverless requirements.
        
        Dataproc Serverless batch IDs must:
        - Contain only lowercase letters, digits, and hyphens
        - Be 4-63 characters long
        - Start with a letter
        """
        import re
        import time
        
        # Convert to lowercase, replace invalid chars with '-', collapse repeats
        s = re.sub(r'[^a-z0-9-]+', '-', raw.lower())
        s = re.sub(r'-{2,}', '-', s).strip('-')
        
        # Must start with a letter
        if not s or not s[0].isalpha():
            s = 'b' + s
        
        # Compose with prefix and timestamp and trim to 63
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{prefix}-{s}-{ts}"
        
        # Ensure length 4-63
        base = base[:63].strip('-')
        if len(base) < 4:
            base = (base + "-bxxx")[:63]
        
        return base
    
    def verify_jar_availability(self, job_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify that required JAR files and dependencies are available.
        
        This method checks:
        1. BigQuery connector availability
        2. Custom JAR files if specified
        3. Python library files
        
        Args:
            job_config: Optional job configuration to check specific requirements
            
        Returns:
            Dictionary with verification results
        """
        verification_results = {
            "bigquery_connector": False,
            "custom_jars": [],
            "python_libs": [],
            "errors": []
        }
        
        try:
            # Check if BigQuery connector is accessible
            try:
                # Test BigQuery format availability
                from pyspark.sql import SparkSession
                
                # Create a minimal Spark session for testing
                test_spark = SparkSession.builder \
                    .appName("JAR-Verification") \
                    .config("spark.sql.adaptive.enabled", "false") \
                    .config("spark.driver.host", "localhost") \
                    .config("spark.driver.bindAddress", "127.0.0.1") \
                    .master("local[1]") \
                    .getOrCreate()
                
                # Test BigQuery format
                test_reader = test_spark.read.format("bigquery")
                verification_results["bigquery_connector"] = True
                logger.info("✅ BigQuery connector is available locally")
                
                test_spark.stop()
                
            except Exception as e:
                verification_results["bigquery_connector"] = False
                verification_results["errors"].append(f"BigQuery connector test failed: {e}")
                logger.warning(f"⚠️ BigQuery connector not available locally: {e}")
            
            # Check custom JAR files if specified
            if job_config and 'jar_uris' in job_config:
                for jar_uri in job_config['jar_uris']:
                    if jar_uri.startswith('gs://'):
                        # Check if GCS file exists
                        try:
                            bucket_name = jar_uri.split('/')[2]
                            blob_name = '/'.join(jar_uri.split('/')[3:])
                            bucket = self.storage_client.bucket(bucket_name)
                            blob = bucket.blob(blob_name)
                            
                            if blob.exists():
                                verification_results["custom_jars"].append({
                                    "uri": jar_uri,
                                    "status": "available",
                                    "size": blob.size
                                })
                                logger.info(f"✅ Custom JAR available: {jar_uri}")
                            else:
                                verification_results["custom_jars"].append({
                                    "uri": jar_uri,
                                    "status": "not_found"
                                })
                                verification_results["errors"].append(f"Custom JAR not found: {jar_uri}")
                                logger.warning(f"⚠️ Custom JAR not found: {jar_uri}")
                        except Exception as e:
                            verification_results["custom_jars"].append({
                                "uri": jar_uri,
                                "status": "error",
                                "error": str(e)
                            })
                            verification_results["errors"].append(f"Error checking custom JAR {jar_uri}: {e}")
                            logger.error(f"❌ Error checking custom JAR {jar_uri}: {e}")
            
            # Check Python library files
            automl_dir = os.path.join(os.path.dirname(__file__), '..', 'automl_pyspark')
            if os.path.exists(automl_dir):
                for root, dirs, files in os.walk(automl_dir):
                    for file in files:
                        if file.endswith('.py'):
                            rel_path = os.path.relpath(os.path.join(root, file), automl_dir)
                            verification_results["python_libs"].append({
                                "file": rel_path,
                                "status": "available",
                                "size": os.path.getsize(os.path.join(root, file))
                            })
            
            logger.info(f"✅ JAR verification completed. Found {len(verification_results['python_libs'])} Python library files")
            
        except Exception as e:
            verification_results["errors"].append(f"Verification failed: {e}")
            logger.error(f"❌ JAR verification failed: {e}")
        
        return verification_results
    
    def _calculate_executor_count(self, job_config: Dict[str, Any]) -> int:
        """Calculate optimal executor count based on job configuration."""
        # Base executor count
        base_count = self.config['executor_count']['min']
        
        # Adjust based on data size
        data_size_mb = job_config.get('data_size_mb', 100)
        if data_size_mb > 1000:
            base_count = min(base_count * 2, self.config['executor_count']['max'])
        if data_size_mb > 10000:
            base_count = min(base_count * 3, self.config['executor_count']['max'])
        
        # Adjust based on model complexity
        models = job_config.get('models', [])
        if len(models) > 3:
            base_count = min(base_count + 2, self.config['executor_count']['max'])
        
        # Adjust based on hyperparameter tuning
        if job_config.get('enable_hyperparameter_tuning', False):
            base_count = min(base_count + 1, self.config['executor_count']['max'])
        
        return base_count
    
    def _store_job_metadata(
        self,
        job_id: str,
        batch_id: str,
        batch: dataproc_v1.Batch,
        job_config: Dict[str, Any]
    ):
        """Store job metadata for tracking."""
        metadata = {
            'job_id': job_id,
            'batch_id': batch_id,
            'status': 'SUBMITTED',
            'submission_time': datetime.now().isoformat(),
            'batch_uri': batch.name,
            'job_config': job_config
        }
        
        self.job_history.append(metadata)
        
        # Store in Cloud Storage for persistence
        bucket = self.storage_client.bucket(self.config['temp_bucket'])
        blob = bucket.blob(f"jobs/{job_id}/metadata.json")
        blob.upload_from_string(
            json.dumps(metadata, indent=2, default=str),
            content_type='application/json'
        )
    
    def get_job_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the current status of a Dataproc Serverless job."""
        try:
            batch = self.dataproc_client.get_batch(name=batch_id)
            
            status_info = {
                'batch_id': batch_id,
                'state': batch.state.name,
                'state_message': batch.state_message,
                'create_time': batch.create_time.isoformat() if batch.create_time else None,
                'update_time': batch.update_time.isoformat() if batch.update_time else None,
                'operation': batch.operation
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get status for batch {batch_id}: {e}")
            return {'error': str(e)}
    
    def list_jobs(self, filter_expr: str = None) -> List[Dict[str, Any]]:
        """List all Dataproc Serverless jobs."""
        try:
            parent = f"projects/{self.config['project_id']}/locations/{self.config['region']}"
            
            if filter_expr:
                request = dataproc_v1.ListBatchesRequest(
                    parent=parent,
                    filter=filter_expr
                )
            else:
                request = dataproc_v1.ListBatchesRequest(parent=parent)
            
            page_result = self.dataproc_client.list_batches(request=request)
            
            jobs = []
            for batch in page_result:
                job_info = {
                    'batch_id': batch.name.split('/')[-1],
                    'state': batch.state.name,
                    'create_time': batch.create_time.isoformat() if batch.create_time else None,
                    'labels': dict(batch.labels) if batch.labels else {}
                }
                jobs.append(job_info)
            
            return jobs
            
        except Exception as e:
            logger.error(f"❌ Failed to list jobs: {e}")
            return []
    
    def cancel_job(self, batch_id: str) -> bool:
        """Cancel a running Dataproc Serverless job."""
        try:
            self.dataproc_client.delete_batch(name=batch_id)
            logger.info(f"✅ Job {batch_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel job {batch_id}: {e}")
            return False
    
    def get_cost_estimate(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the cost of running a job on Dataproc Serverless."""
        # This is a simplified cost estimation
        # In production, you'd want to use the actual pricing API
        
        executor_count = self._calculate_executor_count(job_config)
        estimated_duration_hours = job_config.get('estimated_duration_hours', 1)
        
        # Rough cost estimation (you should adjust based on actual pricing)
        cost_per_executor_hour = 0.10  # USD per executor hour
        estimated_cost = executor_count * estimated_duration_hours * cost_per_executor_hour
        
        return {
            'estimated_cost_usd': round(estimated_cost, 2),
            'executor_count': executor_count,
            'estimated_duration_hours': estimated_duration_hours,
            'cost_per_executor_hour': cost_per_executor_hour
        }
