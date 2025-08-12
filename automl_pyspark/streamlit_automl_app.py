"""
Rapid Modeler Streamlit Application

A comprehensive web interface for the Rapid Modeler PySpark system with two main pages:
1. Job Submission - Configure and submit Rapid Modeler jobs
2. Results Viewer - View results and artifacts from completed jobs
"""

import streamlit as st
import pandas as pd
import os
import yaml
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import glob
import shutil
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
import functools
import hashlib

# Import background job manager
try:
    from background_job_manager import job_manager
except ImportError:
    # Fallback if background job manager is not available
    job_manager = None

# Import the flexible data input system
try:
    from data_input_manager import DataInputManager
    from data_input_integration import enable_flexible_data_input
    FLEXIBLE_INPUT_AVAILABLE = True
except ImportError:
    FLEXIBLE_INPUT_AVAILABLE = False

# Performance optimization: Add caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_read_excel(file_path: str, sheet_name: str = None):
    """Cached Excel file reading to avoid repeated slow operations."""
    try:
        if sheet_name:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def cached_read_json(file_path: str):
    """Cached JSON file reading."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return {}

@st.cache_data(ttl=300)
def cached_glob_files(pattern: str):
    """Cached file globbing to avoid repeated filesystem operations."""
    try:
        return glob.glob(pattern)
    except Exception as e:
        print(f"Error globbing pattern {pattern}: {e}")
        return []

@st.cache_data(ttl=300)
def cached_list_directory(directory: str):
    """Cached directory listing."""
    try:
        return os.listdir(directory)
    except Exception as e:
        print(f"Error listing directory {directory}: {e}")
        return []

# Performance optimization: File discovery cache
_file_discovery_cache = {}

def get_cached_file_discovery(output_dir: str, file_type: str, task_type: str = 'classification'):
    """Get cached file discovery results to avoid repeated filesystem operations."""
    cache_key = f"{output_dir}_{file_type}_{task_type}"
    
    if cache_key not in _file_discovery_cache:
        _file_discovery_cache[cache_key] = {}
    
    # Check if cache is still valid (5 minute TTL)
    current_time = time.time()
    if 'timestamp' not in _file_discovery_cache[cache_key] or \
       current_time - _file_discovery_cache[cache_key]['timestamp'] > 300:
        
        # Refresh cache
        if file_type == 'feature_importance':
            files = discover_feature_importance_files(output_dir)
        elif file_type == 'metrics':
            files = discover_metrics_files(output_dir)
        elif file_type == 'plots':
            files = discover_plot_files(output_dir)
        elif file_type == 'summary':
            files = discover_summary_files(output_dir, task_type)
        elif file_type == 'clustering':
            clustering_data = discover_clustering_files(output_dir)
            files = clustering_data  # Return the dictionary for clustering
        else:
            files = []
        
        _file_discovery_cache[cache_key] = {
            'files': files,
            'timestamp': current_time
        }
    
    return _file_discovery_cache[cache_key]['files']

def discover_feature_importance_files(output_dir: str):
    """Efficiently discover feature importance files."""
    files = []
    
    # Check main directory first
    main_files = [
        os.path.join(output_dir, 'Features_selected_for_modeling.png'),
        os.path.join(output_dir, 'feature_importance.png'),
        os.path.join(output_dir, 'overall_feature_importance.png'),
        os.path.join(output_dir, 'feature_importance_plot.png'),
        os.path.join(output_dir, 'feature_importance.xlsx')
    ]
    
    for file_path in main_files:
        if os.path.exists(file_path):
            files.append(file_path)
    
    # Check subdirectories efficiently
    if os.path.exists(output_dir):
        for item in cached_list_directory(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                # Look for feature importance files in subdirectory
                sub_files = cached_glob_files(os.path.join(item_path, '*feature*importance*.xlsx'))
                sub_files.extend(cached_glob_files(os.path.join(item_path, '*feature*importance*.png')))
                files.extend(sub_files)
    
    return files

def discover_metrics_files(output_dir: str):
    """Efficiently discover metrics files."""
    patterns = [
        os.path.join(output_dir, '*_metrics.json'),
        os.path.join(output_dir, '*', '*_metrics.json'),
        os.path.join(output_dir, '*', '*_multiclass_metrics.json')
    ]
    
    files = []
    for pattern in patterns:
        files.extend(cached_glob_files(pattern))
    
    return files

def discover_plot_files(output_dir: str):
    """Efficiently discover plot files."""
    plot_files = []
    
    # Look for plots in validation_plots directory
    validation_plots_dir = os.path.join(output_dir, 'validation_plots')
    if os.path.exists(validation_plots_dir):
        for file in cached_list_directory(validation_plots_dir):
            if file.endswith('.png'):
                plot_files.append(os.path.join(validation_plots_dir, file))
    
    # Also look for elbow plots in root directory
    elbow_plot_patterns = [
        os.path.join(output_dir, '*_elbow_plot*.png'),
        os.path.join(output_dir, '*_elbow_plot_efficient.png')
    ]
    
    for pattern in elbow_plot_patterns:
        plot_files.extend(cached_glob_files(pattern))
    
    return plot_files

def discover_summary_files(output_dir: str, task_type: str = 'classification'):
    """Efficiently discover summary files."""
    summary_files = []
    
    # Add main summary files
    main_summary_files = [
        os.path.join(output_dir, 'model_selection_summary.txt')
    ]
    
    # Add task-specific summary files
    if task_type == 'regression':
        main_summary_files.append(os.path.join(output_dir, 'champion_challenger_summary.txt'))
    elif task_type == 'clustering':
        main_summary_files.append(os.path.join(output_dir, 'clustering_summary.txt'))
    
    for file_path in main_summary_files:
        if os.path.exists(file_path):
            summary_files.append(file_path)
    
    # Add model-specific summary patterns
    summary_patterns = [
        os.path.join(output_dir, '*', '*_validation_summary.txt'),
        os.path.join(output_dir, '*', '*_multiclass_validation_summary.txt')
    ]
    
    # Add task-specific patterns
    if task_type == 'regression':
        summary_patterns.extend([
            os.path.join(output_dir, '*', '*_regression_validation_summary.txt'),
            os.path.join(output_dir, '*', '*_regression_summary.txt')
        ])
    elif task_type == 'clustering':
        summary_patterns.extend([
            os.path.join(output_dir, '*', '*_clustering_validation_summary.txt'),
            os.path.join(output_dir, '*', '*_clustering_summary.txt'),
            os.path.join(output_dir, '*', '*_dataset_validation_summary.txt'),
            # Add patterns for files in root directory
            os.path.join(output_dir, '*_clustering_validation_summary.txt'),
            os.path.join(output_dir, '*_clustering_summary.txt'),
            os.path.join(output_dir, '*_dataset_validation_summary.txt')
        ])
    
    for pattern in summary_patterns:
        summary_files.extend(cached_glob_files(pattern))
    
    return summary_files

def discover_clustering_files(output_dir: str):
    """Efficiently discover clustering-specific files."""
    clustering_files = []
    
    # Clustering metrics files
    clustering_patterns = [
        os.path.join(output_dir, '*_clustering_metrics.json'),
        os.path.join(output_dir, '*_multi_dataset_validation_metrics.json'),
        os.path.join(output_dir, '*_cross_validation_metrics.json')
    ]
    
    for pattern in clustering_patterns:
        clustering_files.extend(cached_glob_files(pattern))
    
    # Filter out cross-validation and multi-dataset files for main clustering files
    main_clustering_files = [f for f in clustering_files 
                           if f.endswith('_clustering_metrics.json') and 
                           not ('cross_validation' in f or 'multi_dataset_validation' in f)]
    
    # Separate other file types
    multi_dataset_files = [f for f in clustering_files if 'multi_dataset_validation' in f]
    cv_files = [f for f in clustering_files if 'cross_validation' in f]
    
    return {
        'main_clustering': main_clustering_files,
        'multi_dataset': multi_dataset_files,
        'cross_validation': cv_files
    }

# Page configuration
st.set_page_config(
    page_title="Rapid Modeler PySpark System (Unified)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
JOBS_DIR = "automl_jobs"
RESULTS_DIR = "automl_results" 
CONFIG_FILE = "config.yaml"
JOB_COUNTER_FILE = os.path.join(JOBS_DIR, "job_counter.txt")

# Ensure directories exist
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Additional safety check for results directory
if not os.path.exists(RESULTS_DIR):
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Results directory created
    except Exception as e:
        pass  # Could not create results directory

def load_config():
    """Load the YAML configuration file using ConfigManager to handle auto-detection."""
    try:
        from config_manager import ConfigManager
        config_manager = ConfigManager(CONFIG_FILE)
        return config_manager.get_config()
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        # Fallback to simple YAML loading
        try:
            with open(CONFIG_FILE, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"Configuration file {CONFIG_FILE} not found!")
            return {}

def clean_config_for_json(config: Dict) -> Dict:
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

def save_job_config(job_id: str, config: Dict):
    """Save job configuration to file."""
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    
    # Clean config to remove non-JSON-serializable objects
    clean_config = clean_config_for_json(config)
    
    with open(job_file, 'w') as f:
        json.dump(clean_config, f, indent=2)

def get_job_status(job_id: str) -> str:
    """Get the status of a job with improved file-based detection."""
    # Direct file reading is most reliable for persisted jobs
    status_file = os.path.join(JOBS_DIR, f"{job_id}_status.txt")
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = f.read().strip().upper()
                # Map different status formats to standard ones
                if status in ["COMPLETED", "COMPLETE", "SUCCESS"]:
                    return "Completed"
                elif status in ["FAILED", "FAILURE", "ERROR"]:
                    return "Failed"
                elif status in ["RUNNING", "RUN", "ACTIVE"]:
                    return "Running"
                elif status in ["STOPPED", "CANCELLED", "CANCELED"]:
                    return "Stopped"
                else:
                    return status.title() if status else "Unknown"
        except Exception as e:
            pass  # Error reading status file
    
    # Try the background job manager as secondary option
    if job_manager is not None:
        try:
            status = job_manager.get_job_status(job_id)
            if status and status not in ["Submitted", "Unknown"]:
                return status
        except:
            pass
    
    # If no status file exists, check if job has results (assume completed)
    output_dirs_to_check = [
        os.path.join(RESULTS_DIR, job_id),
        os.path.join("automl_output", job_id),
        os.path.join(".", job_id),  # Current directory
        job_id  # Direct path
    ]
    
    for output_dir in output_dirs_to_check:
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            # Check if directory has any result files
            result_files = []
            result_files.extend(glob.glob(os.path.join(output_dir, "*.json")))
            result_files.extend(glob.glob(os.path.join(output_dir, "*.txt")))
            result_files.extend(glob.glob(os.path.join(output_dir, "*.csv")))
            result_files.extend(glob.glob(os.path.join(output_dir, "*", "*.json")))
            
            if result_files:
                return "Completed"
    
    return "Submitted"

def update_job_status(job_id: str, status: str):
    """Update job status."""
    status_file = os.path.join(JOBS_DIR, f"{job_id}_status.txt")
    with open(status_file, 'w') as f:
        f.write(status)

def update_job_progress(job_id: str, current_step: int, total_steps: int, current_task: str):
    """Update job progress with current step and task."""
    progress_file = os.path.join(JOBS_DIR, f"{job_id}_progress.json")
    progress_data = {
        'current_step': current_step,
        'total_steps': total_steps,
        'current_task': current_task,
        'progress_percentage': round((current_step / total_steps) * 100, 1),
        'timestamp': datetime.now().isoformat()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def get_job_progress(job_id: str) -> Dict:
    """Get job progress information."""
    if job_manager is not None:
        return job_manager.get_job_progress(job_id)
    else:
        # Fallback to file-based progress
        progress_file = os.path.join(JOBS_DIR, f"{job_id}_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'current_step': 0,
            'total_steps': 8,  # Default for classification
            'current_task': 'Initializing...',
            'progress_percentage': 0.0,
            'timestamp': datetime.now().isoformat()
        }

def get_total_steps_for_task(task_type: str) -> int:
    """Get total number of steps for a given task type."""
    if task_type == 'classification':
        return 8
    elif task_type == 'regression':
        return 8
    elif task_type == 'clustering':
        return 6
    else:
        return 8  # Default

def get_job_logs(job_id: str, max_lines: int = 50) -> List[str]:
    """Get recent job logs."""
    if job_manager is not None:
        return job_manager.get_job_logs(job_id, max_lines)
    else:
        # Fallback to file-based logs
        log_file = os.path.join(JOBS_DIR, f"{job_id}_log.txt")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return lines[-max_lines:] if len(lines) > max_lines else lines
            except Exception:
                return []
        return []

def get_next_job_number() -> int:
    """Get the next sequential job number and increment the counter."""
    MAX_JOB_NUMBER = 9999  # 4-digit limit
    LOCK_FILE = f"{JOB_COUNTER_FILE}.lock"
    
    # Create jobs directory if it doesn't exist
    os.makedirs(JOBS_DIR, exist_ok=True)
    
    # Simple file-based locking to prevent race conditions
    lock_acquired = False
    max_lock_attempts = 10
    lock_attempt = 0
    
    while not lock_acquired and lock_attempt < max_lock_attempts:
        try:
            # Try to create lock file
            if not os.path.exists(LOCK_FILE):
                with open(LOCK_FILE, 'w') as f:
                    f.write(str(os.getpid()))
                lock_acquired = True
            else:
                # Check if lock is stale (older than 30 seconds)
                lock_age = time.time() - os.path.getmtime(LOCK_FILE)
                if lock_age > 30:
                    # Remove stale lock
                    try:
                        os.remove(LOCK_FILE)
                        with open(LOCK_FILE, 'w') as f:
                            f.write(str(os.getpid()))
                        lock_acquired = True
                    except:
                        pass
                else:
                    # Wait a bit and try again
                    time.sleep(0.1)
                    lock_attempt += 1
        except:
            lock_attempt += 1
            time.sleep(0.1)
    
    if not lock_acquired:
        st.error("❌ Could not acquire job counter lock. Please try again.")
        return 1
    
    try:
        # Read current counter
        if os.path.exists(JOB_COUNTER_FILE):
            with open(JOB_COUNTER_FILE, 'r') as f:
                current_number = int(f.read().strip())
        else:
            current_number = 0
        
        # Check if we've reached the maximum
        if current_number >= MAX_JOB_NUMBER:
            # Reset to 1 and create a backup of the old counter
            backup_file = f"{JOB_COUNTER_FILE}.backup_{int(time.time())}"
            try:
                with open(JOB_COUNTER_FILE, 'r') as f:
                    old_content = f.read()
                with open(backup_file, 'w') as f:
                    f.write(f"Counter reset from {current_number} to 1 at {datetime.now().isoformat()}\n")
                    f.write(f"Previous value: {old_content}\n")
            except:
                pass  # Don't fail if backup creation fails
            
            # Reset to 1
            current_number = 0
            st.warning(f"⚠️ Job counter reached maximum ({MAX_JOB_NUMBER}). Resetting to #0001.")
            st.info("📝 A backup of the previous counter has been created.")
        
        # Increment and save new counter
        next_number = current_number + 1
        with open(JOB_COUNTER_FILE, 'w') as f:
            f.write(str(next_number))
        
        return next_number
    except (ValueError, FileNotFoundError):
        # If there's any issue, start from 1
        with open(JOB_COUNTER_FILE, 'w') as f:
            f.write("1")
        return 1
    finally:
        # Always release the lock
        try:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        except:
            pass

def extract_job_number(job_id: str) -> str:
    """Extract job number from job ID for display purposes."""
    try:
        # New format: job_0001_user_model_timestamp (4-digit)
        if job_id.startswith('job_'):
            parts = job_id.split('_')
            if len(parts) >= 2:
                # Handle both 3-digit and 4-digit formats for backward compatibility
                job_num = parts[1]
                if job_num.isdigit():
                    return f"#{job_num}"
    except:
        pass
    
    # Fallback for old format or unrecognized format
    return "N/A"

def run_automl_job(job_id: str, config: Dict):
    """Run AutoML job using background job manager."""
    if job_manager is None:
        st.error("❌ Background job manager not available. Please ensure background_job_manager.py is in the same directory.")
        return False
    
    try:
        # Start job using background job manager
        success = job_manager.start_job(job_id, config)
        
        if success:
            st.success(f"✅ Job {job_id} started successfully in background!")
        else:
            st.error(f"❌ Failed to start job {job_id}")
            
        return success
        
    except Exception as e:
        st.error(f"❌ Error starting job: {str(e)}")
        return False

# === FLEXIBLE DATA INPUT FUNCTIONS ===
# These functions provide enhanced data input capabilities including BigQuery, file uploads, and existing files

def render_flexible_data_input_section(task_type: str):
    """
    Render the enhanced data input section with flexible data sources.
    """
    if not FLEXIBLE_INPUT_AVAILABLE:
        st.warning("⚠️ Enhanced data input features not available. Using basic functionality.")
        return None
    
    st.header(f"🚀 Enhanced Data Configuration - {task_type.title()}")
    st.markdown("**Choose your data source:**")
    
    # Data source selection
    source_type = st.radio(
        "📊 Data Source Type",
        options=["existing", "upload", "bigquery", "auto"],
        format_func=lambda x: {
            "existing": "📂 Existing Files (Rapid Modeler directory)",
            "upload": "📁 File Upload (CSV, Excel, TSV, JSON, Parquet)", 
            "bigquery": "🔗 GCP BigQuery Tables",
            "auto": "🤖 Auto-detect (Smart detection)"
        }[x],
        help="Select how you want to provide your data"
    )
    
    # Initialize data configuration
    data_config = {
        "source_type": source_type,
        "data_source": None,
        "options": {}
    }
    
    if source_type == "existing":
        data_config = render_existing_files_section()
    elif source_type == "upload":
        data_config = render_enhanced_upload_section()
    elif source_type == "bigquery":
        data_config = render_bigquery_section()
    elif source_type == "auto":
        data_config = render_auto_detection_section()
    
    return data_config

def render_existing_files_section():
    """Render the existing files selection section."""
    st.subheader("📂 Existing Files")
    
    # Discover available files
    available_files = []
    
    # Check current directory
    for ext in ['*.csv', '*.parquet', '*.json', '*.xlsx', '*.xls', '*.tsv']:
        available_files.extend(glob.glob(ext))
    
    # Check AutoML directory for known datasets
    known_datasets = ['bank.csv', 'IRIS.csv', 'bank', 'iris']
    automl_dir = os.path.dirname(os.path.abspath(__file__))
    
    for dataset in known_datasets:
        potential_paths = [
            os.path.join(automl_dir, dataset),
            os.path.join(automl_dir, f"{dataset}.csv")
        ]
        for path in potential_paths:
            if os.path.exists(path):
                available_files.append(os.path.basename(path))
    
    # Remove duplicates and sort
    available_files = sorted(list(set(available_files)))
    
    if available_files:
        selected_file = st.selectbox(
            "📁 Select existing file:",
            options=available_files,
            help="Choose from available files in the Rapid Modeler directory"
        )
        
        # Show file info
        if selected_file:
            st.success(f"✅ Selected: {selected_file}")
            
            # Preview option
            if st.checkbox("👀 Preview file", key="existing_preview"):
                try:
                    # Try to preview the file
                    if selected_file.endswith('.csv'):
                        df_preview = pd.read_csv(selected_file, nrows=5)
                        st.dataframe(df_preview)
                        st.info(f"📋 Showing first 5 rows of {selected_file}")
                    elif selected_file.endswith('.xlsx') or selected_file.endswith('.xls'):
                        df_preview = pd.read_excel(selected_file, nrows=5)
                        st.dataframe(df_preview)
                        st.info(f"📋 Showing first 5 rows of {selected_file}")
                except Exception as e:
                    st.warning(f"Could not preview file: {e}")
        
        return {
            "source_type": "existing",
            "data_source": selected_file,
            "options": {}
        }
    else:
        st.warning("📁 No existing files found.")
        st.info("💡 Available file types: CSV, Excel, TSV, JSON, Parquet")
        st.info("💡 Known datasets: bank.csv, IRIS.csv")
        
        return {
            "source_type": "existing", 
            "data_source": None,
            "options": {}
        }

def render_enhanced_upload_section():
    """Render the enhanced file upload section with multiple formats and delimiters."""
    st.subheader("📁 Enhanced File Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=['csv', 'tsv', 'tab', 'txt', 'parquet', 'json', 'xlsx', 'xls'],
        help="Upload your dataset - supports CSV, TSV, Excel, JSON, Parquet with custom delimiters",
        key="enhanced_upload"
    )
    
    if uploaded_file is not None:
        # File info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size / 1024:.1f} KB")
        
        # Format-specific options
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        options = {}
        
        if file_ext in ['.csv', '.tsv', '.tab', '.txt']:
            st.subheader("🔧 CSV/TSV Options")
            
            col1, col2 = st.columns(2)
            with col1:
                delimiter = st.selectbox(
                    "Delimiter",
                    options=[",", "\t", "|", ";", " "],
                    format_func=lambda x: {
                        ",": "Comma (,)",
                        "\t": "Tab (\\t)", 
                        "|": "Pipe (|)",
                        ";": "Semicolon (;)",
                        " ": "Space ( )"
                    }[x],
                    key="enhanced_delimiter"
                )
                options['delimiter'] = delimiter
                
            with col2:
                encoding = st.selectbox(
                    "Encoding",
                    options=['utf-8', 'latin-1', 'cp1252'],
                    key="enhanced_encoding"
                )
                options['encoding'] = encoding
        
        # Preview uploaded file
        if st.checkbox("👀 Preview uploaded data", key="enhanced_preview"):
            try:
                if file_ext == '.csv':
                    df_preview = pd.read_csv(uploaded_file, nrows=5, delimiter=options.get('delimiter', ','), encoding=options.get('encoding', 'utf-8'))
                elif file_ext in ['.xlsx', '.xls']:
                    df_preview = pd.read_excel(uploaded_file, nrows=5)
                elif file_ext == '.json':
                    df_preview = pd.read_json(uploaded_file, nrows=5)
                elif file_ext == '.parquet':
                    df_preview = pd.read_parquet(uploaded_file)
                    df_preview = df_preview.head(5)
                else:
                    df_preview = pd.read_csv(uploaded_file, nrows=5, delimiter=options.get('delimiter', ','))
                
                st.dataframe(df_preview, use_container_width=True)
                st.info(f"📋 Preview shows first 5 rows")
            except Exception as e:
                st.error(f"❌ Error previewing file: {str(e)}")
        
        return {
            "source_type": "upload",
            "data_source": uploaded_file,
            "options": options
        }
    else:
        return {
            "source_type": "upload",
            "data_source": None,
            "options": {}
        }

def build_bigquery_query(table_ref: str, options: dict = None, is_view: bool = False) -> str:
    """
    Build a BigQuery query with user configurations.
    
    Args:
        table_ref: BigQuery table reference (project.dataset.table or dataset.table)
        options: Dictionary containing query options like WHERE clauses, column selection, etc.
        is_view: Whether the table reference is a view (optional, will be auto-detected if not provided)
        
    Returns:
        str: Configured SQL query
    """
    if not options:
        options = {}
    
    # Start building the query
    select_clause = options.get('select_columns', '*')
    if select_clause and select_clause.strip():
        select_part = select_clause.strip()
    else:
        select_part = '*'
    
    # Build the FROM clause with sampling if specified
    from_part = f"`{table_ref}`"
    sampling_where = ""
    
    if options.get('sample_percent'):
        sample_percent = options['sample_percent']
        if is_view:
            # For views, use RAND() function for sampling
            sampling_where = f"AND RAND() < {sample_percent / 100.0}"
        else:
            # For tables, use TABLESAMPLE SYSTEM
            from_part += f" TABLESAMPLE SYSTEM ({sample_percent} PERCENT)"
    
    # Build the WHERE clause if specified
    where_part = ""
    if options.get('where_clause') and options['where_clause'].strip():
        where_part = f"WHERE {options['where_clause'].strip()}"
        if sampling_where:
            where_part += f" {sampling_where}"
    elif sampling_where:
        # If no WHERE clause but we have sampling, create one
        where_part = f"WHERE {sampling_where.strip('AND ')}"
    
    # Build the LIMIT clause if specified
    limit_part = ""
    if options.get('row_limit'):
        limit_part = f"LIMIT {options['row_limit']}"
    
    # Construct the final query
    query_parts = [f"SELECT {select_part}", f"FROM {from_part}"]
    
    if where_part:
        query_parts.append(where_part)
    
    if limit_part:
        query_parts.append(limit_part)
    
    return " ".join(query_parts)

def check_bigquery_table_size(table_ref: str, project_id: str = None, max_size_gb: float = 10.0, options: dict = None) -> tuple[bool, float, str]:
    """
    Check BigQuery table/view size before loading data using dry run.
    
    Args:
        table_ref: BigQuery table reference (project.dataset.table or dataset.table)
        project_id: GCP project ID (optional, will be extracted from table_ref if not provided)
        max_size_gb: Maximum allowed table size in GB
        options: Dictionary containing query options like WHERE clauses, column selection, etc.
        
    Returns:
        tuple: (is_allowed, size_gb, error_message)
    """
    try:
        from google.cloud import bigquery
        
        # Check if GCP authentication is available
        try:
            # Try to initialize the BigQuery client
            client = bigquery.Client()
        except Exception as auth_error:
            # If authentication fails, return a helpful error message
            if "key.json was not found" in str(auth_error) or "credentials" in str(auth_error).lower():
                return False, 0.0, "GCP authentication not configured. Please run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS environment variable."
            else:
                return False, 0.0, f"GCP authentication error: {str(auth_error)}"
        
        # Extract project_id from table_ref if not provided
        if not project_id:
            table_parts = table_ref.split('.')
            if len(table_parts) >= 3:
                project_id = table_parts[0]
            else:
                return False, 0.0, "Could not extract project ID from table reference"
        
        # Create a dry-run query job to get the actual bytes that would be processed
        # This approach works for both tables and views, and gives us the actual bytes
        # that would be processed when querying the table/view
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        
        # Build the query with user configurations if provided
        if options:
            # First, try to detect if this is a view
            is_view = False
            try:
                # Extract project_id from table_ref if needed
                table_parts = table_ref.split('.')
                if len(table_parts) >= 3:
                    check_project_id = table_parts[0]
                    dataset_id = table_parts[1]
                    table_id = table_parts[2]
                else:
                    # Use default project
                    check_project_id = client.project
                    dataset_id = table_parts[0]
                    table_id = table_parts[1]
                
                # Try to get table info
                table_ref_obj = client.dataset(dataset_id, project=check_project_id).table(table_id)
                table = client.get_table(table_ref_obj)
                is_view = table.table_type == 'VIEW'
            except Exception:
                # If we can't determine if it's a view, assume it's a table
                is_view = False
            
            # Build the query
            query = build_bigquery_query(table_ref, options, is_view)
        else:
            # Create a simple SELECT query to check the table/view
            query = f"SELECT * FROM `{table_ref}`"
        
        # Run the dry run (this doesn't actually execute the query, just estimates the bytes)
        sampling_removed = False
        try:
            query_job = client.query(query, job_config=job_config)
        except Exception as e:
            # If the query failed due to TABLESAMPLE on view, try without sampling
            if "TABLESAMPLE SYSTEM can only be applied directly to base tables" in str(e) and options and options.get('sample_percent'):
                # Remove sampling from options and try again
                options_without_sampling = options.copy()
                if 'sample_percent' in options_without_sampling:
                    del options_without_sampling['sample_percent']
                    sampling_removed = True
                query = build_bigquery_query(table_ref, options_without_sampling, True)  # Force view mode
                query_job = client.query(query, job_config=job_config)
            else:
                # Re-raise the error if it's not related to TABLESAMPLE
                raise e
        
        # Get the processed bytes from the dry run
        bytes_processed = query_job.total_bytes_processed
        
        # Convert to GB
        size_gb = bytes_processed / (1024**3)
        
        # If we have user configurations, run the actual configured query to get accurate size
        if options:
            try:
                # Build the actual query with user configurations
                actual_query = build_bigquery_query(table_ref, options, is_view)
                
                # Run dry run with the actual configured query
                actual_job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
                actual_query_job = client.query(actual_query, actual_job_config)
                actual_bytes_processed = actual_query_job.total_bytes_processed
                actual_size_gb = actual_bytes_processed / (1024**3)
                
                # Use the actual configured query size instead of hardcoded estimates
                size_gb = actual_size_gb
                
                # Log for debugging
                print(f"Original query size: {bytes_processed / (1024**3):.2f} GB")
                print(f"Configured query: {actual_query}")
                print(f"Configured query size: {actual_size_gb:.2f} GB")
                
            except Exception as e:
                print(f"Error running configured query dry run: {e}")
                # If configured query fails, keep the original estimate
                print("Keeping original table size estimate")
                # size_gb remains as the original estimate
        
        # Check if size is within limits
        is_allowed = size_gb <= max_size_gb
        
        # Create error message if sampling was removed
        error_msg = ""
        if sampling_removed:
            error_msg = "Table sampling was automatically disabled because this is a view (TABLESAMPLE only works on base tables)."
        
        return is_allowed, size_gb, error_msg
        
    except ImportError:
        return False, 0.0, "google-cloud-bigquery package not installed. Please install it with: pip install google-cloud-bigquery"
    except Exception as e:
        # Check if this is an authentication error
        if "key.json was not found" in str(e) or "credentials" in str(e).lower():
            return False, 0.0, "GCP authentication not configured. Please run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS environment variable."
        else:
            return False, 0.0, f"Error checking table size: {str(e)}"

def render_bigquery_section():
    """Render the enhanced BigQuery data source section with proven working configuration."""
    st.subheader("🔗 GCP BigQuery Direct Integration")
    
    # Check if BigQuery is available and authenticated
    try:
        from google.cloud import bigquery
        try:
            # Test authentication
            client = bigquery.Client()
            st.success("✅ **BigQuery integration enabled with proven working configuration!**")
            st.info("💡 Connect directly to BigQuery tables without downloading data locally - supports datasets of any size")
        except Exception as auth_error:
            if "key.json was not found" in str(auth_error) or "credentials" in str(auth_error).lower():
                st.warning("⚠️ **BigQuery integration requires GCP authentication**")
                st.error("❌ GCP authentication not configured")
                st.info("🔧 **To enable BigQuery integration:**")
                st.markdown("""
                1. **Install gcloud CLI**: Download from [Google Cloud Console](https://cloud.google.com/sdk/docs/install)
                2. **Authenticate**: Run `gcloud auth application-default login` in your terminal
                3. **Set project**: Run `gcloud config set project YOUR_PROJECT_ID`
                4. **Restart Streamlit**: Restart the application after authentication
                """)
                st.info("💡 **Alternative**: Use file upload or existing files instead of BigQuery")
                return {
                    "source_type": "bigquery", 
                    "data_source": None,
                    "options": {}
                }
            else:
                st.error(f"❌ GCP authentication error: {str(auth_error)}")
                return {
                    "source_type": "bigquery", 
                    "data_source": None,
                    "options": {}
                }
    except ImportError:
        st.warning("⚠️ **BigQuery integration requires additional packages**")
        st.error("❌ google-cloud-bigquery package not installed")
        st.info("🔧 **To enable BigQuery integration:**")
        st.markdown("""
        1. **Install package**: Run `pip install google-cloud-bigquery`
        2. **Authenticate**: Run `gcloud auth application-default login`
        3. **Restart Streamlit**: Restart the application after installation
        """)
        st.info("💡 **Alternative**: Use file upload or existing files instead of BigQuery")
        return {
            "source_type": "bigquery", 
            "data_source": None,
            "options": {}
        }
    
    st.markdown("**Supported formats:** `project.dataset.table` or `dataset.table` (uses default project)")
    
    # Authentication notice
    st.info("🔐 **Authentication Required**: BigQuery integration requires GCP authentication. If you haven't set it up, you'll see authentication errors below. Use the troubleshooting steps above to resolve.")
    
    # Method selection
    input_method = st.radio(
        "🎯 Input Method",
        ["Manual Entry", "Full Table Reference"],
        help="Choose how to specify your BigQuery table"
    )
    
    table_ref = None
    project_id = None  # Initialize to avoid UnboundLocalError
    
    if input_method == "Manual Entry":
        # BigQuery options in columns
        col1, col2 = st.columns(2)
        
        with col1:
            project_id = st.text_input(
                "Project ID", 
                help="Your GCP project ID (leave empty to use default)",
                placeholder="my-gcp-project",
                key="bq_project"
            )
            # Clean project_id to remove any backticks
            if project_id:
                project_id = project_id.strip('`')
            dataset_id = st.text_input(
                "Dataset ID", 
                help="BigQuery dataset name",
                placeholder="my_dataset",
                key="bq_dataset"
            )
            
        with col2:
            table_id = st.text_input(
                "Table ID", 
                help="BigQuery table name",
                placeholder="my_table",
                key="bq_table"
            )
            
        # Construct table reference
        if dataset_id and table_id:
            if project_id:
                table_ref = f"{project_id}.{dataset_id}.{table_id}"
            else:
                table_ref = f"{dataset_id}.{table_id}"
                st.info("ℹ️ Using default project (configure via gcloud or environment)")
    
    else:  # Full Table Reference
        table_ref = st.text_input(
            "Full Table Reference",
            help="Complete BigQuery table reference",
            placeholder="project.dataset.table or dataset.table",
            key="bq_full_ref"
        )
        
        # Clean table_ref to remove any backticks
        if table_ref:
            table_ref = table_ref.strip('`')
        
        # Extract project_id from full table reference if possible
        if table_ref:
            table_parts = table_ref.split('.')
            if len(table_parts) >= 3:
                project_id = table_parts[0]  # Extract project ID from full reference
                # Clean project_id to remove any backticks
                if project_id:
                    project_id = project_id.strip('`')
            # If we can't extract project_id, it stays None (already initialized)
    
    if table_ref:
        # Validate table reference format
        parts = table_ref.split('.')
        if len(parts) not in [2, 3]:
            st.error("❌ Invalid table reference format. Use 'project.dataset.table' or 'dataset.table'")
            return {
                "source_type": "bigquery", 
                "data_source": None,
                "options": {}
            }
        
        st.success(f"✅ Table reference: `{table_ref}`")
        
        # Enhanced BigQuery options - SHOW FIRST
        st.subheader("🔧 BigQuery Configuration")
        
        # General notice about large tables
        st.info("💡 **Pro Tip:** For very large tables (>100GB), if you don't see significant size reduction after applying filters, consider creating a separate filtered table or view for more accurate estimates and better performance.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Data Processing Options**")
            st.info("💡 Configure your data processing options below, then check the size of your filtered data")
            
            # Quick example with user's working table
            with st.expander("📖 Example Configuration", expanded=False):
                st.code("""
# Example working configuration:
Project ID: atus-prism-dev
Dataset: ds_sandbox  
Table: sub_b2c_add_video_dataset_DNA_2504_N02

Full reference: atus-prism-dev.ds_sandbox.sub_b2c_add_video_dataset_DNA_2504_N02
                """)
                st.info("💡 Replace with your own project, dataset, and table names")
            
            # Query limits (optional)
            use_limit = st.checkbox(
                "Limit rows (for testing/development)", 
                help="Apply SQL LIMIT to BigQuery query for faster development/testing. Uses 'SELECT * FROM table LIMIT N' - this gives you exactly N rows, not N rows per partition.",
                key="bq_limit_check"
            )
            
            row_limit = None
            if use_limit:
                row_limit = st.number_input(
                    "Row limit", 
                    min_value=1, 
                    value=10000,
                    step=1000,
                    help="Exact number of rows to process using SQL LIMIT clause",
                    key="bq_limit"
                )
                st.info(f"📋 Will process exactly {row_limit:,} rows using SQL LIMIT")
                st.success("✅ Uses proper SQL LIMIT clause - guarantees exactly the specified number of rows")
            
            # Sampling option
            use_sampling = st.checkbox(
                "Use table sampling (for large datasets)",
                help="Use BigQuery's TABLESAMPLE to randomly sample a percentage of your data - useful for initial model development on very large tables",
                key="bq_sampling"
            )
            
            sample_percent = None
            if use_sampling:
                sample_percent = st.slider(
                    "Sample percentage",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="Percentage of table to sample randomly",
                    key="bq_sample_percent"
                )
                st.info(f"📋 Will sample {sample_percent}% of the table randomly")
                st.markdown("**When to use sampling:**")
                st.write("• Initial model development with billion+ row tables")
                st.write("• Quick prototyping and feature exploration") 
                st.write("• Cost optimization for development environments")
                st.warning("⚠️ **Note:** Table sampling (TABLESAMPLE) only works on base tables, not views. For views, sampling will be automatically disabled.")
        
        with col2:
            st.markdown("**🔍 Filtering & Options**")
            
            # Custom WHERE clause
            custom_where = st.text_area(
                "Custom WHERE clause (optional)",
                help="Add SQL WHERE conditions (without 'WHERE' keyword)",
                placeholder="date_column >= '2023-01-01' AND status = 'active'",
                key="bq_where"
            )
            
            # Custom SELECT columns
            custom_select = st.text_area(
                "Custom column selection (optional)",
                help="Specify columns to select (leave empty for all columns)",
                placeholder="column1, column2, UPPER(column3) as column3_upper",
                key="bq_select"
            )
            
            # Additional BigQuery options
            use_legacy_sql = st.checkbox(
                "Use Legacy SQL", 
                help="Use BigQuery Legacy SQL instead of Standard SQL",
                key="bq_legacy"
            )
        
        # Build options dictionary first (needed for summary and connection test)
        options = {}
        
        if row_limit:
            options['row_limit'] = row_limit
        
        if sample_percent:
            options['sample_percent'] = sample_percent
            
        if custom_where:
            options['where_clause'] = custom_where.strip()
            
        if custom_select:
            options['select_columns'] = custom_select.strip()
            
        if use_legacy_sql:
            options['use_legacy_sql'] = True
        
        # BigQuery connector options
        options['bigquery_options'] = {
            'useAvroLogicalTypes': 'true',
            'viewsEnabled': 'true'
        }
        
        # Store project_id for proper BigQuery connector mapping
        if project_id is not None:
            options['project_id'] = project_id
        
        # Show active options summary
        if options:
            st.success("✅ **BigQuery Options Applied:**")
            active_options = []
            
            if row_limit:
                active_options.append(f"📋 Row limit: {row_limit:,} rows")
            if sample_percent:
                active_options.append(f"📊 Table sampling: {sample_percent}%")
            if custom_where:
                active_options.append(f"🔍 WHERE clause: {custom_where[:50]}{'...' if len(custom_where) > 50 else ''}")
            if custom_select:
                active_options.append(f"📝 Column selection: {custom_select[:50]}{'...' if len(custom_select) > 50 else ''}")
            if use_legacy_sql:
                active_options.append("🔧 Legacy SQL enabled")
            
            for option in active_options:
                st.write(f"   • {option}")
            
            # Show what query will be generated
            if any([sample_percent, custom_where, custom_select]):
                st.info("🔧 **Query Method:** Custom SQL query with your options")
                query = build_bigquery_query(table_ref, options)
                st.code(query)
            else:
                st.info("🔧 **Query Method:** Direct table reference")
                if row_limit:
                    st.code(f"Direct table load with maxRowsPerPartition={row_limit}")
                else:
                    st.code(f"Direct table load: {table_ref}")
        
        # Check table/view size AFTER configuration
        st.subheader("📊 Table/View Size Check")
        max_size_gb = 10.0  # Configurable maximum size limit
        
        # Track size check result
        size_check_passed = False
        
        with st.spinner("Checking BigQuery table/view size with your configurations..."):
            is_allowed, size_gb, error_msg = check_bigquery_table_size(table_ref, project_id, max_size_gb, options)
            
            if error_msg:
                st.error(f"❌ Table/view size check failed: {error_msg}")
                
                # Provide specific guidance based on error type
                if "GCP authentication not configured" in error_msg:
                    st.warning("🔐 **GCP Authentication Required**")
                    st.info("💡 **To enable BigQuery integration:**")
                    st.markdown("""
                    1. **Install gcloud CLI**: Download from [Google Cloud Console](https://cloud.google.com/sdk/docs/install)
                    2. **Authenticate**: Run `gcloud auth application-default login` in your terminal
                    3. **Set project**: Run `gcloud config set project YOUR_PROJECT_ID`
                    4. **Restart Streamlit**: Restart the application after authentication
                    """)
                    st.info("💡 **Alternative**: Use file upload or existing files instead of BigQuery")
                elif "google-cloud-bigquery package not installed" in error_msg:
                    st.warning("📦 **Package Installation Required**")
                    st.info("💡 **To enable BigQuery integration:**")
                    st.markdown("""
                    1. **Install package**: Run `pip install google-cloud-bigquery`
                    2. **Authenticate**: Run `gcloud auth application-default login`
                    3. **Restart Streamlit**: Restart the application after installation
                    """)
                    st.info("💡 **Alternative**: Use file upload or existing files instead of BigQuery")
                else:
                    st.info("💡 **General Troubleshooting:**")
                    st.markdown("""
                    - Ensure you're authenticated with Google Cloud: `gcloud auth application-default login`
                    - Check that the table/view exists and you have access
                    - Verify your project ID and table/view name are correct
                    - Install BigQuery client: `pip install google-cloud-bigquery`
                    """)
                
                return {
                    "source_type": "bigquery", 
                    "data_source": None,
                    "options": {},
                    "size_check_passed": False
                }
            
            # Display size information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Configured Data Size", f"{size_gb:.2f} GB")
            with col2:
                st.metric("Size Limit", f"{max_size_gb} GB")
            with col3:
                status_icon = "✅" if is_allowed else "❌"
                status_text = "Within Limits" if is_allowed else "Exceeds Limits"
                st.metric("Status", f"{status_icon} {status_text}")
            
            # Show size estimation information
            if options and any([options.get('row_limit'), options.get('where_clause'), options.get('select_columns'), options.get('sample_percent')]):
                st.info("📊 **Size Estimation:** Using actual BigQuery dry run with your configured query to get accurate size estimates.")
                st.write("• This reflects the real impact of your filtering options")
                st.write("• No hardcoded estimates - actual query analysis")
                
                # Check if the size is still very large despite configurations
                if size_gb > 100:  # If still over 100GB after configurations
                    st.warning("⚠️ **Large Table Notice:**")
                    st.write("• If you're not seeing significant size reduction after applying filters, the table might be extremely large")
                    st.write("• **Recommendation:** Consider creating a separate table/view with your configurations for more accurate estimates")
                    st.write("• **Example:** `CREATE TABLE `project.dataset.filtered_table` AS SELECT * FROM `project.dataset.original_table` WHERE your_conditions`")
                    st.write("• This approach provides more precise size estimates and better performance")
            
            # Show warning if sampling was removed for views
            if error_msg and "Table sampling was automatically disabled" in error_msg:
                st.warning("⚠️ **View Detected:** Table sampling was automatically disabled because this is a view. TABLESAMPLE only works on base tables.")
            
            if not is_allowed:
                st.error(f"❌ **Configured data size too large!** ({size_gb:.2f} GB > {max_size_gb} GB)")
                st.warning("⚠️ **Job Submission Blocked:**")
                st.markdown(f"""
                Your configured data (with filters, sampling, etc.) is still too large for automated processing. **Job submission is not allowed** until you reduce the data size.
                
                **Required Actions:**
                1. **Contact the ML Team** for assistance with large datasets
                2. **Reduce your data size** by:
                   - Using more restrictive WHERE clauses
                   - Selecting fewer columns
                   - Using smaller table sampling percentages
                   - Adding row limits
                   - Creating a smaller view or table
                
                **Current configured size:** {size_gb:.2f} GB  
                **Maximum allowed:** {max_size_gb} GB
                
                **💡 Tip:** Adjust the filtering options above to further reduce your dataset size.
                """)
                
                # Additional guidance for very large tables
                if size_gb > 100:
                    st.info("💡 **For Very Large Tables (>100GB):**")
                    st.write("• **Option 1:** Create a separate filtered table for more accurate estimates")
                    st.code("""
CREATE TABLE `your-project.your-dataset.filtered_table` AS
SELECT * FROM `your-project.your-dataset.original_table`
WHERE your_filtering_conditions
LIMIT your_row_limit;
                    """, language="sql")
                    st.write("• **Option 2:** Use BigQuery's `CREATE VIEW` for dynamic filtering")
                    st.code("""
CREATE VIEW `your-project.your-dataset.filtered_view` AS
SELECT * FROM `your-project.your-dataset.original_table`
WHERE your_filtering_conditions;
                    """, language="sql")
                    st.write("• **Benefits:** More accurate size estimates, better performance, and cleaner data management")
                    st.write("• **Why this helps:** When tables are extremely large, BigQuery's dry run estimates may not fully reflect the impact of filters")
                
                # Don't return early - continue to show configuration
                size_check_passed = False
            else:
                size_check_passed = True  # Size is within limits
                st.success("✅ **Data size is within limits!** Your configured BigQuery query is ready for processing.")
        

        
        # BigQuery connection test with working configuration
        # Only allow connection test if size check passed or size is reasonable
        can_test_connection = size_check_passed or (size_gb <= 10)  # Allow test if size check passed or table is under 10GB
        
        if can_test_connection:
            if st.checkbox("🔍 Test BigQuery Connection", help="Validate table access and preview schema with your configured options"):
                if table_ref:
                    with st.spinner("Testing BigQuery connection with your configured options..."):
                        try:
                            # Use the proven working BigQuery configuration
                            from pyspark.sql import SparkSession
                            
                            # Create test session with proven working config
                            test_spark = SparkSession.builder \
                                .appName("Streamlit BigQuery Test") \
                                .config("spark.driver.bindAddress", "127.0.0.1") \
                                .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1") \
                                .config("spark.driver.memory", "8g") \
                                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                                .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
                                .config("spark.network.timeout", "600s") \
                                .config("spark.executor.heartbeatInterval", "120s") \
                                .getOrCreate()
                            
                            # Validate Spark session was created successfully
                            if test_spark is None:
                                raise RuntimeError("Failed to create Spark session")
                            
                            # Validate table_ref is not None or empty
                            if not table_ref or table_ref.strip() == "":
                                raise ValueError("Table reference is empty or None")
                            
                            # Validate project_id is properly set
                            if not project_id or project_id.strip() == "":
                                st.warning("⚠️ No project ID specified, using default project")
                                project_id = "default-project"
                            
                            # For connection testing, we'll use the original table reference
                            # but apply the configurations through Spark SQL after loading
                            st.info(f"🔧 **Testing connection with table:** {table_ref}")
                            
                            # Test with direct table reference first
                            test_df = test_spark.read.format("bigquery") \
                                .option("parentProject", project_id or "default-project") \
                                .option("viewsEnabled", "true") \
                                .option("useAvroLogicalTypes", "true") \
                                .option("table", table_ref) \
                                .option("maxRowsPerPartition", 5) \
                                .load()
                            
                            # Apply configurations if they exist
                            if options and any([options.get('row_limit'), options.get('where_clause'), options.get('select_columns')]):
                                st.info("🔧 **Applying your configurations to test data...**")
                                
                                try:
                                    # Apply WHERE clause if specified
                                    if options.get('where_clause'):
                                        # Convert SQL WHERE clause to Spark filter expression
                                        where_clause = options['where_clause']
                                        # For now, we'll skip complex WHERE clauses in testing
                                        # as they require SQL parsing which is complex
                                        st.info(f"⚠️ **Note:** WHERE clause '{where_clause}' will be applied in the actual job, not in this test.")
                                    
                                    # Apply column selection if specified
                                    if options.get('select_columns') and options['select_columns'].strip() != '*':
                                        selected_columns = [col.strip() for col in options['select_columns'].split(',')]
                                        # Check which columns actually exist in the dataframe
                                        existing_columns = [col for col in selected_columns if col in test_df.columns]
                                        if existing_columns:
                                            test_df = test_df.select(*existing_columns)
                                            st.info(f"✅ **Applied column selection:** {', '.join(existing_columns)}")
                                        else:
                                            st.warning(f"⚠️ **Column selection skipped:** None of the specified columns found in the table")
                                    
                                    # Apply row limit if specified
                                    if options.get('row_limit'):
                                        test_df = test_df.limit(options['row_limit'])
                                        st.info(f"✅ **Applied row limit:** {options['row_limit']} rows")
                                    
                                    st.success("✅ **Configurations applied successfully!**")
                                    
                                except Exception as config_error:
                                    st.warning(f"⚠️ **Configuration application warning:** {str(config_error)}")
                                    st.info("💡 **Note:** The connection test will proceed with the original table data. Your configurations will be applied in the actual job.")
                            
                            # Get basic info
                            sample_count = test_df.count()
                            column_count = len(test_df.columns)
                            columns = test_df.columns
                            
                            # Show success
                            st.success(f"✅ BigQuery connection successful!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Sample Rows", sample_count)
                            with col2:
                                st.metric("Total Columns", column_count)
                            
                            st.write("**Available Columns:**")
                            st.write(", ".join(columns[:10]) + ("..." if len(columns) > 10 else ""))
                            
                            if sample_count > 0:
                                if sample_count > 100000:
                                    st.info("📊 **Large dataset detected** - Sample data preview is disabled for datasets with more than 100K rows to improve performance.")
                                else:
                                    st.write("**Sample Data:**")
                                    sample_pandas = test_df.limit(3).toPandas()
                                    st.dataframe(sample_pandas)
                        except Exception as e:
                            st.error(f"❌ BigQuery connection test failed: {str(e)}")
                        finally:
                            # Ensure Spark session is stopped
                            try:
                                if 'test_spark' in locals() and test_spark is not None:
                                    test_spark.stop()
                            except:
                                pass  # Ignore errors during cleanup
                else:
                    st.warning("⚠️ Please specify a BigQuery table reference first")
        else:
            st.info("ℹ️ **Connection Test Disabled:** Table size is too large for safe connection testing. Please reduce the data size or create a filtered table/view first.")
        

        
        # Show configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            config_summary = {
                'table_reference': table_ref,
                'options': options,
                'size_check_passed': size_check_passed
            }
            st.json(config_summary)
        
        return {
            "source_type": "bigquery",
            "data_source": table_ref,
            "options": options,
            "size_check_passed": size_check_passed
        }
    else:
        st.info("📝 Please provide BigQuery table information above")
        return {
            "source_type": "bigquery", 
            "data_source": None,
            "options": {},
            "size_check_passed": False
        }

def render_auto_detection_section():
    """Render the auto-detection data source section."""
    st.subheader("🤖 Auto-detect Data Source")
    
    # Input field for data reference
    data_input = st.text_input(
        "Data Reference",
        help="Enter file path, BigQuery table reference, or data identifier",
        placeholder="e.g., bank.csv, project.dataset.table, /path/to/file.parquet",
        key="auto_detect_input"
    )
    
    if data_input:
        # Attempt to detect source type
        if '.' in data_input and len(data_input.split('.')) == 3:
            # Looks like BigQuery table reference
            detected_type = "bigquery"
            st.info(f"🔗 Detected as BigQuery table: {data_input}")
        elif data_input.endswith(('.csv', '.xlsx', '.json', '.parquet')):
            # Looks like file
            if os.path.exists(data_input):
                detected_type = "existing"
                st.success(f"📂 Detected as existing file: {data_input}")
            else:
                detected_type = "upload"
                st.warning(f"📁 File not found locally - please upload: {data_input}")
        else:
            # Default to existing file search
            detected_type = "existing"
            st.info(f"📂 Searching for existing file: {data_input}")
        
        return {
            "source_type": detected_type,
            "data_source": data_input,
            "options": {}
        }
    else:
        st.info("📝 Enter a data reference to auto-detect the source type")
        return {
            "source_type": "auto",
            "data_source": None,
            "options": {}
        }

# === END FLEXIBLE DATA INPUT FUNCTIONS ===

def create_job_submission_page():
    """Create the job submission page."""
    st.title("🚀 Rapid Modeler Job Submission")
    st.markdown("Configure and submit your Rapid Modeler training job")
    
    config = load_config()
    if not config:
        st.error("Cannot load configuration file!")
        return
    
    # Sidebar for basic settings
    with st.sidebar:
        st.header("📋 Basic Settings")
        
        # Task Type Selection - NEW!
        task_type = st.selectbox(
            "🎯 Task Type",
            ["classification", "regression", "clustering"],
            index=0,
            help="Select the type of machine learning task"
        )
        
        # Job identification
        user_id = st.text_input("User ID", value="automl_user", help="Unique identifier for this user")
        model_name = st.text_input("Model Name", value="automl_model", help="Name for this modeling exercise")
        
        # Environment selection
        config_environments = list(config.get('environments', {}).keys())
        default_environments = ['production', 'development', 'staging']
        
        # Combine and remove duplicates while preserving order
        environments = config_environments.copy()
        for env in default_environments:
            if env not in environments:
                environments.append(env)
        
        environment = st.selectbox(
            "Environment", 
            environments,
            index=0,
            help="Select the execution environment"
        )
        
        # Preset selection
        presets = ['', 'quick', 'comprehensive']
        preset = st.selectbox(
            "Preset Configuration",
            presets,
            help="Quick: Fast training, Comprehensive: Full training"
        )
    
    # Main area - tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data & Target", "🤖 Models", "⚙️ Parameters", "🔧 Advanced"])
    
    with tab1:
        st.header(f"Data Configuration - {task_type.title()}")
        
        # Main data source selection with radio buttons
        st.subheader("📊 Choose Your Data Source")
        
        data_source_method = st.radio(
            "Select how you want to provide data:",
            ["🔗 BigQuery Tables", "📁 Upload Files", "📂 Existing Files"],
            help="Choose your preferred data input method"
        )
        
        # Initialize variables
        enhanced_data_config = None
        data_file = None
        uploaded_file = None
        
        # Handle each data source method
        if data_source_method == "🔗 BigQuery Tables":
            # BigQuery configuration
            st.markdown("---")
            enhanced_data_config = render_bigquery_section()
            
            if enhanced_data_config and enhanced_data_config.get('data_source'):
                st.success("✅ BigQuery configuration completed!")
                st.session_state['enhanced_data_config'] = enhanced_data_config
                
                with st.expander("📋 BigQuery Configuration Summary", expanded=False):
                    st.json(enhanced_data_config)
            else:
                st.info("📝 Complete the BigQuery configuration above to proceed")
                # Don't return - show target column and data processing sections
                
        elif data_source_method == "📁 Upload Files":
            # File upload configuration 
            st.markdown("---")
            enhanced_data_config = render_enhanced_upload_section()
            
            if enhanced_data_config and enhanced_data_config.get('data_source'):
                st.success("✅ File upload configuration completed!")
                st.session_state['enhanced_data_config'] = enhanced_data_config
                
                # Handle the uploaded file
                uploaded_file = enhanced_data_config['data_source']
                data_file = f"uploaded_{uploaded_file.name}"
                
                with st.expander("📋 Upload Configuration Summary", expanded=False):
                    config_summary = {
                        'source_type': enhanced_data_config['source_type'],
                        'file_name': uploaded_file.name,
                        'file_size_kb': round(uploaded_file.size / 1024, 1),
                        'options': enhanced_data_config.get('options', {})
                    }
                    st.json(config_summary)
            else:
                st.info("📝 Please upload a file above to proceed")
                # Don't return - show target column and data processing sections
                
        elif data_source_method == "📂 Existing Files":
            # Existing files configuration
            st.markdown("---")
            enhanced_data_config = render_existing_files_section()
            
            if enhanced_data_config and enhanced_data_config.get('data_source'):
                st.success("✅ Existing file selected!")
                st.session_state['enhanced_data_config'] = enhanced_data_config
                
                # Handle existing file
                data_file = enhanced_data_config['data_source']
                
                with st.expander("📋 File Configuration Summary", expanded=False):
                    st.json(enhanced_data_config)
            else:
                st.info("📝 Please select an existing file above to proceed")
                # Don't return - show target column and data processing sections
        
        # Target column selection section
        st.markdown("---")
        st.subheader("🎯 Target Column Selection")
        
        # Target column (not needed for clustering)
        if task_type == "clustering":
            target_column = None
            st.info("🔍 Clustering doesn't require a target column - the algorithm will discover patterns automatically!")
        else:
            # For classification and regression, we need a target column
            # Check if data source is properly configured
            data_configured = enhanced_data_config and enhanced_data_config.get('data_source')
            
            if not data_configured:
                # Data source not configured yet - show placeholder
                st.info("⬆️ Please configure your data source above first")
                target_column = st.text_input(
                    "🎯 Target Column", 
                    value="target",
                    help=f"Column name to predict ({task_type}) - will be enabled after data source configuration",
                    disabled=True
                )
            elif data_source_method == "📁 Upload Files" and uploaded_file is not None:
                # Try to suggest target column from uploaded data
                try:
                    import pandas as pd
                    if uploaded_file.name.endswith('.csv'):
                        df_temp = pd.read_csv(uploaded_file, nrows=1000)
                    elif uploaded_file.name.endswith('.parquet'):
                        df_temp = pd.read_parquet(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df_temp = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df_temp = pd.read_excel(uploaded_file, nrows=1000)
                    
                    # Show available columns and suggest target
                    st.write("**Available Columns:**")
                    st.write(", ".join(df_temp.columns.tolist()))
                    
                    # Suggest target column based on task type
                    suggested_target = "target"
                    if task_type == "classification":
                        # Look for binary or categorical columns
                        categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
                        binary_cols = [col for col in df_temp.columns if df_temp[col].nunique() == 2]
                        
                        if binary_cols:
                            suggested_target = binary_cols[0]
                        elif categorical_cols:
                            suggested_target = categorical_cols[0]
                        elif len(df_temp.columns) > 0:
                            suggested_target = df_temp.columns[-1]
                    
                    elif task_type == "regression":
                        # Look for numerical columns
                        numerical_cols = df_temp.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        if numerical_cols:
                            suggested_target = numerical_cols[0]
                        elif len(df_temp.columns) > 0:
                            suggested_target = df_temp.columns[-1]
                    
                    # Target column selection with dropdown
                    available_columns = df_temp.columns.tolist()
                    target_column = st.selectbox(
                        "🎯 Target Column",
                        available_columns,
                        index=available_columns.index(suggested_target) if suggested_target in available_columns else 0,
                        help=f"Column to predict ({task_type})"
                    )
                    
                except Exception as e:
                    st.warning(f"Could not analyze uploaded file: {str(e)}")
                    target_column = st.text_input(
                        "🎯 Target Column", 
                        value="target",
                        help=f"Column name to predict ({task_type})"
                    )
            else:
                # For BigQuery and existing files, use text input
                target_column = st.text_input(
                    "🎯 Target Column", 
                    value="target",
                    help=f"Column name to predict ({task_type})"
                )
                
                if data_source_method == "🔗 BigQuery Tables":
                    st.info("💡 Enter the exact column name from your BigQuery table")
                elif data_source_method == "📂 Existing Files":
                    st.info("💡 Enter the exact column name from your selected file")
        
        # OOT (Out-of-Time) Datasets Section - Only for classification and regression
        if task_type in ["classification", "regression"]:
            st.markdown("---")
            st.subheader("📅 Out-of-Time (OOT) Datasets")
            st.info("💡 OOT datasets help evaluate model performance on future/unseen time periods")
            
            oot_enabled = st.checkbox("Include OOT Datasets", value=False, help="Add additional datasets for temporal validation")
            
            oot1_data = None
            oot2_data = None
            oot1_config = None
            oot2_config = None
            
            if oot_enabled:
                # Adapt OOT input method based on main data source method
                if data_source_method == "🔗 BigQuery Tables":
                    # BigQuery OOT input
                    st.info("🔗 Since you're using BigQuery, provide BigQuery table references for OOT datasets")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📅 OOT1 BigQuery Table**")
                        oot1_table = st.text_input(
                            "OOT1 Table Reference",
                            placeholder="project.dataset.oot1_table",
                            help="BigQuery table reference for first out-of-time dataset",
                            key="oot1_bigquery"
                        )
                        
                        if oot1_table:
                            # Validate BigQuery table format
                            if '.' in oot1_table and len(oot1_table.split('.')) >= 2:
                                st.success(f"✅ OOT1 table: {oot1_table}")
                                
                                # Store OOT1 configuration
                                oot1_config = {
                                    "source_type": "bigquery",
                                    "data_source": oot1_table,
                                    "options": enhanced_data_config.get('options', {}) if enhanced_data_config else {}
                                }
                            else:
                                st.error("❌ Invalid table format. Use: project.dataset.table")
                    
                    with col2:
                        st.markdown("**📅 OOT2 BigQuery Table**")
                        oot2_table = st.text_input(
                            "OOT2 Table Reference",
                            placeholder="project.dataset.oot2_table",
                            help="BigQuery table reference for second out-of-time dataset (optional)",
                            key="oot2_bigquery"
                        )
                        
                        if oot2_table:
                            # Validate BigQuery table format
                            if '.' in oot2_table and len(oot2_table.split('.')) >= 2:
                                st.success(f"✅ OOT2 table: {oot2_table}")
                                
                                # Store OOT2 configuration
                                oot2_config = {
                                    "source_type": "bigquery",
                                    "data_source": oot2_table,
                                    "options": enhanced_data_config.get('options', {}) if enhanced_data_config else {}
                                }
                            else:
                                st.error("❌ Invalid table format. Use: project.dataset.table")
                
                elif data_source_method == "📂 Existing Files":
                    # File path OOT input
                    st.info("📂 Since you're using existing files, provide file paths for OOT datasets")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📅 OOT1 File Path**")
                        oot1_file_path = st.text_input(
                            "OOT1 File Path",
                            placeholder="oot1_data.csv",
                            help="File path for first out-of-time dataset",
                            key="oot1_filepath"
                        )
                        
                        if oot1_file_path:
                            if os.path.exists(oot1_file_path):
                                st.success(f"✅ OOT1 file found: {oot1_file_path}")
                                oot1_config = {
                                    "source_type": "existing",
                                    "data_source": oot1_file_path,
                                    "options": {}
                                }
                            else:
                                st.error(f"❌ File not found: {oot1_file_path}")
                    
                    with col2:
                        st.markdown("**📅 OOT2 File Path**")
                        oot2_file_path = st.text_input(
                            "OOT2 File Path",
                            placeholder="oot2_data.csv",
                            help="File path for second out-of-time dataset (optional)",
                            key="oot2_filepath"
                        )
                        
                        if oot2_file_path:
                            if os.path.exists(oot2_file_path):
                                st.success(f"✅ OOT2 file found: {oot2_file_path}")
                                oot2_config = {
                                    "source_type": "existing",
                                    "data_source": oot2_file_path,
                                    "options": {}
                                }
                            else:
                                st.error(f"❌ File not found: {oot2_file_path}")
                
                else:
                    # File upload OOT input (default for uploaded files)
                    st.info("📁 Upload OOT dataset files")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📅 OOT1 Dataset**")
                        oot1_data = st.file_uploader(
                            "Choose OOT1 file",
                            type=['csv', 'xlsx', 'xls', 'tsv', 'json', 'parquet'],
                            help="First out-of-time validation dataset",
                            key="oot1_upload"
                        )
                        
                        if oot1_data is not None:
                            st.success(f"✅ OOT1 uploaded: {oot1_data.name}")
                            st.info(f"📊 Size: {oot1_data.size / 1024:.1f} KB")
                            
                            # Store OOT1 configuration
                            oot1_config = {
                                "source_type": "upload",
                                "data_source": oot1_data,
                                "options": {}
                            }
                            
                            # Preview option for OOT1
                            if st.checkbox("👀 Preview OOT1 data", key="oot1_preview"):
                                try:
                                    if oot1_data.name.endswith('.csv'):
                                        df_preview = pd.read_csv(oot1_data, nrows=3)
                                    elif oot1_data.name.endswith(('.xlsx', '.xls')):
                                        df_preview = pd.read_excel(oot1_data, nrows=3)
                                    elif oot1_data.name.endswith('.json'):
                                        df_preview = pd.read_json(oot1_data, nrows=3)
                                    elif oot1_data.name.endswith('.parquet'):
                                        df_preview = pd.read_parquet(oot1_data).head(3)
                                    else:
                                        df_preview = pd.read_csv(oot1_data, nrows=3)
                                    
                                    st.dataframe(df_preview, use_container_width=True)
                                    st.caption("First 3 rows of OOT1 dataset")
                                except Exception as e:
                                    st.error(f"❌ Error previewing OOT1: {str(e)}")
                    
                    with col2:
                        st.markdown("**📅 OOT2 Dataset**")
                        oot2_data = st.file_uploader(
                            "Choose OOT2 file",
                            type=['csv', 'xlsx', 'xls', 'tsv', 'json', 'parquet'],
                            help="Second out-of-time validation dataset (optional)",
                            key="oot2_upload"
                        )
                        
                        if oot2_data is not None:
                            st.success(f"✅ OOT2 uploaded: {oot2_data.name}")
                            st.info(f"📊 Size: {oot2_data.size / 1024:.1f} KB")
                            
                            # Store OOT2 configuration
                            oot2_config = {
                                "source_type": "upload", 
                                "data_source": oot2_data,
                                "options": {}
                            }
                            
                            # Preview option for OOT2
                            if st.checkbox("👀 Preview OOT2 data", key="oot2_preview"):
                                try:
                                    if oot2_data.name.endswith('.csv'):
                                        df_preview = pd.read_csv(oot2_data, nrows=3)
                                    elif oot2_data.name.endswith(('.xlsx', '.xls')):
                                        df_preview = pd.read_excel(oot2_data, nrows=3)
                                    elif oot2_data.name.endswith('.json'):
                                        df_preview = pd.read_json(oot2_data, nrows=3)
                                    elif oot2_data.name.endswith('.parquet'):
                                        df_preview = pd.read_parquet(oot2_data).head(3)
                                    else:
                                        df_preview = pd.read_csv(oot2_data, nrows=3)
                                    
                                    st.dataframe(df_preview, use_container_width=True)
                                    st.caption("First 3 rows of OOT2 dataset")
                                except Exception as e:
                                    st.error(f"❌ Error previewing OOT2: {str(e)}")
                
                # Store OOT configs in session state
                if oot1_config:
                    st.session_state['oot1_config'] = oot1_config
                if oot2_config:
                    st.session_state['oot2_config'] = oot2_config
                    
                # Show OOT summary
                if oot1_config or oot2_config:
                    with st.expander("📋 OOT Configuration Summary", expanded=False):
                        oot_summary = {}
                        if oot1_config:
                            if oot1_config['source_type'] == 'upload' and hasattr(oot1_config['data_source'], 'name'):
                                oot_summary['oot1'] = {
                                    'source_type': 'File Upload',
                                    'filename': oot1_config['data_source'].name,
                                    'size_kb': round(oot1_config['data_source'].size / 1024, 1),
                                    'type': oot1_config['data_source'].type if hasattr(oot1_config['data_source'], 'type') else 'unknown'
                                }
                            else:
                                oot_summary['oot1'] = {
                                    'source_type': oot1_config['source_type'].title(),
                                    'data_source': oot1_config['data_source'],
                                    'options': oot1_config.get('options', {})
                                }
                        if oot2_config:
                            if oot2_config['source_type'] == 'upload' and hasattr(oot2_config['data_source'], 'name'):
                                oot_summary['oot2'] = {
                                    'source_type': 'File Upload',
                                    'filename': oot2_config['data_source'].name,
                                    'size_kb': round(oot2_config['data_source'].size / 1024, 1),
                                    'type': oot2_config['data_source'].type if hasattr(oot2_config['data_source'], 'type') else 'unknown'
                                }
                            else:
                                oot_summary['oot2'] = {
                                    'source_type': oot2_config['source_type'].title(),
                                    'data_source': oot2_config['data_source'],
                                    'options': oot2_config.get('options', {})
                                }
                        st.json(oot_summary)
        
        # Data processing settings
        st.markdown("---")
        st.subheader("⚙️ Data Processing Settings")
        
        # Check if data source is configured
        data_configured = enhanced_data_config and enhanced_data_config.get('data_source')
        
        if not data_configured:
            st.info("⬆️ Configure your data source above to enable data processing settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_threshold = st.slider(
                "Missing Value Threshold", 
                0.0, 1.0, 0.7, 0.05,
                help="Drop columns with missing values above this threshold",
                disabled=not data_configured
            )
            categorical_threshold = st.number_input(
                "Categorical Threshold", 
                value=10, min_value=1,
                help="Treat columns with unique values below this as categorical",
                disabled=not data_configured
            )
        
        with col2:
            sample_fraction = st.slider(
                "Sample Fraction", 
                0.1, 1.0, 1.0, 0.1,
                help="Fraction of data to use for training",
                disabled=not data_configured
            )
    
    with tab2:
        st.header(f"Model Selection - {task_type.title()}")
        
        # Get model settings from config based on task type
        task_models = config.get(task_type, {}).get('models', {})
        
        # Dynamic model selection based on task type
        model_selections = {}
        
        if task_type == "classification":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Core Models")
                model_selections['run_logistic'] = st.checkbox("Logistic Regression", value=task_models.get('run_logistic', True))
                model_selections['run_random_forest'] = st.checkbox("Random Forest", value=task_models.get('run_random_forest', False))
                model_selections['run_gradient_boosting'] = st.checkbox("Gradient Boosting", value=task_models.get('run_gradient_boosting', False))
                
            with col2:
                st.subheader("Advanced Models")
                model_selections['run_decision_tree'] = st.checkbox("Decision Tree", value=task_models.get('run_decision_tree', False))
                model_selections['run_neural_network'] = st.checkbox("Neural Network", value=task_models.get('run_neural_network', False))
                model_selections['run_xgboost'] = st.checkbox("XGBoost", value=task_models.get('run_xgboost', True))
                model_selections['run_lightgbm'] = st.checkbox("LightGBM", value=task_models.get('run_lightgbm', False))
        
        elif task_type == "regression":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Core Models")
                model_selections['run_linear_regression'] = st.checkbox("Linear Regression", value=task_models.get('run_linear_regression', True))
                model_selections['run_random_forest'] = st.checkbox("Random Forest", value=task_models.get('run_random_forest', False))
                model_selections['run_gradient_boosting'] = st.checkbox("Gradient Boosting", value=task_models.get('run_gradient_boosting', False))
                
            with col2:
                st.subheader("Advanced Models")
                model_selections['run_decision_tree'] = st.checkbox("Decision Tree", value=task_models.get('run_decision_tree', False))
                model_selections['run_xgboost'] = st.checkbox("XGBoost", value=task_models.get('run_xgboost', True))
                model_selections['run_lightgbm'] = st.checkbox("LightGBM", value=task_models.get('run_lightgbm', False))
        
        elif task_type == "clustering":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distance-Based")
                model_selections['run_kmeans'] = st.checkbox("K-Means", value=task_models.get('run_kmeans', True))
                model_selections['run_bisecting_kmeans'] = st.checkbox("Bisecting K-Means", value=task_models.get('run_bisecting_kmeans', False))
                
            with col2:
                st.subheader("Density-Based")
                model_selections['run_dbscan'] = st.checkbox("DBSCAN", value=task_models.get('run_dbscan', False))
                model_selections['run_gaussian_mixture'] = st.checkbox("Gaussian Mixture", value=task_models.get('run_gaussian_mixture', False))
                
            # Clustering specific parameters
            st.subheader("Clustering Parameters")
            k_range_min = st.number_input("Minimum K", value=2, min_value=2, max_value=20, help="Minimum number of clusters to test")
            k_range_max = st.number_input("Maximum K", value=10, min_value=2, max_value=20, help="Maximum number of clusters to test")
            
            # Add elbow method selection
            use_elbow_method = st.checkbox(
                "Use Elbow Method for Optimal K Selection", 
                value=True, 
                help="Automatically determine optimal number of clusters using elbow analysis"
            )
            
            if use_elbow_method:
                st.info("🎯 **Elbow Method**: The system will analyze the elbow plot and recommend the optimal number of clusters based on the 'elbow' point where adding more clusters provides diminishing returns.")
            else:
                st.info("📊 **Manual K Selection**: You will need to manually select the number of clusters based on the elbow plots and metrics.")
            
            st.markdown("---")
            st.markdown("**📋 How Clustering Selection Works:**")
            st.markdown("""
            1. **🎯 Elbow Method** (above): Determines the optimal number of clusters (k)
            2. **📊 Evaluation Metrics** (below): Compares different clustering algorithms using the selected k
            
            **Example**: If elbow method suggests k=5, the system will test K-means, Bisecting K-means, etc. with 5 clusters and pick the best one using the evaluation metric below.
            """)

        # Persist the current model selections in session state so other tabs
        # (particularly the training parameters tab) can access the number of
        # selected models.  This is useful for issuing warnings when
        # hyperparameter tuning is enabled across many models.
        st.session_state['model_selections'] = model_selections
    
    with tab3:
        st.header(f"Training Parameters - {task_type.title()}")
        
        # Get task-specific settings
        task_settings = config.get(task_type, {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            if task_type != "clustering":  # Clustering doesn't use CV in the same way
                st.subheader("Cross Validation")
                cv_settings = task_settings.get('cross_validation', {})
                cv_folds = st.number_input("CV Folds", value=cv_settings.get('cv_folds', 5), min_value=2, max_value=10)
                
                st.subheader("Data Split")
                test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
                validation_size = st.slider("Validation Size", 0.1, 0.3, 0.2, 0.05)
            else:
                st.subheader("Clustering Evaluation")
                eval_settings = task_settings.get('evaluation', {})
                eval_method = st.selectbox(
                    "Model Selection Metric",
                    ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
                    index=0,
                    help="Metric used to compare different clustering algorithms and select the best one"
                )
                
                # Add explanation for each metric
                if eval_method == 'silhouette':
                    st.info("📊 **Silhouette Score**: Measures how similar an object is to its own cluster vs. other clusters. Range: -1 to 1 (higher is better). Best for most datasets.")
                elif eval_method == 'calinski_harabasz':
                    st.info("📊 **Calinski-Harabasz Score**: Ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better-defined clusters.")
                elif eval_method == 'davies_bouldin':
                    st.info("📊 **Davies-Bouldin Score**: Average similarity measure of each cluster with its most similar cluster. Lower values indicate better clustering.")
                
                st.subheader("Data Split (for Evaluation)")
                st.info("📊 Data split is used to evaluate clustering stability across different datasets")
                test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05, help="Split data to test clustering consistency")
                validation_size = st.slider("Validation Size", 0.1, 0.3, 0.2, 0.05, help="Split data for clustering validation")
        
        with col2:
            st.subheader("Hyperparameter Tuning")
            hp_settings = task_settings.get('hyperparameter_tuning', {})
            enable_hp_tuning = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=hp_settings.get('enable_hyperparameter_tuning', False)
            )
            # Persist the hyperparameter tuning flag in session state so it can
            # be inspected elsewhere (e.g. before job submission).  Without
            # this, a local variable would be inaccessible when checking
            # across pages or tabs.  We store as a boolean.
            st.session_state['enable_hp_tuning'] = bool(enable_hp_tuning)
            
            if enable_hp_tuning:
                hp_method = st.selectbox(
                    "Optimization Method",
                    ['optuna', 'random_search', 'grid_search'],
                    index=0
                )
                
                if hp_method == 'optuna':
                    optuna_trials = st.number_input("Optuna Trials", value=hp_settings.get('optuna_trials', 50), min_value=3)
                    optuna_timeout = st.number_input("Optuna Timeout (seconds)", value=hp_settings.get('optuna_timeout', 600))
                elif hp_method == 'random_search':
                    random_trials = st.number_input("Random Search Trials", value=hp_settings.get('random_search_trials', 20), min_value=3)
                
                # Clustering-specific hyperparameter options
                if task_type == "clustering":
                    st.markdown("---")
                    st.markdown("**🔧 Clustering Hyperparameter Ranges**")
                    st.markdown("""
                    **What this does:** Instead of using fixed settings, the system will automatically test different combinations 
                    of these parameters to find the best clustering results. Select which values you want to test for each parameter.
                    """)
                    
                    with st.expander("📚 Quick Guide: Which parameters matter most?"):
                        st.markdown("""
                        **🎯 Most Important:** 
                        - **K Range** (set above): Number of clusters - this has the biggest impact
                        - **Max Iterations**: Start with 50-100 for most datasets
                        
                        **⚙️ Fine-tuning:**
                        - **Tolerance**: Use 1e-4 for balanced speed/quality, 1e-5 for higher precision
                        - **Initialization**: k-means|| usually works better than random
                        - **Min Divisible Size**: Use 1.0-2.0 for balanced clusters, 5.0+ for larger clusters only
                        
                        **💡 Tip:** The defaults are good for most cases. Only customize if you need specific behavior!
                        
                        **🔍 Example:** If you select Max Iterations [50, 100, 200], the system will try:
                        - K-means with 50 iterations
                        - K-means with 100 iterations  
                        - K-means with 200 iterations
                        - And pick the best result automatically!
                        """)
                    
                    st.markdown("**Customize Parameter Ranges (Optional):**")
                    
                    # K-means parameters
                    if st.checkbox("Customize K-means Parameters", value=False, key="kmeans_params_checkbox"):
                        st.markdown("**K-means Hyperparameters:**")
                        st.info("💡 K-means groups data points into clusters by minimizing the distance between points and cluster centers")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Maximum number of times the algorithm will try to improve cluster centers. Higher = more thorough but slower.")
                            kmeans_max_iter_options = st.multiselect(
                                "Choose iteration limits to test:", 
                                [20, 50, 100, 200, 500],
                                default=[20, 50, 100, 200],
                                key="kmeans_max_iter_multiselect"
                            )
                            
                            st.markdown("**Initialization Methods** 🎯")
                            st.caption("How to choose initial cluster centers. k-means|| is usually better but random can find different solutions.")
                            kmeans_init_modes = st.multiselect(
                                "Choose initialization methods:",
                                ["k-means||", "random"],
                                default=["k-means||", "random"],
                                key="kmeans_init_modes_multiselect"
                            )
                        with col2:
                            st.markdown("**Tolerance Values** ⚖️")
                            st.caption("How much improvement is needed to continue. Smaller = more precise but slower. 1e-4 means 0.0001")
                            kmeans_tol_options = st.multiselect(
                                "Choose precision levels:",
                                ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"],
                                default=["1e-5", "1e-4", "1e-3"],
                                key="kmeans_tol_multiselect"
                            )
                    
                    # Bisecting K-means parameters  
                    if st.checkbox("Customize Bisecting K-means Parameters", value=False, key="bisecting_params_checkbox"):
                        st.markdown("**Bisecting K-means Hyperparameters:**")
                        st.info("💡 Bisecting K-means creates clusters by repeatedly splitting existing clusters in half until reaching the target number")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Maximum attempts to improve each cluster split. Higher = more thorough but slower.")
                            bisecting_max_iter_options = st.multiselect(
                                "Choose iteration limits:",
                                [20, 50, 100, 200, 500], 
                                default=[20, 50, 100, 200],
                                key="bisecting_max_iter_multiselect"
                            )
                        with col2:
                            st.markdown("**Min Divisible Cluster Size** ✂️")
                            st.caption("Minimum cluster size that can be split. Higher = fewer small clusters. Use 1.0 for balanced, 5.0 for larger clusters only.")
                            bisecting_min_divisible_options = st.multiselect(
                                "Choose minimum sizes:",
                                [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
                                default=[0.5, 1.0, 2.0, 3.0, 5.0],
                                key="bisecting_min_divisible_multiselect"
                            )
                    
                    # DBSCAN parameters
                    if st.checkbox("Customize DBSCAN Parameters", value=False, key="dbscan_params_checkbox"):
                        st.markdown("**DBSCAN Hyperparameters:**")
                        st.info("💡 DBSCAN groups points that are close together and marks isolated points as noise. Great for finding clusters of arbitrary shapes!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Epsilon (eps)** 📏")
                            st.caption("Maximum distance between points to be considered neighbors. Smaller = more clusters, larger = fewer clusters.")
                            dbscan_eps_options = st.multiselect(
                                "Choose epsilon values:",
                                [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
                                default=[0.1, 0.3, 0.5, 0.7, 1.0],
                                key="dbscan_eps_multiselect"
                            )
                        with col2:
                            st.markdown("**Min Points (minPts)** 🎯")
                            st.caption("Minimum points required to form a cluster. Higher = more noise points, lower = more small clusters.")
                            dbscan_minpts_options = st.multiselect(
                                "Choose minimum points:",
                                [3, 5, 7, 10, 15, 20],
                                default=[3, 5, 7, 10],
                                key="dbscan_minpts_multiselect"
                            )
                    
                    # Gaussian Mixture parameters
                    if st.checkbox("Customize Gaussian Mixture Parameters", value=False, key="gaussian_mixture_params_checkbox"):
                        st.markdown("**Gaussian Mixture Hyperparameters:**")
                        st.info("💡 Gaussian Mixture models data as a mixture of several Gaussian distributions. Great for finding clusters with different shapes and sizes!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Maximum number of iterations for the EM algorithm. Higher = more thorough but slower.")
                            gaussian_max_iter_options = st.multiselect(
                                "Choose iteration limits:",
                                [50, 100, 200, 300, 500],
                                default=[50, 100, 200, 300],
                                key="gaussian_max_iter_multiselect"
                            )
                            
                            st.markdown("**Tolerance Values** ⚖️")
                            st.caption("Convergence tolerance for the EM algorithm. Smaller = more precise but slower.")
                            gaussian_tol_options = st.multiselect(
                                "Choose tolerance levels:",
                                ["1e-6", "1e-5", "1e-4", "1e-3"],
                                default=["1e-5", "1e-4", "1e-3"],
                                key="gaussian_tol_multiselect"
                            )
                        with col2:
                            st.markdown("**Reg Param** 🛡️")
                            st.caption("Regularization parameter for covariance matrices. Higher = more stable but less flexible.")
                            gaussian_reg_param_options = st.multiselect(
                                "Choose regularization values:",
                                [0.001, 0.01, 0.1, 0.5, 1.0],
                                default=[0.001, 0.01, 0.1],
                                key="gaussian_reg_param_multiselect"
                            )

            # After all hyperparameter tuning selections, warn if too many models
            # are selected with tuning enabled.  Long runtimes can lead to job
            # timeout in the Streamlit front end.  Recommend increasing
            # timeout_minutes or deselecting some models.
            if enable_hp_tuning:
                # Determine how many models the user has enabled
                model_selections_state = st.session_state.get('model_selections', {})
                try:
                    num_models_selected = sum(1 for v in model_selections_state.values() if v)
                except Exception:
                    num_models_selected = 0
                # Issue warning if more than three models are selected
                if num_models_selected >= 3:
                    st.warning(
                        "⚠️ You have enabled hyperparameter tuning for multiple models. "
                        "This can significantly increase runtime and may exceed the configured timeout. "
                        "Consider increasing the timeout or reducing the number of selected models before submitting."
                    )
                
                # Classification-specific hyperparameter options
                if task_type == "classification":
                    st.markdown("---")
                    st.markdown("**🔧 Classification Hyperparameter Ranges**")
                    st.markdown("""
                    **What this does:** Instead of using fixed settings, the system will automatically test different combinations 
                    of these parameters to find the best classification results. Select which values you want to test for each parameter.
                    """)
                    
                    with st.expander("📚 Quick Guide: Which parameters matter most?"):
                        st.markdown("""
                        **🎯 Most Important:** 
                        - **Regularization (regParam)**: Controls overfitting - 0.01-1.0 range
                        - **Max Iterations**: Start with 50-100 for most datasets
                        - **Number of Trees (Random Forest)**: More trees = better but slower
                        
                        **⚙️ Fine-tuning:**
                        - **Elastic Net**: 0.0 = Ridge, 1.0 = Lasso, 0.5 = balanced
                        - **Max Depth (Random Forest)**: Controls tree complexity
                        - **Min Instances Per Node**: Controls tree granularity
                        
                        **💡 Tip:** The defaults are good for most cases. Only customize if you need specific behavior!
                        """)
                    
                    st.markdown("**Customize Parameter Ranges (Optional):**")
                    
                    # Logistic Regression parameters
                    if st.checkbox("Customize Logistic Regression Parameters", value=False, key="logistic_params_checkbox"):
                        st.markdown("**Logistic Regression Hyperparameters:**")
                        st.info("💡 Logistic Regression predicts probabilities using a linear model with regularization")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Maximum number of optimization iterations. Higher = more thorough but slower.")
                            logistic_max_iter_options = st.multiselect(
                                "Choose iteration limits to test:", 
                                [10, 20, 50, 100, 200],
                                default=[10, 20, 50, 100],
                                key="logistic_max_iter_multiselect"
                            )
                            
                            st.markdown("**Regularization Parameter** 🛡️")
                            st.caption("Controls overfitting. Lower = less regularization, Higher = more regularization.")
                            logistic_reg_param_options = st.multiselect(
                                "Choose regularization strengths:",
                                [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
                                default=[0.01, 0.1, 0.5, 1.0],
                                key="logistic_reg_param_multiselect"
                            )
                        with col2:
                            st.markdown("**Elastic Net Parameter** 🎯")
                            st.caption("0.0 = Ridge (L2), 1.0 = Lasso (L1), 0.5 = balanced. Controls feature selection.")
                            logistic_elastic_net_options = st.multiselect(
                                "Choose elastic net ratios:",
                                [0.0, 0.25, 0.5, 0.75, 1.0],
                                default=[0.0, 0.25, 0.5, 0.75, 1.0],
                                key="logistic_elastic_net_multiselect"
                            )
                    
                    # Random Forest parameters
                    if st.checkbox("Customize Random Forest Parameters", value=False, key="rf_params_checkbox"):
                        st.markdown("**Random Forest Hyperparameters:**")
                        st.info("💡 Random Forest creates multiple decision trees and combines their predictions")
                        st.success("✅ **Optimized:** Tree-based optimizations are automatically applied to reduce broadcast warnings and improve performance.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Number of Trees** 🌳")
                            st.caption("More trees = better accuracy but slower training. 10-100 is typical.")
                            rf_num_trees_options = st.multiselect(
                                "Choose number of trees to test:",
                                [10, 20, 50, 100, 200],
                                default=[10, 20, 50],  # Reduced from [10, 20, 50, 100]
                                key="rf_num_trees_multiselect"
                            )
                            
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of each tree. Higher = more complex but risk of overfitting.")
                            rf_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 5, 10, 15, 20, 30],
                                default=[3, 5, 10],  # Reduced from [3, 5, 10, 15, 20]
                                key="rf_max_depth_multiselect"
                            )
                        with col2:
                            st.markdown("**Min Instances Per Node** 🎯")
                            st.caption("Minimum samples required to split a node. Higher = simpler trees.")
                            rf_min_instances_options = st.multiselect(
                                "Choose minimum instances:",
                                [1, 2, 5, 10, 20],
                                default=[1, 2, 5],  # Reduced from [1, 2, 5, 10]
                                key="rf_min_instances_multiselect"
                            )
                    
                    # Gradient Boosting parameters
                    if st.checkbox("Customize Gradient Boosting Parameters", value=False, key="gb_params_checkbox"):
                        st.markdown("**Gradient Boosting Hyperparameters:**")
                        st.info("💡 Gradient Boosting builds an ensemble by sequentially adding weak learners")
                        st.success("✅ **Optimized:** Gradient boosting optimizations are automatically applied to reduce broadcast warnings and improve performance.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Number of boosting iterations. More iterations = stronger model but risk of overfitting.")
                            gb_max_iter_options = st.multiselect(
                                "Choose iteration limits to test:",
                                [10, 20, 50, 100, 200],
                                default=[10, 20],  # Further reduced from [10, 20, 30]
                                key="gb_max_iter_multiselect"
                            )
                            
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of each tree. Higher = more complex but risk of overfitting.")
                            gb_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 5, 10, 15, 20],
                                default=[3, 5],  # Further reduced from [3, 5, 7]
                                key="gb_max_depth_multiselect"
                            )
                        with col2:
                            st.markdown("**Step Size** 📈")
                            st.caption("Learning rate for each boosting iteration. Lower = more conservative learning.")
                            gb_step_size_options = st.multiselect(
                                "Choose step sizes:",
                                [0.01, 0.05, 0.1, 0.2, 0.3],
                                default=[0.1],  # Further reduced from [0.1, 0.2]
                                key="gb_step_size_multiselect"
                            )
                    
                    # XGBoost parameters
                    if st.checkbox("Customize XGBoost Parameters", value=False, key="xgb_params_checkbox"):
                        st.markdown("**XGBoost Hyperparameters:**")
                        st.info("💡 XGBoost is a gradient boosting framework that's highly optimized for performance")
                        st.success("✅ **Optimized:** Tree-based optimizations are automatically applied to reduce broadcast warnings and improve performance.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of each tree. Higher = more complex but risk of overfitting.")
                            xgb_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
                                default=[3, 5, 7],  # Reduced from [3, 5, 7, 10]
                                key="xgb_max_depth_multiselect"
                            )
                            
                            st.markdown("**Number of Rounds** 🌳")
                            st.caption("Number of boosting rounds. More rounds = stronger model but slower training.")
                            xgb_num_round_options = st.multiselect(
                                "Choose number of rounds:",
                                [30, 50, 75, 100, 125, 150, 175, 200, 250, 300],
                                default=[30, 50, 75],  # Reduced from [50, 100, 150, 200]
                                key="xgb_num_round_multiselect"
                            )
                            
                            st.markdown("**Eta (Learning Rate)** 📈")
                            st.caption("Step size shrinkage to prevent overfitting. Lower = more conservative.")
                            xgb_eta_options = st.multiselect(
                                "Choose eta values:",
                                [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3],
                                default=[0.01, 0.05, 0.1],  # Reduced from [0.05, 0.1, 0.15, 0.2]
                                key="xgb_eta_multiselect"
                            )
                        with col2:
                            st.markdown("**Subsample** 🎲")
                            st.caption("Fraction of samples used for training trees. Helps prevent overfitting.")
                            xgb_subsample_options = st.multiselect(
                                "Choose subsample ratios:",
                                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                default=[0.7, 0.8, 1.0],  # Reduced from [0.8, 0.9, 1.0]
                                key="xgb_subsample_multiselect"
                            )
                            
                            st.markdown("**Column Sample by Tree** 🎯")
                            st.caption("Fraction of features used for each tree. Helps prevent overfitting.")
                            xgb_colsample_bytree_options = st.multiselect(
                                "Choose column sample ratios:",
                                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                default=[0.7, 0.8, 1.0],  # Reduced from [0.8, 0.9, 1.0]
                                key="xgb_colsample_bytree_multiselect"
                            )
                            
                            st.markdown("**Min Child Weight** ⚖️")
                            st.caption("Minimum sum of instance weight needed in a child. Higher = more conservative.")
                            xgb_min_child_weight_options = st.multiselect(
                                "Choose min child weights:",
                                [1, 2, 3, 5, 7, 10],
                                default=[1, 3, 5],  # Reduced from [1, 2, 3, 5]
                                key="xgb_min_child_weight_multiselect"
                            )
                    
                    # Decision Tree parameters
                    if st.checkbox("Customize Decision Tree Parameters", value=False, key="dt_params_checkbox"):
                        st.markdown("**Decision Tree Hyperparameters:**")
                        st.info("💡 Decision Tree creates a tree structure to make predictions based on feature splits")
                        st.success("✅ **Optimized:** Tree-based optimizations are automatically applied to reduce broadcast warnings and improve performance.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of the tree. Higher = more complex but risk of overfitting.")
                            dt_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 5, 10, 15, 20, 30],
                                default=[3, 5, 10],  # Reduced from [3, 5, 10, 15, 20]
                                key="dt_max_depth_multiselect"
                            )
                            
                            st.markdown("**Min Instances Per Node** 🎯")
                            st.caption("Minimum samples required to split a node. Higher = simpler trees.")
                            dt_min_instances_options = st.multiselect(
                                "Choose minimum instances:",
                                [1, 2, 5, 10, 20],
                                default=[1, 2, 5],  # Reduced from [1, 2, 5, 10]
                                key="dt_min_instances_multiselect"
                            )
                        with col2:
                            st.markdown("**Min Info Gain** 📊")
                            st.caption("Minimum information gain required for a split. Higher = more conservative splits.")
                            dt_min_info_gain_options = st.multiselect(
                                "Choose min info gains:",
                                [0.0, 0.01, 0.05, 0.1, 0.2],
                                default=[0.0, 0.01],  # Reduced from [0.0, 0.01, 0.1]
                                key="dt_min_info_gain_multiselect"
                            )
                    
                    # Neural Network parameters
                    if st.checkbox("Customize Neural Network Parameters", value=False, key="nn_params_checkbox"):
                        st.markdown("**Neural Network Hyperparameters:**")
                        st.info("💡 Neural Network uses multiple layers of neurons to learn complex patterns")
                        
                        # Add feature count guidance
                        st.warning("⚠️ **Important:** Layer sizes will be automatically adjusted if they exceed your dataset's feature count to prevent errors.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Layer Configurations** 🧠")
                            st.caption("Hidden layer architectures. First layer size will be adjusted if it exceeds your feature count.")
                            nn_layers_options = st.multiselect(
                                "Choose layer configurations:",
                                ["[8]", "[16]", "[8, 4]", "[16, 8]", "[16, 8, 4]"],  # Updated to match backend fixes
                                default=["[8]", "[16]", "[8, 4]"],  # More conservative defaults
                                key="nn_layers_multiselect"
                            )
                            
                            # Add layer size guidance
                            if nn_layers_options:
                                st.info("💡 **Layer Guide:**\n- `[8]`: Simple single layer\n- `[16]`: Larger single layer\n- `[8, 4]`: Two layers, decreasing size\n- `[16, 8]`: Two layers, moderate size\n- `[16, 8, 4]`: Three layers, gradually decreasing")
                            
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Maximum training iterations. Higher = more thorough training but slower.")
                            nn_max_iter_options = st.multiselect(
                                "Choose iteration limits:",
                                [50, 100, 200],  # Reduced from [50, 100, 200, 300] to match backend
                                default=[50, 100],  # More conservative defaults
                                key="nn_max_iter_multiselect"
                            )
                        with col2:
                            st.markdown("**Block Size** 📦")
                            st.caption("Number of samples per batch. Affects memory usage and training speed.")
                            nn_block_size_options = st.multiselect(
                                "Choose block sizes:",
                                [32, 64, 128],  # Reduced from [32, 64, 128, 256] to match backend
                                default=[32, 64],  # More conservative defaults
                                key="nn_block_size_multiselect"
                            )
                    
                    # LightGBM parameters
                    if st.checkbox("Customize LightGBM Parameters", value=False, key="lgb_params_checkbox"):
                        st.markdown("**LightGBM Hyperparameters:**")
                        st.info("💡 LightGBM is a gradient boosting framework optimized for efficiency and speed")
                        st.success("✅ **Optimized:** Tree-based optimizations are automatically applied to reduce broadcast warnings and improve performance.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Number of Leaves** 🍃")
                            st.caption("Maximum number of leaves in one tree. Higher = more complex model.")
                            lgb_num_leaves_options = st.multiselect(
                                "Choose number of leaves:",
                                [5, 10, 15, 20, 25, 31, 40, 50, 60, 80, 100, 120],
                                default=[5, 15, 31],  # Reduced from [15, 25, 31, 50]
                                key="lgb_num_leaves_multiselect"
                            )
                            
                            st.markdown("**Number of Iterations** 🔄")
                            st.caption("Number of boosting iterations. More iterations = stronger model.")
                            lgb_num_iterations_options = st.multiselect(
                                "Choose number of iterations:",
                                [30, 50, 75, 100, 125, 150, 175, 200, 250, 300],
                                default=[30, 50, 75],  # Reduced from [50, 100, 150, 200]
                                key="lgb_num_iterations_multiselect"
                            )
                            
                            st.markdown("**Learning Rate** 📈")
                            st.caption("Learning rate for boosting. Lower = more conservative learning.")
                            lgb_learning_rate_options = st.multiselect(
                                "Choose learning rates:",
                                [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3],
                                default=[0.01, 0.05, 0.1],  # Reduced from [0.05, 0.1, 0.15, 0.2]
                                key="lgb_learning_rate_multiselect"
                            )
                        with col2:
                            st.markdown("**Feature Fraction** 🎯")
                            st.caption("Fraction of features used for each iteration. Helps prevent overfitting.")
                            lgb_feature_fraction_options = st.multiselect(
                                "Choose feature fractions:",
                                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                default=[0.7, 0.8, 1.0],  # Reduced from [0.8, 0.9, 1.0]
                                key="lgb_feature_fraction_multiselect"
                            )
                            
                            st.markdown("**Bagging Fraction** 🎲")
                            st.caption("Fraction of data used for each iteration. Helps prevent overfitting.")
                            lgb_bagging_fraction_options = st.multiselect(
                                "Choose bagging fractions:",
                                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                default=[0.7, 0.8, 1.0],  # Reduced from [0.8, 0.9, 1.0]
                                key="lgb_bagging_fraction_multiselect"
                            )
                
                # Regression-specific hyperparameter options
                elif task_type == "regression":
                    st.markdown("---")
                    st.markdown("**🔧 Regression Hyperparameter Ranges**")
                    st.markdown("""
                    **What this does:** Instead of using fixed settings, the system will automatically test different combinations 
                    of these parameters to find the best regression results. Select which values you want to test for each parameter.
                    """)
                    
                    with st.expander("📚 Quick Guide: Which parameters matter most?"):
                        st.markdown("""
                        **🎯 Most Important:** 
                        - **Regularization (regParam)**: Controls overfitting - 0.01-1.0 range
                        - **Max Iterations**: Start with 50-100 for most datasets
                        - **Number of Trees (Random Forest)**: More trees = better but slower
                        
                        **⚙️ Fine-tuning:**
                        - **Elastic Net**: 0.0 = Ridge, 1.0 = Lasso, 0.5 = balanced
                        - **Max Depth (Random Forest)**: Controls tree complexity
                        - **Min Instances Per Node**: Controls tree granularity
                        
                        **💡 Tip:** The defaults are good for most cases. Only customize if you need specific behavior!
                        """)
                    
                    st.markdown("**Customize Parameter Ranges (Optional):**")
                    
                    # Linear Regression parameters
                    if st.checkbox("Customize Linear Regression Parameters", value=False, key="linear_params_checkbox"):
                        st.markdown("**Linear Regression Hyperparameters:**")
                        st.info("💡 Linear Regression predicts continuous values using a linear model with regularization")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Maximum number of optimization iterations. Higher = more thorough but slower.")
                            linear_max_iter_options = st.multiselect(
                                "Choose iteration limits to test:", 
                                [10, 20, 50, 100, 200],
                                default=[10, 20, 50, 100],
                                key="linear_max_iter_multiselect"
                            )
                            
                            st.markdown("**Regularization Parameter** 🛡️")
                            st.caption("Controls overfitting. Lower = less regularization, Higher = more regularization.")
                            linear_reg_param_options = st.multiselect(
                                "Choose regularization strengths:",
                                [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
                                default=[0.01, 0.1, 0.5, 1.0],
                                key="linear_reg_param_multiselect"
                            )
                        with col2:
                            st.markdown("**Elastic Net Parameter** 🎯")
                            st.caption("0.0 = Ridge (L2), 1.0 = Lasso (L1), 0.5 = balanced. Controls feature selection.")
                            linear_elastic_net_options = st.multiselect(
                                "Choose elastic net ratios:",
                                [0.0, 0.25, 0.5, 0.75, 1.0],
                                default=[0.0, 0.25, 0.5, 0.75, 1.0],
                                key="linear_elastic_net_multiselect"
                            )
                    
                    # Random Forest Regression parameters
                    if st.checkbox("Customize Random Forest Regression Parameters", value=False, key="rf_reg_params_checkbox"):
                        st.markdown("**Random Forest Regression Hyperparameters:**")
                        st.info("💡 Random Forest creates multiple decision trees and combines their predictions")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Number of Trees** 🌳")
                            st.caption("More trees = better accuracy but slower training. 10-100 is typical.")
                            rf_reg_num_trees_options = st.multiselect(
                                "Choose number of trees to test:",
                                [10, 20, 50, 100, 200],
                                default=[10, 20, 50, 100],
                                key="rf_reg_num_trees_multiselect"
                            )
                            
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of each tree. Higher = more complex but risk of overfitting.")
                            rf_reg_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 5, 10, 15, 20, 30],
                                default=[3, 5, 10, 15, 20],
                                key="rf_reg_max_depth_multiselect"
                            )
                        with col2:
                            st.markdown("**Min Instances Per Node** 🎯")
                            st.caption("Minimum samples required to split a node. Higher = simpler trees.")
                            rf_reg_min_instances_options = st.multiselect(
                                "Choose minimum instances:",
                                [1, 2, 5, 10, 20],
                                default=[1, 2, 5, 10],
                                key="rf_reg_min_instances_multiselect"
                            )
                    
                    # Gradient Boosting Regression parameters
                    if st.checkbox("Customize Gradient Boosting Regression Parameters", value=False, key="gb_reg_params_checkbox"):
                        st.markdown("**Gradient Boosting Regression Hyperparameters:**")
                        st.info("💡 Gradient Boosting builds trees sequentially, each correcting the previous one's errors")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Iterations** 🔄")
                            st.caption("Number of boosting stages. Higher = more trees but risk of overfitting.")
                            gb_reg_max_iter_options = st.multiselect(
                                "Choose iteration limits to test:",
                                [50, 100, 150, 200, 300],
                                default=[50, 100, 150, 200],
                                key="gb_reg_max_iter_multiselect"
                            )
                            
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of each tree. Higher = more complex but risk of overfitting.")
                            gb_reg_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 5, 8, 10, 12, 15],
                                default=[3, 5, 8, 10, 12],
                                key="gb_reg_max_depth_multiselect"
                            )
                        with col2:
                            st.markdown("**Step Size** 📈")
                            st.caption("Learning rate for each boosting stage. Lower = slower but more stable.")
                            gb_reg_step_size_options = st.multiselect(
                                "Choose step sizes:",
                                [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                                default=[0.05, 0.1, 0.15, 0.2],
                                key="gb_reg_step_size_multiselect"
                            )
                    
                    # Decision Tree Regression parameters
                    if st.checkbox("Customize Decision Tree Regression Parameters", value=False, key="dt_reg_params_checkbox"):
                        st.markdown("**Decision Tree Regression Hyperparameters:**")
                        st.info("💡 Decision Tree creates a tree structure to make predictions")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of the tree. Higher = more complex but risk of overfitting.")
                            dt_reg_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 5, 10, 15, 20, 30],
                                default=[3, 5, 10, 15, 20],
                                key="dt_reg_max_depth_multiselect"
                            )
                            
                            st.markdown("**Min Instances Per Node** 🎯")
                            st.caption("Minimum samples required to split a node. Higher = simpler trees.")
                            dt_reg_min_instances_options = st.multiselect(
                                "Choose minimum instances:",
                                [1, 2, 5, 10, 20],
                                default=[1, 2, 5, 10],
                                key="dt_reg_min_instances_multiselect"
                            )
                        with col2:
                            st.markdown("**Min Info Gain** 📊")
                            st.caption("Minimum information gain for a split. Higher = more conservative splits.")
                            dt_reg_min_info_gain_options = st.multiselect(
                                "Choose minimum info gains:",
                                [0.0, 0.01, 0.05, 0.1, 0.2],
                                default=[0.0, 0.01, 0.05, 0.1],
                                key="dt_reg_min_info_gain_multiselect"
                            )
                    
                    # XGBoost Regression parameters
                    if st.checkbox("Customize XGBoost Regression Parameters", value=False, key="xgb_reg_params_checkbox"):
                        st.markdown("**XGBoost Regression Hyperparameters:**")
                        st.info("💡 XGBoost is an optimized gradient boosting implementation")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Max Depth** 📏")
                            st.caption("Maximum depth of each tree. Higher = more complex but risk of overfitting.")
                            xgb_reg_max_depth_options = st.multiselect(
                                "Choose max depths:",
                                [3, 5, 7, 10, 12, 15],
                                default=[3, 5, 7, 10],
                                key="xgb_reg_max_depth_multiselect"
                            )
                            
                            st.markdown("**Number of Rounds** 🔄")
                            st.caption("Number of boosting rounds. Higher = more trees but risk of overfitting.")
                            xgb_reg_num_round_options = st.multiselect(
                                "Choose number of rounds:",
                                [50, 100, 150, 200, 300],
                                default=[50, 100, 150, 200],
                                key="xgb_reg_num_round_multiselect"
                            )
                        with col2:
                            st.markdown("**Learning Rate (Eta)** 📈")
                            st.caption("Step size shrinkage. Lower = slower but more stable training.")
                            xgb_reg_eta_options = st.multiselect(
                                "Choose learning rates:",
                                [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                                default=[0.05, 0.1, 0.15, 0.2],
                                key="xgb_reg_eta_multiselect"
                            )
                            
                            st.markdown("**Subsample** 🎲")
                            st.caption("Fraction of samples used for training trees. Lower = more regularization.")
                            xgb_reg_subsample_options = st.multiselect(
                                "Choose subsample ratios:",
                                [0.5, 0.7, 0.8, 0.9, 1.0],
                                default=[0.8, 0.9, 1.0],
                                key="xgb_reg_subsample_multiselect"
                            )
                    
                    # LightGBM Regression parameters
                    if st.checkbox("Customize LightGBM Regression Parameters", value=False, key="lgb_reg_params_checkbox"):
                        st.markdown("**LightGBM Regression Hyperparameters:**")
                        st.info("💡 LightGBM is a fast gradient boosting framework")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Number of Leaves** 🍃")
                            st.caption("Maximum number of leaves in one tree. Higher = more complex.")
                            lgb_reg_num_leaves_options = st.multiselect(
                                "Choose number of leaves:",
                                [15, 25, 31, 50, 80, 100],
                                default=[15, 25, 31, 50],
                                key="lgb_reg_num_leaves_multiselect"
                            )
                            
                            st.markdown("**Number of Iterations** 🔄")
                            st.caption("Number of boosting iterations. Higher = more trees.")
                            lgb_reg_num_iterations_options = st.multiselect(
                                "Choose number of iterations:",
                                [50, 100, 150, 200, 300],
                                default=[50, 100, 150, 200],
                                key="lgb_reg_num_iterations_multiselect"
                            )
                        with col2:
                            st.markdown("**Learning Rate** 📈")
                            st.caption("Learning rate for boosting. Lower = slower but more stable.")
                            lgb_reg_learning_rate_options = st.multiselect(
                                "Choose learning rates:",
                                [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                                default=[0.05, 0.1, 0.15, 0.2],
                                key="lgb_reg_learning_rate_multiselect"
                            )
                            
                            st.markdown("**Feature Fraction** 🎯")
                            st.caption("Fraction of features used for each tree. Lower = more regularization.")
                            lgb_reg_feature_fraction_options = st.multiselect(
                                "Choose feature fractions:",
                                [0.5, 0.7, 0.8, 0.9, 1.0],
                                default=[0.8, 0.9, 1.0],
                                key="lgb_reg_feature_fraction_multiselect"
                            )
                    
                    # Disable hyperparameter tuning by default
                    st.session_state['enable_hp_tuning'] = False
    
    with tab4:
        st.header("Advanced Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Initialize auto_balance with default value for all task types
            auto_balance = False  # Default value for non-classification tasks
            
            # Data balancing only for classification
            if task_type == "classification":
                st.subheader("Data Balancing")
                balance_settings = task_settings.get('data_balancing', {})
                auto_balance = st.checkbox("Auto Balance", value=balance_settings.get('auto_balance', True))
                
                if auto_balance:
                    balance_method = st.selectbox(
                        "Balance Method",
                        ['oversample', 'smote', 'disabled'],
                        index=0
                    )
                    imbalance_threshold = st.slider("Imbalance Threshold", 0.01, 0.2, 0.05, 0.01)
            else:
                st.subheader("Feature Scaling")
                st.info(f"📊 {task_type.title()} typically benefits from feature scaling")
                feature_scaling = st.checkbox("Enable Feature Scaling", value=True)
        
        with col2:
            st.subheader("Performance")
            perf_settings = config.get('performance', {})
            parallel_jobs = st.number_input("Parallel Jobs (-1 for all cores)", value=perf_settings.get('parallel_jobs', -1))
            timeout_minutes = st.number_input("Timeout (minutes)", value=perf_settings.get('timeout_minutes', 60), min_value=10)
    
        # Variable Inclusion/Exclusion Settings (full width)
        st.markdown("---")
        st.subheader("🎯 Variable Selection")
        st.markdown("**Control which variables are included or excluded from your model:**")
        
        # Get current data processing settings for defaults
        data_proc_settings = config.get('global', {}).get('data_processing', {})
        
        var_col1, var_col2 = st.columns(2)
        
        with var_col1:
            st.markdown("**📥 Include Variables**")
            
            # Include specific variables
            include_vars_input = st.text_area(
                "Include Variables (one per line)",
                value="\n".join(data_proc_settings.get('include_vars', [])),
                help="Specify exact variable names to include. If specified, only these variables will be used.",
                placeholder="variable_1\nvariable_2\nvariable_3"
            )
            include_vars = [var.strip() for var in include_vars_input.split('\n') if var.strip()]
            
            # Include variables by prefix
            include_prefix_input = st.text_area(
                "Include by Prefix (one per line)",
                value="\n".join(data_proc_settings.get('include_prefix', [])),
                help="Include variables that start with these prefixes",
                placeholder="feat_\nnum_\ncategory_"
            )
            include_prefix = [prefix.strip() for prefix in include_prefix_input.split('\n') if prefix.strip()]
            
            # Include variables by suffix
            include_suffix_input = st.text_area(
                "Include by Suffix (one per line)",
                value="\n".join(data_proc_settings.get('include_suffix', [])),
                help="Include variables that end with these suffixes",
                placeholder="_score\n_ratio\n_flag"
            )
            include_suffix = [suffix.strip() for suffix in include_suffix_input.split('\n') if suffix.strip()]
        
        with var_col2:
            st.markdown("**📤 Exclude Variables**")
            
            # Exclude specific variables
            exclude_vars_input = st.text_area(
                "Exclude Variables (one per line)",
                value="\n".join(data_proc_settings.get('exclude_vars', [])),
                help="Specify exact variable names to exclude from the model",
                placeholder="id_column\ntimestamp\ninternal_id"
            )
            exclude_vars = [var.strip() for var in exclude_vars_input.split('\n') if var.strip()]
            
            # Exclude variables by prefix
            exclude_prefix_input = st.text_area(
                "Exclude by Prefix (one per line)",
                value="\n".join(data_proc_settings.get('exclude_prefix', [])),
                help="Exclude variables that start with these prefixes",
                placeholder="temp_\naux_\ndebug_"
            )
            exclude_prefix = [prefix.strip() for prefix in exclude_prefix_input.split('\n') if prefix.strip()]
            
            # Exclude variables by suffix
            exclude_suffix_input = st.text_area(
                "Exclude by Suffix (one per line)",
                value="\n".join(data_proc_settings.get('exclude_suffix', [])),
                help="Exclude variables that end with these suffixes",
                placeholder="_id\n_timestamp\n_internal"
            )
            exclude_suffix = [suffix.strip() for suffix in exclude_suffix_input.split('\n') if suffix.strip()]
        
        # Show summary of variable selection rules
        if any([include_vars, exclude_vars, include_prefix, exclude_prefix, include_suffix, exclude_suffix]):
            st.markdown("**📋 Variable Selection Summary:**")
            
            summary_items = []
            if include_vars:
                summary_items.append(f"✅ Include {len(include_vars)} specific variable(s)")
            if include_prefix:
                summary_items.append(f"✅ Include variables with {len(include_prefix)} prefix(es)")
            if include_suffix:
                summary_items.append(f"✅ Include variables with {len(include_suffix)} suffix(es)")
            if exclude_vars:
                summary_items.append(f"❌ Exclude {len(exclude_vars)} specific variable(s)")
            if exclude_prefix:
                summary_items.append(f"❌ Exclude variables with {len(exclude_prefix)} prefix(es)")
            if exclude_suffix:
                summary_items.append(f"❌ Exclude variables with {len(exclude_suffix)} suffix(es)")
            
            for item in summary_items:
                st.markdown(f"   • {item}")
            
            st.info("💡 Include rules are applied first, then exclude rules. If no include rules are specified, all variables are initially included.")
        
        # Data Preview Section (optional)
        if data_file and st.checkbox("🔍 Preview Variable Selection", help="Show which variables would be included/excluded based on current settings"):
            try:
                # Try to load a preview of the data to show variable selection
                if data_file.endswith('.csv'):
                    import pandas as pd
                    df_preview = pd.read_csv(data_file, nrows=0)  # Just get column names
                elif data_file.endswith('.xlsx') or data_file.endswith('.xls'):
                    import pandas as pd
                    df_preview = pd.read_excel(data_file, nrows=0)  # Just get column names
                elif uploaded_file is not None:
                    if uploaded_file.name.endswith('.csv'):
                        import pandas as pd
                        df_preview = pd.read_csv(uploaded_file, nrows=0)
                    elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                        import pandas as pd
                        df_preview = pd.read_excel(uploaded_file, nrows=0)
                    else:
                        df_preview = None
                else:
                    df_preview = None
                
                if df_preview is not None:
                    # Get all variables except target
                    all_vars = [col for col in df_preview.columns if col != target_column]
                    
                    # Apply include/exclude logic (same as in data processor)
                    filtered_vars = all_vars.copy()
                    
                    # Apply include filters
                    if include_vars:
                        filtered_vars = [var for var in filtered_vars if var in include_vars]
                    
                    if include_prefix:
                        prefix_vars = []
                        for prefix in include_prefix:
                            prefix_vars.extend([var for var in filtered_vars if var.startswith(prefix)])
                        filtered_vars = list(set(filtered_vars) & set(prefix_vars))
                    
                    if include_suffix:
                        suffix_vars = []
                        for suffix in include_suffix:
                            suffix_vars.extend([var for var in filtered_vars if var.endswith(suffix)])
                        filtered_vars = list(set(filtered_vars) & set(suffix_vars))
                    
                    # Apply exclude filters
                    if exclude_vars:
                        filtered_vars = [var for var in filtered_vars if var not in exclude_vars]
                    
                    if exclude_prefix:
                        for prefix in exclude_prefix:
                            filtered_vars = [var for var in filtered_vars if not var.startswith(prefix)]
                    
                    if exclude_suffix:
                        for suffix in exclude_suffix:
                            filtered_vars = [var for var in filtered_vars if not var.endswith(suffix)]
                    
                    # Show results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**✅ Variables to Include ({len(filtered_vars)}):**")
                        if filtered_vars:
                            for var in sorted(filtered_vars)[:20]:  # Show first 20
                                st.markdown(f"   • {var}")
                            if len(filtered_vars) > 20:
                                st.markdown(f"   ... and {len(filtered_vars) - 20} more")
                        else:
                            st.warning("No variables would be included with current settings!")
                    
                    with col2:
                        excluded_vars = [var for var in all_vars if var not in filtered_vars]
                        st.markdown(f"**❌ Variables to Exclude ({len(excluded_vars)}):**")
                        if excluded_vars:
                            for var in sorted(excluded_vars)[:20]:  # Show first 20
                                st.markdown(f"   • {var}")
                            if len(excluded_vars) > 20:
                                st.markdown(f"   ... and {len(excluded_vars) - 20} more")
                        else:
                            st.info("No variables would be excluded.")
                    
                    if len(filtered_vars) == 0:
                        st.error("⚠️ Warning: No variables would be included with current settings. Please adjust your filters.")
                    elif len(filtered_vars) < 5:
                        st.warning(f"⚠️ Warning: Only {len(filtered_vars)} variables would be included. Consider expanding your filters for better model performance.")
                
            except Exception as e:
                st.error(f"Error previewing data: {str(e)}")
    
    # Submit button and pre-submission warnings
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    # Warn the user when hyperparameter tuning is enabled and many models are selected.
    # Hyperparameter tuning dramatically increases training time as the pipeline
    # must train multiple configurations per model.  If more than two or
    # three models are selected concurrently, the default timeout may be
    # exceeded.  We surface this warning outside of any specific tab so it
    # appears immediately before job submission.
    try:
        enable_hp_flag = bool(st.session_state.get('enable_hp_tuning', False))
        model_selections_state = st.session_state.get('model_selections', {})
        num_models_selected = sum(1 for v in model_selections_state.values() if v)
        # Issue warning if hyperparameter tuning is enabled and three or more models
        # are selected.  This threshold can be adjusted based on performance.
        if enable_hp_flag and num_models_selected >= 3:
            st.warning(
                "⚠️ You have enabled hyperparameter tuning for multiple models ("
                f"{num_models_selected}). This can significantly increase runtime "
                "and may exceed the configured timeout. Consider increasing the "
                "timeout in your configuration or reducing the number of selected "
                "models before submitting."
            )
    except Exception:
        # Silently ignore any errors when computing the warning
        pass

    with col2:
        # Use session state to prevent double submission
        if 'job_submission_in_progress' not in st.session_state:
            st.session_state.job_submission_in_progress = False
        
        # Show submission status and reset option
        if st.session_state.job_submission_in_progress:
            st.warning("🔄 Job submission in progress... Please wait.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset Submission State", help="Click if submission seems stuck"):
                    st.session_state.job_submission_in_progress = False
                    st.rerun()
            with col2:
                if st.button("🔄 Force Reset & Continue", help="Force reset and continue with submission"):
                    st.session_state.job_submission_in_progress = False
                    st.success("✅ Submission state reset. You can now submit again.")
        
        if st.button("🚀 Submit Rapid Modeler Job", type="primary", use_container_width=True) and not st.session_state.job_submission_in_progress:
            # Validate data source configuration before submission
            enhanced_data_config = st.session_state.get('enhanced_data_config')
            
            if not enhanced_data_config or not enhanced_data_config.get('data_source'):
                st.error("❌ Please configure your data source first!")
                st.error("Select and configure either BigQuery Tables, Upload Files, or Existing Files before submitting.")
                return
            
            if task_type != "clustering" and not target_column:
                st.error("❌ Please specify a target column for your model!")
                return
            
            # Set flag to prevent double submission
            st.session_state.job_submission_in_progress = True
            
            # Add timeout mechanism
            if 'submission_start_time' not in st.session_state or st.session_state.submission_start_time is None:
                st.session_state.submission_start_time = time.time()
            
            # Check if submission has been stuck for too long (5 minutes)
            current_time = time.time()
            # Extra safety check to prevent TypeError
            if (hasattr(st.session_state, 'submission_start_time') and 
                st.session_state.submission_start_time is not None and 
                isinstance(st.session_state.submission_start_time, (int, float)) and
                current_time - st.session_state.submission_start_time > 300):  # 5 minutes
                st.error("⏰ Job submission timeout. Please try again.")
                st.session_state.job_submission_in_progress = False
                st.session_state.submission_start_time = None
                return
            
            # Generate job ID with sequential job number
            job_number = get_next_job_number()
            job_id = f"job_{job_number:04d}_{user_id}_{model_name}_{int(time.time())}"
            
            # Create output directory
            output_dir = os.path.join(RESULTS_DIR, job_id)
            
            # Check for existing jobs with same user_id and model_name
            existing_jobs = []
            job_files = glob.glob(os.path.join(JOBS_DIR, "*.json"))
            for job_file in job_files:
                try:
                    with open(job_file, 'r') as f:
                        existing_job = json.load(f)
                        if (existing_job.get('user_id') == user_id and 
                            existing_job.get('model_name') == model_name and
                            existing_job.get('task_type') == task_type):
                            existing_jobs.append(existing_job)
                except:
                    continue
            
            if existing_jobs:
                # Check if any existing job has results
                jobs_with_results = []
                for job in existing_jobs:
                    job_output_dir = job.get('output_dir', '')
                    if os.path.exists(job_output_dir):
                        result_files = []
                        result_files.extend(glob.glob(os.path.join(job_output_dir, "*.json")))
                        result_files.extend(glob.glob(os.path.join(job_output_dir, "*.txt")))
                        result_files.extend(glob.glob(os.path.join(job_output_dir, "*.csv")))
                        result_files.extend(glob.glob(os.path.join(job_output_dir, "*", "*.json")))
                        result_files.extend(glob.glob(os.path.join(job_output_dir, "*", "*.txt")))
                        if result_files:
                            jobs_with_results.append(job)
                
                if jobs_with_results:
                    st.warning(f"""
                    ⚠️ **Similar Job Already Exists!**
                    
                    Found {len(jobs_with_results)} existing job(s) with the same configuration:
                    - User ID: `{user_id}`
                    - Model Name: `{model_name}`
                    - Task Type: `{task_type}`
                    
                    📁 **Existing jobs with results:**
                    """)
                    
                    for i, job in enumerate(jobs_with_results[:3]):  # Show first 3
                        job_number = extract_job_number(job['job_id'])
                        st.write(f"   {i+1}. Job #{job_number} - {job.get('timestamp', 'Unknown time')}")
                    
                    if len(jobs_with_results) > 3:
                        st.write(f"   ... and {len(jobs_with_results) - 3} more")
                    
                    st.write(f"""
                    🕒 **To avoid overwriting previous work:**
                    - Use a different **Model Name** or **User ID**
                    - Check the **Results** page to view existing results
                    - Or manually delete the existing results folders
                    
                    💡 **Tip:** Adding a version number or date to your model name can help distinguish between runs.
                    """)
                    return
            
            # Check if output directory already exists and contains results
            if os.path.exists(output_dir):
                # Check if directory contains any result files
                result_files = []
                result_files.extend(glob.glob(os.path.join(output_dir, "*.json")))
                result_files.extend(glob.glob(os.path.join(output_dir, "*.txt")))
                result_files.extend(glob.glob(os.path.join(output_dir, "*.csv")))
                result_files.extend(glob.glob(os.path.join(output_dir, "*", "*.json")))
                result_files.extend(glob.glob(os.path.join(output_dir, "*", "*.txt")))
                
                if result_files:
                    st.warning(f"""
                    ⚠️ **Job Already Exists!**
                    
                    The output directory `{job_id}` already exists and contains results from a previous run.
                    
                    📁 **Existing files found:** {len(result_files)} files
                    🕒 **To avoid overwriting previous work:**
                    - Use a different **Model Name** or **User ID**
                    - Or manually delete the existing results folder
                    - Or check the **Results** page to view existing results
                    
                    🔄 **Current settings that would create this directory:**
                    - User ID: `{user_id}`
                    - Model Name: `{model_name}`
                    - Task Type: `{task_type}`
                    """)
                    return
            
            # Create the directory first (before handling uploaded files)
            os.makedirs(output_dir, exist_ok=True)
            
            # Handle data file preparation based on enhanced data configuration
            final_data_file = data_file
            
            if enhanced_data_config and enhanced_data_config.get('source_type') == 'bigquery':
                # For BigQuery, we don't need to handle file uploads - data comes directly from BigQuery
                final_data_file = enhanced_data_config['data_source']  # Use BigQuery table reference
                st.success(f"🔗 BigQuery data source configured: {final_data_file}")
                
                # Check if BigQuery table/view size check passed
                size_check_passed = enhanced_data_config.get('size_check_passed', False)
                if not size_check_passed:
                    st.error("❌ **BigQuery table/view size check failed!**")
                    st.warning("⚠️ **Job submission blocked:** Table/view size exceeds the maximum allowed limit.")
                    st.info("💡 **Next steps:**")
                    st.markdown("""
                    1. **Contact the ML Team** for assistance with large datasets
                    2. **Reduce your data size** using the filtering options in the BigQuery configuration
                    3. **Use a smaller subset** of your data for initial model development
                    """)
                    # Reset the submission flag and prevent job submission
                    st.session_state.job_submission_in_progress = False
                    return
                
            elif enhanced_data_config and enhanced_data_config.get('source_type') == 'upload':
                # Handle enhanced uploaded files
                if uploaded_file is not None and data_file.startswith("uploaded_"):
                    # Save uploaded file to output directory
                    uploaded_file_path = os.path.join(output_dir, uploaded_file.name)
                    try:
                        # Ensure the output directory exists
                        os.makedirs(os.path.dirname(uploaded_file_path), exist_ok=True)
                        
                        # Save the file
                        with open(uploaded_file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Verify the file was saved
                        if os.path.exists(uploaded_file_path):
                            final_data_file = uploaded_file_path
                            st.success(f"💾 Enhanced uploaded file saved to: {uploaded_file_path}")
                            st.info(f"📊 File size: {os.path.getsize(uploaded_file_path)} bytes")
                            st.info(f"🔧 Enhanced options: {enhanced_data_config.get('options', {})}")
                        else:
                            raise Exception("File was not created successfully")
                            
                    except Exception as e:
                        st.error(f"❌ Failed to save enhanced uploaded file: {str(e)}")
                        st.error(f"📁 Output directory: {output_dir}")
                        st.error(f"📄 Target file path: {uploaded_file_path}")
                        # Reset the submission flag on failure
                        st.session_state.job_submission_in_progress = False
                        return
                else:
                    st.error(f"❌ Enhanced upload configuration issue: file not properly uploaded")
                    st.session_state.job_submission_in_progress = False
                    return
                    
            else:
                # Standard file handling (backward compatibility)
                if uploaded_file is not None and data_file.startswith("uploaded_"):
                    # Save uploaded file to output directory
                    uploaded_file_path = os.path.join(output_dir, uploaded_file.name)
                    try:
                        # Ensure the output directory exists
                        os.makedirs(os.path.dirname(uploaded_file_path), exist_ok=True)
                        
                        # Save the file
                        with open(uploaded_file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Verify the file was saved
                        if os.path.exists(uploaded_file_path):
                            final_data_file = uploaded_file_path
                            st.success(f"💾 Uploaded file saved to: {uploaded_file_path}")
                            st.info(f"📊 File size: {os.path.getsize(uploaded_file_path)} bytes")
                        else:
                            raise Exception("File was not created successfully")
                            
                    except Exception as e:
                        st.error(f"❌ Failed to save uploaded file: {str(e)}")
                        st.error(f"📁 Output directory: {output_dir}")
                        st.error(f"📄 Target file path: {uploaded_file_path}")
                        # Reset the submission flag on failure
                        st.session_state.job_submission_in_progress = False
                        return
                elif not os.path.exists(data_file):
                    st.error(f"❌ Data file not found: {data_file}")
                    st.info("📁 Please upload a file or ensure the file exists in the current directory.")
                    # Reset the submission flag on failure
                    st.session_state.job_submission_in_progress = False
                    return
            
            # Check if enhanced data input is being used
            enhanced_data_config = st.session_state.get('enhanced_data_config')
            
            # Handle OOT datasets (only for classification and regression)
            oot1_file_path = None
            oot2_file_path = None
            oot1_bigquery_table = None
            oot2_bigquery_table = None
            oot1_config = st.session_state.get('oot1_config')
            oot2_config = st.session_state.get('oot2_config')
            
            if task_type in ["classification", "regression"]:
                # Handle OOT1 dataset based on source type
                if oot1_config and oot1_config.get('data_source'):
                    if oot1_config['source_type'] == 'upload':
                        # File upload - save to disk
                        try:
                            oot1_file = oot1_config['data_source']
                            oot1_file_path = os.path.join(output_dir, f"oot1_{oot1_file.name}")
                            
                            with open(oot1_file_path, 'wb') as f:
                                f.write(oot1_file.getbuffer())
                            
                            if os.path.exists(oot1_file_path):
                                st.success(f"💾 OOT1 dataset saved to: {oot1_file_path}")
                            else:
                                raise Exception("OOT1 file was not created successfully")
                                
                        except Exception as e:
                            st.error(f"❌ Failed to save OOT1 file: {str(e)}")
                            st.session_state.job_submission_in_progress = False
                            return
                    
                    elif oot1_config['source_type'] == 'bigquery':
                        # BigQuery table - use table reference directly
                        oot1_bigquery_table = oot1_config['data_source']
                        st.success(f"🔗 OOT1 BigQuery table configured: {oot1_bigquery_table}")
                    
                    elif oot1_config['source_type'] == 'existing':
                        # Existing file - use file path directly
                        oot1_file_path = oot1_config['data_source']
                        st.success(f"📂 OOT1 existing file configured: {oot1_file_path}")
                
                # Handle OOT2 dataset based on source type
                if oot2_config and oot2_config.get('data_source'):
                    if oot2_config['source_type'] == 'upload':
                        # File upload - save to disk
                        try:
                            oot2_file = oot2_config['data_source']
                            oot2_file_path = os.path.join(output_dir, f"oot2_{oot2_file.name}")
                            
                            with open(oot2_file_path, 'wb') as f:
                                f.write(oot2_file.getbuffer())
                            
                            if os.path.exists(oot2_file_path):
                                st.success(f"💾 OOT2 dataset saved to: {oot2_file_path}")
                            else:
                                raise Exception("OOT2 file was not created successfully")
                                
                        except Exception as e:
                            st.error(f"❌ Failed to save OOT2 file: {str(e)}")
                            st.session_state.job_submission_in_progress = False
                            return
                    
                    elif oot2_config['source_type'] == 'bigquery':
                        # BigQuery table - use table reference directly
                        oot2_bigquery_table = oot2_config['data_source']
                        st.success(f"🔗 OOT2 BigQuery table configured: {oot2_bigquery_table}")
                    
                    elif oot2_config['source_type'] == 'existing':
                        # Existing file - use file path directly
                        oot2_file_path = oot2_config['data_source']
                        st.success(f"📂 OOT2 existing file configured: {oot2_file_path}")
            
            # Prepare job configuration
            job_config = {
                'job_id': job_id,
                'user_id': user_id,
                'model_name': model_name,
                'task_type': task_type,  # NEW: Add task type
                'environment': environment,
                'preset': preset,
                'data_file': final_data_file,  # Use the final data file path
                'target_column': target_column,
                'output_dir': output_dir,
                'timestamp': datetime.now().isoformat(),
                'config_path': CONFIG_FILE,
                'model_params': model_selections,  # Use dynamic model selections
                # OOT datasets (support both files and BigQuery tables)
                'oot1_file': oot1_file_path,
                'oot2_file': oot2_file_path,
                'oot1_bigquery_table': oot1_bigquery_table,
                'oot2_bigquery_table': oot2_bigquery_table,
                # OOT configuration for background job manager
                'oot1_config': oot1_config,
                'oot2_config': oot2_config,
                'data_params': {
                    'test_size': locals().get('test_size', 0.2),
                    'validation_size': locals().get('validation_size', 0.2)
                } if task_type != "clustering" else {
                    'k_range': list(range(locals().get('k_range_min', 2), locals().get('k_range_max', 10) + 1)),
                    'use_elbow_method': locals().get('use_elbow_method', True),
                    'evaluation_method': locals().get('eval_method', 'silhouette')
                },
                'advanced_params': {
                    **({'cv_folds': cv_folds} if task_type != "clustering" else {}),
                    'enable_hyperparameter_tuning': enable_hp_tuning,
                    'auto_balance': auto_balance,
                    'missing_threshold': missing_threshold,
                    'categorical_threshold': categorical_threshold,
                    'sample_fraction': sample_fraction,
                    'parallel_jobs': parallel_jobs,
                    'timeout_minutes': timeout_minutes,
                    # Hyperparameter tuning settings
                    'hp_method': locals().get('hp_method', 'optuna') if enable_hp_tuning else None,
                    'optuna_trials': locals().get('optuna_trials', 50) if enable_hp_tuning and locals().get('hp_method') == 'optuna' else None,
                    'optuna_timeout': locals().get('optuna_timeout', 600) if enable_hp_tuning and locals().get('hp_method') == 'optuna' else None,
                    'random_trials': locals().get('random_trials', 20) if enable_hp_tuning and locals().get('hp_method') == 'random_search' else None,
                    
                    # Task-specific hyperparameter ranges (if configured)
                    **({
                        'clustering_hp_ranges': {
                            'kmeans': {
                                'max_iter_range': locals().get('kmeans_max_iter_options', [20, 50, 100, 200]),
                                'init_modes': locals().get('kmeans_init_modes', ["k-means||", "random"]),
                                'tol_range': [float(t) for t in locals().get('kmeans_tol_options', ["1e-5", "1e-4", "1e-3"])]
                            },
                            'bisecting_kmeans': {
                                'max_iter_range': locals().get('bisecting_max_iter_options', [20, 50, 100, 200]),
                                'min_divisible_range': locals().get('bisecting_min_divisible_options', [0.5, 1.0, 2.0, 3.0, 5.0])
                            },
                            'dbscan': {
                                'eps_values': locals().get('dbscan_eps_options', [0.1, 0.3, 0.5, 0.7, 1.0]),
                                'minPts_values': locals().get('dbscan_minpts_options', [3, 5, 7, 10])
                            },
                            'gaussian_mixture': {
                                'max_iter_range': locals().get('gaussian_max_iter_options', [50, 100, 200, 300]),
                                'tol_range': [float(t) for t in locals().get('gaussian_tol_options', ["1e-5", "1e-4", "1e-3"])],
                                'reg_param_range': locals().get('gaussian_reg_param_options', [0.001, 0.01, 0.1])
                            }
                        }
                    } if task_type == "clustering" else {}),
                    **({
                        'classification_hp_ranges': {
                            'logistic': {
                                'maxIter': locals().get('logistic_max_iter_options', [10, 20, 50, 100]),
                                'regParam': locals().get('logistic_reg_param_options', [0.01, 0.1, 0.5, 1.0]),
                                'elasticNetParam': locals().get('logistic_elastic_net_options', [0.0, 0.25, 0.5, 0.75, 1.0])
                            },
                            'random_forest': {
                                'numTrees': locals().get('rf_num_trees_options', [10, 20, 50, 100]),
                                'maxDepth': locals().get('rf_max_depth_options', [3, 5, 10, 15, 20]),
                                'minInstancesPerNode': locals().get('rf_min_instances_options', [1, 2, 5, 10])
                            },
                            'gradient_boosting': {
                                'maxIter': locals().get('gb_max_iter_options', [10, 20, 50, 100]),
                                'maxDepth': locals().get('gb_max_depth_options', [3, 5, 10, 15]),
                                'stepSize': locals().get('gb_step_size_options', [0.1, 0.2, 0.3])
                            },
                            'decision_tree': {
                                'maxDepth': locals().get('dt_max_depth_options', [3, 5, 10, 15, 20]),
                                'minInstancesPerNode': locals().get('dt_min_instances_options', [1, 2, 5, 10]),
                                'minInfoGain': locals().get('dt_min_info_gain_options', [0.0, 0.01, 0.1])
                            },
                            'neural_network': {
                                'layers': [eval(layer) for layer in locals().get('nn_layers_options', ["[10]", "[20]", "[10, 10]"])],
                                'maxIter': locals().get('nn_max_iter_options', [50, 100, 200]),
                                'blockSize': locals().get('nn_block_size_options', [32, 64, 128])
                            },
                            'xgboost': {
                                'maxDepth': locals().get('xgb_max_depth_options', [3, 5, 7, 10]),
                                'numRound': locals().get('xgb_num_round_options', [50, 100, 150, 200]),
                                'eta': locals().get('xgb_eta_options', [0.05, 0.1, 0.15, 0.2]),
                                'subsample': locals().get('xgb_subsample_options', [0.8, 0.9, 1.0]),
                                'colsampleBytree': locals().get('xgb_colsample_bytree_options', [0.8, 0.9, 1.0]),
                                'minChildWeight': locals().get('xgb_min_child_weight_options', [1, 2, 3, 5])
                            },
                            'lightgbm': {
                                'numLeaves': locals().get('lgb_num_leaves_options', [15, 25, 31, 50]),
                                'numIterations': locals().get('lgb_num_iterations_options', [50, 100, 150, 200]),
                                'learningRate': locals().get('lgb_learning_rate_options', [0.05, 0.1, 0.15, 0.2]),
                                'featureFraction': locals().get('lgb_feature_fraction_options', [0.8, 0.9, 1.0]),
                                'baggingFraction': locals().get('lgb_bagging_fraction_options', [0.8, 0.9, 1.0])
                            }
                        }
                    } if task_type == "classification" else {}),
                    **({
                        'regression_hp_ranges': {
                            'linear': {
                                'maxIter': locals().get('linear_max_iter_options', [10, 20, 50, 100]),
                                'regParam': locals().get('linear_reg_param_options', [0.01, 0.1, 0.5, 1.0]),
                                'elasticNetParam': locals().get('linear_elastic_net_options', [0.0, 0.25, 0.5, 0.75, 1.0])
                            },
                            'random_forest': {
                                'numTrees': locals().get('rf_reg_num_trees_options', [10, 20, 50, 100]),
                                'maxDepth': locals().get('rf_reg_max_depth_options', [3, 5, 10, 15, 20]),
                                'minInstancesPerNode': locals().get('rf_reg_min_instances_options', [1, 2, 5, 10])
                            },
                            'gradient_boosting': {
                                'maxIter': locals().get('gb_reg_max_iter_options', [50, 100, 150, 200]),
                                'maxDepth': locals().get('gb_reg_max_depth_options', [3, 5, 8, 10, 12]),
                                'stepSize': locals().get('gb_reg_step_size_options', [0.05, 0.1, 0.15, 0.2])
                            },
                            'decision_tree': {
                                'maxDepth': locals().get('dt_reg_max_depth_options', [3, 5, 10, 15, 20]),
                                'minInstancesPerNode': locals().get('dt_reg_min_instances_options', [1, 2, 5, 10]),
                                'minInfoGain': locals().get('dt_reg_min_info_gain_options', [0.0, 0.01, 0.05, 0.1])
                            },
                            'xgboost': {
                                'max_depth': locals().get('xgb_reg_max_depth_options', [3, 5, 7, 10]),
                                'n_estimators': locals().get('xgb_reg_num_round_options', [50, 100, 150, 200]),
                                'learning_rate': locals().get('xgb_reg_eta_options', [0.05, 0.1, 0.15, 0.2]),
                                'subsample': locals().get('xgb_reg_subsample_options', [0.8, 0.9, 1.0])
                            },
                            'lightgbm': {
                                'numLeaves': locals().get('lgb_reg_num_leaves_options', [15, 25, 31, 50]),
                                'numIterations': locals().get('lgb_reg_num_iterations_options', [50, 100, 150, 200]),
                                'learningRate': locals().get('lgb_reg_learning_rate_options', [0.05, 0.1, 0.15, 0.2]),
                                'featureFraction': locals().get('lgb_reg_feature_fraction_options', [0.8, 0.9, 1.0])
                            }
                        }
                    } if task_type == "regression" else {}),
                    # Variable selection settings
                    'include_vars': include_vars,
                    'exclude_vars': exclude_vars,
                    'include_prefix': include_prefix,
                    'exclude_prefix': exclude_prefix,
                    'include_suffix': include_suffix,
                    'exclude_suffix': exclude_suffix
                },
                # Enhanced data configuration for BigQuery and other advanced sources
                'enhanced_data_config': enhanced_data_config
            }
            
            # Save job configuration
            save_job_config(job_id, job_config)
            
            # Start background job using job manager
            success = run_automl_job(job_id, job_config)
            
            if not success:
                st.error("❌ Failed to start job. Please check the error messages above.")
                # Reset the submission flag on failure
                st.session_state.job_submission_in_progress = False
                st.session_state.submission_start_time = None
                return
            
            # Show success message with OOT dataset info
            oot_info = ""
            if oot1_file_path or oot2_file_path or oot1_bigquery_table or oot2_bigquery_table:
                oot_info = "\n📅 **Out-of-Time Datasets:**"
                if oot1_file_path:
                    oot_info += f"\n   • OOT1: {os.path.basename(oot1_file_path)}"
                elif oot1_bigquery_table:
                    oot_info += f"\n   • OOT1: {oot1_bigquery_table}"
                if oot2_file_path:
                    oot_info += f"\n   • OOT2: {os.path.basename(oot2_file_path)}"
                elif oot2_bigquery_table:
                    oot_info += f"\n   • OOT2: {oot2_bigquery_table}"
                oot_info += "\n"
                
            st.success(f"""
            ✅ **Job #{job_number} Submitted Successfully!**
            
            🏷️ **Job Number:** `#{job_number}`  
            📋 **Full Job ID:** `{job_id}`  
            👤 **User:** {user_id}  
            🎯 **Model:** {model_name}  
            📊 **Data:** {final_data_file}  
            🎯 **Target:** {target_column}{oot_info}  
            
                            🔄 Your Rapid Modeler job is now running in the background. You can:
            - Check the status in the **Results** page
            - Look for **Job #{job_number}** to track this specific job
            - Come back later to view your model results
            - Submit additional jobs if needed
            
            ⏱️ **Estimated Time:** 15-60 minutes depending on data size and configuration
            """)
            
            # Auto-refresh to results page in 3 seconds
            st.balloons()
            
            # Reset the submission flag after successful submission
            st.session_state.job_submission_in_progress = False
            st.session_state.submission_start_time = None
            
            time.sleep(2)

def create_results_page():
    """Create the results viewing page."""
    st.title("📊 Rapid Modeler Results Viewer")
    st.markdown("View and analyze results from your Rapid Modeler jobs")
    
    # Show data sources being scanned
    with st.expander("📁 Data Sources", expanded=False):
        st.markdown("**Scanning for results in:**")
        sources = [
            f"• `{JOBS_DIR}/` - Job configurations",
            f"• `{RESULTS_DIR}/` - Primary results directory", 
            "• `automl_output/` - Alternative output directory",
            "• Current directory - Direct job folders"
        ]
        for source in sources:
            exists = os.path.exists(source.split('`')[1].split('/')[0]) if '`' in source else True
            status = "✅" if exists else "❌"
            st.markdown(f"{status} {source}")
        
        st.info("💡 Results are automatically discovered from all available directories")
    
    # Get list of job files from multiple sources
    job_files = []
    
    # Primary source: automl_jobs directory (exclude progress files)
    all_job_files = glob.glob(os.path.join(JOBS_DIR, "job_*.json"))
    job_files.extend([f for f in all_job_files if not f.endswith('_progress.json')])
    
    # Also scan for orphaned result directories without job files
    result_dirs_to_scan = [
        RESULTS_DIR,
        "automl_output",
        ".",  # Current directory
    ]
    
    # Auto-discover jobs from result directories
    discovered_jobs = set()
    for result_base_dir in result_dirs_to_scan:
        if os.path.exists(result_base_dir):
            for item in os.listdir(result_base_dir):
                item_path = os.path.join(result_base_dir, item)
                if os.path.isdir(item_path) and item.startswith("job_"):
                    job_id = item
                    discovered_jobs.add(job_id)
    
    if not job_files and not discovered_jobs:
        st.info("No jobs found. Submit a job first!")
        return
    
    # Load all jobs
    jobs = []
    
    # Load jobs from job files
    for job_file in job_files:
        try:
            with open(job_file, 'r') as f:
                job = json.load(f)
                
                # Ensure all required fields exist with safe defaults
                if 'job_id' not in job:
                    job['job_id'] = os.path.basename(job_file).replace('.json', '')
                
                # Set safe defaults for missing fields
                job.setdefault('user_id', 'unknown_user')
                job.setdefault('model_name', 'unknown_model')
                job.setdefault('task_type', 'unknown')
                job.setdefault('environment', 'unknown')
                job.setdefault('timestamp', datetime.now().isoformat())
                
                # Find the correct output directory
                job_id = job['job_id']
                output_dir = None
                for result_base_dir in result_dirs_to_scan:
                    potential_dir = os.path.join(result_base_dir, job_id)
                    if os.path.exists(potential_dir):
                        output_dir = potential_dir
                        break
                
                if output_dir:
                    job['output_dir'] = output_dir
                
                job['status'] = get_job_status(job['job_id'])
                jobs.append(job)
                discovered_jobs.discard(job_id)  # Remove from discovered since we have job file
        except Exception as e:
            pass  # Error loading job file
            continue
    
    # Create placeholder jobs for discovered result directories without job files
    for job_id in discovered_jobs:
        try:
            # Extract user and model info from job_id
            parts = job_id.split('_')
            if len(parts) >= 4:
                job_num = parts[1]
                user_id = parts[2]
                model_name = '_'.join(parts[3:-1])  # Everything except timestamp
                timestamp = parts[-1]
            else:
                job_num = "0000"
                user_id = "unknown"
                model_name = "discovered_job"
                timestamp = str(int(datetime.now().timestamp()))
            
            # Find the output directory
            output_dir = None
            for result_base_dir in result_dirs_to_scan:
                potential_dir = os.path.join(result_base_dir, job_id)
                if os.path.exists(potential_dir):
                    output_dir = potential_dir
                    break
            
            if output_dir:
                # Create a minimal job config for discovered jobs
                job = {
                    'job_id': job_id,
                    'user_id': user_id,
                    'model_name': model_name,
                    'task_type': 'unknown',
                    'environment': 'unknown',
                    'data_file': 'unknown',
                    'target_column': 'unknown',
                    'output_dir': output_dir,
                    'timestamp': datetime.fromtimestamp(int(timestamp)).isoformat() if timestamp.isdigit() else datetime.now().isoformat(),
                    'status': get_job_status(job_id)
                }
                jobs.append(job)
        except Exception as e:
            pass  # Error creating placeholder for discovered job
            continue
    
    # Remove any duplicate jobs by job_id
    seen_job_ids = set()
    unique_jobs = []
    for job in jobs:
        job_id = job.get('job_id', 'unknown')
        if job_id not in seen_job_ids:
            seen_job_ids.add(job_id)
            unique_jobs.append(job)
    
    jobs = unique_jobs
    
    if not jobs:
        st.info("No valid jobs found.")
        return
    
    # Sidebar for job selection
    with st.sidebar:
        st.header("🔍 Select Job")
        
        # Filter by status
        status_filter = st.selectbox(
            "Filter by Status",
            ['All', 'Completed', 'Running', 'Failed', 'Submitted']
        )
        
        # Filter jobs
        filtered_jobs = jobs
        if status_filter != 'All':
            filtered_jobs = [j for j in jobs if j['status'] == status_filter]
        
        if not filtered_jobs:
            st.warning(f"No jobs with status '{status_filter}'")
            return
        
        # Job selection dropdown with placeholder (with safe field access)
        job_options = ["-- Select a job to view results --"]
        for job in filtered_jobs:
            # Safely extract fields with defaults
            job_id = job.get('job_id', 'unknown_job')
            user_id = job.get('user_id', 'unknown_user')
            model_name = job.get('model_name', 'unknown_model')
            status = job.get('status', 'unknown')
            timestamp = job.get('timestamp', '2024-01-01T00:00:00')[:16]
            
            job_display = f"{extract_job_number(job_id)} | {user_id} | {model_name} | {status} | {timestamp}"
            job_options.append(job_display)
        
        selected_job_idx = st.selectbox(
            "Select Job",
            range(len(job_options)),
            format_func=lambda x: job_options[x]
        )
        
        # Check if a valid job is selected (not the placeholder)
        if selected_job_idx == 0:
            # No job selected - show placeholder info
            st.markdown("---")
            st.markdown("**📋 Job Selection Required**")
            st.info("Please select a job from the dropdown above to view its results.")
            
            # Show available jobs summary
            st.markdown("**Available Jobs:**")
            st.text(f"📊 Total: {len(filtered_jobs)}")
            status_summary = {}
            for job in filtered_jobs:
                status = job['status']
                status_summary[status] = status_summary.get(status, 0) + 1
            
            for status, count in status_summary.items():
                emoji = {'Completed': '✅', 'Running': '🔄', 'Failed': '❌', 'Submitted': '⏳'}.get(status, '📝')
                st.text(f"{emoji} {status}: {count}")
            
            # Main area message
            st.markdown("---")
            st.info("👆 **Please select a job from the sidebar dropdown to view its results and performance metrics.**")
            st.markdown("""
            ### 🔍 How to View Results:
            
            1. **Filter by Status** (optional): Choose 'All', 'Completed', 'Running', etc.
            2. **Select Job**: Pick a specific job from the dropdown
            3. **Explore Results**: View performance metrics, logs, and generated files
            
            ### 📊 Available Information:
            - **📈 Model Performance**: Metrics comparison and visualizations
            - **🔍 File Viewer**: Browse all generated files and artifacts  
            - **💾 Scoring Code**: Download ready-to-use scoring scripts
            - **📜 Logs**: View execution logs and error details
            """)
            return
        
        # Valid job selected
        selected_job = filtered_jobs[selected_job_idx - 1]  # Adjust for placeholder
        
        # Job info (with safe field access)
        st.markdown("---")
        st.markdown("**Job Information:**")
        st.text(f"Job Number: {extract_job_number(selected_job.get('job_id', 'unknown'))}")
        st.text(f"Full ID: {selected_job.get('job_id', 'unknown')}")
        st.text(f"User: {selected_job.get('user_id', 'unknown_user')}")
        st.text(f"Model: {selected_job.get('model_name', 'unknown_model')}")
        st.text(f"Status: {selected_job.get('status', 'unknown')}")
        st.text(f"Environment: {selected_job.get('environment', 'N/A')}")
    
    # Main results area (with safe field access)
    job_id = selected_job.get('job_id', 'unknown_job')
    output_dir = selected_job.get('output_dir', '')
    
    # Status indicator
    status = selected_job.get('status', 'unknown')
    job_number = extract_job_number(job_id)
    
    if status == 'Completed':
        st.success(f"✅ Job {job_number} completed successfully!")
    elif status == 'Running':
        # Get job progress information
        progress_info = get_job_progress(job_id)
        current_step = progress_info.get('current_step', 0)
        task_type = selected_job.get('task_type', 'classification')
        total_steps = get_total_steps_for_task(task_type)
        current_task = progress_info.get('current_task', 'Initializing...')
        progress_percentage = progress_info.get('progress_percentage', 0.0)
        
        # Enhanced status message with user ID and model name (safe field access)
        user_display = selected_job.get('user_id', 'Unknown')
        model_display = selected_job.get('model_name', 'Unknown')
        st.info(f"🔄 **Job {job_number}** - **{user_display}** - **{model_display}** job is currently running...")
        
        # Progress bar and task information
        st.subheader("📊 Progress Tracking")
        
        # Progress bar
        progress_value = progress_percentage / 100.0
        st.progress(progress_value)
        
        # Progress details
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric(
                label="Progress", 
                value=f"{progress_percentage}%",
                delta=f"Step {current_step}/{total_steps}"
            )
        
        with col2:
            st.metric(
                label="Current Task",
                value=current_task
            )
        
        # Task breakdown based on task type
        task_type = selected_job.get('task_type', 'classification')
        if task_type == 'classification':
            tasks = [
                "1. Data Preprocessing",
                "2. Feature Selection", 
                "3. Data Splitting and Scaling",
                "4. Preparing Out-of-Time Datasets",
                "5. Model Building and Validation",
                "6. Model Selection",
                "7. Generating Scoring Code",
                "8. Saving Model Configuration"
            ]
        elif task_type == 'regression':
            tasks = [
                "1. Data Preprocessing",
                "2. Feature Selection",
                "3. Data Splitting and Scaling", 
                "4. Preparing Out-of-Time Datasets",
                "5. Model Building and Validation",
                "6. Model Selection",
                "7. Generating Scoring Code",
                "8. Saving Model Configuration"
            ]
        else:  # clustering
            tasks = [
                "1. Data Preprocessing",
                "2. Feature Scaling",
                "3. Clustering Analysis",
                "4. Model Building and Validation", 
                "5. Model Selection",
                "6. Saving Model Configuration"
            ]
        
        # Show task list with current task highlighted
        st.subheader("📋 Task Breakdown")
        for i, task in enumerate(tasks):
            if i < current_step:
                st.markdown(f"✅ {task}")
            elif i == current_step:
                st.markdown(f"🔄 **{task}** (Current)")
            else:
                st.markdown(f"⏳ {task}")
        
        # Real-time logs section
        st.subheader("📜 Real-time Job Logs")
        
        # Get recent logs
        logs = get_job_logs(job_id, max_lines=30)
        
        if logs:
            # Create a log display area
            log_text = "".join(logs)
            
            # Show logs in an expandable section
            with st.expander("📋 View Recent Logs", expanded=True):
                st.code(log_text, language='text')
                
                # Add download button for full logs
                if st.button("📥 Download Full Logs", key=f"download_{job_id}"):
                    full_logs = get_job_logs(job_id, max_lines=1000)
                    full_log_text = "".join(full_logs)
                    st.download_button(
                        label="⬇️ Download Complete Log File",
                        data=full_log_text,
                        file_name=f"{job_id}_complete_log.txt",
                        mime='text/plain'
                    )
        else:
            st.info("📝 No logs available yet. Logs will appear as the job progresses.")
        
        # Control buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("🔄 Refresh Status", key=f"refresh_{job_id}", use_container_width=True):
                st.rerun()
        
        with col3:
            if st.button("🛑 Stop Job", key=f"stop_{job_id}", use_container_width=True):
                if job_manager is not None:
                    if job_manager.stop_job(job_id):
                        st.success("✅ Job stopped successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to stop job")
                else:
                    st.warning("⚠️ Job manager not available for stopping jobs")
        
        st.info("💡 Click 'Refresh Status' to check for updates, or 'Stop Job' to cancel")
        st.success("✅ You're on the Results page - your job is running in the background!")
        return
    elif status == 'Failed':
        st.error(f"❌ Job {job_number} failed!")
        # Show error log if available
        error_file = os.path.join(JOBS_DIR, f"{job_id}_error.log")
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                error_log = f.read()
            st.code(error_log, language='text')
        return
    else:
        st.warning(f"⏳ Job {job_number} is queued...")
        return
    
    # Check if results exist
    if not os.path.exists(output_dir):
        st.warning("Results directory not found!")
        return
    
    # Results tabs - add cluster profiling tab for clustering tasks
    task_type = selected_job.get('task_type', 'classification')
    
    if task_type == 'clustering':
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Model Performance", "🔬 Cluster Profiling", "🔍 File Viewer", "💾 Scoring Code", "📜 Logs"])
        
        with tab1:
            display_model_performance(output_dir, task_type)
        
        with tab2:
            display_cluster_profiling(output_dir)
        
        with tab3:
            display_file_viewer(output_dir)
        
        with tab4:
            display_scoring_code(output_dir)
        
        with tab5:
            display_logs(output_dir, job_id)
    else:
        # For classification and regression, validation plots are now integrated into Model Performance tab
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Performance", "🔍 File Viewer", "💾 Scoring Code", "📜 Logs"])
        
        with tab1:
            display_model_performance(output_dir, task_type)
        
        with tab2:
            display_file_viewer(output_dir)
        
        with tab3:
            display_scoring_code(output_dir)
        
        with tab4:
            display_logs(output_dir, job_id)

def display_feature_importance_section(output_dir: str, task_type: str = 'classification'):
    """Display feature importance plots and data as the first section in model performance."""
    
    # Use cached file discovery for better performance
    feature_importance_files = []
    seen_files = set()  # Track seen file paths to avoid duplicates
    
    # Get cached feature importance files
    all_files = get_cached_file_discovery(output_dir, 'feature_importance', task_type)
    
    for file_path in all_files:
        if file_path not in seen_files:
            filename = os.path.basename(file_path)
            filepath = os.path.dirname(file_path)
            
            # Determine file type and create display name
            if file_path.endswith('.png'):
                file_type = 'plot'
                if filepath == output_dir:
                    display_name = 'Overall Feature Importance'
                else:
                    model_dir = os.path.basename(filepath)
                    display_name = f"{model_dir.replace('_', ' ').title()} - Feature Importance"
            elif file_path.endswith('.xlsx'):
                file_type = 'excel'
                if filepath != output_dir:
                    # File is in a subdirectory (likely model-specific)
                    model_dir = os.path.basename(filepath)
                    display_name = f"{model_dir.replace('_', ' ').title()} - Feature Importance"
                else:
                    # File is in main directory
                    display_name = filename.replace('_', ' ').replace('.xlsx', '').title()
                    if 'feature' not in display_name.lower():
                        display_name = f"Feature Importance - {display_name}"
            else:
                continue
            
            feature_importance_files.append((display_name, file_path, file_type))
            seen_files.add(file_path)
    
    if not feature_importance_files:
        # Don't show anything if no feature importance files found
        return
    
    # Display feature importance section
    st.subheader("🎯 Feature Importance Analysis")
    st.markdown("Understanding which features contribute most to model predictions")
    
    # If only one file, display directly
    if len(feature_importance_files) == 1:
        display_name, file_path, file_type = feature_importance_files[0]
        display_single_feature_importance(display_name, file_path, file_type)
    
    else:
        # Multiple files - create tabs or selection
        st.markdown("**Select Feature Importance View:**")
        
        # Create selection options
        display_options = [name for name, _, _ in feature_importance_files]
        selected_option = st.selectbox(
            "Choose feature importance view:",
            display_options,
            help="Select different feature importance analyses to view"
        )
        
        # Find and display selected file
        for display_name, file_path, file_type in feature_importance_files:
            if display_name == selected_option:
                display_single_feature_importance(display_name, file_path, file_type)
                break
    
    # Add separator
    st.markdown("---")


def display_single_feature_importance(display_name: str, file_path: str, file_type: str):
    """Display a single feature importance file (plot or excel)."""
    
    try:
        if file_type == 'plot':
            # Display image
            st.markdown(f"**{display_name}**")
            
            # Check if file exists and display
            if os.path.exists(file_path):
                try:
                    st.image(file_path, use_container_width=True)
                    
                    # Add download button for the plot
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label=f"📥 Download {display_name} Plot",
                            data=f.read(),
                            file_name=os.path.basename(file_path),
                            mime='image/png'
                        )
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
            else:
                st.warning(f"Feature importance plot not found: {file_path}")
        
        elif file_type == 'excel':
            # Display Excel data
            st.markdown(f"**{display_name}**")
            
            if os.path.exists(file_path):
                try:
                    # Use cached Excel reading for better performance
                    df = cached_read_excel(file_path)
                    
                    # Display as interactive table
                    if len(df) > 0:
                        st.dataframe(
                            df, 
                            use_container_width=True,
                            height=min(400, len(df) * 35 + 100)
                        )
                        
                        # Note: Interactive plots removed per user preference - keeping only Excel data display
                        
                        # Add download button
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label=f"📥 Download {display_name} Data",
                                data=f.read(),
                                file_name=os.path.basename(file_path),
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                    else:
                        st.warning("Excel file is empty")
                        
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
            else:
                st.warning(f"Feature importance Excel file not found: {file_path}")
    
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

def display_model_performance(output_dir: str, task_type: str = 'classification'):
    """Display model performance metrics and visualizations."""
    st.header(f"Model Performance Overview - {task_type.title()}")
    
    # 1. FEATURE IMPORTANCE SECTION (First priority)
    with st.spinner("Loading feature importance analysis..."):
        display_feature_importance_section(output_dir, task_type)
    
    # 2. Model summary files section
    # Use cached summary file discovery for better performance
    summary_files = get_cached_file_discovery(output_dir, 'summary', task_type)
    
    # Display summary files
    if summary_files:
        st.subheader("📊 Model Validation Summaries")
        
        # Create list of available summaries with their names
        available_summaries = []
        for summary_file in summary_files:
            if os.path.exists(summary_file):
                model_name = os.path.basename(os.path.dirname(summary_file)) if os.path.dirname(summary_file) != output_dir else "Overall"
                available_summaries.append((model_name, summary_file))
        
        if available_summaries:
            # Remove duplicates by keeping only unique model names
            unique_summaries = {}
            for name, file_path in available_summaries:
                if name not in unique_summaries:
                    unique_summaries[name] = file_path
            
            # For classification, regression, and clustering, show only the overall summary
            if task_type in ['classification', 'regression', 'clustering']:
                # Find the overall summary
                overall_summary = None
                
                if task_type == 'clustering':
                    # For clustering, find the best model's summary based on silhouette score
                    best_model_summary = None
                    best_silhouette = -1
                    
                    # Get clustering metrics to determine best model
                    clustering_data = get_cached_file_discovery(output_dir, 'clustering', 'clustering')
                    clustering_files = clustering_data.get('main_clustering', [])
                    
                    for metrics_file in clustering_files:
                        try:
                            metrics = cached_read_json(metrics_file)
                            if metrics and 'datasets' in metrics and 'train' in metrics['datasets']:
                                silhouette = metrics['datasets']['train'].get('silhouette_score', -1)
                                if silhouette > best_silhouette:
                                    best_silhouette = silhouette
                                    # Find corresponding summary file
                                    model_name = os.path.basename(metrics_file).replace('_clustering_metrics.json', '')
                                    for name, file_path in unique_summaries.items():
                                        if model_name in file_path and "dataset_validation_summary" in file_path.lower():
                                            best_model_summary = file_path
                                            break
                        except:
                            continue
                    
                    # If no best model found, try to find any clustering summary file
                    if not best_model_summary:
                        for name, file_path in unique_summaries.items():
                            if ("dataset_validation_summary" in file_path.lower() or 
                                "clustering_summary" in file_path.lower()):
                                best_model_summary = file_path
                                break
                    
                    overall_summary = best_model_summary
                else:
                    # For classification and regression, use existing logic
                    for name, file_path in unique_summaries.items():
                        if (name == "Overall" or 
                            "model_selection" in file_path.lower() or 
                            "champion_challenger" in file_path.lower()):
                            overall_summary = file_path
                            break
                
                if overall_summary:
                    if overall_summary.endswith('.json'):
                        # Handle JSON summary files
                        try:
                            summary_data = cached_read_json(overall_summary)
                            if summary_data:
                                st.json(summary_data)
                            else:
                                st.info("Could not load JSON summary file.")
                        except Exception as e:
                            st.error(f"Error loading JSON summary: {str(e)}")
                    else:
                        # Handle text summary files
                        with open(overall_summary, 'r') as f:
                            summary_content = f.read()
                        st.code(summary_content, language='text')
                else:
                    st.info(f"No overall summary found for {task_type} task.")
    
    if task_type == 'clustering':
        # Special handling for enhanced clustering metrics with progress indicators
        with st.spinner("Loading clustering metrics..."):
            display_enhanced_clustering_metrics(output_dir)
        
        # Always display clustering plots (similar to classification confusion matrices and ROC curves)
        with st.spinner("Loading clustering plots..."):
            display_clustering_elbow_plots(output_dir)
            display_clustering_scatter_plots(output_dir)
            display_clustering_distribution_plots(output_dir)
            
    else:
        # Use cached metrics file discovery for better performance
        with st.spinner("Loading model metrics..."):
            metrics_files = get_cached_file_discovery(output_dir, 'metrics', task_type)
            
            if metrics_files:
                metrics_data = []
                for metrics_file in metrics_files:
                    try:
                        # Use cached JSON reading for better performance
                        metrics = cached_read_json(metrics_file)
                        if metrics:  # Only process if we got valid data
                            # Extract model name from file path
                            filename = os.path.basename(metrics_file)
                            if filename.endswith('_multiclass_metrics.json'):
                                model_name = filename.replace('_multiclass_metrics.json', '')
                            else:
                                model_name = filename.replace('_metrics.json', '')
                            metrics['model'] = model_name
                            metrics_data.append(metrics)
                    except:
                        continue
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                # Remove duplicate models (keep first occurrence)
                df = df.drop_duplicates(subset=['model'], keep='first')
                
                # For classification, reorganize the display
                if task_type == 'classification':
                    display_classification_metrics(df, output_dir)
                elif task_type == 'regression':
                    # Use the new regression metrics display (consistent with classification)
                    display_regression_metrics(df, output_dir)
            else:
                st.info("No metrics files found. The job may still be running or failed.")
    
    # Always display regression plots (similar to classification confusion matrices and ROC curves)
    if task_type == 'regression':
        # Add performance optimization toggle
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            load_all_plots = st.checkbox("Load all plots at once", value=False, 
                                       help="Check to load all regression plots immediately. Uncheck for faster initial loading.")
        with col2:
            if not load_all_plots:
                st.info("💡 **Performance Tip**: Uncheck 'Load all plots at once' for faster initial page load. Plots will load when you expand each section.")
        
        # Get sorted model files for consistent ordering
        sorted_model_files = get_sorted_regression_model_files(output_dir)
        
        if load_all_plots:
            # Load all plots at once with progress indicator
            with st.spinner("Loading all regression plots..."):
                display_regression_qq_plots(output_dir, sorted_model_files)
                display_regression_actual_vs_predicted_plots(output_dir, sorted_model_files)
                display_regression_residuals_plots(output_dir, sorted_model_files)
        else:
            # Lazy loading - load plots on demand
            with st.expander("📈 QQ Plots (Click to load)", expanded=False):
                with st.spinner("Loading QQ plots..."):
                    display_regression_qq_plots(output_dir, sorted_model_files)
            
            with st.expander("📊 Actual vs Predicted Plots (Click to load)", expanded=False):
                with st.spinner("Loading actual vs predicted plots..."):
                    display_regression_actual_vs_predicted_plots(output_dir, sorted_model_files)
            
            with st.expander("📈 Residuals Plots (Click to load)", expanded=False):
                with st.spinner("Loading residuals plots..."):
                    display_regression_residuals_plots(output_dir, sorted_model_files)
    
    # Always display KS decile tables, confusion matrices, and ROC curves for classification
    if task_type == 'classification':
        with st.spinner("Loading classification metrics and tables..."):
            display_ks_decile_tables(output_dir)

    # -- SHAP Explainability Section --
    # Display SHAP summary plot and values for the best model if available.  SHAP
    # values are computed by the AutoML pipeline and saved to the output
    # directory as ``shap_summary_<task_type>.png`` and
    # ``shap_values_<task_type>.csv``.  This section appears for all task types.
    shap_summary_path = os.path.join(output_dir, f"shap_summary_{task_type}.png")
    shap_values_path = os.path.join(output_dir, f"shap_values_{task_type}.csv")
    if os.path.exists(shap_summary_path):
        st.markdown("---")
        st.subheader("🔍 Model Explainability (SHAP)")
        try:
            st.image(shap_summary_path, caption="SHAP Summary Plot for the Best Model", use_column_width=True)
        except Exception as e:
            st.warning(f"Unable to load SHAP summary plot: {e}")
        # Display SHAP values table if available
        if os.path.exists(shap_values_path):
            try:
                shap_df = pd.read_csv(shap_values_path)
                st.markdown("**Top SHAP Contributions (first 50 rows):**")
                st.dataframe(shap_df.head(50))
                st.caption("Note: Each row corresponds to a sample used for SHAP computation. Columns represent feature contributions.")
            except Exception as e:
                st.warning(f"Unable to load SHAP values CSV: {e}")
        else:
            st.info("SHAP values CSV not found.")

def display_classification_metrics(df, output_dir: str = ""):
    """Display classification metrics with reorganized layout and grouped charts."""
    # Check if we have any relevant metrics to display
    classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roc_auc', 'accuracy_test', 'f1_test', 'precision_test', 'recall_test']
    available_metrics = [col for col in classification_metrics if col in df.columns]
    
    if not available_metrics:
        return  # Don't show anything if no relevant metrics are available
    
    # Define the desired order for model types (same as KS tables)
    model_order = [
        'logistic', 'logistic_default', 'logistic_tuned',
        'random_forest', 'random_forest_default', 'random_forest_tuned',
        'decision_tree', 'decision_tree_default', 'decision_tree_tuned',
        'gradient_boosting', 'gradient_boosting_default', 'gradient_boosting_tuned',
        'xgboost', 'xgboost_default', 'xgboost_tuned',
        'lightgbm', 'lightgbm_default', 'lightgbm_tuned',
        'neural_network', 'neural_network_default', 'neural_network_tuned'
    ]
    
    # Sort models according to the desired order
    sorted_models = []
    used_models = set()
    
    for model_type in model_order:
        for model in df['model']:
            if model_type.lower() in model.lower() and model not in used_models:
                sorted_models.append(model)
                used_models.add(model)
                break
    
    # Add any remaining models that don't match the pattern
    for model in df['model']:
        if model not in used_models:
            sorted_models.append(model)
            used_models.add(model)
    
    # Reorder dataframe by sorted models
    df_sorted = df.set_index('model').reindex(sorted_models).reset_index()
    
    # 1. Reorganize metrics table by metric type
    st.subheader("📈 Classification Metrics Comparison")
    
    # Group metrics by type (accuracy, f1, precision, recall)
    metric_groups = {}
    for col in df_sorted.columns:
        if col != 'model':
            # Extract base metric name (remove _train, _test, _valid suffixes)
            base_metric = col.split('_')[0] if '_' in col else col
            if base_metric not in metric_groups:
                metric_groups[base_metric] = []
            metric_groups[base_metric].append(col)
    
    # Create reorganized dataframe
    reorganized_data = []
    for model in df_sorted['model']:
        model_data = {'Model': model}
        for base_metric, metric_cols in metric_groups.items():
            for metric_col in metric_cols:
                if metric_col in df_sorted.columns:
                    model_data[metric_col] = df_sorted[df_sorted['model'] == model][metric_col].iloc[0]
        reorganized_data.append(model_data)
    
    reorganized_df = pd.DataFrame(reorganized_data)
    st.dataframe(reorganized_df.round(4), use_container_width=True)
    
    # 2. Create grouped bar charts for performance metrics
    st.subheader("🎯 Classification Performance Metrics")
    
    # Create grouped bar charts for each metric type
    for base_metric, metric_cols in metric_groups.items():
        if len(metric_cols) > 1:  # Only create charts if we have multiple datasets
            # Prepare data for grouped bar chart
            chart_data = []
            for model in df_sorted['model']:
                for metric_col in metric_cols:
                    if metric_col in df_sorted.columns:
                        value = df_sorted[df_sorted['model'] == model][metric_col].iloc[0]
                        # Extract dataset name from metric column
                        dataset = metric_col.replace(f'{base_metric}_', '') if '_' in metric_col else 'overall'
                        chart_data.append({
                            'Model': model,
                            'Dataset': dataset,
                            'Value': value,
                            'Metric': base_metric.upper() if base_metric in ['roc', 'ks'] else base_metric.title()
                        })
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                
                # Create enhanced bar chart with better styling
                fig = go.Figure()
                
                # Get unique datasets for color coding
                datasets = chart_df['Dataset'].unique()
                colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4']
                
                for i, dataset in enumerate(datasets):
                    dataset_data = chart_df[chart_df['Dataset'] == dataset]
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Bar(
                        x=dataset_data['Model'],
                        y=dataset_data['Value'],
                        name=dataset.title(),
                        marker_color=color,
                        marker_line_color='black',
                        marker_line_width=1,
                        text=[f'{val:.2f}' for val in dataset_data['Value']],
                        textposition='outside',
                        textfont=dict(size=10, color='black')
                    ))
                
                # Update layout for better appearance
                metric_title = base_metric.upper() if base_metric in ['roc', 'ks'] else base_metric.title()
                fig.update_layout(
                    title=f'{metric_title} by Model and Dataset',
                    xaxis_title='Model',
                    yaxis_title=f'{metric_title} Score',
                    barmode='group',
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12),
                    height=500
                )
                
                # Update axes
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                )
                
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black'
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"{base_metric}_grouped_chart")

def display_ks_decile_tables(output_dir: str):
    """Display KS decile tables from Excel files for binary classification models."""
    try:
        # Use cached file discovery for better performance
        ks_files = cached_glob_files(os.path.join(output_dir, '*', 'KS_Charts.xlsx'))
        
        if not ks_files:
            return  # No KS files found
        
        # Initialize sorted_model_files to avoid UnboundLocalError
        sorted_model_files = []
        
        st.subheader("📊 KS Decile Tables")
        st.markdown("Kolmogorov-Smirnov (KS) decile analysis for binary classification models")
        
        # Deduplicate by model directory - only keep one KS file per model
        unique_ks_files = []
        seen_dirs = set()
        for ks_file in ks_files:
            model_dir = os.path.dirname(ks_file)
            if model_dir not in seen_dirs:
                unique_ks_files.append(ks_file)
                seen_dirs.add(model_dir)
        
        # Create tabs for each model's KS table
        if len(unique_ks_files) == 1:
            # Single model - display directly
            display_single_ks_table(unique_ks_files[0])
            # Create single-item list for sorted_model_files
            model_name = os.path.basename(os.path.dirname(unique_ks_files[0]))
            sorted_model_files = [(model_name, unique_ks_files[0])]
        else:
            # Multiple models - create tabs with consistent ordering
            model_files = []
            for ks_file in unique_ks_files:
                model_name = os.path.basename(os.path.dirname(ks_file))
                model_files.append((model_name, ks_file))
            
            # Model files found and processed
            
            # Define the desired order for model types
            model_order = [
                'logistic', 'logistic_default', 'logistic_tuned',
                'random_forest', 'random_forest_default', 'random_forest_tuned',
                'decision_tree', 'decision_tree_default', 'decision_tree_tuned',
                'gradient_boosting', 'gradient_boosting_default', 'gradient_boosting_tuned',
                'xgboost', 'xgboost_default', 'xgboost_tuned',
                'lightgbm', 'lightgbm_default', 'lightgbm_tuned',
                'neural_network', 'neural_network_default', 'neural_network_tuned'
            ]
            
            # Remove duplicates first
            unique_model_files = []
            seen_models = set()
            for model_name, ks_file in model_files:
                if model_name not in seen_models:
                    unique_model_files.append((model_name, ks_file))
                    seen_models.add(model_name)
            
            # Unique models after deduplication
            
            # Sort model files according to the desired order
            sorted_model_files = []
            used_models = set()
            
            for model_type in model_order:
                for model_name, ks_file in unique_model_files:
                    if model_type.lower() in model_name.lower() and model_name not in used_models:
                        sorted_model_files.append((model_name, ks_file))
                        used_models.add(model_name)
            
            # Add any remaining models that don't match the pattern
            for model_name, ks_file in unique_model_files:
                if model_name not in used_models:
                    sorted_model_files.append((model_name, ks_file))
                    used_models.add(model_name)
            
            # Create tabs with sorted names
            tab_names = [model_name for model_name, _ in sorted_model_files]
            tabs = st.tabs(tab_names)
            
            for i, (tab, (model_name, ks_file)) in enumerate(zip(tabs, sorted_model_files)):
                with tab:
                    display_single_ks_table(ks_file)
        
        # Display confusion matrices in separate section with tabs
        display_confusion_matrices_with_tabs(output_dir, sorted_model_files)
        
        # Display ROC curves in separate section with tabs
        display_roc_curves_with_tabs(output_dir, sorted_model_files)
        
    except Exception as e:
        st.warning(f"Could not load KS decile tables: {str(e)}")

def display_confusion_matrices_with_tabs(output_dir: str, sorted_model_files: list = None):
    """Display confusion matrices for each model using tab format."""
    try:
        # If sorted_model_files is provided, use the same ordering as KS tables
        if sorted_model_files:
            # Filter to only include models that have confusion matrix files
            model_dirs_with_files = []
            for model_name, _ in sorted_model_files:
                model_dir = os.path.join(output_dir, model_name)
                if os.path.exists(model_dir):
                    # Check if this model directory has confusion matrix files
                    confusion_matrix_files = []
                    for file in os.listdir(model_dir):
                        if 'confusion matrix' in file.lower() and file.endswith('.png'):
                            confusion_matrix_files.append(os.path.join(model_dir, file))
                    
                    # Only include directories that have confusion matrix files
                    if confusion_matrix_files:
                        model_dirs_with_files.append((model_name, model_dir))
        else:
            # Fallback: Find all model directories that actually have confusion matrix files
            model_dirs_with_files = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and any(model_type in item.lower() for model_type in [
                    'logistic', 'xgboost', 'lightgbm', 'random_forest', 'randomforest', 
                    'naivebayes', 'decision_tree', 'decisiontree', 'gradient_boosting', 
                    'gradientboosting', 'neural_network', 'neuralnetwork'
                ]):
                    # Check if this model directory has confusion matrix files
                    confusion_matrix_files = []
                    for file in os.listdir(item_path):
                        if 'confusion matrix' in file.lower() and file.endswith('.png'):
                            confusion_matrix_files.append(os.path.join(item_path, file))
                    
                    # Only include directories that have confusion matrix files
                    if confusion_matrix_files:
                        model_dirs_with_files.append((item, item_path))
            
            # Sort model directories for consistent display
            model_dirs_with_files.sort(key=lambda x: x[0])
        
        if not model_dirs_with_files:
            return  # No model directories with confusion matrix files found
        
        st.subheader("🎯 Confusion Matrices")
        st.markdown("Confusion matrix analysis for classification models")
        
        # Create tabs for each model's confusion matrices
        if len(model_dirs_with_files) == 1:
            # Single model - display directly
            display_single_model_confusion_matrices(model_dirs_with_files[0][1])
        else:
            # Multiple models - create tabs using the same names as KS tables
            model_names = [model_name for model_name, _ in model_dirs_with_files]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, model_dir) in zip(tabs, model_dirs_with_files):
                with tab:
                    display_single_model_confusion_matrices(model_dir)
        
    except Exception as e:
        st.error(f"Error displaying confusion matrices: {str(e)}")

def display_single_model_confusion_matrices(model_dir: str):
    """Display confusion matrices for a single model."""
    try:
        model_name = os.path.basename(model_dir)
        
        # Find confusion matrix images
        confusion_matrix_files = []
        for file in os.listdir(model_dir):
            if 'confusion matrix' in file.lower() and file.endswith('.png'):
                confusion_matrix_files.append(os.path.join(model_dir, file))
        
        if not confusion_matrix_files:
            st.info(f"No confusion matrix images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in confusion_matrix_files:
                if dataset in file.lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in confusion_matrix_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 3:
            num_columns = num_datasets
        elif num_datasets == 4:
            num_columns = 4
        else:  # 5 or more datasets
            num_columns = 5
        
        # Create columns and display confusion matrices
        cols = st.columns(num_columns)
        
        for i, file_path in enumerate(sorted_files):
            col_index = i % num_columns
            with cols[col_index]:
                dataset_name = os.path.basename(file_path).replace('.png', '').replace('Confusion Matrix for ', '').replace(' data', '')
                st.markdown(f"**{dataset_name.title()}**")
                st.image(file_path, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying confusion matrices for {model_name}: {str(e)}")

def display_confusion_matrices(output_dir: str):
    """Display confusion matrices for each model."""
    try:
        # Find all model directories
        model_dirs = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and any(model_type in item.lower() for model_type in [
                'logistic', 'xgboost', 'lightgbm', 'random_forest', 'randomforest', 
                'naivebayes', 'decision_tree', 'decisiontree', 'gradient_boosting', 
                'gradientboosting', 'neural_network', 'neuralnetwork'
            ]):
                model_dirs.append(item_path)
        
        if not model_dirs:
            st.info("No model directories found for confusion matrices.")
            return
        
        # Sort model directories for consistent display
        model_dirs.sort()
        
        st.subheader("🎯 Confusion Matrices")
        
        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            st.markdown(f"**{model_name.replace('_', ' ').title()}**")
            
            # Find confusion matrix images
            confusion_matrix_files = []
            for file in os.listdir(model_dir):
                if 'confusion matrix' in file.lower() and file.endswith('.png'):
                    confusion_matrix_files.append(os.path.join(model_dir, file))
            
            if confusion_matrix_files:
                # Sort files by dataset type for consistent display
                dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
                sorted_files = []
                
                for dataset in dataset_order:
                    for file in confusion_matrix_files:
                        if dataset in file.lower():
                            sorted_files.append(file)
                            break
                
                # Add any remaining files
                for file in confusion_matrix_files:
                    if file not in sorted_files:
                        sorted_files.append(file)
                
                # Display confusion matrices in columns
                if len(sorted_files) <= 3:
                    cols = st.columns(len(sorted_files))
                    for i, file_path in enumerate(sorted_files):
                        with cols[i]:
                            dataset_name = os.path.basename(file_path).replace('.png', '').replace('Confusion Matrix for ', '').replace(' data', '')
                            st.markdown(f"**{dataset_name.title()}**")
                            st.image(file_path, use_container_width=True)
                else:
                    # Display in rows for many files
                    for file_path in sorted_files:
                        dataset_name = os.path.basename(file_path).replace('.png', '').replace('Confusion Matrix for ', '').replace(' data', '')
                        st.markdown(f"**{dataset_name.title()}**")
                        st.image(file_path, use_container_width=True)
                        st.markdown("---")
            else:
                st.info(f"No confusion matrix images found for {model_name}")
            
            st.markdown("---")
            
    except Exception as e:
        st.error(f"Error displaying confusion matrices: {str(e)}")

def convert_sheet_name_to_display(sheet_name: str) -> str:
    """Convert abbreviated sheet name to readable display name."""
    # Map abbreviations to full names
    model_mapping = {
        'RF': 'Random Forest',
        'GB': 'Gradient Boosting', 
        'DT': 'Decision Tree',
        'NN': 'Neural Network',
        'LR': 'Logistic Regression',
        'XGB': 'XGBoost',
        'LGB': 'LightGBM'
    }
    
    # Split the sheet name (e.g., "RF_def_train" -> ["RF", "def", "train"])
    parts = sheet_name.split('_')
    
    if len(parts) >= 2:
        model_abbr = parts[0]
        data_type = parts[-1]  # Last part is always data type
        
        # Get full model name
        model_name = model_mapping.get(model_abbr, model_abbr)
        
        # Add variant if present
        if len(parts) >= 3 and parts[1] in ['def', 'tuned']:
            variant = 'Default' if parts[1] == 'def' else 'Tuned'
            # Handle special dataset names
            if data_type == 'oot1':
                return f"{model_name} {variant} - Out-of-Time 1"
            elif data_type == 'oot2':
                return f"{model_name} {variant} - Out-of-Time 2"
            else:
                return f"{model_name} {variant} - {data_type.title()}"
        else:
            # Handle special dataset names without variant
            if data_type == 'oot1':
                return f"{model_name} - Out-of-Time 1"
            elif data_type == 'oot2':
                return f"{model_name} - Out-of-Time 2"
            else:
                return f"{model_name} - {data_type.title()}"
    
    # Fallback to original name if parsing fails
    return sheet_name


def display_single_ks_table(ks_file: str):
    """Display a single KS decile table from Excel file."""
    try:
        import pandas as pd
        import openpyxl
        
        # Read the Excel file
        wb = openpyxl.load_workbook(ks_file)
        sheet_names = wb.sheetnames
        
        if not sheet_names:
            st.warning("No sheets found in KS file")
            return
        
        # Define the desired order for datasets (including out-of-time datasets)
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        
        # Remove duplicate sheet names first
        unique_sheet_names = list(dict.fromkeys(sheet_names))  # Preserves order while removing duplicates
        
        # Sort sheet names according to the desired order
        sorted_sheet_names = []
        for dataset in dataset_order:
            for sheet_name in unique_sheet_names:
                if dataset.lower() in sheet_name.lower():
                    sorted_sheet_names.append(sheet_name)
                    break
        
        # Add any remaining sheets that don't match the pattern
        for sheet_name in unique_sheet_names:
            if sheet_name not in sorted_sheet_names:
                sorted_sheet_names.append(sheet_name)
        
        # Display each sheet as a separate table in the specified order
        for sheet_name in sorted_sheet_names:
            # Convert abbreviated sheet name to readable display name
            display_name = convert_sheet_name_to_display(sheet_name)
            st.markdown(f"**{display_name}**")
            
            # Read the sheet using cached Excel reading for better performance
            df = cached_read_excel(ks_file, sheet_name=sheet_name)
            
            # Format column names to sentence case
            df.columns = [col.replace('_', ' ').title() for col in df.columns]
            
            # Create a copy for styling (keep original for display)
            styled_df = df.copy()
            
            # Convert numeric columns back to numeric for styling and handle NaN values
            for col in styled_df.columns:
                if col in ['Count', 'Target', 'Spread']:
                    styled_df[col] = pd.to_numeric(styled_df[col], errors='coerce')
                    # Fill NaN values with 0 for styling purposes
                    styled_df[col] = styled_df[col].fillna(0)
            
            # Start with base styling
            style_chain = styled_df.style
            
            # Apply bar styling to Count column - ALWAYS use bars for consistency
            if 'Count' in styled_df.columns:
                # Ensure we have valid numeric data for bar charts
                count_data = styled_df['Count'].fillna(0)
                if count_data.max() > 0:  # Only apply bars if we have positive values
                    style_chain = style_chain.bar(
                        subset=['Count'], 
                        color='#3B82F6',  # Blue - vibrant and consistent
                        align='left',
                        width=100,
                        vmin=0,
                        vmax=count_data.max()
                    )
            
            # Apply bar styling to Target column
            if 'Target' in styled_df.columns:
                # Ensure we have valid numeric data for bar charts
                target_data = styled_df['Target'].fillna(0)
                if target_data.max() > 0:  # Only apply bars if we have positive values
                    style_chain = style_chain.bar(
                        subset=['Target'], 
                        color='#10B981',  # Emerald green - vibrant
                        align='left',
                        width=100,
                        vmin=0,
                        vmax=target_data.max()
                    )
            
            # Highlight max value in Spread column
            if 'Spread' in styled_df.columns:
                spread_data = styled_df['Spread'].fillna(0)
                if spread_data.max() > 0:  # Only highlight if we have positive values
                    # Define highlight function for maximum value
                    def highlight_max(s):
                        is_max = s == s.max()
                        return ['background-color: orange' if v else '' for v in is_max]
                    
                    # Apply highlighting and formatting to Spread column
                    style_chain = style_chain.apply(highlight_max, subset=['Spread'])
                    style_chain = style_chain.format({'Spread': '{:.2f}'})
                    style_chain = style_chain.hide(axis="index")  # Hide the index column
                    style_chain = style_chain.set_table_styles([
                        {'selector': 'th', 'props': [('font-weight', 'normal')]}
                    ])  # Make column headers normal weight instead of bold
            
            # Format numeric columns for display (after styling)
            display_df = df.copy()
            for col in display_df.columns:
                if display_df[col].dtype in ['float64', 'float32']:
                    # Format decimal columns to 2 decimal places
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                elif display_df[col].dtype in ['int64', 'int32']:
                    # Format integer columns to whole numbers
                    display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else x)
            
            # Apply the same formatting to the styled dataframe
            for col in styled_df.columns:
                if col in ['Count', 'Target', 'Spread']:
                    # Keep numeric for styling but format for display
                    continue
                elif styled_df[col].dtype in ['float64', 'float32']:
                    styled_df[col] = styled_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
                elif styled_df[col].dtype in ['int64', 'int32']:
                    styled_df[col] = styled_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else x)
            
            # Display the styled table using HTML rendering
            try:
                # Convert styled dataframe to HTML for better rendering
                html = style_chain.to_html()
                st.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                pass  # Styling failed
                # Fallback to regular dataframe if styling fails
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Add some spacing between tables
            st.markdown("---")
            
    except ImportError:
        st.warning("pandas and openpyxl are required to display KS tables. Please install: `pip install pandas openpyxl`")
    except Exception as e:
        st.error(f"Error reading KS file {ks_file}: {str(e)}")

def display_regression_metrics(df, output_dir: str = ""):
    """Display regression metrics with reorganized layout and grouped charts similar to classification."""
    
    # Check if we have any relevant metrics to display (more flexible detection)
    regression_metrics = ['rmse', 'mae', 'r2', 'mse', 'mape', 'explained_variance']
    
    # Find metrics more flexibly - check if any column contains the metric name
    available_metrics = []
    for metric in regression_metrics:
        metric_columns = [col for col in df.columns if metric in col.lower()]
        if metric_columns:
            available_metrics.append(metric)
    
    if not available_metrics:
        return  # Don't show anything if no relevant metrics are available
    
    # Define the desired order for model types (similar to classification)
    model_order = [
        'linear_regression', 'linear_regression_default', 'linear_regression_tuned',
        'random_forest', 'random_forest_default', 'random_forest_tuned',
        'decision_tree', 'decision_tree_default', 'decision_tree_tuned',
        'gradient_boosting', 'gradient_boosting_default', 'gradient_boosting_tuned',
        'xgboost', 'xgboost_default', 'xgboost_tuned',
        'lightgbm', 'lightgbm_default', 'lightgbm_tuned'
    ]
    
    # Sort models according to the desired order
    sorted_models = []
    used_models = set()
    
    for model_type in model_order:
        for model in df['model']:
            if model_type.lower() in model.lower() and model not in used_models:
                sorted_models.append(model)
                used_models.add(model)
                break
    
    # Add any remaining models that don't match the pattern
    for model in df['model']:
        if model not in used_models:
            sorted_models.append(model)
            used_models.add(model)
    
    # Reorder dataframe by sorted models
    df_sorted = df.set_index('model').reindex(sorted_models).reset_index()
    
    # 1. Reorganize metrics table by metric type
    st.subheader("📈 Regression Metrics Comparison")
    
    # Group metrics by type (rmse, mae, r2)
    metric_groups = {}
    for col in df_sorted.columns:
        if col != 'model':
            # Extract base metric name (remove _train, _test, _valid suffixes)
            for base_metric in regression_metrics:
                if base_metric in col.lower():
                    if base_metric not in metric_groups:
                        metric_groups[base_metric] = []
                    metric_groups[base_metric].append(col)
                    break
    
    # Create reorganized dataframe
    reorganized_data = []
    for model in df_sorted['model']:
        model_data = {'Model': model}
        for base_metric, metric_cols in metric_groups.items():
            for metric_col in metric_cols:
                if metric_col in df_sorted.columns:
                    model_data[metric_col] = df_sorted[df_sorted['model'] == model][metric_col].iloc[0]
        reorganized_data.append(model_data)
    
    reorganized_df = pd.DataFrame(reorganized_data)
    st.dataframe(reorganized_df.round(4), use_container_width=True)
    
    # 2. Create grouped bar charts for performance metrics (focus on key metrics: RMSE, R2, MAE)
    st.subheader("🎯 Regression Performance Metrics")
    
    # Focus on the three key metrics requested by user
    key_metrics = ['rmse', 'r2', 'mae']
    
    # Create grouped bar charts for each key metric type
    for base_metric in key_metrics:
        if base_metric in metric_groups and len(metric_groups[base_metric]) > 1:
            metric_cols = metric_groups[base_metric]
            
            # Prepare data for grouped bar chart
            chart_data = []
            for model in df_sorted['model']:
                for metric_col in metric_cols:
                    if metric_col in df_sorted.columns:
                        value = df_sorted[df_sorted['model'] == model][metric_col].iloc[0]
                        # Extract dataset name from metric column
                        dataset = metric_col.replace(f'{base_metric}_', '') if '_' in metric_col else 'overall'
                        # Handle different naming patterns
                        for pattern in [f'{base_metric}_', f'_{base_metric}']:
                            if pattern in metric_col:
                                dataset = metric_col.replace(pattern, '')
                                break
                        chart_data.append({
                            'Model': model,
                            'Dataset': dataset,
                            'Value': value,
                            'Metric': base_metric.upper()
                        })
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                
                # Create enhanced bar chart with better styling
                fig = go.Figure()
                
                # Get unique datasets for color coding
                datasets = chart_df['Dataset'].unique()
                colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4']
                
                for i, dataset in enumerate(datasets):
                    dataset_data = chart_df[chart_df['Dataset'] == dataset]
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Bar(
                        x=dataset_data['Model'],
                        y=dataset_data['Value'],
                        name=dataset.title(),
                        marker_color=color,
                        marker_line_color='black',
                        marker_line_width=1,
                        text=[f'{val:.3f}' for val in dataset_data['Value']],
                        textposition='outside',
                        textfont=dict(size=10, color='black')
                    ))
                
                # Update layout for better appearance
                metric_title = base_metric.upper()
                better_indicator = "(Lower is Better)" if base_metric in ['rmse', 'mae'] else "(Higher is Better)"
                
                fig.update_layout(
                    title=f'{metric_title} by Model and Dataset {better_indicator}',
                    xaxis_title='Model',
                    yaxis_title=f'{metric_title} Score',
                    barmode='group',
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12),
                    height=500
                )
                
                # Update axes
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                )
                
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black'
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"{base_metric}_grouped_chart")

@st.cache_data(ttl=300)
def get_sorted_regression_model_files(output_dir: str):
    """Get sorted regression model files for consistent display ordering."""
    try:
        # Find all model directories using cached directory listing
        model_dirs = []
        for item in cached_list_directory(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and any(model_type in item.lower() for model_type in [
                'linear_regression', 'random_forest', 'randomforest', 'decision_tree', 'decisiontree', 
                'gradient_boosting', 'gradientboosting', 'xgboost', 'lightgbm'
            ]):
                model_dirs.append((item, item_path))
        
        if not model_dirs:
            return []
        
        # Define the desired order for model types
        model_order = [
            'linear_regression', 'linear_regression_default', 'linear_regression_tuned',
            'random_forest', 'random_forest_default', 'random_forest_tuned',
            'decision_tree', 'decision_tree_default', 'decision_tree_tuned',
            'gradient_boosting', 'gradient_boosting_default', 'gradient_boosting_tuned',
            'xgboost', 'xgboost_default', 'xgboost_tuned',
            'lightgbm', 'lightgbm_default', 'lightgbm_tuned'
        ]
        
        # Sort model directories according to the desired order
        sorted_model_files = []
        used_models = set()
        
        for model_type in model_order:
            for model_name, model_path in model_dirs:
                if model_type.lower() in model_name.lower() and model_name not in used_models:
                    sorted_model_files.append((model_name, model_path))
                    used_models.add(model_name)
        
        # Add any remaining models that don't match the pattern
        for model_name, model_path in model_dirs:
            if model_name not in used_models:
                sorted_model_files.append((model_name, model_path))
                used_models.add(model_name)
        
        return sorted_model_files
        
    except Exception as e:
        pass  # Error getting sorted regression model files
        return []

def display_regression_charts(df):
    """Display regression-specific charts - now calls the new grouped metrics display."""
    # Use the new consistent regression metrics display
    display_regression_metrics(df)

def display_enhanced_clustering_metrics(output_dir: str):
    """Display enhanced clustering metrics with dataset-specific results."""
    st.subheader("🔍 Enhanced Clustering Performance Metrics")
    
    # Use cached clustering file discovery for better performance
    clustering_data = get_cached_file_discovery(output_dir, 'clustering', 'clustering')
    
    clustering_files = clustering_data.get('main_clustering', [])
    multi_dataset_files = clustering_data.get('multi_dataset', [])
    cv_files = clustering_data.get('cross_validation', [])
    
    if not clustering_files:
        st.info("No enhanced clustering metrics found. The job may still be running or failed.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "📈 Dataset Analysis", "🔄 Cross-Validation"])
    
    with tab1:
        st.subheader("📊 Model Comparison Overview")
        display_clustering_model_comparison(clustering_files)
    
    with tab2:
        st.subheader("📈 Dataset-Specific Analysis")
        display_clustering_dataset_analysis(clustering_files)
    
    with tab3:
        st.subheader("🔄 Cross-Validation Results")
        if cv_files:
            display_clustering_cross_validation(cv_files)
        else:
            st.info("No cross-validation results found.")

def display_clustering_model_comparison(clustering_files):
    """Display model comparison for clustering results."""
    models_data = []
    
    for file_path in clustering_files:
        try:
            # Use cached JSON reading for better performance
            data = cached_read_json(file_path)
            if not data:  # Skip if no valid data
                continue
            
            model_name = os.path.basename(file_path).replace('_clustering_metrics.json', '')
            
            # Extract overall metrics or use validation dataset if available
            if 'datasets' in data:
                # Use validation dataset metrics if available, otherwise use train
                if 'validation' in data['datasets']:
                    metrics = data['datasets']['validation'].copy()
                    metrics['dataset_source'] = 'validation'
                elif 'test' in data['datasets']:
                    metrics = data['datasets']['test'].copy()
                    metrics['dataset_source'] = 'test'  
                else:
                    metrics = data['datasets']['train'].copy()
                    metrics['dataset_source'] = 'train'
            else:
                # Fallback to direct metrics
                metrics = data.copy()
                metrics['dataset_source'] = 'direct'
            
            # Only add if we have a valid silhouette score
            if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
                metrics['model'] = model_name
                models_data.append(metrics)
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    if models_data:
        df = pd.DataFrame(models_data)
        
        # Filter out any remaining None values
        df = df.dropna(subset=['silhouette_score'])
        
        if df.empty:
            st.info("No valid clustering metrics found for comparison.")
            return
        
        # Display comparison charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'silhouette_score' in df.columns:
                fig = px.bar(df, x='model', y='silhouette_score', 
                            title='Silhouette Score by Model (Higher is Better)',
                            color='dataset_source',
                            color_discrete_sequence=['blue', 'green', 'orange'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'calinski_harabasz_score' in df.columns:
                fig = px.bar(df, x='model', y='calinski_harabasz_score', 
                            title='Calinski-Harabasz Index (Higher is Better)',
                            color='dataset_source',
                            color_discrete_sequence=['green', 'blue', 'orange'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if 'davies_bouldin_score' in df.columns:
                fig = px.bar(df, x='model', y='davies_bouldin_score', 
                            title='Davies-Bouldin Index (Lower is Better)',
                            color='dataset_source',
                            color_discrete_sequence=['red', 'orange', 'blue'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if 'num_clusters' in df.columns:
                fig = px.bar(df, x='model', y='num_clusters', title='Number of Clusters by Model',
                            color='dataset_source',
                            color_discrete_sequence=['purple', 'blue', 'green'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'inertia' in df.columns:
                fig = px.bar(df, x='model', y='inertia', title='Inertia (WCSS) by Model (Lower is Better)',
                            color='dataset_source',
                            color_discrete_sequence=['orange', 'red', 'blue'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Comprehensive metrics table
        st.subheader("📊 Comprehensive Metrics Comparison")
        clustering_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
                            'inertia', 'num_clusters', 'cluster_balance', 'dataset_source']
        available_metrics = [col for col in clustering_metrics if col in df.columns]
        
        if available_metrics:
            metrics_df = df[['model'] + available_metrics].round(4)
            
            # Replace any remaining None values with "N/A" for display
            metrics_df = metrics_df.fillna("N/A")
            
            # Only show rows where silhouette_score is not "N/A"
            metrics_df = metrics_df[metrics_df['silhouette_score'] != "N/A"]
            
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No valid metrics data to display.")
    else:
        st.info("No clustering models found for comparison.")

def display_clustering_dataset_analysis(clustering_files):
    """Display dataset-specific clustering analysis."""
    # First, collect all model data
    model_data = []
    for file_path in clustering_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            model_name = os.path.basename(file_path).replace('_clustering_metrics.json', '')
            
            if 'datasets' not in data:
                continue
                
            model_data.append({
                'name': model_name,
                'data': data,
                'file_path': file_path
            })
                
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    if not model_data:
        st.info("No dataset analysis data found.")
        return
    
    # Create tabs for each model
    model_names = [model['name'].upper() for model in model_data]
    model_tabs = st.tabs([f"🎯 {name}" for name in model_names])
    
    for idx, (model_tab, model_info) in enumerate(zip(model_tabs, model_data)):
        with model_tab:
            model_name = model_info['name']
            data = model_info['data']
            
            datasets = data['datasets']
            dataset_names = list(datasets.keys())
            
            # Create metrics comparison across datasets
            dataset_metrics = []
            for dataset_name, metrics in datasets.items():
                metrics_row = metrics.copy()
                metrics_row['dataset'] = dataset_name
                dataset_metrics.append(metrics_row)
            
            if dataset_metrics:
                df = pd.DataFrame(dataset_metrics)
                
                # Filter out rows with None silhouette scores
                df = df.dropna(subset=['silhouette_score'])
                
                if df.empty:
                    st.info(f"No valid metrics found for {model_name}.")
                    continue
                
                # Display metrics by dataset
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'silhouette_score' in df.columns:
                        fig = px.bar(df, x='dataset', y='silhouette_score',
                                    title=f'{model_name} - Silhouette Score by Dataset',
                                    color_discrete_sequence=['blue'])
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'sample_count' in df.columns:
                        fig = px.bar(df, x='dataset', y='sample_count',
                                    title=f'{model_name} - Sample Count by Dataset',
                                    color_discrete_sequence=['green'])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Dataset metrics table
                st.markdown(f"**📊 {model_name} - Detailed Dataset Metrics:**")
                metrics_cols = ['dataset', 'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 
                              'inertia', 'sample_count', 'num_clusters']
                available_cols = [col for col in metrics_cols if col in df.columns]
                
                if available_cols:
                    display_df = df[available_cols].round(4)
                    # Remove any remaining None values
                    display_df = display_df.dropna()
                    
                    if not display_df.empty:
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info(f"No valid data to display for {model_name}.")
                
                # Cross-dataset comparison insights
                if len(dataset_names) > 1:
                    st.markdown(f"**🔍 {model_name} - Cross-Dataset Insights:**")
                    
                    if 'silhouette_score' in df.columns:
                        best_dataset = df.loc[df['silhouette_score'].idxmax(), 'dataset']
                        best_score = df['silhouette_score'].max()
                        worst_dataset = df.loc[df['silhouette_score'].idxmin(), 'dataset']
                        worst_score = df['silhouette_score'].min()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best Performing Dataset", best_dataset, f"{best_score:.4f}")
                        with col2:
                            st.metric("Worst Performing Dataset", worst_dataset, f"{worst_score:.4f}")
                        with col3:
                            score_std = df['silhouette_score'].std()
                            st.metric("Score Consistency (Std)", f"{score_std:.4f}", 
                                    "Good" if score_std < 0.1 else "Variable")

def display_clustering_cross_validation(cv_files):
    """Display cross-validation results for clustering models."""
    # First, collect all CV data
    cv_data_list = []
    for file_path in cv_files:
        try:
            # Use cached JSON reading for better performance
            cv_data = cached_read_json(file_path)
            if not cv_data:  # Skip if no valid data
                continue
            
            model_name = os.path.basename(file_path).replace('_cross_validation_metrics.json', '')
            
            cv_data_list.append({
                'name': model_name,
                'data': cv_data,
                'file_path': file_path
            })
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    if not cv_data_list:
        st.info("No cross-validation data found.")
        return
    
    # Create tabs for each model
    model_names = [model['name'].upper() for model in cv_data_list]
    cv_tabs = st.tabs([f"🔄 {name}" for name in model_names])
    
    for idx, (cv_tab, cv_info) in enumerate(zip(cv_tabs, cv_data_list)):
        with cv_tab:
            model_name = cv_info['name']
            cv_data = cv_info['data']
            
            # Display CV summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'mean_silhouette_score' in cv_data:
                    st.metric("Mean Silhouette Score", 
                             f"{cv_data['mean_silhouette_score']:.4f}",
                             f"±{cv_data.get('std_silhouette_score', 0):.4f}")
            
            with col2:
                if 'cv_folds' in cv_data:
                    st.metric("CV Folds", cv_data['cv_folds'])
            
            with col3:
                if 'n_folds' in cv_data:
                    st.metric("Completed Folds", cv_data['n_folds'])
            
            with col4:
                if 'validation_type' in cv_data:
                    st.metric("Validation Type", cv_data['validation_type'])
            
            # Display fold-by-fold results if available
            if 'fold_results' in cv_data and cv_data['fold_results']:
                st.markdown("**📊 Fold-by-Fold Results:**")
                
                fold_df = pd.DataFrame(cv_data['fold_results'])
                
                # Fold performance chart
                if 'silhouette_score' in fold_df.columns:
                    fig = px.line(fold_df, x='fold', y='silhouette_score', 
                                 title=f'{model_name} - Silhouette Score by CV Fold',
                                 markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Fold details table
                display_cols = ['fold', 'silhouette_score', 'train_samples', 'val_samples']
                available_cols = [col for col in display_cols if col in fold_df.columns]
                
                if available_cols:
                    st.dataframe(fold_df[available_cols].round(4), use_container_width=True)

def display_clustering_summary_reports(summary_files):
    """Display clustering validation summary reports."""
    # First, collect all summary data
    summary_data_list = []
    for file_path in summary_files:
        try:
            model_name = os.path.basename(file_path).replace('_dataset_validation_summary.txt', '')
            
            with open(file_path, 'r') as f:
                summary_content = f.read()
            
            summary_data_list.append({
                'name': model_name,
                'content': summary_content,
                'file_path': file_path
            })
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
    
    if not summary_data_list:
        st.info("No summary reports found.")
        return
    
    # Create tabs for each model
    model_names = [model['name'].upper() for model in summary_data_list]
    summary_tabs = st.tabs([f"📋 {name}" for name in model_names])
    
    for idx, (summary_tab, summary_info) in enumerate(zip(summary_tabs, summary_data_list)):
        with summary_tab:
            model_name = summary_info['name']
            summary_content = summary_info['content']
            
            st.code(summary_content, language='text')
            
            # Add download button for the summary
            st.download_button(
                label=f"⬇️ Download {model_name} Summary",
                data=summary_content,
                file_name=f"{model_name}_validation_summary.txt",
                mime='text/plain',
                key=f"download_{model_name}_{idx}"  # Unique key to avoid conflicts
            )

def display_clustering_charts(df):
    """Display clustering-specific charts."""
    st.subheader("🔍 Clustering Performance Metrics")
    
    # Filter out rows with None values in key metrics
    df = df.dropna(subset=['silhouette_score'], how='any')
    
    if df.empty:
        st.info("No valid clustering metrics available for display.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Silhouette score comparison (higher is better)
        if 'silhouette_score' in df.columns:
            fig = px.bar(df, x='model', y='silhouette_score', 
                        title='Silhouette Score by Model (Higher is Better)',
                        color_discrete_sequence=['blue'])
            st.plotly_chart(fig, use_container_width=True, key="silhouette_chart")
    
    with col2:
        # Calinski-Harabasz Index (higher is better)
        if 'calinski_harabasz_score' in df.columns:
            fig = px.bar(df, x='model', y='calinski_harabasz_score', 
                        title='Calinski-Harabasz Index (Higher is Better)',
                        color_discrete_sequence=['green'])
            st.plotly_chart(fig, use_container_width=True, key="calinski_chart")
    
    with col3:
        # Davies-Bouldin Index (lower is better)
        if 'davies_bouldin_score' in df.columns:
            fig = px.bar(df, x='model', y='davies_bouldin_score', 
                        title='Davies-Bouldin Index (Lower is Better)',
                        color_discrete_sequence=['red'])
            st.plotly_chart(fig, use_container_width=True, key="davies_chart")
    
    # Number of clusters and additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        if 'num_clusters' in df.columns:
            fig = px.bar(df, x='model', y='num_clusters', title='Number of Clusters by Model',
                        color_discrete_sequence=['purple'])
            st.plotly_chart(fig, use_container_width=True, key="num_clusters_chart")
    
    with col2:
        if 'inertia' in df.columns:
            fig = px.bar(df, x='model', y='inertia', title='Inertia (WCSS) by Model (Lower is Better)',
                        color_discrete_sequence=['orange'])
            st.plotly_chart(fig, use_container_width=True, key="inertia_chart")
    
    # Comprehensive clustering metrics table
    if len(df) > 0:
        st.subheader("📊 Comprehensive Clustering Metrics")
        
        # Select relevant metrics for display
        clustering_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
                            'inertia', 'num_clusters', 'cluster_balance']
        available_metrics = [col for col in clustering_metrics if col in df.columns]
        
        if available_metrics:
            metrics_df = df[['model'] + available_metrics].round(4)
            
            # Remove any remaining None values
            metrics_df = metrics_df.dropna()
            
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No valid metrics data available after filtering.")
        
        # Cluster statistics if available
        cluster_stats = [col for col in df.columns if 'cluster_size' in col or 'inter_cluster' in col]
        if cluster_stats:
            st.subheader("📏 Cluster Statistics")
            stats_df = df[['model'] + cluster_stats].round(4)
            st.dataframe(stats_df, use_container_width=True)

def display_validation_plots(output_dir: str, task_type: str):
    """Display validation plots based on task type with improved organization."""
    st.header(f"📊 Validation Plots - {task_type.title()}")
    
    # Look for validation plots directory
    plots_dir = os.path.join(output_dir, 'validation_plots')
    
    if not os.path.exists(plots_dir):
        st.info("No validation plots directory found. Plots may not have been generated yet.")
        return
    
    # Find plot files with error handling
    try:
        plot_files = glob.glob(os.path.join(plots_dir, '*.png'))
        # Filter out any invalid files
        plot_files = [f for f in plot_files if os.path.isfile(f) and os.path.exists(f)]
    except Exception as e:
        st.error(f"Error accessing validation plots directory: {str(e)}")
        return
    
    if not plot_files:
        st.info("No validation plots found. They may not have been generated yet.")
        return
    
    # Show plot count and download option
    st.success(f"🎉 Found {len(plot_files)} validation plots!")
    
    # Add download all plots option
    if st.button("📥 Download All Plots", help="Download all validation plots as a zip file"):
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for plot_file in plot_files:
                zip_file.write(plot_file, os.path.basename(plot_file))
        
        st.download_button(
            label="⬇️ Download Plots ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{task_type}_validation_plots.zip",
            mime='application/zip'
        )
    
    # Categorize plots by type and model
    plot_categories = {}
    
    for plot_file in plot_files:
        filename = os.path.basename(plot_file)
        
        # Extract model type and plot type from filename
        if task_type == 'regression':
            if 'actual_vs_predicted' in filename:
                category = "📈 Actual vs Predicted"
            elif 'residuals_distribution' in filename:
                category = "📊 Residuals Distribution"
            elif 'residuals.png' in filename:
                category = "🔍 Residuals Plot"
            elif 'qq_plot' in filename:
                category = "📏 Q-Q Plot"
            elif 'error_analysis' in filename:
                category = "⚠️ Error Analysis"
            else:
                category = "📋 Other Plots"
                
        elif task_type == 'clustering':
            # Extract model name from filename first, then categorize by plot type
            model_name = None
            plot_type = None
            
            # Extract model name (preserving default/tuned suffixes)
            parts = filename.split('_')
            for i in range(len(parts) - 2):
                potential_model = '_'.join(parts[:i+1])
                if any(model_type in potential_model.lower() for model_type in [
                    'kmeans', 'bisecting_kmeans', 'gaussian_mixture', 'dbscan'
                ]):
                    # Check if the next part is 'default' or 'tuned'
                    if i + 1 < len(parts) and parts[i + 1] in ['default', 'tuned']:
                        model_name = potential_model + '_' + parts[i + 1]
                    else:
                        model_name = potential_model
                    break
            
            # Determine plot type
            if 'train_clustering_scatter' in filename:
                plot_type = 'scatter_train'
            elif 'validation_clustering_scatter' in filename:
                plot_type = 'scatter_validation'
            elif 'test_clustering_scatter' in filename:
                plot_type = 'scatter_test'
            elif 'oot1_clustering_scatter' in filename:
                plot_type = 'scatter_oot1'
            elif 'oot2_clustering_scatter' in filename:
                plot_type = 'scatter_oot2'
            elif 'train_distribution' in filename:
                plot_type = 'distribution_train'
            elif 'validation_distribution' in filename:
                plot_type = 'distribution_validation'
            elif 'test_distribution' in filename:
                plot_type = 'distribution_test'
            elif 'oot1_distribution' in filename:
                plot_type = 'distribution_oot1'
            elif 'oot2_distribution' in filename:
                plot_type = 'distribution_oot2'
            elif 'elbow_plot' in filename or 'elbow' in filename:
                plot_type = 'elbow'
            elif 'silhouette' in filename:
                plot_type = 'silhouette'
            elif 'pca' in filename.lower():
                plot_type = 'pca'
            elif 'tsne' in filename.lower():
                plot_type = 'tsne'
            elif 'cluster_center' in filename or 'centers' in filename:
                plot_type = 'cluster_centers'
            elif 'inertia' in filename:
                plot_type = 'inertia'
            elif 'calinski' in filename:
                plot_type = 'calinski'
            elif 'davies' in filename:
                plot_type = 'davies'
            else:
                plot_type = 'other'
            
            # Create category based on model and plot type
            if model_name:
                if plot_type.startswith('scatter_'):
                    category = f"🎯 {model_name} - Scatter Plots"
                elif plot_type.startswith('distribution_'):
                    category = f"📊 {model_name} - Distribution Plots"
                elif plot_type == 'elbow':
                    category = f"📐 {model_name} - Elbow Plot"
                elif plot_type == 'silhouette':
                    category = f"📊 {model_name} - Silhouette Analysis"
                elif plot_type == 'pca':
                    category = f"🎯 {model_name} - PCA Visualization"
                elif plot_type == 'tsne':
                    category = f"🎯 {model_name} - t-SNE Visualization"
                elif plot_type == 'cluster_centers':
                    category = f"🔥 {model_name} - Cluster Centers"
                elif plot_type == 'inertia':
                    category = f"📉 {model_name} - Inertia Analysis"
                elif plot_type == 'calinski':
                    category = f"📈 {model_name} - Calinski-Harabasz Analysis"
                elif plot_type == 'davies':
                    category = f"📊 {model_name} - Davies-Bouldin Analysis"
                else:
                    category = f"📋 {model_name} - Other Plots"
            else:
                # Fallback to original categorization if no model name found
                if 'train_clustering_scatter' in filename:
                    category = "🎯 Training Dataset Plots"
                elif 'validation_clustering_scatter' in filename:
                    category = "✅ Validation Dataset Plots"  
                elif 'test_clustering_scatter' in filename:
                    category = "🧪 Test Dataset Plots"
                elif 'oot1_clustering_scatter' in filename:
                    category = "📅 OOT1 Dataset Plots"
                elif 'oot2_clustering_scatter' in filename:
                    category = "📅 OOT2 Dataset Plots"
                elif 'train_distribution' in filename:
                    category = "📊 Training Distribution Plots"
                elif 'validation_distribution' in filename:
                    category = "📊 Validation Distribution Plots"
                elif 'test_distribution' in filename:
                    category = "📊 Test Distribution Plots"
                elif 'oot1_distribution' in filename:
                    category = "📊 OOT1 Distribution Plots"
                elif 'oot2_distribution' in filename:
                    category = "📊 OOT2 Distribution Plots"
                elif 'elbow_plot' in filename or 'elbow' in filename:
                    category = "📐 Elbow Plot"
                elif 'silhouette' in filename:
                    category = "📊 Silhouette Analysis"
                elif 'pca' in filename.lower():
                    category = "🎯 PCA Visualization"
                elif 'tsne' in filename.lower():
                    category = "🎯 t-SNE Visualization"
                elif 'cluster_center' in filename or 'centers' in filename:
                    category = "🔥 Cluster Centers"
                elif 'inertia' in filename:
                    category = "📉 Inertia Analysis"
                elif 'calinski' in filename:
                    category = "📈 Calinski-Harabasz Analysis"
                elif 'davies' in filename:
                    category = "📊 Davies-Bouldin Analysis"
                else:
                    category = "📋 Other Plots"
                
        else:  # classification
            if 'confusion_matrix' in filename:
                category = "📊 Confusion Matrix"
            elif 'roc_curve' in filename:
                category = "📈 ROC Curve"
            elif 'precision_recall' in filename:
                category = "⚖️ Precision-Recall Curve"
            elif 'feature_importance' in filename:
                category = "📊 Feature Importance"
            else:
                category = "📋 Other Plots"
        
        if category not in plot_categories:
            plot_categories[category] = []
        plot_categories[category].append(plot_file)
    
    # Sort plots within each category by dataset type for clustering
    if task_type == 'clustering':
        def sort_plots_by_dataset(plots):
            """Sort clustering plots by dataset type: train, valid, test, oot1, oot2."""
            dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
            
            def get_dataset_priority(plot_file):
                filename = os.path.basename(plot_file).lower()
                for i, dataset in enumerate(dataset_order):
                    if dataset in filename:
                        return i
                return len(dataset_order)  # Put unknown datasets at the end
            
            return sorted(plots, key=get_dataset_priority)
        
        # Sort plots in each category
        for category in plot_categories:
            plot_categories[category] = sort_plots_by_dataset(plot_categories[category])
    
    # Create tabs for different plot categories
    if len(plot_categories) > 1:
        category_names = list(plot_categories.keys())
        plot_tabs = st.tabs(category_names)
        
        for i, (category, plots) in enumerate(plot_categories.items()):
            with plot_tabs[i]:
                st.subheader(f"{category} ({len(plots)} plots)")
                
                # Display plots in a grid with better layout and error handling
                def safe_display_image(plot_file, caption):
                    """Safely display an image with error handling."""
                    try:
                        if os.path.exists(plot_file):
                            st.image(plot_file, caption=caption, use_container_width=True)
                        else:
                            st.error(f"Image file not found: {os.path.basename(plot_file)}")
                    except Exception as e:
                        st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
                
                if len(plots) == 1:
                    safe_display_image(plots[0], os.path.basename(plots[0]))
                elif len(plots) == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        safe_display_image(plots[0], os.path.basename(plots[0]))
                    with col2:
                        safe_display_image(plots[1], os.path.basename(plots[1]))
                elif len(plots) == 3:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        safe_display_image(plots[0], os.path.basename(plots[0]))
                    with col2:
                        safe_display_image(plots[1], os.path.basename(plots[1]))
                    with col3:
                        safe_display_image(plots[2], os.path.basename(plots[2]))
                elif len(plots) == 4:
                    col1, col2 = st.columns(2)
                    with col1:
                        safe_display_image(plots[0], os.path.basename(plots[0]))
                        safe_display_image(plots[2], os.path.basename(plots[2]))
                    with col2:
                        safe_display_image(plots[1], os.path.basename(plots[1]))
                        safe_display_image(plots[3], os.path.basename(plots[3]))
                elif len(plots) == 5:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        safe_display_image(plots[0], os.path.basename(plots[0]))
                        safe_display_image(plots[3], os.path.basename(plots[3]))
                    with col2:
                        safe_display_image(plots[1], os.path.basename(plots[1]))
                        safe_display_image(plots[4], os.path.basename(plots[4]))
                    with col3:
                        safe_display_image(plots[2], os.path.basename(plots[2]))
                else:
                    # For 6+ plots, use a more flexible layout
                    cols = st.columns(3)
                    for j, plot_file in enumerate(plots):
                        with cols[j % 3]:
                            safe_display_image(plot_file, os.path.basename(plot_file))
    else:
        # Single category - display directly
        category, plots = list(plot_categories.items())[0]
        st.subheader(f"{category} ({len(plots)} plots)")
        
        # Display plots in a grid with better layout and error handling
        def safe_display_image(plot_file, caption):
            """Safely display an image with error handling."""
            try:
                if os.path.exists(plot_file):
                    st.image(plot_file, caption=caption, use_container_width=True)
                else:
                    st.error(f"Image file not found: {os.path.basename(plot_file)}")
            except Exception as e:
                st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
        
        if len(plots) == 1:
            safe_display_image(plots[0], os.path.basename(plots[0]))
        elif len(plots) == 2:
            col1, col2 = st.columns(2)
            with col1:
                safe_display_image(plots[0], os.path.basename(plots[0]))
            with col2:
                safe_display_image(plots[1], os.path.basename(plots[1]))
        elif len(plots) == 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                safe_display_image(plots[0], os.path.basename(plots[0]))
            with col2:
                safe_display_image(plots[1], os.path.basename(plots[1]))
            with col3:
                safe_display_image(plots[2], os.path.basename(plots[2]))
        elif len(plots) == 4:
            col1, col2 = st.columns(2)
            with col1:
                safe_display_image(plots[0], os.path.basename(plots[0]))
                safe_display_image(plots[2], os.path.basename(plots[2]))
            with col2:
                safe_display_image(plots[1], os.path.basename(plots[1]))
                safe_display_image(plots[3], os.path.basename(plots[3]))
        elif len(plots) == 5:
            col1, col2, col3 = st.columns(3)
            with col1:
                safe_display_image(plots[0], os.path.basename(plots[0]))
                safe_display_image(plots[3], os.path.basename(plots[3]))
            with col2:
                safe_display_image(plots[1], os.path.basename(plots[1]))
                safe_display_image(plots[4], os.path.basename(plots[4]))
            with col3:
                safe_display_image(plots[2], os.path.basename(plots[2]))
        else:
            # For 6+ plots, use a more flexible layout
            cols = st.columns(3)
            for i, plot_file in enumerate(plots):
                with cols[i % 3]:
                    safe_display_image(plot_file, os.path.basename(plot_file))
    
    # Special handling for elbow plot in clustering
    if task_type == 'clustering':
        elbow_json = os.path.join(output_dir, '*_elbow_analysis.json')
        elbow_files = glob.glob(elbow_json)
        
        if elbow_files:
            st.subheader("🔍 Elbow Analysis Results")
            try:
                with open(elbow_files[0], 'r') as f:
                    elbow_data = json.load(f)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Clustering Metrics by K:**")
                    if 'k_range' in elbow_data:
                        metrics_df = pd.DataFrame({
                            'K': elbow_data['k_range'],
                            'Inertia (WCSS)': elbow_data.get('inertias', []),
                            'Silhouette Score': elbow_data.get('silhouette_scores', []),
                            'Calinski-Harabasz': elbow_data.get('calinski_scores', []),
                            'Davies-Bouldin': elbow_data.get('davies_bouldin_scores', [])
                        })
                        st.dataframe(metrics_df.round(4))
                
                with col2:
                    st.markdown("**🎯 Recommendation:**")
                    if 'recommendation' in elbow_data:
                        st.text(elbow_data['recommendation'])
                    else:
                        st.info("No specific recommendation available.")
                        
            except Exception as e:
                st.error(f"Could not load elbow analysis: {str(e)}")

def display_clustering_scatter_plots(output_dir: str):
    """Display clustering scatter plots for each model using tab format."""
    try:
        # Use cached plot file discovery for better performance
        all_plot_files = get_cached_file_discovery(output_dir, 'plots', 'clustering')
        
        # Filter for clustering scatter plot files
        scatter_plot_files = [file for file in all_plot_files if 'clustering_scatter' in os.path.basename(file).lower()]
        
        if not scatter_plot_files:
            return  # No scatter plot files found
        
        # Group files by model type
        model_plot_groups = {}
        for plot_file in scatter_plot_files:
            filename = os.path.basename(plot_file)
            
            # Extract model name from filename (preserving default/tuned suffixes)
            parts = filename.split('_')
            model_name = None
            for i in range(len(parts) - 2):
                potential_model = '_'.join(parts[:i+1])
                if any(model_type in potential_model.lower() for model_type in [
                    'kmeans', 'bisecting_kmeans', 'gaussian_mixture', 'dbscan'
                ]):
                    # Check if the next part is 'default' or 'tuned'
                    if i + 1 < len(parts) and parts[i + 1] in ['default', 'tuned']:
                        model_name = potential_model + '_' + parts[i + 1]
                    else:
                        model_name = potential_model
                    break
            
            if model_name:
                if model_name not in model_plot_groups:
                    model_plot_groups[model_name] = []
                model_plot_groups[model_name].append(plot_file)
        
        if not model_plot_groups:
            return  # No valid model groups found
        
        # Define the desired order for model types
        model_order = [
            'kmeans', 'kmeans_default', 'kmeans_tuned',
            'bisecting_kmeans', 'bisecting_kmeans_default', 'bisecting_kmeans_tuned',
            'gaussian_mixture', 'gaussian_mixture_default', 'gaussian_mixture_tuned',
            'dbscan', 'dbscan_default', 'dbscan_tuned'
        ]
        
        st.subheader("🎯 Clustering Scatter Plots")
        st.markdown("Scatter plot visualizations for clustering models")
        
        # Sort model groups according to the desired order
        sorted_model_groups = []
        used_models = set()
        
        for model_type in model_order:
            for model_name in model_plot_groups.keys():
                if model_type.lower() in model_name.lower() and model_name not in used_models:
                    sorted_model_groups.append((model_name, model_plot_groups[model_name]))
                    used_models.add(model_name)
        
        # Add any remaining models that don't match the pattern
        for model_name, plot_files in model_plot_groups.items():
            if model_name not in used_models:
                sorted_model_groups.append((model_name, plot_files))
                used_models.add(model_name)
        
        # Create tabs for each model's scatter plots
        if len(sorted_model_groups) == 1:
            # Single model - display directly
            display_single_model_clustering_scatter_plots(sorted_model_groups[0][0], sorted_model_groups[0][1])
        else:
            # Multiple models - create tabs
            model_names = [model_name for model_name, _ in sorted_model_groups]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, plot_files) in zip(tabs, sorted_model_groups):
                with tab:
                    display_single_model_clustering_scatter_plots(model_name, plot_files)
        
    except Exception as e:
        st.error(f"Error displaying clustering scatter plots: {str(e)}")

def display_single_model_clustering_scatter_plots(model_name: str, plot_files: list):
    """Display scatter plots for a single clustering model organized by dataset."""
    try:
        if not plot_files:
            st.info(f"No scatter plot images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in plot_files:
                if dataset in os.path.basename(file).lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                st.markdown(f"**{dataset_name}**")
                
                # Display the image
                try:
                    if os.path.exists(plot_file):
                        st.image(plot_file, use_container_width=True)
                    else:
                        st.error(f"Image file not found: {os.path.basename(plot_file)}")
                except Exception as e:
                    st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying scatter plots for {model_name}: {str(e)}")

def display_clustering_distribution_plots(output_dir: str):
    """Display clustering distribution plots for each model using tab format."""
    try:
        # Use cached plot file discovery for better performance
        all_plot_files = get_cached_file_discovery(output_dir, 'plots', 'clustering')
        
        # Filter for clustering distribution plot files
        distribution_plot_files = [file for file in all_plot_files if 'distribution' in os.path.basename(file).lower()]
        
        if not distribution_plot_files:
            return  # No distribution plot files found
        
        # Group files by model type
        model_plot_groups = {}
        for plot_file in distribution_plot_files:
            filename = os.path.basename(plot_file)
            
            # Extract model name from filename (preserving default/tuned suffixes)
            parts = filename.split('_')
            model_name = None
            for i in range(len(parts) - 2):
                potential_model = '_'.join(parts[:i+1])
                if any(model_type in potential_model.lower() for model_type in [
                    'kmeans', 'bisecting_kmeans', 'gaussian_mixture', 'dbscan'
                ]):
                    # Check if the next part is 'default' or 'tuned'
                    if i + 1 < len(parts) and parts[i + 1] in ['default', 'tuned']:
                        model_name = potential_model + '_' + parts[i + 1]
                    else:
                        model_name = potential_model
                    break
            
            if model_name:
                if model_name not in model_plot_groups:
                    model_plot_groups[model_name] = []
                model_plot_groups[model_name].append(plot_file)
        
        if not model_plot_groups:
            return  # No valid model groups found
        
        # Define the desired order for model types
        model_order = [
            'kmeans', 'kmeans_default', 'kmeans_tuned',
            'bisecting_kmeans', 'bisecting_kmeans_default', 'bisecting_kmeans_tuned',
            'gaussian_mixture', 'gaussian_mixture_default', 'gaussian_mixture_tuned',
            'dbscan', 'dbscan_default', 'dbscan_tuned'
        ]
        
        st.subheader("📊 Clustering Distribution Plots")
        st.markdown("Distribution plot visualizations for clustering models")
        
        # Sort model groups according to the desired order
        sorted_model_groups = []
        used_models = set()
        
        for model_type in model_order:
            for model_name in model_plot_groups.keys():
                if model_type.lower() in model_name.lower() and model_name not in used_models:
                    sorted_model_groups.append((model_name, model_plot_groups[model_name]))
                    used_models.add(model_name)
        
        # Add any remaining models that don't match the pattern
        for model_name, plot_files in model_plot_groups.items():
            if model_name not in used_models:
                sorted_model_groups.append((model_name, plot_files))
                used_models.add(model_name)
        
        # Create tabs for each model's distribution plots
        if len(sorted_model_groups) == 1:
            # Single model - display directly
            display_single_model_clustering_distribution_plots(sorted_model_groups[0][0], sorted_model_groups[0][1])
        else:
            # Multiple models - create tabs
            model_names = [model_name for model_name, _ in sorted_model_groups]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, plot_files) in zip(tabs, sorted_model_groups):
                with tab:
                    display_single_model_clustering_distribution_plots(model_name, plot_files)
        
    except Exception as e:
        st.error(f"Error displaying clustering distribution plots: {str(e)}")

def display_single_model_clustering_distribution_plots(model_name: str, plot_files: list):
    """Display distribution plots for a single clustering model organized by dataset."""
    try:
        if not plot_files:
            st.info(f"No distribution plot images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in plot_files:
                if dataset in os.path.basename(file).lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                st.markdown(f"**{dataset_name}**")
                
                # Display the image
                try:
                    if os.path.exists(plot_file):
                        st.image(plot_file, use_container_width=True)
                    else:
                        st.error(f"Image file not found: {os.path.basename(plot_file)}")
                except Exception as e:
                    st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying distribution plots for {model_name}: {str(e)}")

def display_clustering_elbow_plots(output_dir: str):
    """Display clustering elbow plots for each model using tab format."""
    try:
        # Use cached plot file discovery for better performance
        all_plot_files = get_cached_file_discovery(output_dir, 'plots', 'clustering')
        
        # Filter for elbow plot files
        elbow_plot_files = [file for file in all_plot_files if 'elbow' in os.path.basename(file).lower()]
        
        if not elbow_plot_files:
            return  # No elbow plot files found
        
        # Group files by model type
        model_plot_groups = {}
        for plot_file in elbow_plot_files:
            filename = os.path.basename(plot_file)
            
            # Extract model name from filename
            parts = filename.split('_')
            model_name = None
            for i in range(len(parts) - 2):
                potential_model = '_'.join(parts[:i+1])
                if any(model_type in potential_model.lower() for model_type in [
                    'kmeans', 'bisecting_kmeans', 'gaussian_mixture', 'dbscan'
                ]):
                    model_name = potential_model
                    break
            
            if model_name:
                if model_name not in model_plot_groups:
                    model_plot_groups[model_name] = []
                model_plot_groups[model_name].append(plot_file)
        
        if not model_plot_groups:
            return  # No valid model groups found
        
        # Define the desired order for model types
        model_order = [
            'kmeans', 'kmeans_default', 'kmeans_tuned',
            'bisecting_kmeans', 'bisecting_kmeans_default', 'bisecting_kmeans_tuned',
            'gaussian_mixture', 'gaussian_mixture_default', 'gaussian_mixture_tuned',
            'dbscan', 'dbscan_default', 'dbscan_tuned'
        ]
        
        st.subheader("📊 Clustering Elbow Plots")
        st.markdown("Elbow method plots for optimal cluster number selection")
        
        # Sort model groups according to the desired order
        sorted_model_groups = []
        used_models = set()
        
        for model_type in model_order:
            for model_name in model_plot_groups.keys():
                if model_type.lower() in model_name.lower() and model_name not in used_models:
                    sorted_model_groups.append((model_name, model_plot_groups[model_name]))
                    used_models.add(model_name)
        
        # Add any remaining models that don't match the pattern
        for model_name, plot_files in model_plot_groups.items():
            if model_name not in used_models:
                sorted_model_groups.append((model_name, plot_files))
                used_models.add(model_name)
        
        # Create tabs for each model's elbow plots
        if len(sorted_model_groups) == 1:
            # Single model - display directly
            display_single_model_clustering_elbow_plots(sorted_model_groups[0][0], sorted_model_groups[0][1])
        else:
            # Multiple models - create tabs
            model_names = [model_name for model_name, _ in sorted_model_groups]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, plot_files) in zip(tabs, sorted_model_groups):
                with tab:
                    display_single_model_clustering_elbow_plots(model_name, plot_files)
        
    except Exception as e:
        st.error(f"Error displaying clustering elbow plots: {str(e)}")

def display_single_model_clustering_elbow_plots(model_name: str, plot_files: list):
    """Display elbow plots for a single clustering model."""
    try:
        if not plot_files:
            st.info(f"No elbow plot images found for {model_name}")
            return
        
        # Remove duplicates while preserving order
        unique_plot_files = []
        seen_files = set()
        for plot_file in plot_files:
            if plot_file not in seen_files:
                unique_plot_files.append(plot_file)
                seen_files.add(plot_file)
        
        # Display each unique elbow plot
        for i, plot_file in enumerate(unique_plot_files):
            try:
                if os.path.exists(plot_file):
                    # Create a descriptive caption
                    filename = os.path.basename(plot_file)
                    caption = f"{model_name.replace('_', ' ').title()} - Elbow Analysis"
                    
                    # Display the image
                    st.image(plot_file, caption=caption, use_container_width=True)
                    
                    # Try to find and display the corresponding JSON analysis file
                    json_file = plot_file.replace('_plot_efficient.png', '_analysis_efficient.json')
                    if os.path.exists(json_file):
                        try:
                            analysis_data = cached_read_json(json_file)
                            if analysis_data and 'recommended_k' in analysis_data:
                                # Display additional metrics if available
                                if 'k_range' in analysis_data and 'silhouette_scores' in analysis_data:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("K Range", f"{min(analysis_data['k_range'])} - {max(analysis_data['k_range'])}")
                                    with col2:
                                        best_silhouette = max(analysis_data['silhouette_scores'])
                                        best_k = analysis_data['k_range'][analysis_data['silhouette_scores'].index(best_silhouette)]
                                        st.metric("Best Silhouette Score", f"{best_silhouette:.3f}", f"at k={best_k}")
                                    with col3:
                                        # Display the selection method instead of technical method
                                        method = analysis_data.get('method', 'Elbow Method')
                                        if 'precomputed_efficient' in method:
                                            # Determine if it's elbow or silhouette based on the analysis
                                            if 'silhouette' in method or 'silhouette_scores' in analysis_data:
                                                display_method = "Silhouette Method"
                                            else:
                                                display_method = "Elbow Method"
                                        else:
                                            display_method = method
                                        st.metric("Selection Method", display_method)
                        except Exception as e:
                            st.warning(f"Could not load elbow analysis data: {str(e)}")
                    
                    # Add download button with unique key including index
                    st.download_button(
                        label=f"📥 Download {model_name} Elbow Plot",
                        data=open(plot_file, 'rb').read(),
                        file_name=filename,
                        mime="image/png",
                        key=f"download_elbow_{model_name}_{i}_{filename}"  # Include index for uniqueness
                    )
                else:
                    st.error(f"Image file not found: {os.path.basename(plot_file)}")
            except Exception as e:
                st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying elbow plots for {model_name}: {str(e)}")

def display_cluster_profiling(output_dir: str):
    """Display cluster profiling results including numeric and categorical variable analysis."""
    st.header("🔬 Cluster Profiling")
    
    # Look for profiling files
    numeric_profiles = os.path.join(output_dir, 'cluster_numeric_profiles.csv')
    categorical_profiles = os.path.join(output_dir, 'cluster_categorical_profiles.csv')
    profiling_summary = os.path.join(output_dir, 'cluster_profiling_summary.json')
    profiling_report = os.path.join(output_dir, 'cluster_profiling_report.txt')
    
    if not any(os.path.exists(f) for f in [numeric_profiles, categorical_profiles, profiling_summary]):
        st.info("📊 No cluster profiling results found. Profiling may not have been generated yet.")
        return
    
    # Display profiling summary if available
    if os.path.exists(profiling_summary):
        try:
            with open(profiling_summary, 'r') as f:
                summary = json.load(f)
            
            st.subheader("📊 Cluster Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'cluster_sizes' in summary:
                    st.markdown("**Cluster Distribution:**")
                    for cluster, size in summary['cluster_sizes'].items():
                        percentage = summary.get('cluster_percentages', {}).get(cluster, 0)
                        st.write(f"• Cluster {cluster}: {size:,} samples ({percentage:.1f}%)")
            
            with col2:
                if 'total_samples' in summary:
                    st.metric("Total Samples", f"{summary['total_samples']:,}")
                if 'num_clusters' in summary:
                    st.metric("Number of Clusters", summary['num_clusters'])
                    
        except Exception as e:
            st.error(f"Could not load profiling summary: {str(e)}")
    
    # Create tabs for different profiling aspects
    tabs = []
    tab_content = []
    
    if os.path.exists(numeric_profiles):
        tabs.append("📈 Numeric Variables")
        tab_content.append(('numeric', numeric_profiles))
    
    if os.path.exists(categorical_profiles):
        tabs.append("🏷️ Categorical Variables")  
        tab_content.append(('categorical', categorical_profiles))
        
    if os.path.exists(profiling_report):
        tabs.append("📋 Full Report")
        tab_content.append(('report', profiling_report))
    
    if tabs:
        profile_tabs = st.tabs(tabs)
        
        for i, (tab_type, file_path) in enumerate(tab_content):
            with profile_tabs[i]:
                if tab_type == 'numeric':
                    st.subheader("📈 Numeric Variable Profiles")
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Add explanation about pct_diff_from_overall
                        with st.expander("ℹ️ Understanding the Results", expanded=True):
                            st.markdown("""
                            **What do these metrics mean?**
                            
                            - **mean**: Average value of the variable within each cluster
                            - **std**: Standard deviation (measure of spread/variability)
                            - **median**: Middle value when data is sorted
                            - **min/max**: Lowest and highest values in the cluster
                            - **count**: Number of samples in each cluster
                            
                            **🔍 pct_diff_from_overall**: This shows how much each cluster differs from the overall average!
                            - **Positive values** (e.g., +15.2%): Cluster average is **higher** than overall average
                            - **Negative values** (e.g., -12.8%): Cluster average is **lower** than overall average
                            - **Larger absolute values** indicate more distinctive clusters
                            
                            **💡 How to interpret:**
                            - Look for variables with large percentage differences - these are key distinguishing features
                            - Compare clusters to understand what makes each unique
                            """)
                        
                        # Show key insights with better formatting
                        st.markdown("### 🎯 Key Insights")
                        
                        # Find most distinctive variables (highest absolute pct_diff)
                        if 'pct_diff_from_overall' in df.columns:
                            df['abs_pct_diff'] = df['pct_diff_from_overall'].abs()
                            most_distinctive = df.nlargest(3, 'abs_pct_diff')
                            
                            col1, col2, col3 = st.columns(3)
                            for i, (_, row) in enumerate(most_distinctive.iterrows()):
                                with [col1, col2, col3][i]:
                                    pct_diff = row['pct_diff_from_overall']
                                    direction = "🔺 Higher" if pct_diff > 0 else "🔻 Lower"
                                    st.metric(
                                        label=f"Most Distinctive: {row['variable']}",
                                        value=f"{pct_diff:.1f}%",
                                        delta=direction,
                                        help=f"Cluster {row['cluster']} has {abs(pct_diff):.1f}% {'higher' if pct_diff > 0 else 'lower'} {row['variable']} than overall average"
                                    )
                        
                        # Show cluster comparison by variable
                        st.markdown("### 📊 Cluster Comparison by Variable")
                        
                        # Get unique variables
                        variables = df['variable'].unique()
                        
                        for var in variables:
                            with st.expander(f"📈 {var} Analysis", expanded=False):
                                var_data = df[df['variable'] == var]
                                
                                # Create comparison chart
                                fig = px.bar(
                                    var_data, 
                                    x='cluster', 
                                    y='mean',
                                    color='cluster',
                                    title=f'{var} - Mean Values by Cluster',
                                    labels={'mean': f'{var} Mean', 'cluster': 'Cluster'},
                                    text=var_data['mean'].round(3)
                                )
                                fig.update_traces(textposition='outside')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show percentage differences
                                if 'pct_diff_from_overall' in var_data.columns:
                                    fig2 = px.bar(
                                        var_data,
                                        x='cluster',
                                        y='pct_diff_from_overall',
                                        color='pct_diff_from_overall',
                                        title=f'{var} - Percentage Difference from Overall Average',
                                        labels={'pct_diff_from_overall': 'Percentage Difference (%)', 'cluster': 'Cluster'},
                                        text=var_data['pct_diff_from_overall'].round(1).astype(str) + '%',
                                        color_continuous_scale='RdYlBu'
                                    )
                                    fig2.update_traces(textposition='outside')
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                # Show detailed stats
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**📋 Detailed Statistics:**")
                                    display_df = var_data[['cluster', 'mean', 'std', 'median', 'min', 'max', 'count']].copy()
                                    display_df.columns = ['Cluster', 'Mean', 'Std Dev', 'Median', 'Min', 'Max', 'Count']
                                    st.dataframe(display_df.round(3), use_container_width=True)
                                
                                with col2:
                                    st.markdown("**📊 Percentage Differences:**")
                                    if 'pct_diff_from_overall' in var_data.columns:
                                        pct_df = var_data[['cluster', 'pct_diff_from_overall']].copy()
                                        pct_df.columns = ['Cluster', 'Diff from Overall (%)']
                                        pct_df['Diff from Overall (%)'] = pct_df['Diff from Overall (%)'].round(1)
                                        st.dataframe(pct_df, use_container_width=True)
                        
                        # Show complete table in a collapsible section
                        with st.expander("📋 Complete Numeric Profiles Table", expanded=False):
                            st.markdown("**Full dataset with all metrics:**")
                            st.dataframe(df.round(3), use_container_width=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Numeric Profiles",
                            data=df.to_csv(index=False),
                            file_name="cluster_numeric_profiles.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Could not load numeric profiles: {str(e)}")
                
                elif tab_type == 'categorical':
                    st.subheader("🏷️ Categorical Variable Profiles")
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Add explanation about categorical profiling
                        with st.expander("ℹ️ Understanding Categorical Results", expanded=True):
                            # Check which columns are available to customize the explanation
                            available_cols = df.columns.tolist()
                            
                            # Build explanation with proper markdown formatting
                            explanation_parts = []
                            
                            explanation_parts.append("**What do these metrics mean?**")
                            explanation_parts.append("")
                            explanation_parts.append("- **category**: The specific value/category of the variable")
                            explanation_parts.append("- **cluster**: Which cluster this category belongs to")
                            explanation_parts.append("- **count**: Number of samples with this category in this cluster")
                            explanation_parts.append("- **percentage**: Percentage of this category within the cluster")
                            
                            if 'overall_percentage' in available_cols:
                                explanation_parts.append("- **overall_percentage**: Percentage of this category across all data")
                            
                            if 'ratio' in available_cols:
                                explanation_parts.append("- **ratio**: How over/under-represented this category is in the cluster")
                            
                            explanation_parts.append("")
                            explanation_parts.append("**🔍 Key Insights to Look For:**")
                            
                            if 'ratio' in available_cols:
                                explanation_parts.append("- **High ratios (>1.5)**: Category is over-represented in this cluster")
                                explanation_parts.append("- **Low ratios (<0.7)**: Category is under-represented in this cluster")
                            
                            explanation_parts.append("- **Large percentage differences**: Strong distinguishing features")
                            explanation_parts.append("- **Cluster-specific patterns**: What makes each cluster unique")
                            
                            explanation_parts.append("")
                            explanation_parts.append("**💡 How to interpret:**")
                            explanation_parts.append("- Look for categories that are much more common in one cluster vs others")
                            
                            if 'ratio' in available_cols:
                                explanation_parts.append("- High ratios indicate key characteristics of each cluster")
                            
                            explanation_parts.append("- Compare percentages to understand cluster composition")
                            
                            # Join with proper line breaks
                            explanation = "\n".join(explanation_parts)
                            st.markdown(explanation)
                        
                        # Show cluster comparison by variable
                        st.markdown("### 📊 Cluster Comparison by Variable")
                        
                        # Group by variable for better display
                        variables = df['variable'].unique()
                        
                        for var in variables:
                            with st.expander(f"📊 {var} Analysis", expanded=False):
                                var_data = df[df['variable'] == var]
                                
                                # Create pivot table for better visualization
                                pivot_data = var_data.pivot_table(
                                    index='category', 
                                    columns='cluster', 
                                    values='percentage', 
                                    fill_value=0
                                )
                                
                                # Show ratio analysis if available
                                if 'ratio' in var_data.columns:
                                    ratio_pivot = var_data.pivot_table(
                                        index='category', 
                                        columns='cluster', 
                                        values='ratio', 
                                        fill_value=1
                                    )
                                
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Show as bar chart
                                    if len(pivot_data) > 0:
                                        fig = px.bar(
                                            pivot_data.reset_index().melt(id_vars='category'), 
                                            x='category', 
                                            y='value', 
                                            color='cluster',
                                            title=f'{var} Distribution by Cluster',
                                            labels={'value': 'Percentage (%)', 'category': var}
                                        )
                                        fig.update_layout(xaxis_tickangle=-45)
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Show data table with better formatting
                                    st.markdown("**📋 Percentage Distribution:**")
                                    display_pivot = pivot_data.round(1)
                                    st.dataframe(display_pivot, use_container_width=True)
                                
                                # Show ratio analysis if available
                                if 'ratio' in var_data.columns:
                                    st.markdown("**🔍 Representation Ratios:**")
                                    st.markdown("*Values >1 indicate over-representation, <1 indicate under-representation*")
                                    
                                    col3, col4 = st.columns([2, 1])
                                    
                                    with col3:
                                        # Create heatmap for ratios
                                        fig2 = px.imshow(
                                            ratio_pivot,
                                            title=f'{var} - Representation Ratios by Cluster',
                                            labels=dict(x="Cluster", y="Category", color="Ratio"),
                                            color_continuous_scale='RdYlBu',
                                            aspect="auto"
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                    
                                    with col4:
                                        # Show ratio table
                                        display_ratio = ratio_pivot.round(2)
                                        st.dataframe(display_ratio, use_container_width=True)
                                
                                # Show detailed statistics
                                st.markdown("**📊 Detailed Statistics:**")
                                # Check which columns are available
                                available_cols = var_data.columns.tolist()
                                detail_cols = ['category', 'cluster', 'count', 'percentage']
                                
                                # Add optional columns if they exist
                                if 'overall_percentage' in available_cols:
                                    detail_cols.append('overall_percentage')
                                if 'ratio' in available_cols:
                                    detail_cols.append('ratio')
                                
                                detail_df = var_data[detail_cols].copy()
                                
                                # Create column names based on available columns
                                col_names = ['Category', 'Cluster', 'Count', 'Cluster %']
                                if 'overall_percentage' in available_cols:
                                    col_names.append('Overall %')
                                if 'ratio' in available_cols:
                                    col_names.append('Ratio')
                                
                                detail_df.columns = col_names
                                st.dataframe(detail_df.round(2), use_container_width=True)
                        
                        # Show complete table in a collapsible section
                        with st.expander("📋 Complete Categorical Profiles Table", expanded=False):
                            st.markdown("**Full dataset with all metrics:**")
                            st.dataframe(df.round(3), use_container_width=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Categorical Profiles",
                            data=df.to_csv(index=False),
                            file_name="cluster_categorical_profiles.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Could not load categorical profiles: {str(e)}")
                        
                elif tab_type == 'report':
                    st.subheader("📋 Complete Profiling Report")
                    try:
                        with open(file_path, 'r') as f:
                            report_content = f.read()
                        
                        st.text_area(
                            "Full Report", 
                            report_content, 
                            height=400,
                            help="Complete cluster profiling analysis"
                        )
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Full Report",
                            data=report_content,
                            file_name="cluster_profiling_report.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Could not load profiling report: {str(e)}")

def display_file_viewer(output_dir: str):
    """Display file viewer for all generated artifacts."""
    st.header("🔍 File Viewer")
    
    # Show the output directory path for debugging
    st.info(f"📁 Looking for files in: {output_dir}")
    
    # List all files in output directory (including subdirectories)
    files = glob.glob(os.path.join(output_dir, '*'))
    
            # Also search for common Rapid Modeler output files (excluding scoring code and model summaries)
    common_files = []
    common_patterns = [
        os.path.join(output_dir, '*.json'),
        os.path.join(output_dir, '*.csv'),
        os.path.join(output_dir, '*.log'),
        os.path.join(output_dir, '*_metrics.*'),
        os.path.join(output_dir, 'validation_plots', '*'),
        os.path.join(output_dir, '*', '*.json'),  # JSON files in subdirectories
    ]
    
    for pattern in common_patterns:
        common_files.extend(glob.glob(pattern))
    
    # Remove duplicates and add to files list
    all_files = list(set(files + common_files))
    
    if not all_files:
        st.warning(f"❌ No files found in output directory: {output_dir}")
        st.info("💡 This could mean:")
        st.info("   • The job hasn't completed yet")
        st.info("   • The job failed and didn't generate output files")
        st.info("   • The output directory path is incorrect")
        return
    
    # Show total file count
    st.success(f"🎉 Found {len(all_files)} total files in output directory!")
    
    # Add download all files button
    import zipfile
    import io
    
    def create_zip_file():
        """Create a zip file containing all job files."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in all_files:
                if os.path.isfile(file_path):
                    # Get relative path from output directory
                    arcname = os.path.relpath(file_path, output_dir)
                    zip_file.write(file_path, arcname)
                elif os.path.isdir(file_path):
                    # Add all files in directory
                    for root, dirs, files in os.walk(file_path):
                        for file in files:
                            file_path_full = os.path.join(root, file)
                            arcname = os.path.relpath(file_path_full, output_dir)
                            zip_file.write(file_path_full, arcname)
        return zip_buffer.getvalue()
    
    # Download all files button
    if st.button("📥 Download All Job Files (ZIP)", help="Download all files generated by this job as a ZIP archive"):
        try:
            zip_data = create_zip_file()
            job_name = os.path.basename(output_dir)
            st.download_button(
                label="⬇️ Download Complete Job Archive",
                data=zip_data,
                file_name=f"{job_name}_complete_archive.zip",
                mime='application/zip',
                key="download_all_files"
            )
        except Exception as e:
            st.error(f"Error creating ZIP file: {str(e)}")
    
    # Categorize files
    model_files = [f for f in all_files if os.path.isdir(f) and '_model' in os.path.basename(f)]
    log_files = [f for f in all_files if f.endswith('.log')]
    json_files = [f for f in all_files if f.endswith('.json')]
    csv_files = [f for f in all_files if f.endswith('.csv')]
    other_files = [f for f in all_files if f not in model_files + log_files + json_files + csv_files and os.path.isfile(f)]
    
    # Show file counts
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Model Dirs", len(model_files))
    with col2:
        st.metric("📄 JSON Files", len(json_files))
    with col3:
        st.metric("📊 CSV Files", len(csv_files))
    with col4:
        st.metric("📜 Log Files", len(log_files))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗂️ Available Files")
        
        if model_files:
            st.markdown("**Model Directories:**")
            for f in model_files:
                st.text(f"📁 {os.path.basename(f)}")
        
        if json_files:
            st.markdown("**JSON Files:**")
            for f in json_files:
                st.text(f"📄 {os.path.basename(f)}")
        
        if csv_files:
            st.markdown("**CSV Files:**")
            for f in csv_files:
                st.text(f"📊 {os.path.basename(f)}")
        
        if log_files:
            st.markdown("**Log Files:**")
            for f in log_files:
                st.text(f"📜 {os.path.basename(f)}")
        
        if other_files:
            st.markdown("**Other Files:**")
            for f in other_files:
                st.text(f"📋 {os.path.basename(f)}")
    
    with col2:
        st.subheader("🔍 File Viewer")
        
        viewable_files = json_files + csv_files + log_files
        if viewable_files:
            st.success(f"📋 {len(viewable_files)} viewable files found!")
            selected_file = st.selectbox("Select file to view", viewable_files, format_func=os.path.basename)
            
            if selected_file.endswith('.json'):
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    st.json(data)
                    
                    # Add download button for individual JSON file
                    st.download_button(
                        label=f"📥 Download {os.path.basename(selected_file)}",
                        data=json.dumps(data, indent=2, ensure_ascii=False),
                        file_name=os.path.basename(selected_file),
                        mime="application/json"
                    )
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                    # Try to show as text if JSON parsing fails
                    try:
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.text_area("File content (as text):", content, height=300)
                    except Exception as e2:
                        st.error(f"Could not read file as text either: {str(e2)}")
                except Exception as e:
                    st.error(f"Could not read JSON file: {str(e)}")
                    # Try to show as text if file reading fails
                    try:
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.text_area("File content (as text):", content, height=300)
                    except Exception as e2:
                        st.error(f"Could not read file as text either: {str(e2)}")
            
            elif selected_file.endswith('.csv'):
                try:
                    df = pd.read_csv(selected_file)
                    st.dataframe(df)
                    
                    # Add download button for individual CSV file
                    st.download_button(
                        label=f"📥 Download {os.path.basename(selected_file)}",
                        data=df.to_csv(index=False),
                        file_name=os.path.basename(selected_file),
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Could not read CSV file: {str(e)}")
            
            elif selected_file.endswith('.log'):
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    st.code(content, language='text')
                    
                    # Add download button for individual log file
                    st.download_button(
                        label=f"📥 Download {os.path.basename(selected_file)}",
                        data=content,
                        file_name=os.path.basename(selected_file),
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Could not read log file: {str(e)}")
        else:
            st.warning("⚠️ No viewable files found (JSON, CSV, or LOG files)")
            st.info("💡 The File Viewer shows JSON, CSV, and LOG files")
            st.info("   • Model directories are shown in the left panel")
            st.info("   • Model summaries are shown in the Model Performance tab")
            st.info("   • Scoring code is shown in the Scoring Code tab")
            st.info("   • Other file types are listed but not viewable")

def display_scoring_code(output_dir: str):
    """Display generated scoring code with dropdown selection for multiple models."""
    st.header("💾 Scoring Code")
    
    # Look for scoring scripts (both classification/regression and clustering patterns)
    scoring_files = glob.glob(os.path.join(output_dir, '*_scoring.py'))
    scoring_files.extend(glob.glob(os.path.join(output_dir, '*_clustering_scorer.py')))
    
    if not scoring_files:
        st.info("No scoring scripts found. They may not have been generated yet.")
        return
    
    # Show total count of scoring scripts
    st.success(f"🎉 Found {len(scoring_files)} scoring script(s)!")
    
    # Create selection for multiple scoring scripts
    if len(scoring_files) == 1:
        selected_file = scoring_files[0]
        st.info(f"📄 Single scoring script: {os.path.basename(selected_file)}")
    else:
        # Multiple scoring scripts - use dropdown
        file_options = [os.path.basename(f) for f in scoring_files]
        selected_filename = st.selectbox(
            "Select Scoring Script:",
            file_options,
            help="Choose which model's scoring script to view"
        )
        selected_file = os.path.join(output_dir, selected_filename)
    
    # Display selected scoring script
    filename = os.path.basename(selected_file)
    if '_clustering_scorer.py' in filename:
        model_type = filename.replace('_clustering_scorer.py', '')
    else:
        model_type = filename.replace('_model_scoring.py', '').replace('_scoring.py', '')
    
    st.subheader(f"🐍 {model_type.title()} Scoring Script")
    
    try:
        with open(selected_file, 'r') as f:
            code_content = f.read()
        
        # Add download button
        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(
                label=f"⬇️ Download Script",
                data=code_content,
                file_name=os.path.basename(selected_file),
                mime='text/plain',
                use_container_width=True
            )
        
        with col2:
            st.info(f"📄 File: {os.path.basename(selected_file)} | 📏 Lines: {len(code_content.split(chr(10)))}")
        
        # Display code
        st.code(code_content, language='python')
        
    except Exception as e:
        st.error(f"Could not read scoring file: {e}")

def display_logs(output_dir: str, job_id: str):
    """Display job logs and execution details with radio button selection."""
    st.header("📜 Logs")
    
    # Create radio button for log/script selection
    log_options = []
    
    # Add execution script
    script_log = os.path.join(JOBS_DIR, f"{job_id}_script.py")
    if os.path.exists(script_log):
        log_options.append("🔧 Execution Script")
    
    # Add job execution log
    job_log = os.path.join(JOBS_DIR, f"{job_id}_log.txt")
    if os.path.exists(job_log):
        log_options.append("📋 Job Execution Log")
    
    # Add error log if exists
    error_log = os.path.join(JOBS_DIR, f"{job_id}_error.log")
    if os.path.exists(error_log):
        log_options.append("❌ Error Log")
    
            # Add Rapid Modeler logs from output directory
    log_files = glob.glob(os.path.join(output_dir, '*.log'))
    for log_file in log_files:
        log_name = os.path.basename(log_file)
        log_options.append(f"📜 {log_name}")
    
    if not log_options:
        st.info("No logs or scripts found for this job.")
        return
    
    # Show total count
    st.success(f"🎉 Found {len(log_options)} log/script file(s)!")
    
    # Radio button selection
    selected_log = st.radio(
        "Select Log/Script to View:",
        log_options,
        help="Choose which log or script to display"
    )
    
    # Display selected log/script
    if selected_log == "🔧 Execution Script":
        st.subheader("🔧 Execution Script")
        try:
            with open(script_log, 'r') as f:
                script_content = f.read()
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="⬇️ Download Script",
                    data=script_content,
                    file_name=f"{job_id}_script.py",
                    mime='text/plain',
                    use_container_width=True
                )
            
            with col2:
                st.info(f"📄 File: {job_id}_script.py | 📏 Lines: {len(script_content.split(chr(10)))}")
            
            st.code(script_content, language='python')
                
        except Exception as e:
            st.error(f"Could not read execution script: {e}")
    
    elif selected_log == "📋 Job Execution Log":
        st.subheader("📋 Job Execution Log")
        try:
            with open(job_log, 'r') as f:
                log_content = f.read()
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="⬇️ Download Job Log",
                    data=log_content,
                    file_name=f"{job_id}_log.txt",
                    mime='text/plain',
                    use_container_width=True
                )
            
            with col2:
                st.info(f"📄 File: {job_id}_log.txt | 📏 Lines: {len(log_content.split(chr(10)))}")
            
            st.code(log_content, language='text')
                
        except Exception as e:
            st.error(f"Could not read job log: {e}")
    
    elif selected_log == "❌ Error Log":
        st.subheader("❌ Error Log")
        try:
            with open(error_log, 'r') as f:
                error_content = f.read()
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="⬇️ Download Error Log",
                    data=error_content,
                    file_name=f"{job_id}_error.log",
                    mime='text/plain',
                    use_container_width=True
                )
            
            with col2:
                st.info(f"📄 File: {job_id}_error.log | 📏 Lines: {len(error_content.split(chr(10)))}")
            
            st.code(error_content, language='text')
                
        except Exception as e:
            st.error(f"Could not read error log: {e}")
    
    else:
        # Rapid Modeler log file
        log_name = selected_log.replace("📜 ", "")
        log_file = os.path.join(output_dir, log_name)
        
        st.subheader(f"📜 {log_name}")
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="⬇️ Download Log",
                    data=log_content,
                    file_name=log_name,
                    mime='text/plain',
                    use_container_width=True
                )
            
            with col2:
                st.info(f"📄 File: {log_name} | 📏 Lines: {len(log_content.split(chr(10)))}")
            
            # Show last 200 lines for large logs
            lines = log_content.split('\n')
            if len(lines) > 200:
                st.warning(f"Log file has {len(lines)} lines. Showing last 200 lines.")
                log_content = '\n'.join(lines[-200:])
            
            st.code(log_content, language='text')
                
        except Exception as e:
            st.error(f"Could not read log file {log_name}: {e}")

def display_roc_curves_with_tabs(output_dir: str, sorted_model_files: list = None):
    """Display ROC curves for each model using tab format."""
    try:
        # If sorted_model_files is provided, use the same ordering as KS tables
        if sorted_model_files:
            # Filter to only include models that have ROC curve files
            model_dirs_with_files = []
            for model_name, _ in sorted_model_files:
                model_dir = os.path.join(output_dir, model_name)
                if os.path.exists(model_dir):
                    # Check if this model directory has ROC curve files
                    roc_files = []
                    for file in os.listdir(model_dir):
                        if 'roc' in file.lower() and file.endswith('.png'):
                            roc_files.append(os.path.join(model_dir, file))
                    
                    # Only include directories that have ROC curve files
                    if roc_files:
                        model_dirs_with_files.append((model_name, model_dir))
        else:
            # Fallback: Find all model directories that actually have ROC curve files
            model_dirs_with_files = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and any(model_type in item.lower() for model_type in [
                    'logistic', 'xgboost', 'lightgbm', 'random_forest', 'randomforest', 
                    'naivebayes', 'decision_tree', 'decisiontree', 'gradient_boosting', 
                    'gradientboosting', 'neural_network', 'neuralnetwork'
                ]):
                    # Check if this model directory has ROC curve files
                    roc_files = []
                    for file in os.listdir(item_path):
                        if 'roc' in file.lower() and file.endswith('.png'):
                            roc_files.append(os.path.join(item_path, file))
                    
                    # Only include directories that have ROC curve files
                    if roc_files:
                        model_dirs_with_files.append((item, item_path))
            
            # Sort model directories for consistent display
            model_dirs_with_files.sort(key=lambda x: x[0])
        
        if not model_dirs_with_files:
            return  # No model directories with ROC curve files found
        
        st.subheader("📈 ROC Curves")
        st.markdown("Receiver Operating Characteristic (ROC) curves for classification models")
        
        # Create tabs for each model's ROC curves
        if len(model_dirs_with_files) == 1:
            # Single model - display directly
            display_single_model_roc_curves(model_dirs_with_files[0][1])
        else:
            # Multiple models - create tabs using the same names as KS tables
            model_names = [model_name for model_name, _ in model_dirs_with_files]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, model_dir) in zip(tabs, model_dirs_with_files):
                with tab:
                    display_single_model_roc_curves(model_dir)
        
    except Exception as e:
        st.error(f"Error displaying ROC curves: {str(e)}")

def display_single_model_roc_curves(model_dir: str):
    """Display ROC curves for a single model."""
    try:
        model_name = os.path.basename(model_dir)
        
        # Find ROC curve images
        roc_files = []
        for file in os.listdir(model_dir):
            if 'roc' in file.lower() and file.endswith('.png'):
                roc_files.append(os.path.join(model_dir, file))
        
        if not roc_files:
            st.info(f"No ROC curve images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in roc_files:
                if dataset in file.lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in roc_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 3:
            num_columns = num_datasets
        elif num_datasets == 4:
            num_columns = 4
        else:  # 5 or more datasets
            num_columns = 5
        
        # Create columns and display ROC curves
        cols = st.columns(num_columns)
        
        for i, file_path in enumerate(sorted_files):
            col_index = i % num_columns
            with cols[col_index]:
                dataset_name = os.path.basename(file_path).replace('.png', '').replace('ROC for ', '').replace(' data', '')
                st.markdown(f"**{dataset_name.title()}**")
                st.image(file_path, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying ROC curves for {model_name}: {str(e)}")

def display_regression_qq_plots(output_dir: str, sorted_model_files: list = None):
    """Display QQ plots for regression models using tab format."""
    try:
        # Use cached plot file discovery for better performance
        all_plot_files = get_cached_file_discovery(output_dir, 'plots', 'regression')
        
        # Filter for QQ plot files
        qq_plot_files = [file for file in all_plot_files if 'qq_plot' in os.path.basename(file).lower()]
        
        if not qq_plot_files:
            return  # No QQ plot files found
        
        # Group files by model type
        model_plot_groups = {}
        for plot_file in qq_plot_files:
            filename = os.path.basename(plot_file)
            
            # Extract model name from filename, preserving default/tuned suffixes
            parts = filename.split('_')
            model_name = None
            
            # Look for the full model name including default/tuned
            for i in range(len(parts) - 2):
                potential_model = '_'.join(parts[:i+1])
                if any(model_type in potential_model.lower() for model_type in [
                    'linear_regression', 'random_forest', 'decision_tree', 
                    'gradient_boosting', 'xgboost', 'lightgbm'
                ]):
                    # Check if the next part is 'default' or 'tuned'
                    if i + 1 < len(parts) and parts[i + 1] in ['default', 'tuned']:
                        model_name = potential_model + '_' + parts[i + 1]
                    else:
                        model_name = potential_model
                    break
            
            if model_name:
                if model_name not in model_plot_groups:
                    model_plot_groups[model_name] = []
                model_plot_groups[model_name].append(plot_file)
        
        if not model_plot_groups:
            return  # No valid model groups found
        
        # Define the desired order for model types
        model_order = [
            'linear_regression', 'linear_regression_default', 'linear_regression_tuned',
            'random_forest', 'random_forest_default', 'random_forest_tuned',
            'decision_tree', 'decision_tree_default', 'decision_tree_tuned',
            'gradient_boosting', 'gradient_boosting_default', 'gradient_boosting_tuned',
            'xgboost', 'xgboost_default', 'xgboost_tuned',
            'lightgbm', 'lightgbm_default', 'lightgbm_tuned'
        ]
        
        st.subheader("📈 QQ Plots")
        st.markdown("Q-Q plots for regression model residuals")
        
        # Sort model groups according to the desired order
        sorted_model_groups = []
        used_models = set()
        
        for model_type in model_order:
            for model_name in model_plot_groups.keys():
                if model_type.lower() in model_name.lower() and model_name not in used_models:
                    sorted_model_groups.append((model_name, model_plot_groups[model_name]))
                    used_models.add(model_name)
        
        # Add any remaining models that don't match the pattern
        for model_name, plot_files in model_plot_groups.items():
            if model_name not in used_models:
                sorted_model_groups.append((model_name, plot_files))
                used_models.add(model_name)
        
        # Create tabs for each model's QQ plots
        if len(sorted_model_groups) == 1:
            # Single model - display directly
            display_single_model_regression_qq_plots(sorted_model_groups[0][0], sorted_model_groups[0][1])
        else:
            # Multiple models - create tabs
            model_names = [model_name for model_name, _ in sorted_model_groups]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, plot_files) in zip(tabs, sorted_model_groups):
                with tab:
                    display_single_model_regression_qq_plots(model_name, plot_files)
        
    except Exception as e:
        st.error(f"Error displaying regression QQ plots: {str(e)}")

def display_single_model_regression_qq_plots(model_name: str, plot_files: list):
    """Display QQ plots for a single regression model organized by dataset."""
    try:
        if not plot_files:
            st.info(f"No QQ plot images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in plot_files:
                if dataset in os.path.basename(file).lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                st.markdown(f"**{dataset_name}**")
                
                # Display the image
                try:
                    if os.path.exists(plot_file):
                        st.image(plot_file, use_container_width=True)
                    else:
                        st.error(f"Image file not found: {os.path.basename(plot_file)}")
                except Exception as e:
                    st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying QQ plots for {model_name}: {str(e)}")

def display_regression_actual_vs_predicted_plots(output_dir: str, sorted_model_files: list = None):
    """Display actual vs predicted plots for regression models using tab format."""
    try:
        # Use cached plot file discovery for better performance
        all_plot_files = get_cached_file_discovery(output_dir, 'plots', 'regression')
        
        # Filter for actual vs predicted plot files
        actual_vs_predicted_files = [file for file in all_plot_files if 'actual_vs_predicted' in os.path.basename(file).lower()]
        
        if not actual_vs_predicted_files:
            return  # No actual vs predicted plot files found
        
        # Group files by model type
        model_plot_groups = {}
        for plot_file in actual_vs_predicted_files:
            filename = os.path.basename(plot_file)
            
            # Extract model name from filename, preserving default/tuned suffixes
            parts = filename.split('_')
            model_name = None
            
            # Look for the full model name including default/tuned
            for i in range(len(parts) - 2):
                potential_model = '_'.join(parts[:i+1])
                if any(model_type in potential_model.lower() for model_type in [
                    'linear_regression', 'random_forest', 'decision_tree', 
                    'gradient_boosting', 'xgboost', 'lightgbm'
                ]):
                    # Check if the next part is 'default' or 'tuned'
                    if i + 1 < len(parts) and parts[i + 1] in ['default', 'tuned']:
                        model_name = potential_model + '_' + parts[i + 1]
                    else:
                        model_name = potential_model
                    break
            
            if model_name:
                if model_name not in model_plot_groups:
                    model_plot_groups[model_name] = []
                model_plot_groups[model_name].append(plot_file)
        
        if not model_plot_groups:
            return  # No valid model groups found
        
        # Define the desired order for model types
        model_order = [
            'linear_regression', 'linear_regression_default', 'linear_regression_tuned',
            'random_forest', 'random_forest_default', 'random_forest_tuned',
            'decision_tree', 'decision_tree_default', 'decision_tree_tuned',
            'gradient_boosting', 'gradient_boosting_default', 'gradient_boosting_tuned',
            'xgboost', 'xgboost_default', 'xgboost_tuned',
            'lightgbm', 'lightgbm_default', 'lightgbm_tuned'
        ]
        
        st.subheader("🎯 Actual vs Predicted Plots")
        st.markdown("Actual vs predicted plots for regression models")
        
        # Sort model groups according to the desired order
        sorted_model_groups = []
        used_models = set()
        
        for model_type in model_order:
            for model_name in model_plot_groups.keys():
                if model_type.lower() in model_name.lower() and model_name not in used_models:
                    sorted_model_groups.append((model_name, model_plot_groups[model_name]))
                    used_models.add(model_name)
        
        # Add any remaining models that don't match the pattern
        for model_name, plot_files in model_plot_groups.items():
            if model_name not in used_models:
                sorted_model_groups.append((model_name, plot_files))
                used_models.add(model_name)
        
        # Create tabs for each model's actual vs predicted plots
        if len(sorted_model_groups) == 1:
            # Single model - display directly
            display_single_model_regression_actual_vs_predicted_plots(sorted_model_groups[0][0], sorted_model_groups[0][1])
        else:
            # Multiple models - create tabs
            model_names = [model_name for model_name, _ in sorted_model_groups]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, plot_files) in zip(tabs, sorted_model_groups):
                with tab:
                    display_single_model_regression_actual_vs_predicted_plots(model_name, plot_files)
        
    except Exception as e:
        st.error(f"Error displaying regression actual vs predicted plots: {str(e)}")

def display_single_model_regression_actual_vs_predicted_plots(model_name: str, plot_files: list):
    """Display actual vs predicted plots for a single regression model organized by dataset."""
    try:
        if not plot_files:
            st.info(f"No actual vs predicted plot images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in plot_files:
                if dataset in os.path.basename(file).lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                st.markdown(f"**{dataset_name}**")
                
                # Display the image
                try:
                    if os.path.exists(plot_file):
                        st.image(plot_file, use_container_width=True)
                    else:
                        st.error(f"Image file not found: {os.path.basename(plot_file)}")
                except Exception as e:
                    st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying actual vs predicted plots for {model_name}: {str(e)}")

def display_regression_residuals_plots(output_dir: str, sorted_model_files: list = None):
    """Display residuals plots for regression models using tab format."""
    try:
        # Use cached plot file discovery for better performance
        all_plot_files = get_cached_file_discovery(output_dir, 'plots', 'regression')
        
        # Filter for residuals plot files (exclude residuals_distribution)
        residuals_plot_files = [file for file in all_plot_files 
                               if 'residuals' in os.path.basename(file).lower() 
                               and 'residuals_distribution' not in os.path.basename(file).lower()]
        
        if not residuals_plot_files:
            return  # No residuals plot files found
        
        # Group files by model type
        model_plot_groups = {}
        for plot_file in residuals_plot_files:
            filename = os.path.basename(plot_file)
            
            # Extract model name from filename, preserving default/tuned suffixes
            parts = filename.split('_')
            model_name = None
            
            # Look for the full model name including default/tuned
            for i in range(len(parts) - 2):
                potential_model = '_'.join(parts[:i+1])
                if any(model_type in potential_model.lower() for model_type in [
                    'linear_regression', 'random_forest', 'decision_tree', 
                    'gradient_boosting', 'xgboost', 'lightgbm'
                ]):
                    # Check if the next part is 'default' or 'tuned'
                    if i + 1 < len(parts) and parts[i + 1] in ['default', 'tuned']:
                        model_name = potential_model + '_' + parts[i + 1]
                    else:
                        model_name = potential_model
                    break
            
            if model_name:
                if model_name not in model_plot_groups:
                    model_plot_groups[model_name] = []
                model_plot_groups[model_name].append(plot_file)
        
        if not model_plot_groups:
            return  # No valid model groups found
        
        # Define the desired order for model types
        model_order = [
            'linear_regression', 'linear_regression_default', 'linear_regression_tuned',
            'random_forest', 'random_forest_default', 'random_forest_tuned',
            'decision_tree', 'decision_tree_default', 'decision_tree_tuned',
            'gradient_boosting', 'gradient_boosting_default', 'gradient_boosting_tuned',
            'xgboost', 'xgboost_default', 'xgboost_tuned',
            'lightgbm', 'lightgbm_default', 'lightgbm_tuned'
        ]
        
        st.subheader("📊 Residuals Plots")
        st.markdown("Residuals plots for regression models")
        
        # Sort model groups according to the desired order
        sorted_model_groups = []
        used_models = set()
        
        for model_type in model_order:
            for model_name in model_plot_groups.keys():
                if model_type.lower() in model_name.lower() and model_name not in used_models:
                    sorted_model_groups.append((model_name, model_plot_groups[model_name]))
                    used_models.add(model_name)
        
        # Add any remaining models that don't match the pattern
        for model_name, plot_files in model_plot_groups.items():
            if model_name not in used_models:
                sorted_model_groups.append((model_name, plot_files))
                used_models.add(model_name)
        
        # Create tabs for each model's residuals plots
        if len(sorted_model_groups) == 1:
            # Single model - display directly
            display_single_model_regression_residuals_plots(sorted_model_groups[0][0], sorted_model_groups[0][1])
        else:
            # Multiple models - create tabs
            model_names = [model_name for model_name, _ in sorted_model_groups]
            tabs = st.tabs(model_names)
            
            for tab, (model_name, plot_files) in zip(tabs, sorted_model_groups):
                with tab:
                    display_single_model_regression_residuals_plots(model_name, plot_files)
        
    except Exception as e:
        st.error(f"Error displaying regression residuals plots: {str(e)}")

def display_single_model_regression_residuals_plots(model_name: str, plot_files: list):
    """Display residuals plots for a single regression model organized by dataset."""
    try:
        if not plot_files:
            st.info(f"No residuals plot images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in plot_files:
                if dataset in os.path.basename(file).lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                st.markdown(f"**{dataset_name}**")
                
                # Display the image
                try:
                    if os.path.exists(plot_file):
                        st.image(plot_file, use_container_width=True)
                    else:
                        st.error(f"Image file not found: {os.path.basename(plot_file)}")
                except Exception as e:
                    st.error(f"Error displaying image {os.path.basename(plot_file)}: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying residuals plots for {model_name}: {str(e)}")

def display_plot_type_section(plot_type: str, model_plot_groups: dict, model_order: list):
    """Display a specific plot type section with models as tabs."""
    plot_type_display_names = {
        'qq_plot': 'QQ Plots',
        'actual_vs_predicted': 'Actual vs Predicted',
        'residuals': 'Residuals'
    }
    
    st.subheader(f"📊 {plot_type_display_names[plot_type]}")
    
    # Sort model groups according to the desired order
    sorted_model_groups = []
    used_models = set()
    
    for model_type in model_order:
        for model_name in model_plot_groups.keys():
            if model_type.lower() in model_name.lower() and model_name not in used_models:
                sorted_model_groups.append((model_name, model_plot_groups[model_name]))
                used_models.add(model_name)
    
    # Add any remaining models that don't match the pattern
    for model_name, plot_files in model_plot_groups.items():
        if model_name not in used_models:
            sorted_model_groups.append((model_name, plot_files))
            used_models.add(model_name)
    
    if len(sorted_model_groups) == 1:
        # Single model - display directly
        display_model_plots_by_dataset(sorted_model_groups[0][0], sorted_model_groups[0][1], plot_type)
    else:
        # Multiple models - create tabs
        model_names = [model_name for model_name, _ in sorted_model_groups]
        tabs = st.tabs(model_names)
        
        for tab, (model_name, plot_files) in zip(tabs, sorted_model_groups):
            with tab:
                display_model_plots_by_dataset(model_name, plot_files, plot_type)

def display_model_plots_by_dataset(model_name: str, plot_files: list, plot_type: str):
    """Display plots for a specific model organized by dataset."""
    try:
        if not plot_files:
            st.info(f"No {plot_type} images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in plot_files:
                if dataset in os.path.basename(file).lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                st.markdown(f"**{dataset_name}**")
                
                # Display the image
                try:
                    with open(plot_file, 'rb') as f:
                        image_data = f.read()
                    st.image(image_data, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {str(e)}")
                
                # Add download button for individual plot
                st.download_button(
                    label=f"📥 Download {dataset_name}",
                    data=image_data if 'image_data' in locals() else b"",
                    file_name=f"{model_name}_{plot_type}_{dataset_name.lower()}.png",
                    mime="image/png",
                    key=f"download_{model_name}_{plot_type}_{dataset_name}_{i}"
                )
        
    except Exception as e:
        st.error(f"Error displaying {plot_type} plots for {model_name}: {str(e)}")

def display_single_model_regression_validation_plots_from_files(model_name: str, plot_files: list):
    """Display regression validation plots for a single model from a list of files."""
    try:
        if not plot_files:
            st.info(f"No validation plot images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in plot_files:
                if dataset in os.path.basename(file).lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                # Determine plot type
                plot_type = "Validation Plot"
                if 'qq_plot' in filename.lower():
                    plot_type = "QQ Plot"
                elif 'actual_vs_predicted' in filename.lower():
                    plot_type = "Actual vs Predicted"
                elif 'residuals' in filename.lower() and 'distribution' not in filename.lower():
                    plot_type = "Residuals"
                elif 'residuals_distribution' in filename.lower():
                    plot_type = "Residuals Distribution"
                
                st.markdown(f"**{plot_type} - {dataset_name}**")
                
                # Display the image
                try:
                    with open(plot_file, 'rb') as f:
                        image_data = f.read()
                    st.image(image_data, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {str(e)}")
                
                # Add download button for individual plot
                st.download_button(
                    label=f"📥 Download {plot_type} - {dataset_name}",
                    data=image_data if 'image_data' in locals() else b"",
                    file_name=f"{model_name}_{plot_type.lower().replace(' ', '_')}_{dataset_name.lower()}.png",
                    mime="image/png",
                    key=f"download_{model_name}_{plot_type}_{dataset_name}_{i}"
                )
        
    except Exception as e:
        st.error(f"Error displaying validation plots for {model_name}: {str(e)}")

def display_single_model_regression_validation_plots(model_dir: str):
    """Display regression validation plots for a single model."""
    try:
        model_name = os.path.basename(model_dir)
        
        # Find validation plot images
        validation_plot_files = []
        for file in os.listdir(model_dir):
            if any(plot_type in file.lower() for plot_type in ['qqplot', 'actual_vs_predicted', 'residuals']) and file.endswith('.png'):
                validation_plot_files.append(os.path.join(model_dir, file))
        
        if not validation_plot_files:
            st.info(f"No validation plot images found for {model_name}")
            return
        
        # Sort files by dataset type for consistent display
        dataset_order = ['train', 'valid', 'test', 'oot1', 'oot2']
        sorted_files = []
        
        for dataset in dataset_order:
            for file in validation_plot_files:
                if dataset in file.lower():
                    sorted_files.append(file)
                    break
        
        # Add any remaining files
        for file in validation_plot_files:
            if file not in sorted_files:
                sorted_files.append(file)
        
        # Determine number of columns based on available datasets
        num_datasets = len(sorted_files)
        if num_datasets == 0:
            return
        
        # Use appropriate number of columns based on dataset count
        if num_datasets <= 2:
            cols = st.columns(num_datasets)
        elif num_datasets <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(5)
        
        # Display plots in columns
        for i, plot_file in enumerate(sorted_files):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Extract dataset name from filename
                filename = os.path.basename(plot_file)
                dataset_name = "Unknown"
                for dataset in dataset_order:
                    if dataset in filename.lower():
                        dataset_name = dataset.title()
                        break
                
                # Determine plot type
                plot_type = "Validation Plot"
                if 'qqplot' in filename.lower():
                    plot_type = "QQ Plot"
                elif 'actual_vs_predicted' in filename.lower():
                    plot_type = "Actual vs Predicted"
                elif 'residuals' in filename.lower():
                    plot_type = "Residuals"
                
                st.markdown(f"**{plot_type} - {dataset_name}**")
                
                # Display the image
                try:
                    with open(plot_file, 'rb') as f:
                        image_data = f.read()
                    st.image(image_data, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {str(e)}")
                
                # Add download button for individual plot
                st.download_button(
                    label=f"📥 Download {plot_type} - {dataset_name}",
                    data=image_data if 'image_data' in locals() else b"",
                    file_name=f"{model_name}_{plot_type.lower().replace(' ', '_')}_{dataset_name.lower()}.png",
                    mime="image/png",
                    key=f"download_{model_name}_{plot_type}_{dataset_name}_{i}"
                )
        
    except Exception as e:
        st.error(f"Error displaying validation plots for {model_name}: {str(e)}")

def main():
    """Main application function."""
    # Clean up completed jobs if job manager is available
    if job_manager is not None:
        job_manager.cleanup_completed_jobs()
    
    # Sidebar navigation
    st.sidebar.title("🤖 Rapid Modeler System")
    
    # Initialize session state for page selection
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🚀 Submit Job"
    
    # Page selection with session state persistence
    page = st.sidebar.radio(
        "Navigate to:",
        ["🚀 Submit Job", "📊 View Results"],
        index=0 if st.session_state.current_page == "🚀 Submit Job" else 1
    )
    
    # Update session state when page changes
    if page != st.session_state.current_page:
        st.session_state.current_page = page
    
    # Add some spacing
    st.sidebar.markdown("---")
    
    # Status overview in sidebar
    st.sidebar.markdown("**📈 System Status:**")
    
    # Count jobs by status - only count active/completed jobs
    job_files = glob.glob(os.path.join(JOBS_DIR, "*.json"))
    status_counts = {'Completed': 0, 'Running': 0, 'Failed': 0}
    
    for job_file in job_files:
        try:
            job_id = os.path.basename(job_file).replace('.json', '')
            status = get_job_status(job_id)
            
            # Only count jobs that are actually running, completed, or failed
            # Ignore "Submitted" status to avoid double counting
            if status == 'Running':
                status_counts['Running'] += 1
            elif status == 'Completed':
                status_counts['Completed'] += 1
            elif status == 'Failed':
                status_counts['Failed'] += 1
            # Note: "Submitted" jobs are not counted to avoid double counting
        except:
            continue
    
    total_jobs = sum(status_counts.values())
    st.sidebar.text(f"📊 Total Jobs: {total_jobs}")
    
    for status, count in status_counts.items():
        emoji = {'Completed': '✅', 'Running': '🔄', 'Failed': '❌'}[status]
        st.sidebar.text(f"{emoji} {status}: {count}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Rapid Modeler PySpark System v1.0*")
    
    # Route to appropriate page
    if page == "🚀 Submit Job":
        create_job_submission_page()
    else:
        create_results_page()

if __name__ == "__main__":
    main() 