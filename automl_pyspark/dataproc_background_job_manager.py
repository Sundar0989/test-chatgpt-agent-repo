"""
Dataproc Serverless Background Job Manager

This module extends the existing background job management system to use
Google Cloud Dataproc Serverless for executing AutoML jobs.

Features:
- Submit jobs to Dataproc Serverless instead of local execution
- Real-time job status monitoring
- Cost estimation and optimization
- Integration with existing Streamlit UI
- Automatic cleanup and resource management
"""

import os
import json
import time
import threading
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataproc_serverless_manager import DataprocServerlessManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataprocBackgroundJobManager:
    """
    Manages background AutoML jobs using Google Cloud Dataproc Serverless.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dataproc background job manager.
        
        Args:
            config: Configuration dictionary for Dataproc Serverless
        """
        self.config = config or {}
        self.dataproc_manager = DataprocServerlessManager(config)
        
        # Job tracking
        self.active_jobs = {}  # job_id -> batch_id mapping
        self.job_status = {}   # job_id -> status info
        self.job_results = {}  # job_id -> results info
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("🚀 Dataproc Background Job Manager initialized")
    
    def submit_job(
        self,
        job_config: Dict[str, Any],
        job_id: str,
        data_files: List[str] = None,
        dependencies: List[str] = None
    ) -> Dict[str, Any]:
        """
        Submit an AutoML job to Dataproc Serverless.
        
        Args:
            job_config: AutoML job configuration
            job_id: Unique job identifier
            data_files: List of data file paths
            dependencies: List of dependency files
            
        Returns:
            Dictionary with submission status and batch ID
        """
        try:
            logger.info(f"🚀 Submitting AutoML job {job_id} to Dataproc Serverless")
            
            # Validate job configuration
            self._validate_job_config(job_config)
            
            # Estimate cost
            cost_estimate = self.dataproc_manager.get_cost_estimate(job_config)
            
            # Submit to Dataproc Serverless
            batch_id = self.dataproc_manager.submit_spark_job(
                job_config, job_id, data_files, dependencies
            )
            
            # Track job
            self.active_jobs[job_id] = batch_id
            self.job_status[job_id] = {
                'status': 'SUBMITTED',
                'batch_id': batch_id,
                'submission_time': datetime.now().isoformat(),
                'cost_estimate': cost_estimate,
                'progress': 0
            }
            
            # Save job metadata
            self._save_job_metadata(job_id, job_config, batch_id)
            
            logger.info(f"✅ Job {job_id} submitted successfully. Batch ID: {batch_id}")
            
            return {
                'success': True,
                'batch_id': batch_id,
                'cost_estimate': cost_estimate,
                'message': f'Job submitted to Dataproc Serverless with batch ID: {batch_id}'
            }
            
        except Exception as e:
            error_msg = f"Failed to submit job {job_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Update job status
            self.job_status[job_id] = {
                'status': 'FAILED',
                'error': str(e),
                'submission_time': datetime.now().isoformat()
            }
            
            return {
                'success': False,
                'error': str(e),
                'message': error_msg
            }
    
    def _validate_job_config(self, job_config: Dict[str, Any]):
        """Validate the job configuration."""
        required_fields = ['user_id', 'model_name', 'task_type']
        
        for field in required_fields:
            if field not in job_config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate task type
        valid_task_types = ['classification', 'regression', 'clustering']
        if job_config['task_type'] not in valid_task_types:
            raise ValueError(f"Invalid task type: {job_config['task_type']}. Must be one of: {valid_task_types}")
        
        # Validate models
        if 'models' not in job_config or not job_config['models']:
            raise ValueError("At least one model must be specified")
    
    def _save_job_metadata(self, job_id: str, job_config: Dict[str, Any], batch_id: str):
        """Save job metadata for tracking."""
        metadata = {
            'job_id': job_id,
            'batch_id': batch_id,
            'submission_time': datetime.now().isoformat(),
            'job_config': job_config,
            'execution_platform': 'dataproc_serverless'
        }
        
        # Save to local file system for compatibility with existing UI
        os.makedirs('automl_jobs', exist_ok=True)
        with open(f'automl_jobs/{job_id}.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save status file
        with open(f'automl_jobs/{job_id}_status.txt', 'w') as f:
            f.write('SUBMITTED')
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the current status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with job status information
        """
        if job_id not in self.active_jobs:
            return {'error': 'Job not found'}
        
        try:
            batch_id = self.active_jobs[job_id]
            dataproc_status = self.dataproc_manager.get_job_status(batch_id)
            
            # Update local status
            if 'error' not in dataproc_status:
                self.job_status[job_id].update({
                    'status': dataproc_status['state'],
                    'state_message': dataproc_status.get('state_message', ''),
                    'last_update': datetime.now().isoformat()
                })
                
                # Update progress based on state
                progress = self._calculate_progress(dataproc_status['state'])
                self.job_status[job_id]['progress'] = progress
                
                # Check if job is completed
                if dataproc_status['state'] in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    self._handle_job_completion(job_id, dataproc_status)
            
            return self.job_status[job_id]
            
        except Exception as e:
            logger.error(f"❌ Failed to get status for job {job_id}: {e}")
            return {'error': str(e)}
    
    def _calculate_progress(self, state: str) -> int:
        """Calculate progress percentage based on job state."""
        progress_map = {
            'PENDING': 10,
            'RUNNING': 50,
            'SUCCEEDED': 100,
            'FAILED': 100,
            'CANCELLED': 100
        }
        return progress_map.get(state, 0)
    
    def _handle_job_completion(self, job_id: str, dataproc_status: Dict[str, Any]):
        """Handle job completion and cleanup."""
        try:
            if dataproc_status['state'] == 'SUCCEEDED':
                # Download results from Cloud Storage
                self._download_job_results(job_id)
                
                # Update status
                self.job_status[job_id]['status'] = 'COMPLETED'
                self.job_status[job_id]['completion_time'] = datetime.now().isoformat()
                self.job_status[job_id]['progress'] = 100
                
                # Update status file
                with open(f'automl_jobs/{job_id}_status.txt', 'w') as f:
                    f.write('COMPLETED')
                
                logger.info(f"🎉 Job {job_id} completed successfully")
                
            elif dataproc_status['state'] in ['FAILED', 'CANCELLED']:
                # Update status
                self.job_status[job_id]['status'] = 'FAILED'
                self.job_status[job_id]['error'] = dataproc_status.get('state_message', 'Unknown error')
                self.job_status[job_id]['completion_time'] = datetime.now().isoformat()
                
                # Update status file
                with open(f'automl_jobs/{job_id}_status.txt', 'w') as f:
                    f.write('FAILED')
                
                logger.error(f"❌ Job {job_id} failed: {dataproc_status.get('state_message', 'Unknown error')}")
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
                
        except Exception as e:
            logger.error(f"❌ Error handling job completion for {job_id}: {e}")
    
    def _download_job_results(self, job_id: str):
        """Download job results from Cloud Storage."""
        try:
            # This would download results from the results bucket
            # For now, we'll just mark it as completed
            logger.info(f"📥 Results for job {job_id} would be downloaded here")
            
            # Create results directory structure
            os.makedirs(f'automl_results/{job_id}', exist_ok=True)
            
            # Create a placeholder results file
            with open(f'automl_results/{job_id}/results_summary.json', 'w') as f:
                json.dump({
                    'job_id': job_id,
                    'status': 'completed',
                    'completion_time': datetime.now().isoformat(),
                    'execution_platform': 'dataproc_serverless'
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to download results for job {job_id}: {e}")
    
    def list_jobs(self, filter_status: str = None) -> List[Dict[str, Any]]:
        """
        List all jobs with optional status filtering.
        
        Args:
            filter_status: Optional status filter (e.g., 'RUNNING', 'COMPLETED')
            
        Returns:
            List of job information dictionaries
        """
        jobs = []
        
        for job_id, status in self.job_status.items():
            if filter_status and status.get('status') != filter_status:
                continue
                
            job_info = {
                'job_id': job_id,
                'batch_id': status.get('batch_id'),
                'status': status.get('status'),
                'submission_time': status.get('submission_time'),
                'completion_time': status.get('completion_time'),
                'progress': status.get('progress', 0),
                'cost_estimate': status.get('cost_estimate'),
                'error': status.get('error')
            }
            jobs.append(job_info)
        
        return jobs
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with cancellation status
        """
        if job_id not in self.active_jobs:
            return {'success': False, 'error': 'Job not found or not active'}
        
        try:
            batch_id = self.active_jobs[job_id]
            success = self.dataproc_manager.cancel_job(batch_id)
            
            if success:
                # Update local status
                self.job_status[job_id]['status'] = 'CANCELLED'
                self.job_status[job_id]['completion_time'] = datetime.now().isoformat()
                
                # Remove from active jobs
                del self.active_jobs[job_id]
                
                # Update status file
                with open(f'automl_jobs/{job_id}_status.txt', 'w') as f:
                    f.write('CANCELLED')
                
                logger.info(f"✅ Job {job_id} cancelled successfully")
                return {'success': True, 'message': f'Job {job_id} cancelled successfully'}
            else:
                return {'success': False, 'error': 'Failed to cancel job'}
                
        except Exception as e:
            error_msg = f"Error cancelling job {job_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def start_monitoring(self):
        """Start the background monitoring thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
            self.monitoring_thread.start()
            logger.info("🔍 Job monitoring started")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🔍 Job monitoring stopped")
    
    def _monitor_jobs(self):
        """Background thread to monitor job status."""
        while self.monitoring_active:
            try:
                # Check status of all active jobs
                for job_id in list(self.active_jobs.keys()):
                    try:
                        self.get_job_status(job_id)
                    except Exception as e:
                        logger.error(f"❌ Error monitoring job {job_id}: {e}")
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring thread: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for all jobs."""
        total_cost = 0
        job_costs = {}
        
        for job_id, status in self.job_status.items():
            cost_estimate = status.get('cost_estimate', {})
            estimated_cost = cost_estimate.get('estimated_cost_usd', 0)
            
            if status.get('status') == 'COMPLETED':
                # For completed jobs, use actual cost if available
                actual_cost = status.get('actual_cost_usd', estimated_cost)
                total_cost += actual_cost
                job_costs[job_id] = actual_cost
            else:
                # For running/pending jobs, use estimated cost
                total_cost += estimated_cost
                job_costs[job_id] = estimated_cost
        
        return {
            'total_cost_usd': round(total_cost, 2),
            'job_costs': job_costs,
            'active_jobs_count': len(self.active_jobs),
            'completed_jobs_count': len([j for j in self.job_status.values() if j.get('status') == 'COMPLETED'])
        }
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up completed jobs older than specified age."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            jobs_to_cleanup = []
            
            for job_id, status in self.job_status.items():
                if status.get('status') in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    completion_time = status.get('completion_time')
                    if completion_time:
                        try:
                            job_completion = datetime.fromisoformat(completion_time)
                            if job_completion < cutoff_time:
                                jobs_to_cleanup.append(job_id)
                        except ValueError:
                            # Skip jobs with invalid completion time
                            continue
            
            # Clean up old jobs
            for job_id in jobs_to_cleanup:
                self._cleanup_job(job_id)
            
            logger.info(f"🧹 Cleaned up {len(jobs_to_cleanup)} old jobs")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def _cleanup_job(self, job_id: str):
        """Clean up a specific job."""
        try:
            # Remove from tracking
            if job_id in self.job_status:
                del self.job_status[job_id]
            
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Remove local files
            job_files = [
                f'automl_jobs/{job_id}.json',
                f'automl_jobs/{job_id}_status.txt',
                f'automl_jobs/{job_id}_script.py',
                f'automl_jobs/{job_id}_error.log'
            ]
            
            for file_path in job_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info(f"🧹 Cleaned up job {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up job {job_id}: {e}")
    
    def __del__(self):
        """Cleanup when the manager is destroyed."""
        self.stop_monitoring()
