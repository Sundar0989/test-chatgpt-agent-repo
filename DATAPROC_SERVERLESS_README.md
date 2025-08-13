# AutoML PySpark with Google Cloud Dataproc Serverless

This guide explains how to integrate your AutoML PySpark system with **Google Cloud Dataproc Serverless** to achieve true autoscaling of Spark executors and distributed workload management.

## 🚀 Why Dataproc Serverless?

### **Benefits Over Traditional Clusters**
- **True Autoscaling**: Automatically scales from 0 to thousands of executors based on workload
- **No Cluster Management**: No need to manage cluster lifecycle, scaling, or maintenance
- **Cost Optimization**: Pay only for actual compute time (per-second billing)
- **Instant Availability**: Jobs start immediately without waiting for cluster provisioning
- **Resource Isolation**: Separate Spark processing from your Cloud Run UI

### **Perfect for AutoML Workloads**
- **Variable Resource Needs**: AutoML jobs have different computational requirements
- **Batch Processing**: Ideal for training multiple models with different parameters
- **Cost Control**: Stop paying when jobs complete
- **Scalability**: Handle multiple concurrent users without resource conflicts

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Job Scheduler   │───▶│ Dataproc Serverless │
│   (Cloud Run)   │    │  (Cloud Run)     │    │  (Spark Jobs)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Results Viewer  │◀───│  Cloud Storage   │◀───│  Job Outputs    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Setup Instructions

### 1. **Prerequisites**
- Google Cloud Project with billing enabled
- `gcloud` CLI installed and authenticated
- Docker (optional, for local testing)

### 2. **Run the Setup Script**
```bash
# Make the script executable
chmod +x setup_dataproc_serverless.sh

# Run the setup
./setup_dataproc_serverless.sh
```

The script will:
- ✅ Enable required Google Cloud APIs
- ✅ Create Cloud Storage buckets for job files and results
- ✅ Create service account with proper IAM roles
- ✅ Set up Dataproc Metastore
- ✅ Generate environment configuration

### 3. **Environment Configuration**
After setup, you'll have a `.env.dataproc` file with:
```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCP_TEMP_BUCKET=automl-temp-your-project-id
GCP_RESULTS_BUCKET=automl-results-your-project-id
GCP_SERVICE_ACCOUNT_EMAIL=automl-dataproc-sa@your-project-id.iam.gserviceaccount.com
```

## 🔧 Integration with Existing AutoML System

### **Option 1: Replace Background Job Manager**
```python
# In your Streamlit app, replace the existing background job manager
from automl_pyspark.dataproc_background_job_manager import DataprocBackgroundJobManager

# Initialize with Dataproc Serverless
job_manager = DataprocBackgroundJobManager(config={
    'project_id': os.environ['GCP_PROJECT_ID'],
    'region': os.environ['GCP_REGION'],
    'temp_bucket': os.environ['GCP_TEMP_BUCKET'],
    'results_bucket': os.environ['GCP_RESULTS_BUCKET']
})

# Submit jobs as usual
result = job_manager.submit_job(job_config, job_id, data_files)
```

### **Option 2: Hybrid Approach**
```python
# Keep existing local execution for small jobs, use Dataproc for large ones
if job_config.get('data_size_mb', 0) > 1000:
    # Use Dataproc Serverless for large datasets
    job_manager = DataprocBackgroundJobManager(config)
else:
    # Use local execution for small datasets
    job_manager = BackgroundJobManager()
```

## 📊 Job Execution Flow

### **1. Job Submission**
```python
# Job is submitted through Streamlit UI
job_config = {
    'user_id': 'user123',
    'model_name': 'customer_churn',
    'task_type': 'classification',
    'models': ['Random Forest', 'XGBoost', 'LightGBM'],
    'data_size_mb': 2500,  # Helps determine executor count
    'enable_hyperparameter_tuning': True
}

result = job_manager.submit_job(job_config, job_id, data_files)
```

### **2. Dataproc Serverless Processing**
- **File Upload**: Job files uploaded to Cloud Storage
- **Batch Creation**: Spark batch job created in Dataproc Serverless
- **Autoscaling**: Executors scale automatically based on workload
- **Execution**: AutoML pipeline runs on distributed Spark cluster
- **Results**: Output saved to Cloud Storage

### **3. Status Monitoring**
```python
# Real-time status updates
status = job_manager.get_job_status(job_id)
print(f"Job {job_id}: {status['status']} - {status['progress']}% complete")

# List all jobs
jobs = job_manager.list_jobs(filter_status='RUNNING')
```

## 💰 Cost Optimization

### **Cost Estimation**
```python
# Get cost estimate before job submission
cost_estimate = job_manager.dataproc_manager.get_cost_estimate(job_config)
print(f"Estimated cost: ${cost_estimate['estimated_cost_usd']}")
print(f"Executors: {cost_estimate['executor_count']}")
```

### **Cost Control Strategies**
- **Executor Limits**: Set min/max executor counts
- **Job Timeouts**: Prevent runaway jobs
- **Resource Monitoring**: Track actual vs. estimated costs
- **Cleanup**: Automatic cleanup of completed jobs

## 🔍 Monitoring and Debugging

### **Job Status Tracking**
```python
# Monitor job progress
while True:
    status = job_manager.get_job_status(job_id)
    if status['status'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
        break
    time.sleep(30)
    print(f"Progress: {status['progress']}%")
```

### **Logs and Debugging**
- **Dataproc Logs**: View in Google Cloud Console
- **Job Metadata**: Stored in Cloud Storage
- **Error Handling**: Comprehensive error reporting
- **Retry Logic**: Automatic retry for transient failures

## 🚀 Deployment to Cloud Run

### **Updated Dockerfile**
```dockerfile
# Add Dataproc Serverless dependencies
RUN pip install google-cloud-dataproc google-cloud-storage

# Copy configuration files
COPY dataproc_config.yaml /app/
COPY .env.dataproc /app/
```

### **Environment Variables**
```bash
gcloud run deploy rapid-modeler \
  --image gcr.io/YOUR_PROJECT_ID/rapid-modeler:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars-file .env.dataproc \
  --allow-unauthenticated
```

## 📈 Performance Benefits

### **Before (Local Execution)**
- ❌ Limited to Cloud Run container resources
- ❌ No parallel job execution
- ❌ Resource contention with UI
- ❌ Fixed executor count

### **After (Dataproc Serverless)**
- ✅ Unlimited executor scaling (0 to thousands)
- ✅ True parallel job execution
- ✅ No resource contention
- ✅ Dynamic executor allocation
- ✅ Pay-per-use pricing

## 🔧 Configuration Options

### **Dataproc Serverless Settings**
```yaml
dataproc_serverless:
  enabled: true
  spark_version: "3.4"
  runtime_config_version: "1.0"
  
resources:
  executor_count:
    min: 2
    max: 100
  executor_memory: "4g"
  executor_cpu: "2"
```

### **AutoML Integration**
```yaml
automl:
  supported_tasks:
    - classification
    - regression
    - clustering
  
  supported_models:
    classification:
      - "Random Forest"
      - "XGBoost"
      - "LightGBM"
      # ... more models
```

## 🧪 Testing

### **Local Testing**
```bash
# Set environment variables
export $(cat .env.dataproc | xargs)

# Test Dataproc Serverless manager
python -c "
from automl_pyspark.dataproc_serverless_manager import DataprocServerlessManager
manager = DataprocServerlessManager()
print('✅ Dataproc Serverless manager initialized successfully')
"
```

### **Test Job Submission**
```python
# Create test job configuration
test_config = {
    'user_id': 'test_user',
    'model_name': 'test_model',
    'task_type': 'classification',
    'models': ['Random Forest'],
    'data_size_mb': 100
}

# Submit test job
result = job_manager.submit_job(test_config, 'test_job_001')
print(f"Test job result: {result}")
```

## 🚨 Troubleshooting

### **Common Issues**

1. **"Permission Denied"**
   - Check service account IAM roles
   - Verify project permissions

2. **"Bucket Not Found"**
   - Ensure Cloud Storage buckets exist
   - Check bucket names in configuration

3. **"API Not Enabled"**
   - Run setup script to enable required APIs
   - Check API status in Google Cloud Console

4. **"Job Stuck in PENDING"**
   - Check Dataproc Serverless quotas
   - Verify service account permissions

### **Debug Commands**
```bash
# Check Dataproc Serverless status
gcloud dataproc batches list --location=us-central1

# View job logs
gcloud dataproc batches describe BATCH_ID --location=us-central1

# Check service account
gcloud iam service-accounts describe automl-dataproc-sa@PROJECT_ID.iam.gserviceaccount.com
```

## 📚 Additional Resources

- [Dataproc Serverless Documentation](https://cloud.google.com/dataproc-serverless)
- [Spark on Dataproc](https://cloud.google.com/dataproc/docs/concepts/components/spark)
- [Cloud Storage Integration](https://cloud.google.com/dataproc/docs/concepts/connectors/cloud-storage)
- [Cost Optimization](https://cloud.google.com/dataproc-serverless/docs/guides/cost-optimization)

## 🎯 Next Steps

1. **Run the setup script** to create infrastructure
2. **Update your AutoML configuration** to use Dataproc Serverless
3. **Test with small jobs** to verify integration
4. **Monitor costs and performance** to optimize settings
5. **Scale up** to handle larger datasets and more users

## 💡 Best Practices

- **Start Small**: Begin with small jobs to test the setup
- **Monitor Costs**: Use cost estimation before job submission
- **Set Limits**: Configure reasonable executor limits
- **Clean Up**: Implement automatic job cleanup
- **Error Handling**: Add comprehensive error handling and retry logic

---

**Ready to scale your AutoML system?** 🚀

Run the setup script and start enjoying the benefits of true autoscaling with Dataproc Serverless!
