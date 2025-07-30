# Deploying Rapid Modeler PySpark on Google Cloud Run

This document provides step‑by‑step instructions for deploying the Rapid Modeler PySpark application to [Google Cloud Run](https://cloud.google.com/run). Cloud Run allows you to run containerised applications in a fully managed environment without managing infrastructure.

## Prerequisites

1. **Google Cloud Account** – Ensure you have an active Google Cloud project and billing enabled.
2. **gcloud CLI** – Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_GCP_PROJECT_ID
   ```
3. **Docker** – Docker is required for building the container image locally. If using Cloud Build you can skip local Docker installation.

## Build the Container Image

1. Clone or copy this repository locally. From the repository root, build the Docker image using either Docker or Cloud Build.

### Option A: Build with Docker

```bash
docker build -t gcr.io/YOUR_GCP_PROJECT_ID/rapid‑modeler:latest .
```

### Option B: Build with Cloud Build

```bash
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/rapid‑modeler:latest .
```

Cloud Build will automatically build the container using the `Dockerfile` in the project root.

## Deploy to Cloud Run

Once the image is built and pushed to Container Registry (or Artifact Registry), deploy it to Cloud Run:

```bash
gcloud run deploy rapid‑modeler \
  --image gcr.io/YOUR_GCP_PROJECT_ID/rapid‑modeler:latest \
  --platform managed \
  --region us‑central1 \
  --allow‑unauthenticated \
  --port 8080
```

- `--allow‑unauthenticated` makes the application publicly accessible. Omit this flag if you want to restrict access.
- Change `--region` to your preferred region.

After deployment, Cloud Run will provide a service URL where you can access the Streamlit UI.

## Configuring Message Queue (Optional)

This application supports offloading background AutoML jobs to an external queue.  There are two supported dispatch mechanisms: **Pub/Sub** and **Cloud Tasks**.  Both modes allow you to decouple long‑running training jobs from the Streamlit UI and handle them in a scalable manner.  If neither mode is enabled, jobs run in the Cloud Run container using Python threads.

### Enabling Pub/Sub dispatch

1. Create a Pub/Sub topic (if not already created):
   ```bash
   gcloud pubsub topics create automl‑jobs
   ```
2. Grant the Cloud Run service account the **Pub/Sub Publisher** role for this topic.
3. Set the following environment variables when deploying your service:
   - `USE_GCP_QUEUE=true` – enable queue‑based dispatch.
   - `GCP_PUBSUB_TOPIC=projects/YOUR_GCP_PROJECT_ID/topics/automl‑jobs` – full topic path.
4. Deploy with these variables:
   ```bash
   gcloud run deploy rapid‑modeler \
     --image gcr.io/YOUR_GCP_PROJECT_ID/rapid‑modeler:latest \
     --platform managed \
     --region us‑central1 \
     --set‑env‑vars USE_GCP_QUEUE=true,GCP_PUBSUB_TOPIC=projects/YOUR_GCP_PROJECT_ID/topics/automl‑jobs \
     --allow‑unauthenticated
   ```
5. Implement a worker service (e.g. another Cloud Run instance or Cloud Function) subscribed to this topic.  The worker reads the job configuration from Cloud Storage (the `automl_jobs` directory) and executes the job script.  The worker should run in the same Python environment and can invoke the generated job script directly.

### Enabling Cloud Tasks dispatch

Cloud Tasks allows you to schedule HTTP requests directly to a service.  The AutoML codebase includes a helper (`automl_pyspark/gcp_helpers.py`) and extended dispatch logic to support Cloud Tasks.  To enable this mode:

1. Create a Cloud Tasks queue:
   ```bash
   gcloud tasks queues create automl‑jobs --location=us‑east1
   ```
2. Grant the Cloud Run service account the **Cloud Tasks Enqueuer** role, and ensure the target service has the appropriate IAM binding to accept the OIDC token (e.g. `run.invoker`).
3. Set the following environment variables when deploying:
   - `USE_GCP_QUEUE=true` – enable queue dispatch.
   - `USE_GCP_TASKS=true` – select Cloud Tasks instead of Pub/Sub.
   - `GCP_TASKS_PROJECT=YOUR_GCP_PROJECT_ID` – project ID containing the queue.
   - `GCP_TASKS_LOCATION=us‑east1` – location/region of the queue.
   - `GCP_TASKS_QUEUE=automl‑jobs` – name of the queue.
   - `CLOUD_RUN_BASE_URL=https://rapid‑modeler-<hash>-uc.a.run.app` – base URL of the service that will process the job.  This should point to your deployed Streamlit service.
   - `SERVICE_ACCOUNT_EMAIL=my-service@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com` – (optional) service account email for the OIDC token.  If omitted, the helper will derive it from the credentials file.
4. Deploy with the above variables:
   ```bash
   gcloud run deploy rapid‑modeler \
     --image gcr.io/YOUR_GCP_PROJECT_ID/rapid‑modeler:latest \
     --platform managed \
     --region us‑central1 \
     --set‑env‑vars USE_GCP_QUEUE=true,USE_GCP_TASKS=true,GCP_TASKS_PROJECT=YOUR_GCP_PROJECT_ID,GCP_TASKS_LOCATION=us‑east1,GCP_TASKS_QUEUE=automl‑jobs,CLOUD_RUN_BASE_URL=https://rapid‑modeler-<hash>-uc.a.run.app,SERVICE_ACCOUNT_EMAIL=my-service@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com \
     --allow‑unauthenticated
   ```
5. Implement a job‑processing endpoint in your service (e.g. `/run-job`) to handle the POST requests from Cloud Tasks.  The AutoML tasks manager will include the job ID and configuration in the JSON payload.  Use the helper `create_http_task` in `automl_pyspark/gcp_helpers.py` if you need to enqueue tasks programmatically from within Python.

Without setting `USE_GCP_QUEUE`, the background jobs run inside the Cloud Run container using Python threads.  This may be sufficient for small workloads but will compete for CPU and memory with the Streamlit UI.  For larger or multiple concurrent jobs, using a queue with a separate worker is recommended.

## Notes

- Cloud Run instances are stateless and may be terminated between requests. The `automl_jobs` and `automl_results` directories are stored on the container’s filesystem, which is an in‑memory file system and will not persist across instance restarts. For production use, configure Cloud Storage buckets for job metadata and results, and update the configuration accordingly.
- To secure the Streamlit UI, consider enabling Cloud Run IAM authentication or a Cloud Armor policy.
 - For BigQuery support, ensure the Cloud Run service account has the necessary BigQuery read permissions.

## Using a Service Account Key File

In some environments you may need to provide explicit service account credentials rather than relying on the Cloud Run runtime identity.  For example, when connecting to BigQuery from PySpark or using Google Cloud clients (BigQuery, Pub/Sub, Cloud Tasks) directly, you can mount a JSON key file and point the Google SDKs to it.

1. **Create or download a service account key** with the required roles (e.g. BigQuery Data Viewer, Pub/Sub Publisher) from the Google Cloud Console.
2. **Include the key in your container.**  You can copy it into the image or mount it as a secret at runtime.  In your Dockerfile you might add:

   ```dockerfile
   # Copy service account key into the image
   COPY service-account.json /app/service-account.json
   # Tell client libraries to use this key
   ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
   ```

   Alternatively, when deploying to Cloud Run you can mount the key as a secret and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the mounted path.

3. **Deploy with the environment variable set.**  For example:

   ```bash
   gcloud run deploy rapid‑modeler \
     --image gcr.io/YOUR_GCP_PROJECT_ID/rapid‑modeler:latest \
     --platform managed \
     --region us‑central1 \
     --allow‑unauthenticated \
     --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
   ```

When `GOOGLE_APPLICATION_CREDENTIALS` is set, Google client libraries (including the Spark BigQuery connector and Pub/Sub or Cloud Tasks clients) will automatically use the specified credentials.  You can also instantiate clients manually in your code:

```python
from google.oauth2 import service_account
from google.cloud import bigquery
import json, os

with open(os.environ['GOOGLE_APPLICATION_CREDENTIALS'], 'r') as f:
    info = json.load(f)
credentials = service_account.Credentials.from_service_account_info(info)

client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
```

Similarly, you can create Cloud Tasks or Pub/Sub clients with the same credentials.

## Optional Cloud Tasks Integration

The included `BackgroundJobManager` can dispatch jobs to an external queue using Google Pub/Sub by default.  As described above you can enable [Cloud Tasks](https://cloud.google.com/tasks) dispatch by setting `USE_GCP_TASKS=true` and supplying the necessary environment variables.  Internally the AutoML package uses the helper functions in `automl_pyspark/gcp_helpers.py` to create and enqueue tasks.  Below is a simplified example of how to create a Cloud Tasks HTTP task manually:

```python
from google.cloud import tasks_v2
from google.oauth2 import service_account
import json

credentials = service_account.Credentials.from_service_account_file('service-account.json')
client = tasks_v2.CloudTasksClient(credentials=credentials)

task = tasks_v2.Task(
    http_request=tasks_v2.HttpRequest(
        http_method=tasks_v2.HttpMethod.POST,
        url="https://service-url/run-job",
        headers={"Content-Type": "application/json"},
        body=json.dumps(job_payload).encode(),
        oidc_token=tasks_v2.OidcToken(
            service_account_email="my-service@project.iam.gserviceaccount.com"
        )
    )
)

client.create_task(
    parent=client.queue_path(PROJECT_ID, REGION, QUEUE_ID),
    task=task
)
```

If you prefer to enqueue tasks yourself rather than relying on the built‑in dispatch logic, you can call `create_http_task` directly from Python.  Make sure to set `GOOGLE_APPLICATION_CREDENTIALS` or provide a `service_account_email` so that the OIDC token can be generated.  See `automl_pyspark/gcp_helpers.py` for details.

## Summary

1. Build the Docker image (`docker build` or `gcloud builds submit`).
2. Deploy the image to Cloud Run using `gcloud run deploy`.
3. Optionally, enable message queue dispatch with `USE_GCP_QUEUE` and configure a Pub/Sub topic and worker.
4. Access the UI via the Cloud Run service URL, configure your job and start exploring your data!