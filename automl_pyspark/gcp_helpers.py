"""
Google Cloud helper utilities for AutoML PySpark
================================================

This module provides thin wrappers around the Google Cloud Python
client libraries to support the following scenarios:

* Authenticating with a service account key file for use from Cloud
  Run or other containerized environments.  When the
  ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable is set to
  the path of a JSON key file, these helpers will load the key and
  instantiate clients accordingly.
* Creating BigQuery client instances via the `google-cloud-bigquery`
  library.  In the AutoML project the Spark BigQuery connector is
  used to load training and scoring data.  However, some downstream
  workflows may require access to BigQuery outside of Spark (for
  example when materialising results back to BigQuery).  The
  ``get_bigquery_client`` helper makes it easy to obtain a
  ``bigquery.Client`` either with application default credentials or a
  provided service account key.
* Creating Cloud Tasks clients and HTTP tasks.  The
  ``CloudTasksSingleton`` caches a single ``tasks_v2.CloudTasksClient``
  per process to avoid repeated initialisation overhead.  The
  ``create_http_task`` function builds an HTTP task with an OIDC
  token and schedules it on the specified queue.  This is provided
  as an optional alternative to the Pub/Sub based dispatch logic in
  ``background_job_manager.py``.  To enable task‑based dispatch you
  can set the environment variable ``USE_GCP_TASKS=true`` and supply
  the necessary environment variables described below.

Environment variables
---------------------

The following environment variables control behaviour:

``GOOGLE_APPLICATION_CREDENTIALS``
    Path to a JSON service account key file.  If set, the key will
    be loaded and used to authenticate both BigQuery and Cloud Tasks
    clients.  If not set, Application Default Credentials will be
    used (as is the case when running in Cloud Run with Workload
    Identity).

``GCP_TASKS_PROJECT``
    The Google Cloud project where the Cloud Tasks queue resides.

``GCP_TASKS_LOCATION``
    The location/region of the queue, e.g. ``us-east1``.

``GCP_TASKS_QUEUE``
    The name of the Cloud Tasks queue to which jobs should be
    submitted.

``CLOUD_RUN_BASE_URL``
    Base URL of your deployed Cloud Run service.  Tasks created via
    ``create_http_task`` will POST to this URL suffixed with the
    ``target_path`` argument.

``SERVICE_ACCOUNT_EMAIL``
    (Optional) Service account email used for generating the OIDC
    token on Cloud Tasks.  Defaults to the account associated with
    ``GOOGLE_APPLICATION_CREDENTIALS`` if provided.

Usage
-----

You can import these helpers wherever you need them::

    from automl_pyspark.gcp_helpers import get_bigquery_client, create_http_task

    bq_client = get_bigquery_client(project="my-project")
    # Perform BigQuery operations ...

    # Create a Cloud Tasks HTTP task
    create_http_task(
        project=os.environ["GCP_TASKS_PROJECT"],
        location=os.environ["GCP_TASKS_LOCATION"],
        queue=os.environ["GCP_TASKS_QUEUE"],
        target_path="/run-job",
        json_payload={"job_id": job_id, "config": job_config},
    )

These helpers are entirely optional and do not affect the default
behaviour of the AutoML pipeline.  They are provided to make it
easier to integrate with other Google Cloud services when running
inside Cloud Run or other container environments.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional

try:
    from google.cloud import bigquery  # type: ignore
    from google.cloud import tasks_v2  # type: ignore
    from google.oauth2 import service_account  # type: ignore
    from google.protobuf import timestamp_pb2, duration_pb2  # type: ignore
except ImportError:
    # These imports are optional.  If they fail it simply means
    # google-cloud packages are not installed.  Clients will not
    # initialise unless imported explicitly.
    bigquery = None  # type: ignore
    tasks_v2 = None  # type: ignore
    service_account = None  # type: ignore
    timestamp_pb2 = None  # type: ignore
    duration_pb2 = None  # type: ignore


def _load_credentials_from_env() -> Optional[service_account.Credentials]:
    """Load service account credentials from the file specified in
    GOOGLE_APPLICATION_CREDENTIALS.  Returns None if the environment
    variable is not set or the google-cloud library is unavailable.
    """
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not key_path.strip() or not service_account:
        return None
    try:
        return service_account.Credentials.from_service_account_file(key_path)
    except Exception:
        return None


def get_bigquery_client(project: str, credentials_path: Optional[str] = None):
    """Return a BigQuery client.

    If a credentials path is provided or the ``GOOGLE_APPLICATION_CREDENTIALS``
    environment variable is set, a service account credential will be
    loaded and used.  Otherwise the client will rely on application
    default credentials.

    Parameters
    ----------
    project : str
        The Google Cloud project ID.
    credentials_path : Optional[str]
        Path to a service account JSON key.  This parameter overrides
        the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable if
        provided.  If set to None, the function will fall back to the
        environment variable or ADC.

    Returns
    -------
    bigquery.Client
        An authenticated BigQuery client.
    """
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is not installed. Run 'pip install google-cloud-bigquery' to use this feature.")

    creds = None
    # Use explicit path if provided
    if credentials_path:
        if service_account is None:
            raise RuntimeError("google-auth library is required for service account credentials")
        creds = service_account.Credentials.from_service_account_file(credentials_path)
    else:
        creds = _load_credentials_from_env()

    return bigquery.Client(project=project, credentials=creds)


class CloudTasksSingleton:
    """Singleton wrapper for ``tasks_v2.CloudTasksClient``.

    Cloud Tasks clients maintain gRPC connections and should ideally
    be reused.  This singleton ensures that only one client is
    created per process.  When a service account key is provided via
    ``GOOGLE_APPLICATION_CREDENTIALS``, it will be used to
    authenticate the client.  Otherwise the client uses default
    credentials.
    """

    _instance: Optional[CloudTasksSingleton] = None
    _lock = threading.Lock()

    def __new__(cls) -> tasks_v2.CloudTasksClient:
        if tasks_v2 is None:
            raise RuntimeError(
                "google-cloud-tasks is not installed. Run 'pip install google-cloud-tasks' to use Cloud Tasks"
            )
        with cls._lock:
            if cls._instance is None:
                creds = _load_credentials_from_env()
                if creds:
                    client = tasks_v2.CloudTasksClient(credentials=creds)
                else:
                    client = tasks_v2.CloudTasksClient()
                cls._instance = super().__new__(cls)
                cls._instance.client = client
            return cls._instance.client


def create_http_task(
    project: str,
    location: str,
    queue: str,
    target_path: str,
    json_payload: Dict,
    scheduled_seconds_from_now: Optional[int] = None,
    task_id: Optional[str] = None,
    deadline_in_seconds: Optional[int] = None,
    service_account_email: Optional[str] = None,
) -> "tasks_v2.Task":
    """Create and enqueue an HTTP POST task for Cloud Run.

    This helper constructs a Cloud Tasks HTTP task that will POST a JSON
    payload to the specified Cloud Run endpoint.  The OIDC token used
    for authentication is automatically generated from the service
    account associated with the credentials loaded via
    ``GOOGLE_APPLICATION_CREDENTIALS`` or the email provided.

    Parameters
    ----------
    project : str
        GCP project ID where the queue is located.
    location : str
        Location/region of the queue (e.g. ``us-east1``).
    queue : str
        ID of the Cloud Tasks queue.
    target_path : str
        Path on the Cloud Run service to which the task should send
        the POST request.  This is appended to the ``CLOUD_RUN_BASE_URL``.
    json_payload : Dict
        JSON payload to include in the POST body.
    scheduled_seconds_from_now : Optional[int]
        Delay execution of the task by this many seconds.
    task_id : Optional[str]
        Custom task ID.  If not provided, Cloud Tasks will auto-generate one.
    deadline_in_seconds : Optional[int]
        Deadline for task execution.
    service_account_email : Optional[str]
        Email of the service account to use for authentication.  If
        omitted, the account associated with the credentials file (if
        any) will be used.

    Returns
    -------
    tasks_v2.Task
        The created task.
    """
    if tasks_v2 is None or timestamp_pb2 is None or duration_pb2 is None:
        raise RuntimeError(
            "google-cloud-tasks or protobuf dependencies are not installed. Run 'pip install google-cloud-tasks google-cloud-core'"
        )
    client = CloudTasksSingleton()
    # Construct the full URL for the task
    base_url = os.getenv("CLOUD_RUN_BASE_URL")
    if not base_url:
        raise RuntimeError("CLOUD_RUN_BASE_URL environment variable must be set to create Cloud Tasks HTTP requests.")
    url = base_url.rstrip("/") + "/" + target_path.lstrip("/")
    # Determine which service account to use for the OIDC token
    if service_account_email is None:
        # Attempt to derive from credentials file
        creds = _load_credentials_from_env()
        if creds and hasattr(creds, "service_account_email"):
            service_account_email = creds.service_account_email
    # Create task definition
    http_request = tasks_v2.HttpRequest(
        http_method=tasks_v2.HttpMethod.POST,
        url=url,
        headers={"Content-type": "application/json"},
        body=json.dumps(json_payload).encode("utf-8"),
    )
    # Attach OIDC token if service account email is available
    if service_account_email:
        http_request.oidc_token = tasks_v2.OidcToken(
            service_account_email=service_account_email,
        )
    # Build the task object
    task = tasks_v2.Task(
        http_request=http_request,
    )
    # Assign a custom name if provided
    if task_id:
        task.name = client.task_path(project, location, queue, task_id)
    # Schedule for future execution if needed
    if scheduled_seconds_from_now is not None:
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(datetime.utcnow() + timedelta(seconds=scheduled_seconds_from_now))
        task.schedule_time = timestamp
    # Set deadline if provided
    if deadline_in_seconds is not None:
        duration = duration_pb2.Duration()
        duration.FromSeconds(deadline_in_seconds)
        task.dispatch_deadline = duration
    # Enqueue the task
    response = client.create_task(
        tasks_v2.CreateTaskRequest(
            parent=client.queue_path(project, location, queue),
            task=task,
        )
    )
    return response
