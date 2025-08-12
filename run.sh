#!/bin/bash
docker stop rapid_modeler && docker rm rapid_modeler
docker build . -t rapid_modeler
docker run -p 8080:8080 \
  -v ./automl_pyspark/gcp-creds/service_account_key.json:/root/.config/gcloud/key.json \
  -v ./automl_pyspark/automl_results:/app/automl_pyspark/automl_results \
  -v ./automl_pyspark/automl_jobs:/app/automl_pyspark/automl_jobs \
  -e GOOGLE_CLOUD_PROJECT=atus-prism-dev \
  --name rapid_modeler rapid_modeler:latest