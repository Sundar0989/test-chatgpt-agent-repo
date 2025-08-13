#!/bin/bash

# Stop and remove existing container if it exists
docker stop rapid_modeler 2>/dev/null || true
docker rm rapid_modeler 2>/dev/null || true

# Build the Docker image
echo "🔨 Building Docker image..."
docker build . -t rapid_modeler

# Run the container with service account key and proper environment variables
echo "🚀 Starting Rapid Modeler container..."
docker run -p 8080:8080 \
  -v $(pwd)/prism-dev-dq-sa-key-21.json:/app/automl_pyspark/prism-dev-dq-sa-key-21.json:ro \
  -v $(pwd)/automl_pyspark/automl_results:/app/automl_pyspark/automl_results \
  -v $(pwd)/automl_pyspark/automl_jobs:/app/automl_pyspark/automl_jobs \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/automl_pyspark/prism-dev-dq-sa-key-21.json \
  -e GOOGLE_CLOUD_PROJECT=atus-prism-dev \
  --name rapid_modeler rapid_modeler:latest

echo "✅ Rapid Modeler container started successfully!"
echo "🌐 Access the application at: http://localhost:8080"
echo "🔑 Service account key mounted and configured"