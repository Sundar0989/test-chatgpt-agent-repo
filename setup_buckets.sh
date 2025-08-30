#!/bin/bash

# Setup script for Rapid Modeler GCS buckets
PROJECT_ID="atus-prism-dev"
REGION="us-east1"

# Bucket names - using existing bucket
TEMP_BUCKET="rapid_modeler_app"
RESULTS_BUCKET="rapid_modeler_app"

echo "🪣 Setting up GCS buckets for Rapid Modeler..."
echo "📋 Configuration:"
echo "   Project ID: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Temp Bucket: ${TEMP_BUCKET}"
echo "   Results Bucket: ${RESULTS_BUCKET}"

# Check if gcloud is configured
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ gcloud is not authenticated. Please run:"
    echo "   gcloud auth login"
    echo "   gcloud config set project ${PROJECT_ID}"
    exit 1
fi

# Check if project is set correctly
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo "⚠️ Current project is ${CURRENT_PROJECT}, setting to ${PROJECT_ID}..."
    gcloud config set project ${PROJECT_ID}
fi

# Check if the bucket exists
echo "🔍 Checking if bucket exists: ${TEMP_BUCKET}"
if gsutil ls -b gs://${TEMP_BUCKET} >/dev/null 2>&1; then
    echo "✅ Bucket already exists: gs://${TEMP_BUCKET}"
else
    echo "❌ Bucket does not exist: gs://${TEMP_BUCKET}"
    echo "   Please create the bucket manually or contact your GCP admin."
    exit 1
fi

# Create default folders in the existing bucket
echo "📁 Creating default folder structure in existing bucket..."

# Temp bucket folders
gsutil -m cp -r /dev/null gs://${TEMP_BUCKET}/jobs/ 2>/dev/null || echo "⚠️ Could not create jobs folder"
gsutil -m cp -r /dev/null gs://${TEMP_BUCKET}/temp/ 2>/dev/null || echo "⚠️ Could not create temp folder"
echo "✅ Created temp bucket folders"

# Results bucket folders
gsutil -m cp -r /dev/null gs://${TEMP_BUCKET}/results/ 2>/dev/null || echo "⚠️ Could not create results folder"
gsutil -m cp -r /dev/null gs://${TEMP_BUCKET}/models/ 2>/dev/null || echo "⚠️ Could not create models folder"
echo "✅ Created results bucket folders"

echo ""
echo "🎉 GCS bucket setup completed successfully!"
echo ""
echo "📋 Bucket Details:"
echo "   Temp Bucket: gs://${TEMP_BUCKET}"
echo "   Results Bucket: gs://${RESULTS_BUCKET}"
echo ""
echo "🔍 Verify bucket structure:"
echo "   gsutil ls gs://${TEMP_BUCKET}"
echo ""
echo "🚀 You can now deploy to Cloud Run using:"
echo "   ./gcp-push.sh"
