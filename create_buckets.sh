#!/bin/bash

# Script to verify and set up the existing rapid_modeler_app bucket
PROJECT_ID="atus-prism-dev"
REGION="us-east1"

echo "🪣 Using existing GCS bucket for Rapid Modeler..."
echo "Bucket: gs://rapid_modeler_app"

# Check if the bucket exists
if gsutil ls -b gs://rapid_modeler_app >/dev/null 2>&1; then
    echo "✅ Bucket exists and is accessible"
    
    # Create default folder structure
    echo "📁 Creating default folder structure..."
    gsutil -m cp -r /dev/null gs://rapid_modeler_app/jobs/ 2>/dev/null || echo "⚠️ Could not create jobs folder"
    gsutil -m cp -r /dev/null gs://rapid_modeler_app/temp/ 2>/dev/null || echo "⚠️ Could not create temp folder"
    gsutil -m cp -r /dev/null gs://rapid_modeler_app/results/ 2>/dev/null || echo "⚠️ Could not create results folder"
    gsutil -m cp -r /dev/null gs://rapid_modeler_app/models/ 2>/dev/null || echo "⚠️ Could not create models folder"
    
    echo "✅ Bucket setup completed!"
    echo "🚀 You can now deploy to Cloud Run using: ./gcp-push.sh"
else
    echo "❌ Bucket does not exist or is not accessible"
    echo "Please contact your GCP admin to create the bucket: rapid_modeler_app"
fi
