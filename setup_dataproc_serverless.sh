#!/bin/bash

# Setup Dataproc Serverless for AutoML PySpark
# This script sets up the necessary infrastructure for running Spark jobs on Dataproc Serverless

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some features may not work."
    fi
    
    print_success "Prerequisites check completed"
}

# Get project configuration
get_project_config() {
    print_status "Getting project configuration..."
    
    # Get current project
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        print_error "No project is set. Please run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
    
    # Get current region
    REGION=$(gcloud config get-value compute/region 2>/dev/null)
    if [ -z "$REGION" ]; then
        REGION="us-central1"
        print_warning "No region set, using default: $REGION"
    fi
    
    print_success "Project: $PROJECT_ID, Region: $REGION"
}

# Enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    APIs=(
        "dataproc.googleapis.com"
        "storage.googleapis.com"
        "bigquery.googleapis.com"
        "compute.googleapis.com"
        "iam.googleapis.com"
        "metastore.googleapis.com"
    )
    
    for api in "${APIs[@]}"; do
        if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
            print_status "API $api is already enabled"
        else
            print_status "Enabling API $api..."
            gcloud services enable "$api"
            print_success "API $api enabled"
        fi
    done
}

# Create Cloud Storage buckets
create_storage_buckets() {
    print_status "Creating Cloud Storage buckets..."
    
    # Create temp bucket for job files
    TEMP_BUCKET="automl-temp-${PROJECT_ID}"
    if gsutil ls -b "gs://$TEMP_BUCKET" &>/dev/null; then
        print_status "Bucket $TEMP_BUCKET already exists"
    else
        print_status "Creating bucket $TEMP_BUCKET..."
        gsutil mb -l "$REGION" "gs://$TEMP_BUCKET"
        print_success "Bucket $TEMP_BUCKET created"
    fi
    
    # Create results bucket
    RESULTS_BUCKET="automl-results-${PROJECT_ID}"
    if gsutil ls -b "gs://$RESULTS_BUCKET" &>/dev/null; then
        print_status "Bucket $RESULTS_BUCKET already exists"
    else
        print_status "Creating bucket $RESULTS_BUCKET..."
        gsutil mb -l "$REGION" "gs://$RESULTS_BUCKET"
        print_success "Bucket $RESULTS_BUCKET created"
    fi
    
    # Set bucket labels
    gsutil label ch -l "project:$PROJECT_ID" -l "environment:automl" "gs://$TEMP_BUCKET"
    gsutil label ch -l "project:$PROJECT_ID" -l "environment:automl" "gs://$RESULTS_BUCKET"
}

# Create service account
create_service_account() {
    print_status "Creating service account for Dataproc Serverless..."
    
    SA_NAME="automl-dataproc-sa"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    if gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
        print_status "Service account $SA_EMAIL already exists"
    else
        print_status "Creating service account $SA_EMAIL..."
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="AutoML Dataproc Serverless Service Account" \
            --description="Service account for running AutoML jobs on Dataproc Serverless"
        print_success "Service account created"
    fi
    
    # Grant necessary roles
    print_status "Granting necessary IAM roles..."
    
    ROLES=(
        "roles/dataproc.worker"
        "roles/storage.objectViewer"
        "roles/storage.objectCreator"
        "roles/bigquery.dataViewer"
        "roles/bigquery.jobUser"
        "roles/metastore.user"
    )
    
    for role in "${ROLES[@]}"; do
        print_status "Granting role $role..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role"
    done
    
    print_success "IAM roles granted"
}

# Create Dataproc Metastore
create_metastore() {
    print_status "Setting up Dataproc Metastore..."
    
    METASTORE_NAME="automl-metastore"
    METASTORE_URI="projects/${PROJECT_ID}/locations/${REGION}/services/${METASTORE_NAME}"
    
    if gcloud metastore services describe "$METASTORE_NAME" --location="$REGION" &>/dev/null; then
        print_status "Metastore $METASTORE_NAME already exists"
    else
        print_status "Creating metastore $METASTORE_NAME..."
        gcloud metastore services create "$METASTORE_NAME" \
            --location="$REGION" \
            --hive-metastore-version="3.1.2" \
            --tier="DEVELOPER"
        print_success "Metastore created"
    fi
}

# Create Dataproc Serverless environment
create_dataproc_environment() {
    print_status "Setting up Dataproc Serverless environment..."
    
    # This is mostly configuration - Dataproc Serverless doesn't require pre-creation
    # but we can set up some default configurations
    
    print_success "Dataproc Serverless environment is ready"
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration file..."
    
    cat > .env.dataproc << EOF
# Dataproc Serverless Environment Configuration
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION
GCP_TEMP_BUCKET=$TEMP_BUCKET
GCP_RESULTS_BUCKET=$RESULTS_BUCKET
GCP_SERVICE_ACCOUNT_EMAIL=$SA_EMAIL

# Dataproc Serverless Configuration
DATAPROC_SERVERLESS_ENABLED=true
DATAPROC_SPARK_VERSION=3.4
DATAPROC_RUNTIME_VERSION=1.0

# Resource Limits
DATAPROC_MAX_EXECUTORS=100
DATAPROC_MIN_EXECUTORS=2
DATAPROC_EXECUTOR_MEMORY=4g
DATAPROC_EXECUTOR_CPU=2
EOF
    
    print_success "Environment file created: .env.dataproc"
}

# Test Dataproc Serverless
test_dataproc_serverless() {
    print_status "Testing Dataproc Serverless setup..."
    
    # Create a simple test job
    TEST_JOB_NAME="test-automl-job"
    
    print_status "Creating test job..."
    
    # This would create a simple test job to verify the setup
    # For now, we'll just check if the environment is accessible
    
    if gcloud dataproc batches list --location="$REGION" &>/dev/null; then
        print_success "Dataproc Serverless is accessible"
    else
        print_error "Dataproc Serverless is not accessible. Please check your setup."
        exit 1
    fi
}

# Main execution
main() {
    print_status "Starting Dataproc Serverless setup for AutoML PySpark..."
    
    check_prerequisites
    get_project_config
    enable_apis
    create_storage_buckets
    create_service_account
    create_metastore
    create_dataproc_environment
    create_env_file
    test_dataproc_serverless
    
    print_success "Dataproc Serverless setup completed successfully!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Source the environment file: source .env.dataproc"
    print_status "2. Update your AutoML configuration to use Dataproc Serverless"
    print_status "3. Deploy your Cloud Run service with the new configuration"
    print_status ""
    print_status "Environment variables have been saved to .env.dataproc"
    print_status "Project ID: $PROJECT_ID"
    print_status "Region: $REGION"
    print_status "Temp Bucket: $TEMP_BUCKET"
    print_status "Results Bucket: $RESULTS_BUCKET"
    print_status "Service Account: $SA_EMAIL"
}

# Run main function
main "$@"
