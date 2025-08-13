FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    default-jdk \
    ca-certificates \
    ca-certificates-java \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Update CA certificates
RUN update-ca-certificates

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r automl_pyspark/requirements.txt \
    && pip install --no-cache-dir -r requirements.txt || true \
    && pip install setuptools


# Expose port for Cloud Run
ENV PORT=8080
# Set the path to service account key (will be overridden at runtime)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/automl_pyspark/gcp-creds/service_account_key.json
# Set the GCP project ID
ENV GOOGLE_CLOUD_PROJECT=atus-prism-dev

# Set PySpark environment variables
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYSPARK_SUBMIT_ARGS="--driver-memory 2g --executor-memory 2g pyspark-shell"

# Set Maven and SSL environment variables
ENV MAVEN_OPTS="-Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true"
ENV JAVA_OPTS="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts -Djavax.net.ssl.trustStorePassword=changeit"

# Copy JAR files to avoid Maven download issues
RUN mkdir -p /app/automl_pyspark/libs
COPY automl_pyspark/libs/*.jar /app/automl_pyspark/libs/

WORKDIR /app/automl_pyspark
# Default command to run the Streamlit application
CMD ["streamlit", "run", "streamlit_automl_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]