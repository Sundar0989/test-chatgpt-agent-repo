FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    openjdk-21-jdk-headless \
    ca-certificates \
    ca-certificates-java \
    openssl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update CA certificates
RUN update-ca-certificates

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY automl_pyspark/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core ML dependencies
RUN pip install --no-cache-dir \
    pyspark==3.5.6 \
    py4j==0.10.9.7 \
    xgboost==3.0.3 \
    scikit-learn==1.7.1 \
    scipy==1.16.1 \
    numpy==1.26.4 \
    pandas==2.3.1

# Copy project files first (needed for automl_pyspark installation)
COPY . .

# Install automl_pyspark package from local source
RUN pip install --no-cache-dir -e automl_pyspark/

# Install remaining requirements (excluding automl_pyspark since it's already installed)
RUN pip install --no-cache-dir -r requirements.txt || true

# Ensure streamlit is installed
RUN pip install --no-cache-dir streamlit==1.47.1

# Copy JAR files
RUN mkdir -p /app/automl_pyspark/libs
COPY automl_pyspark/libs/*.jar /app/automl_pyspark/libs/

# Note: setup.py is already available in automl_pyspark/ directory

# Copy and make startup script executable
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Set environment variables
ENV PLATFORM_ARCH=auto
ENV ENABLE_SYNAPSEML_LIGHTGBM=true
ENV ENABLE_SPARK_XGBOOST=true
ENV ENABLE_NATIVE_SPARK_ML=true
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYSPARK_SUBMIT_ARGS="--driver-memory 4g --executor-memory 4g pyspark-shell"

# Set the GCP project ID
ENV GOOGLE_CLOUD_PROJECT=atus-prism-dev

# Set Maven and SSL environment variables
ENV MAVEN_OPTS="-Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true"
ENV JAVA_OPTS="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts -Djavax.net.ssl.trustStorePassword=changeit -Xmx4g -Xms2g"

# Disable Maven downloads since we're using local JARs
ENV DISABLE_MAVEN_DOWNLOADS=true

# Set Streamlit environment variables for Cloud Run
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Verify installations
RUN python --version \
    && java -version \
    && python -c "import xgboost; print(f'✅ XGBoost {xgboost.__version__}')" \
    && python -c "import pyspark; print(f'✅ PySpark {pyspark.__version__}')" \
    && python -c "import synapse.ml; print('✅ SynapseML')" || echo "⚠️ SynapseML not available" \
    && python -c "import streamlit; print(f'✅ Streamlit {streamlit.__version__}')" \
    && streamlit --version

WORKDIR /app

# Create a health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/_stcore/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Use startup script instead of direct streamlit command
CMD ["/app/startup.sh"]