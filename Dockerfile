FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r automl_pyspark/requirements.txt \
    && pip install --no-cache-dir -r requirements.txt || true

# Expose port for Cloud Run
ENV PORT=8080

# Default command to run the Streamlit application
CMD ["streamlit", "run", "automl_pyspark/streamlit_automl_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]