#!/bin/bash

# Startup script for Rapid Modeler on Cloud Run
echo "🚀 Starting Rapid Modeler service..."

# Set environment variables
export STREAMLIT_SERVER_PORT=8080
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Wait for any background processes to complete
sleep 5

# Start Streamlit app in background
echo "📱 Starting Streamlit application..."

streamlit run automl_pyspark/streamlit_automl_app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to start..."
sleep 15

# Check if Streamlit is running
if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "❌ Streamlit failed to start!"
    exit 1
fi

# Wait for Streamlit to be ready
echo "🔍 Waiting for Streamlit to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8080/_stcore/health >/dev/null 2>&1; then
        echo "✅ Streamlit is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Streamlit failed to become ready within timeout!"
        kill $STREAMLIT_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

# Keep the script running and monitor Streamlit
echo "🔄 Monitoring Streamlit process..."
wait $STREAMLIT_PID
