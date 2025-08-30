#!/bin/bash

echo "🛑 Stopping Rapid Modeler container..."

# Stop the container
docker stop rapid_modeler 2>/dev/null || true

# Remove the container
docker rm rapid_modeler 2>/dev/null || true

echo "✅ Rapid Modeler container stopped and removed successfully!"
echo "🚀 To start again, run: ./run.sh"
