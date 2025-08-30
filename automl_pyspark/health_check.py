#!/usr/bin/env python3
"""
Health check endpoint for Rapid Modeler service
This can be used by Cloud Run to verify the service is healthy
"""

import json
import time
from datetime import datetime
import os

def health_check():
    """Perform health check and return status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "rapid-modeler",
        "version": "1.0.0",
        "environment": {
            "python_version": os.sys.version,
            "platform": os.sys.platform,
            "working_directory": os.getcwd()
        }
    }
    
    # Check if critical components are available
    try:
        import pyspark
        health_status["components"] = {
            "pyspark": "available",
            "pyspark_version": pyspark.__version__
        }
    except ImportError as e:
        health_status["components"] = {
            "pyspark": f"unavailable: {str(e)}"
        }
    
    try:
        import streamlit
        health_status["components"]["streamlit"] = "available"
        health_status["components"]["streamlit_version"] = streamlit.__version__
    except ImportError as e:
        health_status["components"]["streamlit"] = f"unavailable: {str(e)}"
    
    try:
        import xgboost
        health_status["components"]["xgboost"] = "available"
        health_status["components"]["xgboost_version"] = xgboost.__version__
    except ImportError as e:
        health_status["components"]["xgboost"] = f"unavailable: {str(e)}"
    
    # Check if we can access the working directory
    try:
        os.listdir(".")
        health_status["filesystem"] = "accessible"
    except Exception as e:
        health_status["filesystem"] = f"error: {str(e)}"
    
    return health_status

if __name__ == "__main__":
    """Run health check and print results."""
    try:
        result = health_check()
        print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        if result["status"] == "healthy":
            exit(0)
        else:
            exit(1)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, indent=2))
        exit(1)
