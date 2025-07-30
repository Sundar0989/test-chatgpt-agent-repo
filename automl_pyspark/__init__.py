"""
AutoML PySpark Package

A comprehensive automated machine learning package for PySpark that provides:
- Automated feature selection and engineering
- Multiple classification algorithms (Logistic Regression, Random Forest, Gradient Boosting, Neural Network)
- Model validation and performance metrics
- Model selection and comparison
- Production-ready scoring code generation

Author: AutoML PySpark Team
Version: 1.0.0
"""

from .classification import AutoMLClassifier
from .regression import AutoMLRegressor
from .clustering import AutoMLClusterer

# Also import modules for compatibility
from . import classification
from . import regression
from . import clustering

__all__ = [
    'AutoMLClassifier',
    'AutoMLRegressor', 
    'AutoMLClusterer',
    'classification',
    'regression',
    'clustering',
]

__version__ = '1.0.0'
__author__ = 'AutoML PySpark Team'
__license__ = 'MIT'
 