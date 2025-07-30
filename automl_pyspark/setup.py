"""
Setup script for AutoML PySpark Package

This package provides automated machine learning capabilities for PySpark,
including feature selection, model building, validation, and scoring code generation.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AutoML PySpark Package - Automated machine learning for PySpark"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="automl-pyspark",
    version="1.0.0",
    author="AutoML PySpark Team",
    author_email="automl-pyspark@example.com",
    description="Automated machine learning package for PySpark classification, regression, and clustering tasks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/automl-pyspark/automl-pyspark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pyspark>=3.3.0,<4.0.0",
        "pandas>=1.3.0,<3.0.0",
        "numpy>=1.21.0,<2.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "openpyxl>=3.0.0",
        "xlsxwriter>=3.0.0",
        "PyYAML>=6.0.0",
        "optuna>=3.0.0",
        "xgboost>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0"
        ]
    },
    include_package_data=True,
    package_data={
        "automl_pyspark": ["*.py", "*.md", "*.txt"],
    },
    entry_points={
        "console_scripts": [
            "automl-pyspark=automl_pyspark.cli:main",
        ],
    },
    keywords=[
        "automl",
        "machine-learning",
        "pyspark",
        "classification",
        "regression",
        "clustering",
        "feature-selection",
        "model-selection",
        "automated-ml",
    ],
    project_urls={
        "Bug Reports": "https://github.com/automl-pyspark/automl-pyspark/issues",
        "Source": "https://github.com/automl-pyspark/automl-pyspark",
        "Documentation": "https://automl-pyspark.readthedocs.io/",
        "Metrics & Visualization Guide": "https://github.com/automl-pyspark/automl-pyspark/blob/main/METRICS_AND_VISUALIZATION_GUIDE.md",
    },
) 