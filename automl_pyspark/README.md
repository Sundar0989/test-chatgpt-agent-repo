# AutoML PySpark

A comprehensive, production-grade Automated Machine Learning (AutoML) system for PySpark, supporting classification, regression, and clustering with advanced metrics, visualizations, and a modern Streamlit dashboard.

---

## 🌟 Features

- **Automated Model Selection** for classification, regression, and clustering
- **Comprehensive Metrics**: 35+ metrics across all task types
- **Rich Visualizations**: Actual vs Predicted, Residuals, Elbow Plot, Cluster Visualizations, and more
- **Intelligent Model Comparison**: Multi-criteria, statistical significance, overfitting checks
- **Hyperparameter Tuning**: Grid, Random, and Bayesian (Optuna) search
- **Streamlit Dashboard**: Interactive, task-adaptive UI for results, plots, and logs
- **Parallel Job Execution**: Multi-user, multi-job support
- **Production Ready**: Robust error handling, logging, and artifact management

---

## 🚀 Quick Start

Follow these steps to get started with AutoML PySpark:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optimize for hyperparameter tuning (recommended):**
   ```python
   # Reduce "Broadcasting large task binary" warnings
   python optimize_hyperparameter_tuning.py tips
   ```

3. **Basic usage:**
   ```python
   from classification.automl_classifier import AutoMLClassifier
   
   # Create classifier with optimized Spark session
   automl = AutoMLClassifier(preset='comprehensive')
   
   # Train model
   automl.fit(train_data='your_data.csv', target_column='target')
   ```

## 📊 Feature Importance Analysis

Both classification and regression tasks now include **comprehensive feature importance analysis** that helps you understand which features contribute most to your model's predictions.

### 🎯 What's Generated Automatically

During the feature selection phase, the AutoML pipeline automatically generates:

1. **📊 Excel File** (`feature_importance.xlsx`): 
   - Feature names and importance scores
   - Top 30 most important features
   - Downloadable spreadsheet format

2. **📈 Visualization** (`Features_selected_for_modeling.png`):
   - Horizontal bar chart with color gradients
   - Statistical summaries (mean, std, top feature)
   - Value labels on each bar
   - Professional styling with grid lines

# Download the BigQuery connector JAR
wget https://storage.googleapis.com/hadoop-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.32.2.jar

# Move to a directory accessible by Spark
mkdir -p ~/spark-jars
mv spark-bigquery-with-dependencies_2.12-0.32.2.jar ~/spark-jars/

### 🔍 Accessing Feature Importance

**Method 1: Using the AutoML Object**
```python
# After training your model
automl = AutoMLRegressor()  # or AutoMLClassifier()
automl.fit(train_data='your_data.csv', target_column='target')

# Get feature importance data
feature_importance = automl.get_feature_importance()

# This displays:
# ✅ Feature importance Excel file found: /path/to/feature_importance.xlsx
# ✅ Feature importance plot found: /path/to/Features_selected_for_modeling.png
# 🎯 Top 10 Most Important Features:
# 1. feature_name_1             | 0.234567
# 2. feature_name_2             | 0.198765
# ...
```

**Method 2: Direct File Access**
```python
import pandas as pd

# Load Excel file directly
df = pd.read_excel('automl_output/feature_importance.xlsx')
print(df.head(10))  # Top 10 features

# Display plot
from IPython.display import Image, display
display(Image('automl_output/Features_selected_for_modeling.png'))
```

**Method 3: Manual Generation**
```python
from feature_selection import save_feature_importance
import pandas as pd

# Create your own feature importance data
importance_data = {
    'name': ['feature1', 'feature2', 'feature3'],
    'Importance_Score': [0.5, 0.3, 0.2]
}
df = pd.DataFrame(importance_data)

# Generate files
excel_path, plot_path = save_feature_importance(
    output_dir='my_output', 
    user_id='my_user', 
    model_literal='my_model', 
    importance_df=df
)
```

### 🎨 Enhanced Visualization Features

The feature importance plots now include:

- **Color Gradients**: More important features are darker blue
- **Value Labels**: Exact importance scores displayed on bars
- **Statistics Box**: Mean, standard deviation, and top feature name
- **Smart Sizing**: Adapts to number of features (max 30 shown)
- **High Resolution**: 300 DPI for publication-quality images
- **Grid Lines**: For easier reading of values

### 🔬 Behind the Scenes

**For Classification**: Uses Random Forest Classifier during feature selection
**For Regression**: Uses Random Forest Regressor during feature selection

The feature importance is calculated using the **Mean Decrease in Impurity** method:
- Features that split the data most effectively get higher scores
- Scores are normalized so they sum to 1.0
- Only available for numerical features (categorical features are excluded)

### 📁 File Locations

Feature importance files are saved in your output directory:
```
your_output_directory/
├── feature_importance.xlsx           # 📊 Spreadsheet with scores
├── Features_selected_for_modeling.png # 📈 Visualization
└── other_model_files...
```

### 🧪 Testing Feature Importance

Run the test script to verify functionality:
```bash
python test_feature_importance.py
```

This script will:
- ✅ Test both classification and regression
- ✅ Generate sample files
- ✅ Demonstrate manual usage
- ✅ Check for existing files

### 🎯 Use Cases

**Understanding Your Model**:
- Which features drive predictions?
- Are important features domain-relevant?
- Can you simplify by removing low-importance features?

**Feature Engineering**:
- Focus engineering efforts on high-importance features
- Identify redundant or noisy features
- Guide data collection for future models

**Model Explanation**:
- Share insights with stakeholders
- Validate business logic
- Support regulatory requirements

**Example Output**:
```
📊 Feature Importance for Regression Model
==================================================
💾 Feature importance files should be in: automl_output
✅ Feature importance Excel file found: automl_output/feature_importance.xlsx
✅ Feature importance plot found: automl_output/Features_selected_for_modeling.png

🎯 Top 10 Most Important Features:
----------------------------------------
 1. account_balance           | 0.245613
 2. credit_score              | 0.198347
 3. income_annual             | 0.156789
 4. employment_years          | 0.123456
 5. debt_to_income_ratio      | 0.098765
 6. previous_defaults         | 0.087654
 7. age                       | 0.076543
 8. education_level           | 0.054321
 9. marital_status            | 0.043210
10. home_ownership           | 0.034567
```

### 🎯 Hyperparameter Tuning Optimizations

### Reducing "Broadcasting Large Task Binary" Warnings

If you see warnings like:
```
25/07/22 19:05:46 WARN DAGScheduler: Broadcasting large task binary with size 1001.1 KiB
```

**Solution:** Use the built-in optimization utility:

```python
# Automatic optimization (recommended)
from optimize_hyperparameter_tuning import optimize_spark_for_hyperparameter_tuning

# This automatically applies optimizations based on model type and trial count
spark = optimize_spark_for_hyperparameter_tuning(
    model_type="gradient_boosting",  # or "random_forest", "xgboost", etc.
    n_trials=50
)

# Use with AutoML
automl = AutoMLRegressor(spark_session=spark)
```

### Manual Spark Optimization

For manual control:

```python
from spark_optimization_config import apply_tuning_optimizations, create_optimized_spark_session

# Create optimized session
spark = create_optimized_spark_session("AutoML Tuning")

# Apply hyperparameter tuning specific optimizations
apply_tuning_optimizations(spark, aggressive=True)  # For tree-based models

# Use with AutoML
automl = AutoMLRegressor(spark_session=spark)
```

### Configuration Recommendations

**For Tree-Based Models (Gradient Boosting, Random Forest, XGBoost):**
- Use `aggressive=True` optimizations
- Limit trials: `optuna_trials: 20-50` instead of 100+
- Reduce model complexity: `maxDepth: 3-10`, `maxIter: 50-100`

**For Linear Models:**
- Standard optimizations sufficient
- Can use higher trial counts: `optuna_trials: 50-100`

### Optimization Utility Script

Run the optimization utility for guided setup:

```bash
# Interactive mode
python optimize_hyperparameter_tuning.py

# Test optimizations
python optimize_hyperparameter_tuning.py test bank.csv gradient_boosting 3

# Show optimization tips
python optimize_hyperparameter_tuning.py tips
```

---

## 📖 Usage Guide

### **Job Submission Workflow**
1. Configure job (data, target, models, hyperparameters) in the Streamlit UI
2. Submit job and monitor progress in real time
3. Explore results, metrics, plots, scoring code, and logs in the dashboard

### **Results Viewing Workflow**
- **Model Performance**: Compare models with interactive charts
- **Detailed Results**: Browse all generated files and artifacts
- **Validation Plots**: Task-specific diagnostic plots (see below)
- **Scoring Code**: Download ready-to-use scoring scripts
- **Logs**: View execution logs and error details

---

## 📊 Metrics & Visualization Summary

### 🎯 **Classification**
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, Specificity, Log Loss, MCC, etc.
- **Visualizations**: 
  - **Performance Charts**: Grouped bar charts comparing metrics across datasets (train/valid/test/oot1/oot2)
  - **Diagnostic Plots**: Confusion Matrix, ROC Curve, Precision-Recall Curve  
  - **Feature Analysis**: Feature Importance plots and Excel exports
  - **KS Analysis**: Decile tables for binary classification

### 📈 **Regression** 
- **Metrics**: RMSE, MAE, R², MSE, MAPE, Explained Variance, Max Error, Median AE, MSLE, Residual Stats
- **Visualizations**: 
  - **Performance Charts**: Grouped bar charts comparing RMSE/R²/MAE across datasets (train/valid/test/oot1/oot2)
  - **Diagnostic Plots**: Actual vs Predicted, Residuals Plot, Residuals Distribution, Q-Q Plot, Error Analysis
  - **Feature Analysis**: Feature Importance plots and Excel exports (same as classification)
  - **Consistency**: Visualization style matches classification for easy comparison

### 🔍 **Clustering**
- **Metrics**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin, Inertia, Cluster Sizes, Balance, Inter-Cluster Distances
- **Visualizations**: Elbow Plot (with k-recommendation), PCA/t-SNE Cluster Visualization, Silhouette Analysis, Cluster Statistics, Cluster Centers Heatmap

### 🎨 **Visualization Consistency**
Both **Classification** and **Regression** now feature:
- ✅ **Grouped Bar Charts**: Compare model performance across multiple datasets
- ✅ **Dataset Color Coding**: Consistent colors for Train/Valid/Test/OOT1/OOT2
- ✅ **Full-Height Charts**: 500px height for better readability  
- ✅ **Performance Indicators**: Clear "Higher is Better" / "Lower is Better" labels
- ✅ **Overfitting Detection**: Easy comparison of train vs validation performance
- ✅ **Model Stability**: Track performance degradation across time periods

See [`METRICS_AND_VISUALIZATION_GUIDE.md`](./METRICS_AND_VISUALIZATION_GUIDE.md) for full details.

---

## 🌐 Streamlit Dashboard Highlights

- **Task-Adaptive Tabs**: Only relevant metrics and plots shown for each task
- **Validation Plots Tab**: All diagnostic plots, categorized and interactive
- **Elbow Analysis Panel**: For clustering, with multi-metric k recommendation
- **Enhanced Metrics Tables**: Color-coded, comprehensive, and easy to interpret
- **File Explorer**: Browse and view all result files and artifacts
- **Real-Time Logs**: Debug and monitor jobs easily

---

## 🗂️ Directory Structure

```
automl_pyspark/
├── classification/           # Classification logic
├── clustering/               # Clustering logic
├── regression/               # Regression logic
├── streamlit_automl_app.py   # Main Streamlit dashboard
├── requirements.txt          # Core dependencies
├── streamlit_requirements.txt# Streamlit dependencies
├── METRICS_AND_VISUALIZATION_GUIDE.md # Metrics/plots documentation
├── ... (other modules, configs, and data)
```

---

## ⚙️ Configuration

- **config.yaml**: All parameters for data, models, tuning, and environment
- **Presets**: Quick, Comprehensive, and Custom modes
- **Environment Modes**: Development, Staging, Production

---

## 🐛 Troubleshooting & Support

- **Configuration Issues**: Check `config.yaml` syntax and file paths
- **Model Training Errors**: Review logs in the dashboard
- **Performance Problems**: Adjust environment/preset settings
- **Documentation**: See [`METRICS_AND_VISUALIZATION_GUIDE.md`](./METRICS_AND_VISUALIZATION_GUIDE.md)

---

## 📄 License

MIT License

---

## 🤝 Contributing

Pull requests and issues are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) if available.

---

## 📞 Contact

- **Bug Reports**: https://github.com/automl-pyspark/automl-pyspark/issues
- **Documentation**: https://automl-pyspark.readthedocs.io/ 

## ⚠️ Troubleshooting Common Issues

### Spark Deprecation Warnings

If you see warnings like:
```
WARN SQLConf: The SQL config 'spark.sql.adaptive.coalescePartitions.minPartitionNum' has been deprecated in Spark v3.2 and may be removed in the future. Use 'spark.sql.adaptive.coalescePartitions.minPartitionSize' instead.
```

**Quick Fix:**
```bash
# Run the deprecation warning fix utility
python fix_spark_deprecation_warnings.py
```

**What this does:**
- ✅ Diagnoses current Spark configuration
- ✅ Applies fixes for deprecated parameters  
- ✅ Creates a clean Spark session
- ✅ Generates optimized startup script
- ✅ Tests that warnings are resolved

**Manual Fix:**
If you prefer to fix manually, update your Spark session creation:
```python
from spark_startup_no_warnings import create_optimized_spark_session
spark = create_optimized_spark_session("YourAppName")
```

### Memory Issues

If you encounter memory errors during training:
```bash
# Use aggressive optimizations
python optimize_hyperparameter_tuning.py --aggressive
```

### Large Task Binary Warnings

For "Broadcasting large task binary" warnings during hyperparameter tuning:
- ✅ **Automatically handled** - optimizations are applied during tuning
- ✅ Reduces broadcast thresholds and buffers
- ✅ Enables periodic cleanup and garbage collection 