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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For the Streamlit dashboard:

```bash
pip install -r streamlit_requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run streamlit_automl_app.py
```

The app will open in your browser at `http://localhost:8501`

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
- **Visualizations**: Confusion Matrix, ROC Curve, Precision-Recall Curve, Feature Importance

### 📈 **Regression**
- **Metrics**: RMSE, MAE, R², MSE, MAPE, Explained Variance, Max Error, Median AE, MSLE, Residual Stats
- **Visualizations**: Actual vs Predicted, Residuals Plot, Residuals Distribution, Q-Q Plot, Error Analysis

### 🔍 **Clustering**
- **Metrics**: Silhouette Score, Calinski-Harabasz, Davies-Bouldin, Inertia, Cluster Sizes, Balance, Inter-Cluster Distances
- **Visualizations**: Elbow Plot (with k-recommendation), PCA/t-SNE Cluster Visualization, Silhouette Analysis, Cluster Statistics, Cluster Centers Heatmap

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