# AutoML Streamlit Application

A comprehensive web interface for the AutoML PySpark system that allows users to submit jobs, monitor progress, and view results through an intuitive web interface.

## 🌟 Features

### 📋 **Job Submission Page**
- **Complete Configuration**: Expose all parameters from `config.yaml`
- **Environment Selection**: Development, Staging, Production environments
- **Preset Support**: Quick and Comprehensive presets
- **Model Selection**: Choose from 7+ machine learning models
- **Advanced Parameters**: Hyperparameter tuning, cross-validation, data balancing
- **Parallel Job Support**: Multiple users can submit jobs simultaneously
- **Real-time Status**: Background job execution with status tracking

### 📊 **Results Viewing Page**
- **Multi-user Support**: Dropdown to select different users' model outputs
- **Comprehensive Results**: Model performance, detailed metrics, artifacts
- **Scoring Code Display**: View and download generated scoring scripts
- **Log Viewing**: Complete execution logs and error tracking
- **Interactive Charts**: Performance comparisons using Plotly
- **File Explorer**: Browse all generated files and artifacts

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Streamlit app dependencies
pip install -r streamlit_requirements.txt

# Ensure AutoML system dependencies are installed
pip install -r requirements.txt
```

### 2. Start the Application

```bash
streamlit run streamlit_automl_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### **Job Submission Workflow**

1. **Navigate to "🚀 Submit Job"** page
2. **Configure Basic Settings** (sidebar):
   - User ID (your identifier)
   - Model Name (project name)
   - Environment (development/staging/production)
   - Preset (quick/comprehensive/custom)

3. **Configure Data & Target** (Tab 1):
   - Select CSV/Parquet data file
   - Specify target column
   - Set data processing parameters

4. **Select Models** (Tab 2):
   - Choose from: Logistic Regression, Random Forest, XGBoost, LightGBM, etc.
   - Enable/disable specific models

5. **Set Training Parameters** (Tab 3):
   - Cross-validation settings
   - Data split ratios
   - Hyperparameter tuning configuration

6. **Advanced Configuration** (Tab 4):
   - Data balancing options
   - Performance settings
   - Resource limits

7. **Submit Job**: Click "🚀 Submit AutoML Job"
   - Job runs in background
   - Unique job ID generated
   - Status tracking enabled

### **Results Viewing Workflow**

1. **Navigate to "📊 View Results"** page
2. **Select Job** (sidebar):
   - Filter by status (All/Completed/Running/Failed)
   - Choose specific job from dropdown
   - View job information

3. **Explore Results** (Tabs):
   - **📈 Model Performance**: Metrics comparison, charts
   - **📋 Detailed Results**: File explorer, artifacts viewer
   - **💾 Scoring Code**: Generated scoring scripts
   - **📜 Logs**: Execution logs, error details

## 🗂️ Directory Structure

```
automl_pyspark/
├── streamlit_automl_app.py          # Main Streamlit application
├── streamlit_requirements.txt       # Streamlit dependencies
├── config.yaml                      # AutoML configuration
├── automl_jobs/                     # Job configurations and status
│   ├── {job_id}.json               # Job configuration
│   ├── {job_id}_status.txt         # Job status
│   ├── {job_id}_script.py          # Generated execution script
│   └── {job_id}_error.log          # Error logs (if failed)
└── automl_results/                  # Job results and outputs
    └── {job_id}/                    # Individual job results
        ├── model_info.pkl           # Model metadata
        ├── {model_type}_model/      # Trained models
        ├── *_metrics.json           # Model metrics
        ├── *_scoring.py             # Scoring scripts
        └── *.log                    # AutoML logs
```

## ⚙️ Configuration

### **Environment Presets**

- **Development**: Fast training, limited models, debugging enabled
- **Staging**: Balanced performance, moderate complexity
- **Production**: Full training, all models, optimized for accuracy

### **Quick Presets**

- **Quick**: Fast iteration (Logistic + Random Forest only)
- **Comprehensive**: Full model training with hyperparameter tuning

### **Custom Configuration**

All parameters from `config.yaml` are exposed in the UI:

- **Models**: 7+ algorithms (Logistic, Random Forest, XGBoost, etc.)
- **Hyperparameter Tuning**: Optuna, Random Search, Grid Search
- **Cross Validation**: 2-10 folds
- **Data Balancing**: SMOTE, Oversampling, etc.
- **Performance**: Parallel jobs, timeouts, memory limits

## 🔧 Advanced Features

### **Parallel Job Execution**
- Multiple users can submit jobs simultaneously
- Background processing with threading
- Status tracking and real-time updates

### **Intelligent Results Display**
- Auto-detection of result files
- Categorized file explorer
- Interactive performance charts
- Code syntax highlighting

### **Error Handling**
- Comprehensive error logging
- User-friendly error messages
- Automatic retry mechanisms
- Graceful failure handling

## 🐛 Troubleshooting

### **Common Issues**

1. **"Configuration file not found"**
   - Ensure `config.yaml` exists in the same directory
   - Check file permissions

2. **"No data files found"**
   - Place CSV/Parquet files in the main directory
   - Or use the text input to specify file path

3. **Job fails immediately**
   - Check the error log in the "📜 Logs" tab
   - Verify data file format and target column

4. **Scoring scripts not generated**
   - Ensure job completed successfully
   - Check if models were trained successfully

### **Performance Tips**

1. **Development Environment**: Use for quick testing
2. **Quick Preset**: For rapid prototyping
3. **Limit Models**: Disable heavy models (Neural Network, XGBoost) for speed
4. **Reduce Hyperparameter Trials**: Use fewer trials for faster results

## 🔄 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Job Scheduler   │───▶│  AutoML Engine  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Results Viewer  │◀───│  File Manager    │◀───│  Model Outputs  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📞 Support

- **Configuration Issues**: Check `config.yaml` syntax
- **Model Training Errors**: Review logs in Results page
- **Performance Problems**: Adjust environment/preset settings
- **UI Issues**: Refresh browser, check console logs

## 🎯 Next Steps

1. **Start the app**: `streamlit run streamlit_automl_app.py`
2. **Submit your first job**: Use the quick preset
3. **Monitor progress**: Check the Results page
4. **Download scoring code**: Use generated scripts for production
5. **Scale up**: Submit multiple jobs for different datasets

Happy AutoML! 🤖✨ 