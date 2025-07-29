"""
    validation_and_plots.py

    This code is used to perform
    1. Model validation
    2. Generate ROC chart
    3. Generate KS Chart
    4. Confusion matrix
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import glob
import os
import time
import pandas as pd
import seaborn as sns
from pandas import ExcelWriter
import xlsxwriter

# Try to import from package, fall back to local import
try:
    from automl_pyspark.metrics_calculator import calculate_metrics, highlight_max
except ImportError:
    # Fallback to relative import  
    from metrics_calculator import calculate_metrics, highlight_max


# Generate ROC chart
def draw_roc_plot(output_dir, model_type, y_score, y_true, data_type):
    """
    Generate ROC plot and save to specified output directory.
    
    Args:
        output_dir (str): Directory to save the plot
        model_type (str): Type of model
        y_score (array): Predicted probabilities
        y_true (array): True labels
        data_type (str): Type of data (train/test/validation)
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(f'{model_type} Model - ROC for {data_type} data')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='lower right')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'{model_type} Model - ROC for {data_type} data.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f'ROC plot saved: {filepath}')


# Generate KS Chart
def draw_ks_plot(output_dir, model_type):
    """
    Generate KS plot and save to specified output directory.
    
    Args:
        output_dir (str): Directory to save the plot
        model_type (str): Type of model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    writer = ExcelWriter(os.path.join(output_dir, 'KS_Charts.xlsx'), engine='xlsxwriter')

    # Look for KS files in the output directory
    pattern = os.path.join(output_dir, f'KS {model_type} Model*.xlsx')
    print(f"Looking for KS files with pattern: {pattern}")
    
    files_to_remove = []  # Track files that should be removed
    
    found_files = glob.glob(pattern)
    print(f"Found {len(found_files)} KS files to process")
    
    if not found_files:
        print("No KS files found to consolidate")
        writer.close()
        return
    
    for filename in found_files:
        try:
            excel_file = pd.ExcelFile(filename)
            (_, f_name) = os.path.split(filename)
            (f_short_name, _) = os.path.splitext(f_name)
            
            for sheet_name in excel_file.sheet_names:
                df_excel = pd.read_excel(filename, sheet_name=sheet_name)
                # Apply styling and save to Excel
                df_excel = df_excel.style.apply(highlight_max, subset=['spread'], color='#e6b71e')
                
                # Create a shorter, unique sheet name that fits Excel's 31-character limit
                # Extract model type and data type from filename
                # Example: "KS random_forest_default Model train.xlsx" -> "RF_default_train"
                base_name = os.path.basename(filename)
                name_parts = base_name.replace('.xlsx', '').split(' ')
                
                # Extract model type and data type
                if len(name_parts) >= 4:
                    model_part = name_parts[1]  # e.g., "random_forest_default"
                    data_type = name_parts[-1]  # e.g., "train"
                    
                    # Create abbreviated model name
                    if 'random_forest' in model_part:
                        model_abbr = 'RF'
                    elif 'gradient_boosting' in model_part:
                        model_abbr = 'GB'
                    elif 'decision_tree' in model_part:
                        model_abbr = 'DT'
                    elif 'neural_network' in model_part:
                        model_abbr = 'NN'
                    elif 'logistic' in model_part:
                        model_abbr = 'LR'
                    elif 'xgboost' in model_part:
                        model_abbr = 'XGB'
                    elif 'lightgbm' in model_part:
                        model_abbr = 'LGB'
                    else:
                        model_abbr = model_part[:3].upper()
                    
                    # Add suffix if it's default or tuned
                    if 'default' in model_part:
                        model_abbr += '_def'
                    elif 'tuned' in model_part:
                        model_abbr += '_tuned'
                    
                    # Create final sheet name
                    sheet_name_final = f"{model_abbr}_{data_type}"
                    
                    # Ensure it's within 31 characters
                    if len(sheet_name_final) > 31:
                        sheet_name_final = sheet_name_final[:31]
                    
                    print(f"Sheet name: '{f_short_name}' -> '{sheet_name_final}'")
                else:
                    # Fallback to truncated name if parsing fails
                    sheet_name_final = f_short_name[:31] if len(f_short_name) > 31 else f_short_name
                    print(f"Fallback sheet name: '{f_short_name}' -> '{sheet_name_final}'")
                
                df_excel.to_excel(writer, sheet_name_final, index=False)
                
                # Get the worksheet and apply conditional formatting
                worksheet = writer.sheets[sheet_name_final]
                
                # Apply conditional formatting to columns C and E (good and bad counts)
                # Use xlsxwriter syntax for conditional formatting
                try:
                    # xlsxwriter uses different syntax for conditional formatting
                    worksheet.conditional_format('C2:C11', {'type': 'data_bar', 'bar_color': '#34b5d9'})
                    worksheet.conditional_format('E2:E11', {'type': 'data_bar', 'bar_color': '#366fff'})
                    print(f"✓ Conditional formatting applied to {sheet_name_final}")
                except Exception as e:
                    print(f"✗ Failed to apply conditional formatting to {sheet_name_final}: {e}")
                    # Try alternative xlsxwriter approach
                    try:
                        # Alternative: use xlsxwriter workbook directly
                        workbook = writer.book
                        worksheet_xlsx = workbook.get_worksheet_by_name(sheet_name_final)
                        if worksheet_xlsx:
                            worksheet_xlsx.conditional_format('C2:C11', {'type': 'data_bar', 'bar_color': '#34b5d9'})
                            worksheet_xlsx.conditional_format('E2:E11', {'type': 'data_bar', 'bar_color': '#366fff'})
                            print(f"✓ Conditional formatting applied to {sheet_name_final} (alternative method)")
                    except Exception as e2:
                        print(f"✗ Alternative conditional formatting also failed for {sheet_name_final}: {e2}")
            
            # Mark file for removal (successfully processed)
            files_to_remove.append(filename)
            print(f"✓ Processed and marked for removal: {f_name}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            # Still mark for removal even if processing failed
            files_to_remove.append(filename)
            print(f"✗ Failed to process but marking for removal: {os.path.basename(filename)}")
    
    # Remove all processed files after consolidation
    for filename in files_to_remove:
        try:
            os.remove(filename)
            print(f"✓ Removed: {os.path.basename(filename)}")
        except Exception as e:
            print(f"✗ Failed to remove {filename}: {e}")
    
    print(f"Total files processed: {len(files_to_remove)}")
    
    writer.close()
    print(f'KS charts consolidated in: {os.path.join(output_dir, "KS_Charts.xlsx")}')
    print('Note: Conditional formatting should now be visible in the Excel file')


# Confusion matrix
def draw_confusion_matrix(output_dir, model_type, y_pred, y_true, data_type):
    """
    Generate confusion matrix and save to specified output directory.
    
    Args:
        output_dir (str): Directory to save the plot
        model_type (str): Type of model
        y_pred (array): Predicted labels
        y_true (array): True labels
        data_type (str): Type of data (train/test/validation)
    """
    try:
        AccuracyValue = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
        PrecisionValue = metrics.precision_score(y_pred=y_pred, y_true=y_true, average='weighted')
        RecallValue = metrics.recall_score(y_pred=y_pred, y_true=y_true, average='weighted')
        F1Value = metrics.f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        AccuracyValue = PrecisionValue = RecallValue = F1Value = 0.0

    plt.title(f'{model_type} Model - Confusion Matrix for {data_type} data\n\n'
              f'Accuracy:{AccuracyValue:.3f}   Precision:{PrecisionValue:.3f}   '
              f'Recall:{RecallValue:.3f}   F1 Score:{F1Value:.3f}\n')
    
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'{model_type} Model - Confusion Matrix for {data_type} data.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved: {filepath}')


# Model validation
def model_validation(output_dir, data, y, model, model_type, data_type):
    """
    Perform comprehensive model validation including ROC, KS, and confusion matrix.
    
    Args:
        output_dir (str): Directory to save all validation outputs
        data: Input data for prediction
        y: Target variable
        model: Trained model
        model_type (str): Type of model
        data_type (str): Type of data (train/test/validation)
    
    Returns:
        list: List containing [roc_data, accuracy_data, ks_data]
    """
    start_time = time.time()

    try:
        pred_data = model.transform(data)
        print('Model prediction completed')

        roc_data, accuracy_data, ks_data, y_score, y_pred, y_true, decile_table = calculate_metrics(pred_data, y, data_type)
        
        # Generate plots
        draw_roc_plot(output_dir, model_type, y_score, y_true, data_type)
        
        # Save decile table
        os.makedirs(output_dir, exist_ok=True)
        decile_filename = f'KS {model_type} Model {data_type}.xlsx'
        decile_filepath = os.path.join(output_dir, decile_filename)
        decile_table.to_excel(decile_filepath, index=False, engine='xlsxwriter')
        print(f'Decile table saved: {decile_filepath}')
        
        draw_confusion_matrix(output_dir, model_type, y_pred, y_true, data_type)
        print('All validation metrics computed')

        l = [roc_data, accuracy_data, ks_data]
        end_time = time.time()
        print(f"Model validation process completed in: {end_time - start_time:.2f} seconds")
        return l
        
    except Exception as e:
        print(f"Error during model validation: {e}")
        end_time = time.time()
        print(f"Model validation failed after: {end_time - start_time:.2f} seconds")
        return None


# Legacy function for backward compatibility
def model_validation_legacy(user_id, mdl_ltrl, data, y, model, model_type, data_type):
    """
    Legacy function that maintains the original interface for backward compatibility.
    
    Args:
        user_id (str): User ID (not used in new implementation)
        mdl_ltrl (str): Model literal (not used in new implementation)
        data: Input data for prediction
        y: Target variable
        model: Trained model
        model_type (str): Type of model
        data_type (str): Type of data (train/test/validation)
    
    Returns:
        list: List containing [roc_data, accuracy_data, ks_data]
    """
    # Create output directory based on legacy path structure
    output_dir = os.path.join('/home', user_id, f'mla_{mdl_ltrl}', model_type)
    return model_validation(output_dir, data, y, model, model_type, data_type) 