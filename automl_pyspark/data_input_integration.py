"""
Data Input Integration for AutoML Classes

This module provides enhanced methods for AutoML classes to use the new DataInputManager
for flexible data input from BigQuery, uploaded files, and existing files.

Author: AutoML PySpark Team
"""

from typing import Any, Dict, Optional, Union, Tuple
from pyspark.sql import DataFrame
from .data_input_manager import DataInputManager

def add_data_input_methods():
    """
    Add data input methods to AutoML classes via monkey patching.
    This allows backward compatibility while adding new functionality.
    """
    
    # Import here to avoid circular imports
    from .regression.automl_regressor import AutoMLRegressor
    from .classification.automl_classifier import AutoMLClassifier
    
    # Add methods to AutoMLRegressor
    AutoMLRegressor.load_data_from_source = load_data_from_source_regression
    AutoMLRegressor.fit_with_data_source = fit_with_data_source_regression
    AutoMLRegressor.preview_data_source = preview_data_source
    
    # Add methods to AutoMLClassifier  
    AutoMLClassifier.load_data_from_source = load_data_from_source_classification
    AutoMLClassifier.fit_with_data_source = fit_with_data_source_classification
    AutoMLClassifier.preview_data_source = preview_data_source
    
    print("✅ Enhanced data input methods added to AutoML classes")

def load_data_from_source_regression(self, 
                                   data_source: str,
                                   source_type: str = "auto",
                                   **kwargs) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Load data from various sources for regression tasks.
    
    Args:
        data_source: Data source identifier (BigQuery table, file path, or existing file name)
        source_type: Type of source ('bigquery', 'upload', 'existing', 'auto')
        **kwargs: Additional loading options
        
    Returns:
        Tuple of (DataFrame, metadata)
        
    Example:
        # BigQuery
        df, meta = automl.load_data_from_source("project.dataset.table", "bigquery")
        
        # File upload
        df, meta = automl.load_data_from_source("/path/to/data.csv", "upload", delimiter=",")
        
        # Existing file
        df, meta = automl.load_data_from_source("bank", "existing")
    """
    if not hasattr(self, '_data_input_manager'):
        self._data_input_manager = DataInputManager(
            spark=self.spark,
            output_dir=self.output_dir,
            user_id=getattr(self, 'user_id', 'default_user')
        )
    
    return self._data_input_manager.load_data(data_source, source_type, **kwargs)

def load_data_from_source_classification(self,
                                       data_source: str,
                                       source_type: str = "auto", 
                                       **kwargs) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Load data from various sources for classification tasks.
    
    Args:
        data_source: Data source identifier (BigQuery table, file path, or existing file name)
        source_type: Type of source ('bigquery', 'upload', 'existing', 'auto')
        **kwargs: Additional loading options
        
    Returns:
        Tuple of (DataFrame, metadata)
        
    Example:
        # BigQuery
        df, meta = automl.load_data_from_source("project.dataset.table", "bigquery")
        
        # File upload with custom delimiter
        df, meta = automl.load_data_from_source("/path/to/data.tsv", "upload", delimiter="\t")
        
        # Existing file
        df, meta = automl.load_data_from_source("iris", "existing")
    """
    if not hasattr(self, '_data_input_manager'):
        self._data_input_manager = DataInputManager(
            spark=self.spark,
            output_dir=self.output_dir,
            user_id=getattr(self, 'user_id', 'default_user')
        )
    
    return self._data_input_manager.load_data(data_source, source_type, **kwargs)

def fit_with_data_source_regression(self,
                                   train_data_source: str,
                                   target_column: str,
                                   oot1_data_source: Optional[str] = None,
                                   oot2_data_source: Optional[str] = None,
                                   source_type: str = "auto",
                                   **kwargs) -> 'AutoMLRegressor':
    """
    Fit regression model using data from a single source type.
    
    Args:
        train_data_source: Training data source identifier
        target_column: Name of the target column
        oot1_data_source: Out-of-time 1 data source (optional)
        oot2_data_source: Out-of-time 2 data source (optional)
        source_type: Type of data source (applied to all datasets)
        **kwargs: Additional loading options (applied to all datasets)
        
    Returns:
        Fitted AutoMLRegressor instance
        
    Example:
        # BigQuery training with multiple tables
        automl.fit_with_data_source(
            train_data_source="project.dataset.train_table",
            oot1_data_source="project.dataset.oot1_table", 
            oot2_data_source="project.dataset.oot2_table",
            target_column="price",
            source_type="bigquery"
        )
        
        # File upload with multiple files
        automl.fit_with_data_source(
            train_data_source="/data/train.csv",
            oot1_data_source="/data/oot1.csv",
            target_column="target",
            source_type="upload",
            delimiter=","
        )
    """
    print(f"\n🚀 Starting AutoML Regression with single data source type")
    print(f"   📊 Source Type: {source_type}")
    
    # Load training data
    print(f"📊 Loading training data...")
    train_data, train_meta = self.load_data_from_source(
        train_data_source, source_type, **kwargs
    )
    
    # Load OOT1 data if provided
    oot1_data = None
    if oot1_data_source:
        print(f"📊 Loading OOT1 data...")
        oot1_data, oot1_meta = self.load_data_from_source(
            oot1_data_source, source_type, **kwargs
        )
    
    # Load OOT2 data if provided
    oot2_data = None
    if oot2_data_source:
        print(f"📊 Loading OOT2 data...")
        oot2_data, oot2_meta = self.load_data_from_source(
            oot2_data_source, source_type, **kwargs
        )
    
    # Store metadata for reference
    self.data_sources_metadata = {
        'train': train_meta,
        'oot1': oot1_meta if oot1_data else None,
        'oot2': oot2_meta if oot2_data else None
    }
    
    # Log data source summary
    print(f"📋 Data Source Summary:")
    print(f"   🎯 Source Type: {source_type}")
    print(f"   📊 Training: {train_data_source} ({train_meta['row_count']} rows)")
    if oot1_data_source:
        print(f"   📊 OOT1: {oot1_data_source} ({oot1_meta['row_count']} rows)")
    if oot2_data_source:
        print(f"   📊 OOT2: {oot2_data_source} ({oot2_meta['row_count']} rows)")
    
    # Call the original fit method
    return self.fit(
        train_data=train_data,
        oot1_data=oot1_data,
        oot2_data=oot2_data,
        target_column=target_column
    )

def fit_with_data_source_classification(self,
                                       train_data_source: str,
                                       target_column: str,
                                       oot1_data_source: Optional[str] = None,
                                       oot2_data_source: Optional[str] = None,
                                       source_type: str = "auto",
                                       **kwargs) -> 'AutoMLClassifier':
    """
    Fit classification model using data from a single source type.
    
    Args:
        train_data_source: Training data source identifier
        target_column: Name of the target column
        oot1_data_source: Out-of-time 1 data source (optional)
        oot2_data_source: Out-of-time 2 data source (optional)
        source_type: Type of data source (applied to all datasets)
        **kwargs: Additional loading options (applied to all datasets)
        
    Returns:
        Fitted AutoMLClassifier instance
        
    Example:
        # Existing files with multiple datasets
        automl.fit_with_data_source(
            train_data_source="train_data.csv",
            oot1_data_source="validation_data.csv",
            target_column="target",
            source_type="existing"
        )
        
        # File upload with custom delimiter
        automl.fit_with_data_source(
            train_data_source="/data/train.tsv",
            oot1_data_source="/data/test.tsv",
            target_column="class",
            source_type="upload",
            delimiter="\t"
        )
    """
    print(f"\n🚀 Starting AutoML Classification with single data source type")
    print(f"   📊 Source Type: {source_type}")
    
    # Load training data
    print(f"📊 Loading training data...")
    train_data, train_meta = self.load_data_from_source(
        train_data_source, source_type, **kwargs
    )
    
    # Load OOT1 data if provided
    oot1_data = None
    if oot1_data_source:
        print(f"📊 Loading OOT1 data...")
        oot1_data, oot1_meta = self.load_data_from_source(
            oot1_data_source, source_type, **kwargs
        )
    
    # Load OOT2 data if provided  
    oot2_data = None
    if oot2_data_source:
        print(f"📊 Loading OOT2 data...")
        oot2_data, oot2_meta = self.load_data_from_source(
            oot2_data_source, source_type, **kwargs
        )
    
    # Store metadata for reference
    self.data_sources_metadata = {
        'train': train_meta,
        'oot1': oot1_meta if oot1_data else None,
        'oot2': oot2_meta if oot2_data else None
    }
    
    # Log data source summary
    print(f"📋 Data Source Summary:")
    print(f"   🎯 Source Type: {source_type}")
    print(f"   📊 Training: {train_data_source} ({train_meta['row_count']} rows)")
    if oot1_data_source:
        print(f"   📊 OOT1: {oot1_data_source} ({oot1_meta['row_count']} rows)")
    if oot2_data_source:
        print(f"   📊 OOT2: {oot2_data_source} ({oot2_meta['row_count']} rows)")
    
    # Call the original fit method
    return self.fit(
        train_data=train_data,
        oot1_data=oot1_data,
        oot2_data=oot2_data,
        target_column=target_column
    )

def preview_data_source(self,
                       data_source: str,
                       source_type: str = "auto",
                       num_rows: int = 5,
                       **kwargs) -> None:
    """
    Preview data from a source without loading it fully.
    
    Args:
        data_source: Data source identifier
        source_type: Type of data source
        num_rows: Number of rows to preview
        **kwargs: Additional loading options
        
    Example:
        # Preview BigQuery table
        automl.preview_data_source("project.dataset.table", "bigquery")
        
        # Preview uploaded file
        automl.preview_data_source("/path/to/data.csv", "upload", delimiter="|")
    """
    print(f"\n👀 Previewing data source: {data_source}")
    
    try:
        # Load data
        df, metadata = self.load_data_from_source(data_source, source_type, **kwargs)
        
        # Show preview using the DataInputManager
        if not hasattr(self, '_data_input_manager'):
            self._data_input_manager = DataInputManager(
                spark=self.spark,
                output_dir=self.output_dir,
                user_id=getattr(self, 'user_id', 'default_user')
            )
        
        self._data_input_manager.get_data_preview(df, num_rows)
        
        # Show metadata
        print(f"\n📋 Data Source Metadata:")
        print(f"   Source Type: {metadata['source_type']}")
        print(f"   Format: {metadata.get('file_format', 'N/A')}")
        print(f"   Shape: {metadata['row_count']} rows × {metadata['column_count']} columns")
        
        if metadata['source_type'] == 'bigquery':
            print(f"   BigQuery Table: {metadata['table_reference']}")
        elif metadata['source_type'] == 'upload':
            print(f"   Original File: {metadata['original_file']}")
            print(f"   Saved File: {metadata['saved_file']}")
        elif metadata['source_type'] == 'existing':
            print(f"   File Path: {metadata['file_path']}")
            
    except Exception as e:
        print(f"❌ Failed to preview data source: {str(e)}")
        raise

# Convenience function to initialize the integration
def enable_flexible_data_input():
    """
    Enable flexible data input methods for AutoML classes.
    Call this once after importing automl_pyspark to add the new methods.
    """
    add_data_input_methods()
    print("🎉 Flexible data input enabled! You can now use:")
    print("   • automl.load_data_from_source() - Load from any source")
    print("   • automl.fit_with_data_source() - Train with flexible data sources") 
    print("   • automl.preview_data_source() - Preview data before loading")
    print("\nSupported sources:")
    print("   🔗 BigQuery: 'project.dataset.table'")
    print("   📁 File Upload: '/path/to/file.csv' (CSV, Excel, TSV, JSON, Parquet)")
    print("   📂 Existing Files: 'bank', 'iris', etc.") 
