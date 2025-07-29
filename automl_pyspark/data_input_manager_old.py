"""
DataInputManager for AutoML PySpark

This module provides a unified interface for handling various data input sources:
1. GCP BigQuery tables - Direct Spark integration
2. File uploads - CSV, Excel, TSV with custom delimiters  
3. Existing files - Pre-loaded datasets in AutoML directory

Author: AutoML PySpark Team
"""

import os
import shutil
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
import pandas as pd

class DataInputManager:
    """
    Unified data input manager supporting multiple data sources.
    """
    
    def __init__(self, spark: SparkSession, output_dir: str, user_id: str = "default_user"):
        """
        Initialize DataInputManager.
        
        Args:
            spark: SparkSession instance
            output_dir: Output directory for saving uploaded files
            user_id: User identifier for organizing files
        """
        self.spark = spark
        self.output_dir = output_dir
        self.user_id = user_id
        self.supported_formats = {
            'csv': ['.csv'],
            'excel': ['.xlsx', '.xls'],
            'tsv': ['.tsv', '.tab'],
            'parquet': ['.parquet'],
            'json': ['.json']
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize temporary table tracking
        self.temporary_tables = []
        
        print(f"✅ DataInputManager initialized for user: {user_id}")
        print(f"   📁 Output directory: {output_dir}")
    
    def load_data(self, 
                  data_source: str,
                  source_type: str = "auto",
                  **kwargs) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Load data from various sources.
        
        Args:
            data_source: Path/identifier for the data source
            source_type: Type of data source ('bigquery', 'upload', 'existing', 'auto')
            **kwargs: Additional parameters for specific source types
            
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        print(f"\n📊 Loading data from: {data_source}")
        print(f"   🔍 Source type: {source_type}")
        
        # Auto-detect source type if not specified
        if source_type == "auto":
            source_type = self._detect_source_type(data_source)
            print(f"   🤖 Auto-detected source type: {source_type}")
        
        # Route to appropriate loading method
        if source_type == "bigquery":
            return self._load_from_bigquery(data_source, **kwargs)
        elif source_type == "upload":
            return self._load_from_upload(data_source, **kwargs)
        elif source_type == "existing":
            return self._load_from_existing(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _detect_source_type(self, data_source: str) -> str:
        """
        Auto-detect the data source type based on the input.
        
        Args:
            data_source: Data source identifier
            
        Returns:
            Detected source type
        """
        # Check if it looks like a BigQuery table reference
        if self._is_bigquery_reference(data_source):
            return "bigquery"
        
        # Check if it's an existing file in AutoML directory
        elif self._is_existing_file(data_source):
            return "existing"
        
        # Check if it's a file path for upload
        elif os.path.exists(data_source) or any(data_source.endswith(ext) for exts in self.supported_formats.values() for ext in exts):
            return "upload"
        
        else:
            # Default to existing file lookup
            return "existing"
    
    def _is_bigquery_reference(self, data_source: str) -> bool:
        """Check if data_source looks like a BigQuery table reference."""
        # BigQuery format: project.dataset.table or dataset.table
        parts = data_source.split('.')
        return len(parts) >= 2 and len(parts) <= 3 and all(part.replace('_', '').replace('-', '').isalnum() for part in parts)
    
    def _is_existing_file(self, data_source: str) -> bool:
        """Check if data_source refers to an existing file in AutoML directory."""
        # Check common existing files
        existing_files = ['bank.csv', 'IRIS.csv', 'bank', 'iris']
        
        # Check if it's a known file name
        if data_source.lower() in [f.lower() for f in existing_files]:
            return True
        
        # Check if file exists in current directory or AutoML directory
        automl_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            data_source,
            os.path.join(automl_dir, data_source),
            os.path.join(automl_dir, f"{data_source}.csv"),
            os.path.join(self.output_dir, data_source),
            os.path.join(self.output_dir, f"{data_source}.csv")
        ]
        
        return any(os.path.exists(path) for path in potential_paths)
    
    def _load_from_bigquery(self, table_reference: str, **kwargs) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Load data directly from GCP BigQuery using the user's proven working pattern.
        
        Args:
            table_reference: BigQuery table reference (project.dataset.table)
            **kwargs: Additional BigQuery options including:
                - project_id: GCP project ID
                - row_limit: Limit number of rows (for testing)
                - sample_percent: Percentage for table sampling
                - where_clause: Custom WHERE clause (without 'WHERE' keyword)
                - select_columns: Custom column selection
                - use_legacy_sql: Whether to use Legacy SQL
                - bigquery_options: Additional BigQuery connector options
        """
        print(f"   🔌 Connecting to BigQuery table: {table_reference}")
        
        try:
            # Get BigQuery options from kwargs
            project_id = kwargs.get('project_id')
            row_limit = kwargs.get('row_limit')
            sample_percent = kwargs.get('sample_percent')
            where_clause = kwargs.get('where_clause')
            select_columns = kwargs.get('select_columns')
            use_legacy_sql = kwargs.get('use_legacy_sql', False)
            bigquery_options = kwargs.get('bigquery_options', {})
            
            # Determine final project ID
            if not project_id:
                # If no explicit project_id, try to extract from table reference
                table_parts = table_reference.split('.')
                if len(table_parts) >= 3:
                    project_id = table_parts[0]
                    print(f"   📊 Extracted project ID from table reference: {project_id}")
                else:
                    raise ValueError(f"No project_id provided and cannot extract from table reference: {table_reference}")
            else:
                print(f"   📊 Using project ID from configuration: {project_id}")
            
            print(f"   📋 BigQuery configuration:")
            print(f"      • Project: {project_id}")
            print(f"      • Table: {table_reference}")
            
            # Check if we need to create a temporary table
            needs_temp_table = any([where_clause, select_columns, row_limit, sample_percent])
            
            if needs_temp_table:
                print(f"   🔧 Creating temporary table with applied filters...")
                # Create temporary table with all filters applied
                temp_table_ref = self._create_temporary_bigquery_table(
                    original_table=table_reference,
                    project_id=project_id,
                    where_clause=where_clause,
                    select_columns=select_columns,
                    row_limit=row_limit,
                    sample_percent=sample_percent
                )
                
                # Load from the temporary table (no additional options needed)
                print(f"   📖 Loading from temporary table: {temp_table_ref}")
                reader = self.spark.read \
                    .format("bigquery") \
                    .option("parentProject", project_id) \
                    .option("viewsEnabled", "true") \
                    .option("useAvroLogicalTypes", "true") \
                    .option("table", temp_table_ref)
                
                # Load the data
                print(f"   🔄 Executing BigQuery load operation...")
                try:
                    df = reader.load()
                    print(f"   ✅ BigQuery data loaded successfully from temporary table")
                except Exception as e:
                    print(f"   ❌ Failed to load from temporary table: {e}")
                    raise
            else:
                # Use direct table loading (original approach)
                print(f"   📖 Loading table using direct table reference")
                print(f"      • Full table reference: {table_reference}")
                
                reader = self.spark.read \
                    .format("bigquery") \
                    .option("parentProject", project_id) \
                    .option("viewsEnabled", "true") \
                    .option("useAvroLogicalTypes", "true") \
                    .option("table", table_reference)
                
                # Load the data
                print(f"   🔄 Executing BigQuery load operation...")
                try:
                    df = reader.load()
                    print(f"   ✅ BigQuery data loaded successfully")
                except Exception as e:
                    print(f"   ❌ Failed to load from BigQuery: {e}")
                    raise
                print(f"   🔧 Building custom BigQuery query with options:")
                if where_clause:
                    print(f"      • WHERE clause: {where_clause}")
                if select_columns:
                    print(f"      • Column selection: {select_columns}")
                if use_legacy_sql:
                    print(f"      • Using Legacy SQL")
                
                # Build the SQL query
                if select_columns:
                    select_part = select_columns
                else:
                    select_part = "*"
                
                # Use the table reference directly (sampling will be handled by maxRowsPerPartition)
                table_part = f"`{table_reference}`"
                
                # Build WHERE clause
                where_part = ""
                if where_clause:
                    where_part = f" WHERE {where_clause}"
                
                # Row limiting is handled separately for better compatibility
                limit_part = ""
                
                # Construct the full query
                sql_query = f"SELECT {select_part} FROM {table_part}{where_part}{limit_part}"
                print(f"   📝 Generated SQL query: {sql_query}")
                
                # Use query-based loading
                # Extract dataset from table reference for materialization
                print(f"   🔍 Extracting dataset from table reference: {table_reference}")
                table_parts = table_reference.split('.')
                print(f"   📋 Table parts: {table_parts}")
                
                # Handle different table reference formats
                if len(table_parts) == 3:
                    # Format: project.dataset.table
                    dataset_id = table_parts[1]
                elif len(table_parts) == 2:
                    # Format: dataset.table (project is already specified)
                    dataset_id = table_parts[0]
                else:
                    # Fallback: try to extract from project_id if available
                    print(f"   ⚠️ Cannot extract dataset from table reference, using fallback")
                    dataset_id = "temp_dataset"
                
                # Validate dataset_id is not empty
                if not dataset_id or dataset_id.strip() == "":
                    print(f"   ⚠️ Extracted dataset is empty, using fallback")
                    dataset_id = "temp_dataset"
                
                print(f"   📊 Using dataset '{dataset_id}' for query materialization")
                
                # Validate dataset_id before using it
                if not dataset_id or dataset_id.strip() == "":
                    raise ValueError(f"Cannot determine dataset for table reference: {table_reference}")
                
                # Build the reader with all options
                reader = self.spark.read.format("bigquery")
                
                # Set core options
                reader = reader.option("parentProject", project_id)
                reader = reader.option("dataset", dataset_id)
                reader = reader.option("viewsEnabled", "true")
                reader = reader.option("useAvroLogicalTypes", "true")
                reader = reader.option("query", sql_query)
                
                # Add Legacy SQL option if specified
                if use_legacy_sql:
                    reader = reader.option("useLegacySql", "true")
                
                # Add any additional BigQuery options (but skip dataset to avoid conflicts)
                for key, value in bigquery_options.items():
                    if key.lower() != "dataset":  # Avoid overriding our dataset setting
                        reader = reader.option(key, value)
                
                # Log all options being used
                print(f"   🔧 BigQuery options configured:")
                print(f"      • parentProject: {project_id}")
                print(f"      • dataset: {dataset_id}")
                print(f"      • viewsEnabled: true")
                print(f"      • useAvroLogicalTypes: true")
                print(f"      • query: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
                if use_legacy_sql:
                    print(f"      • useLegacySql: true")
                if bigquery_options:
                    print(f"      • Additional options: {list(bigquery_options.keys())}")
                    
            else:
                # Use direct table loading (original approach)
                print(f"   📖 Loading table using direct table reference")
                print(f"      • Full table reference: {table_reference}")
                
                reader = self.spark.read \
                    .format("bigquery") \
                    .option("parentProject", project_id) \
                    .option("viewsEnabled", "true") \
                    .option("useAvroLogicalTypes", "true") \
                    .option("table", table_reference)
                
                # Handle row limiting and sampling with direct table loading
                if row_limit:
                    print(f"   📋 Row limit specified: {row_limit:,} rows")
                    print(f"      🔧 Using optimized row limiting strategy")
                    # Use a combination of maxRowsPerPartition and post-load limiting
                    # First, try to limit at the partition level
                    estimated_partitions = 5  # Conservative estimate
                    rows_per_partition = max(1, row_limit // estimated_partitions)
                    reader = reader.option("maxRowsPerPartition", rows_per_partition)
                    print(f"      📊 Setting {rows_per_partition:,} rows per partition")
                elif sample_percent:
                    print(f"   📊 Table sampling specified: {sample_percent}% (using maxRowsPerPartition)")
                    # Estimate rows per partition for sampling
                    estimated_rows_per_partition = 10000  # Conservative estimate
                    reader = reader.option("maxRowsPerPartition", estimated_rows_per_partition)
                
                # Add any additional BigQuery options
                for key, value in bigquery_options.items():
                    reader = reader.option(key, value)
            
            # Load the data
            print(f"   🔄 Executing BigQuery load operation...")
            try:
                df = reader.load()
                print(f"   ✅ BigQuery data loaded successfully")
                
                # Apply post-load row limiting as a safety measure
                if row_limit and not use_custom_query:
                    print(f"   📋 Applying final row limit: {row_limit:,} rows")
                    original_count = df.count()
                    if original_count > row_limit:
                        df = df.limit(row_limit)
                        actual_count = df.count()
                        print(f"      📊 Limited from {original_count:,} to {actual_count:,} rows")
                    else:
                        print(f"      📊 Already within limit: {original_count:,} rows")
                    
            except Exception as e:
                print(f"   ❌ Failed to load from BigQuery: {e}")
                print(f"   💡 Troubleshooting information:")
                print(f"      • Table reference: {table_reference}")
                print(f"      • Project ID: {project_id}")
                if 'dataset_id' in locals():
                    print(f"      • Dataset ID: {dataset_id}")
                if 'sql_query' in locals():
                    print(f"      • SQL Query: {sql_query}")
                print(f"      • BigQuery options: {bigquery_options}")
                raise
            
            # Get basic metadata
            print(f"   📊 Calculating dataset statistics...")
            if row_limit:
                print(f"      (Limited to {row_limit:,} rows for testing)")
            elif sample_percent:
                print(f"      (Sampled {sample_percent}% of table)")
            else:
                print(f"      (Processing full dataset)")
            
            # Optimize row counting for large datasets
            try:
                row_count = df.count()
            except Exception as e:
                if row_limit:
                    print(f"   ⚠️ Could not count limited dataset rows: {e}")
                    row_count = f"≤{row_limit:,} (limited)"
                elif sample_percent:
                    print(f"   ⚠️ Could not count sampled dataset rows: {e}")
                    row_count = f"~{sample_percent}% of table (sampled)"
                else:
                    print(f"   ⚠️ Could not count full dataset rows (very large BigQuery dataset): {e}")
                    row_count = "Unknown (very large dataset)"
            
            column_count = len(df.columns)
            
            # Parse table reference for metadata
            table_parts = table_reference.split('.')
            if len(table_parts) == 3:
                _, dataset_name, table_name = table_parts
            elif len(table_parts) == 2:
                dataset_name, table_name = table_parts
            else:
                dataset_name = "unknown"
                table_name = table_parts[-1] if table_parts else "unknown"
            
            metadata = {
                'source_type': 'bigquery',
                'table_reference': table_reference,
                'project_id': project_id,
                'dataset_name': dataset_name,
                'table_name': table_name,
                'row_count': row_count,
                'column_count': column_count,
                'columns': df.columns,
                'schema': df.schema.json(),
                'query_options': {
                    'sample_percent': sample_percent,
                    'where_clause': where_clause,
                    'select_columns': select_columns,
                    'use_legacy_sql': use_legacy_sql,
                    'row_limit': row_limit
                }
            }
            
            print(f"   ✅ BigQuery data loaded successfully")
            print(f"      📊 Shape: {row_count:,} rows × {column_count} columns")
            print(f"      📋 Columns: {df.columns[:5]}{'...' if len(df.columns) > 5 else ''}")
            
            return df, metadata
            
        except Exception as e:
            print(f"   ❌ Failed to load from BigQuery: {str(e)}")
            
            # Enhanced error handling for authentication issues
            if "no JSON input found" in str(e) or "authentication" in str(e).lower():
                print(f"   🔐 Authentication Error Detected!")
                print(f"   💡 This is a Google Cloud authentication issue.")
                print(f"   🛠️ Quick Fix Options:")
                print(f"      1. Run: gcloud auth application-default login")
                print(f"      2. Set environment variable: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
                print(f"      3. Use the BigQuery auth helper: python bigquery_auth_helper.py --auto-fix {project_id} {table_reference}")
                print(f"      4. Check authentication: python bigquery_auth_helper.py --check")
            
            print(f"   💡 General troubleshooting tips:")
            print(f"      • Ensure BigQuery connector JAR (v0.36.1) is in Spark classpath")
            print(f"      • Verify authentication (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)")
            print(f"      • Check table reference format: {table_reference}")
            print(f"      • Ensure table exists and you have access permissions")
            print(f"      • Ensure project_id is provided: {kwargs.get('project_id', 'NOT PROVIDED')}")
            print(f"      • For query-based loading, dataset option is required for materialization")
            print(f"   📖 Query pattern: .option('parentProject', project).option('dataset', dataset).option('query', sql)")
            print(f"   📖 Table pattern: .option('parentProject', project).option('table', full_table_ref)")
            raise
    
    def _load_from_upload(self, file_path: str, **kwargs) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Load data from uploaded file (CSV, Excel, TSV, etc.).
        
        Args:
            file_path: Path to the uploaded file
            **kwargs: File reading options (delimiter, header, etc.)
        """
        print(f"   📁 Processing uploaded file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file format
        file_format = self._detect_file_format(file_path)
        print(f"   🔍 Detected format: {file_format}")
        
        # Copy file to output directory with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        saved_filename = f"{name}_{timestamp}{ext}"
        saved_path = os.path.join(self.output_dir, saved_filename)
        
        shutil.copy2(file_path, saved_path)
        print(f"   💾 File saved to: {saved_path}")
        
        try:
            # Load based on format
            if file_format == 'csv' or file_format == 'tsv':
                df = self._load_csv_file(saved_path, **kwargs)
            elif file_format == 'excel':
                df = self._load_excel_file(saved_path, **kwargs)
            elif file_format == 'parquet':
                df = self._load_parquet_file(saved_path, **kwargs)
            elif file_format == 'json':
                df = self._load_json_file(saved_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Get metadata
            row_count = df.count()
            column_count = len(df.columns)
            
            metadata = {
                'source_type': 'upload',
                'original_file': file_path,
                'saved_file': saved_path,
                'file_format': file_format,
                'row_count': row_count,
                'column_count': column_count,
                'columns': df.columns,
                'schema': df.schema.json(),
                'load_options': kwargs
            }
            
            print(f"   ✅ File loaded successfully")
            print(f"      📊 Shape: {row_count} rows × {column_count} columns")
            print(f"      📋 Columns: {df.columns[:5]}{'...' if len(df.columns) > 5 else ''}")
            
            return df, metadata
            
        except Exception as e:
            print(f"   ❌ Failed to load uploaded file: {str(e)}")
            # Clean up saved file on error
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise
    
    def _load_from_existing(self, file_identifier: str, **kwargs) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Load data from existing files in AutoML directory.
        
        Args:
            file_identifier: Name or identifier of existing file
            **kwargs: Additional loading options
        """
        print(f"   📂 Looking for existing file: {file_identifier}")
        
        # Find the actual file path
        file_path = self._find_existing_file(file_identifier)
        
        if not file_path:
            # List available files for user guidance
            available_files = self._list_available_files()
            raise FileNotFoundError(
                f"Existing file '{file_identifier}' not found.\n"
                f"Available files: {available_files}"
            )
        
        print(f"   📍 Found file: {file_path}")
        
        # Detect format and load
        file_format = self._detect_file_format(file_path)
        
        try:
            if file_format == 'csv':
                df = self._load_csv_file(file_path, **kwargs)
            elif file_format == 'excel':
                df = self._load_excel_file(file_path, **kwargs)
            elif file_format == 'parquet':
                df = self._load_parquet_file(file_path, **kwargs)
            elif file_format == 'json':
                df = self._load_json_file(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Get metadata
            row_count = df.count()
            column_count = len(df.columns)
            
            metadata = {
                'source_type': 'existing',
                'file_identifier': file_identifier,
                'file_path': file_path,
                'file_format': file_format,
                'row_count': row_count,
                'column_count': column_count,
                'columns': df.columns,
                'schema': df.schema.json(),
                'load_options': kwargs
            }
            
            print(f"   ✅ Existing file loaded successfully")
            print(f"      📊 Shape: {row_count} rows × {column_count} columns")
            print(f"      📋 Columns: {df.columns[:5]}{'...' if len(df.columns) > 5 else ''}")
            
            return df, metadata
            
        except Exception as e:
            print(f"   ❌ Failed to load existing file: {str(e)}")
            raise
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        for format_name, extensions in self.supported_formats.items():
            if ext in extensions:
                return format_name
        
        # Default to CSV for unknown extensions
        return 'csv'
    
    def _find_existing_file(self, file_identifier: str) -> Optional[str]:
        """Find the actual path of an existing file."""
        automl_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Potential file locations and names
        potential_paths = [
            # Exact path
            file_identifier,
            # In AutoML directory
            os.path.join(automl_dir, file_identifier),
            os.path.join(automl_dir, f"{file_identifier}.csv"),
            # In output directory
            os.path.join(self.output_dir, file_identifier),
            os.path.join(self.output_dir, f"{file_identifier}.csv"),
            # Common variations
            os.path.join(automl_dir, f"{file_identifier.upper()}.csv"),
            os.path.join(automl_dir, f"{file_identifier.lower()}.csv"),
        ]
        
        # Check each potential path
        for path in potential_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _list_available_files(self) -> List[str]:
        """List available existing files."""
        automl_dir = os.path.dirname(os.path.abspath(__file__))
        available_files = []
        
        # Check AutoML directory
        if os.path.exists(automl_dir):
            for file in os.listdir(automl_dir):
                if any(file.endswith(ext) for exts in self.supported_formats.values() for ext in exts):
                    available_files.append(file)
        
        # Check output directory
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if any(file.endswith(ext) for exts in self.supported_formats.values() for ext in exts):
                    available_files.append(f"output/{file}")
        
        return sorted(available_files)
    
    def _load_csv_file(self, file_path: str, **kwargs) -> DataFrame:
        """Load CSV/TSV file with Spark."""
        delimiter = kwargs.get('delimiter', kwargs.get('sep', ','))
        header = kwargs.get('header', True)
        infer_schema = kwargs.get('infer_schema', True)
        
        # Handle TSV files
        if file_path.endswith('.tsv') or file_path.endswith('.tab'):
            delimiter = kwargs.get('delimiter', kwargs.get('sep', '\t'))
        
        return self.spark.read \
            .option("header", header) \
            .option("inferSchema", infer_schema) \
            .option("delimiter", delimiter) \
            .csv(file_path)
    
    def _load_excel_file(self, file_path: str, **kwargs) -> DataFrame:
        """Load Excel file by converting to pandas first."""
        sheet_name = kwargs.get('sheet_name', 0)
        header = kwargs.get('header', 0)
        
        # Use pandas to read Excel, then convert to Spark DataFrame
        pandas_df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        return self.spark.createDataFrame(pandas_df)
    
    def _load_parquet_file(self, file_path: str, **kwargs) -> DataFrame:
        """Load Parquet file with Spark."""
        return self.spark.read.parquet(file_path)
    
    def _load_json_file(self, file_path: str, **kwargs) -> DataFrame:
        """Load JSON file with Spark."""
        multiline = kwargs.get('multiline', True)
        return self.spark.read.option("multiLine", multiline).json(file_path)
    
    def list_bigquery_tables(self, project_id: str, dataset_id: str) -> List[str]:
        """
        List available BigQuery tables in a dataset.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            
        Returns:
            List of table names
        """
        try:
            # This would require BigQuery client
            print(f"📋 Listing tables in {project_id}.{dataset_id}")
            print("💡 Note: This requires BigQuery client configuration")
            
            # Placeholder - would implement actual BigQuery listing
            return ["table1", "table2", "table3"]
            
        except Exception as e:
            print(f"❌ Failed to list BigQuery tables: {str(e)}")
            return []
    
    def get_data_preview(self, df: DataFrame, num_rows: int = 5) -> None:
        """
        Display a preview of the loaded data.
        
        Args:
            df: Spark DataFrame to preview
            num_rows: Number of rows to show
        """
        print(f"\n📋 Data Preview (first {num_rows} rows):")
        df.show(num_rows, truncate=False)
        
        print(f"\n📊 Schema Information:")
        df.printSchema()
        
        print(f"\n📈 Basic Statistics:")
        print(f"   Total rows: {df.count()}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Column names: {df.columns}")
    
    def _create_temporary_bigquery_table(self, 
                                       original_table: str, 
                                       project_id: str,
                                       where_clause: Optional[str] = None,
                                       select_columns: Optional[str] = None,
                                       row_limit: Optional[int] = None,
                                       sample_percent: Optional[float] = None) -> str:
        """
        Create a temporary BigQuery table with applied filters.
        
        Args:
            original_table: Original BigQuery table reference
            project_id: BigQuery project ID
            where_clause: Optional WHERE clause
            select_columns: Optional column selection
            row_limit: Optional row limit
            sample_percent: Optional sampling percentage
            
        Returns:
            Temporary table reference
        """
        print(f"   🔧 Creating temporary BigQuery table with filters...")
        
        # Parse original table reference
        table_parts = original_table.split('.')
        if len(table_parts) == 3:
            _, dataset_id, table_name = table_parts
        elif len(table_parts) == 2:
            dataset_id, table_name = table_parts
        else:
            raise ValueError(f"Invalid table reference format: {original_table}")
        
        # Create temporary table name
        temp_table_name = f"{table_name}_automl_classifier_conditions"
        temp_table_ref = f"{project_id}.{dataset_id}.{temp_table_name}"
        
        print(f"      📋 Original table: {original_table}")
        print(f"      📋 Temporary table: {temp_table_ref}")
        
        # Build the SQL query for temporary table creation
        select_part = select_columns if select_columns else "*"
        
        # Add table sampling if specified
        if sample_percent:
            table_part = f"`{original_table}` TABLESAMPLE SYSTEM ({sample_percent} PERCENT)"
        else:
            table_part = f"`{original_table}`"
        
        # Build WHERE clause
        where_part = ""
        if where_clause:
            where_part = f" WHERE {where_clause}"
        
        # Add row limit if specified
        limit_part = ""
        if row_limit:
            limit_part = f" LIMIT {row_limit}"
        
        # Construct the full query
        create_query = f"""
        CREATE OR REPLACE TABLE `{temp_table_ref}` AS
        SELECT {select_part}
        FROM {table_part}{where_part}{limit_part}
        """
        
        print(f"      📝 SQL Query: {create_query.strip()}")
        
        # Execute the query using Spark SQL
        try:
            # Register the query as a temporary view
            self.spark.sql(create_query)
            print(f"      ✅ Temporary table created successfully")
            
            # Track the temporary table for cleanup
            self.temporary_tables.append(temp_table_ref)
            
            return temp_table_ref
            
        except Exception as e:
            print(f"      ❌ Failed to create temporary table: {e}")
            raise
    
    def _drop_temporary_tables(self):
        """
        Drop all temporary tables created during this session.
        """
        if not self.temporary_tables:
            return
        
        print(f"   🧹 Cleaning up temporary BigQuery tables...")
        
        for temp_table in self.temporary_tables:
            try:
                drop_query = f"DROP TABLE IF EXISTS `{temp_table}`"
                self.spark.sql(drop_query)
                print(f"      ✅ Dropped temporary table: {temp_table}")
            except Exception as e:
                print(f"      ⚠️ Failed to drop temporary table {temp_table}: {e}")
        
        # Clear the tracking list
        self.temporary_tables.clear()
        print(f"      ✅ Temporary table cleanup completed") 