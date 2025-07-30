"""
Simplified DataInputManager for AutoML PySpark.

This module provides a lightweight implementation of the
``DataInputManager`` used in the AutoML PySpark project.  The main
goal of this rewrite is to streamline loading of BigQuery tables and
files while avoiding the expensive temporary table creation logic
found in the original implementation.  In addition to BigQuery
support, the manager can load data from uploaded files and from
predefined datasets included with the project.

Key features:

* BigQuery loading uses the connector's query interface directly and
  supports optional selection of columns, row limits, sampling
  percentages and WHERE clauses.  Filters and limits are pushed down
  into BigQuery via the SQL query to minimise data transfer.
* File uploads support CSV, TSV, JSON, Parquet and Excel formats.
  Uploaded files are copied into the configured ``output_dir`` for
  persistence and ease of access.  Excel files are read with pandas
  then converted into Spark DataFrames.
* Existing datasets bundled with the project (e.g. ``IRIS.csv``,
  ``bank.csv`` and ``regression_file.csv``) can be loaded by name.

The returned metadata includes basic information about the loaded
dataset such as the number of rows and columns, file format (for
files) and the original table reference (for BigQuery).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from pyspark.sql import DataFrame, SparkSession


class DataInputManager:
    """Unified interface for loading data into the AutoML pipeline."""

    def __init__(self, spark: SparkSession, output_dir: str, user_id: str = "default_user") -> None:
        self.spark = spark
        self.output_dir = output_dir
        self.user_id = user_id
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Supported file formats for upload
        self.supported_extensions = {
            "csv": [".csv"],
            "tsv": [".tsv", ".tab"],
            "json": [".json"],
            "parquet": [".parquet"],
            "excel": [".xlsx", ".xls"]
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(
        self,
        data_source: str,
        source_type: str = "auto",
        **kwargs: Any,
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load data from BigQuery, a file upload or an existing dataset.

        Parameters
        ----------
        data_source : str
            Identifier for the data source.  For BigQuery this should
            be a fully‑qualified table reference (``project.dataset.table``).
            For uploaded files this is a path to the file.  For
            existing datasets it can be a file name (e.g. ``iris``).
        source_type : str, default ``"auto"``
            Explicitly specify ``"bigquery"``, ``"upload"`` or
            ``"existing"``.  When set to ``"auto"``, the manager will
            attempt to infer the source type from the format of
            ``data_source``.
        **kwargs : Any
            Additional parameters passed through to the underlying load
            functions.  For BigQuery these include ``project_id``,
            ``row_limit``, ``sample_percent``, ``where_clause``,
            ``select_columns`` and ``bigquery_options``.

        Returns
        -------
        tuple
            A tuple containing the loaded Spark DataFrame and a
            metadata dictionary describing the source.
        """

        # Infer source type if set to auto
        if source_type == "auto":
            if self._is_bigquery_reference(data_source):
                source_type = "bigquery"
            elif os.path.exists(data_source):
                source_type = "upload"
            else:
                source_type = "existing"

        if source_type == "bigquery":
            df, meta = self._load_from_bigquery(data_source, **kwargs)
        elif source_type == "upload":
            df, meta = self._load_from_upload(data_source, **kwargs)
        elif source_type == "existing":
            df, meta = self._load_from_existing(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

        return df, meta

    def get_data_preview(self, df: DataFrame, num_rows: int = 5) -> None:
        """Display a preview of the data in the console.

        This convenience function prints the first few rows along with
        basic metadata.  It is intended for interactive use in the
        Streamlit application.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The DataFrame to preview.
        num_rows : int, default ``5``
            Number of rows to display.
        """
        try:
            row_count = df.count()
            col_count = len(df.columns)
            print(f"📝 Data preview ({row_count} rows × {col_count} columns):")
            df.show(num_rows, truncate=False)
        except Exception as e:
            print(f"⚠️ Could not preview data: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_bigquery_reference(self, ref: str) -> bool:
        """Return True if the string looks like a BigQuery table reference."""
        parts = ref.split(".")
        return 2 <= len(parts) <= 3

    def _load_from_bigquery(self, table_reference: str, **kwargs: Any) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load data from BigQuery using a pushed‑down SQL query.

        Parameters
        ----------
        table_reference : str
            The fully qualified BigQuery table name (``project.dataset.table``).
        **kwargs : Any
            Additional options such as ``project_id``, ``row_limit``,
            ``sample_percent``, ``where_clause``, ``select_columns`` and
            ``bigquery_options``.

        Returns
        -------
        tuple
            A tuple of (DataFrame, metadata).
        """
        project_id = kwargs.get("project_id")
        row_limit = kwargs.get("row_limit")
        sample_percent = kwargs.get("sample_percent")
        where_clause = kwargs.get("where_clause")
        select_columns = kwargs.get("select_columns")
        bigquery_options = kwargs.get("bigquery_options", {})

        # Attempt to derive project ID from the table reference if not provided
        if not project_id:
            parts = table_reference.split(".")
            if len(parts) == 3:
                project_id = parts[0]
            else:
                raise ValueError(
                    "A BigQuery project_id is required when table_reference"
                    " does not include it."
                )

        # Build SELECT clause
        if select_columns:
            select_part = select_columns
        else:
            select_part = "*"

        # Start constructing the SQL query
        # Use backticks around the table to support names with dashes
        sql = f"SELECT {select_part} FROM `{table_reference}`"

        # Apply table sampling if provided
        if sample_percent:
            # BigQuery TABLESAMPLE supports system sampling percentages
            sql += f" TABLESAMPLE SYSTEM ({float(sample_percent)} PERCENT)"

        # Apply WHERE clause if provided
        if where_clause:
            sql += f" WHERE {where_clause}"

        # Apply row limit at the end of the query
        if row_limit:
            sql += f" LIMIT {int(row_limit)}"

        # Determine the dataset ID for the connector.  BigQuery's Spark
        # connector requires a dataset to be specified when using the
        # ``query`` option.  Extract it from the table reference (which
        # may be ``project.dataset.table`` or ``dataset.table``).
        table_parts = table_reference.split(".")
        # Initialise dataset_id to avoid NameError if parsing fails
        dataset_id: Optional[str] = None
        if len(table_parts) == 3:
            # format: project.dataset.table
            _, dataset_id, _ = table_parts
        elif len(table_parts) == 2:
            # format: dataset.table; project_id must be provided separately
            dataset_id, _ = table_parts
        # After parsing, ensure dataset_id is set
        if not dataset_id:
            raise ValueError(
                f"Unable to determine dataset from table_reference '{table_reference}'."
            )

        # Configure the reader
        reader = (
            self.spark.read.format("bigquery")
            .option("parentProject", project_id)
            .option("dataset", dataset_id)
            .option("viewsEnabled", "true")
            .option("useAvroLogicalTypes", "true")
            .option("query", sql)
        )

        # When using query with the BigQuery connector, materialization
        # options specify where the temporary table for the query
        # execution will live.  Without these, queries against large
        # tables can fail if the default dataset is not available.
        reader = reader.option("materializationDataset", dataset_id)
        reader = reader.option("materializationProject", project_id)

        # Apply any additional BigQuery connector options
        for key, value in bigquery_options.items():
            reader = reader.option(key, value)

        # Load the DataFrame
        df = reader.load()

        # Gather metadata
        try:
            row_count = df.count()
        except Exception:
            row_count = -1  # Unknown for very large tables
        col_count = len(df.columns)
        meta: Dict[str, Any] = {
            "source_type": "bigquery",
            "table_reference": table_reference,
            "project_id": project_id,
            "row_count": row_count,
            "column_count": col_count,
            "query": sql,
        }
        return df, meta

    def _load_from_upload(self, file_path: str, **kwargs: Any) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load data from an uploaded file.

        Supported formats are CSV, TSV, JSON, Parquet and Excel.  Excel
        files are read via pandas then converted to a Spark DataFrame.  The
        uploaded file is copied into ``output_dir`` so that downstream
        tasks (e.g. job reloading) can access it later.
        """
        # Validate that the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        file_format = None
        for fmt, extensions in self.supported_extensions.items():
            if ext in extensions:
                file_format = fmt
                break

        if file_format is None:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Supported extensions: {self.supported_extensions}"
            )

        # Copy file to output_dir with a unique name to avoid collisions
        base_name = os.path.basename(file_path)
        dest_path = os.path.join(self.output_dir, base_name)
        if file_path != dest_path:
            shutil.copy(file_path, dest_path)

        # Read the file into a Spark DataFrame based on its format
        if file_format == "csv":
            delimiter = kwargs.get("delimiter", ",")
            header = str(kwargs.get("header", True)).lower()
            infer_schema = str(kwargs.get("inferSchema", True)).lower()
            df = self.spark.read.option("header", header).option("inferSchema", infer_schema).option(
                "delimiter", delimiter
            ).csv(dest_path)
        elif file_format == "tsv":
            header = str(kwargs.get("header", True)).lower()
            infer_schema = str(kwargs.get("inferSchema", True)).lower()
            df = self.spark.read.option("header", header).option("inferSchema", infer_schema).option(
                "delimiter", "\t"
            ).csv(dest_path)
        elif file_format == "json":
            df = self.spark.read.json(dest_path)
        elif file_format == "parquet":
            df = self.spark.read.parquet(dest_path)
        elif file_format == "excel":
            # Read using pandas then convert to Spark DataFrame
            sheet_name = kwargs.get("sheet_name")
            try:
                pandas_df = pd.read_excel(dest_path, sheet_name=sheet_name)
            except Exception as e:
                raise RuntimeError(f"Error reading Excel file: {e}")
            df = self.spark.createDataFrame(pandas_df)
        else:
            raise ValueError(f"Unhandled file format: {file_format}")

        # Gather metadata
        try:
            row_count = df.count()
        except Exception:
            row_count = -1
        col_count = len(df.columns)
        meta: Dict[str, Any] = {
            "source_type": "upload",
            "file_format": file_format,
            "original_file": file_path,
            "saved_file": dest_path,
            "row_count": row_count,
            "column_count": col_count,
        }
        return df, meta

    def _load_from_existing(self, name: str, **kwargs: Any) -> Tuple[DataFrame, Dict[str, Any]]:
        """Load a built‑in dataset by name or file path.

        The AutoML project ships with a few small example datasets.  If
        ``name`` matches one of these known datasets (case‑insensitive),
        it is loaded directly from the package directory.  Otherwise,
        ``name`` is treated as a path relative to the current working
        directory or to the package directory.  The caller can also
        provide a full path to a file.
        """
        # Known datasets packaged with the project
        known_files = {
            "iris": "IRIS.csv",
            "bank": "bank.csv",
            "regression": "regression_file.csv",
        }

        automl_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = None

        lower_name = name.lower()
        if lower_name in known_files:
            file_path = os.path.join(automl_dir, known_files[lower_name])
        else:
            # Try interpreting name as a direct path or with .csv extension
            candidates = [name, f"{name}.csv"]
            for candidate in candidates:
                # Absolute or relative to working directory
                if os.path.exists(candidate):
                    file_path = candidate
                    break
                # Relative to automl directory
                alt_path = os.path.join(automl_dir, candidate)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break

        if not file_path:
            raise FileNotFoundError(f"Could not find existing dataset: {name}")

        # Read using the upload loader logic based on extension
        return self._load_from_upload(file_path)
