"""
Explainability utilities for AutoML PySpark.

This module provides helper functions to compute SHAP values for models
trained using the AutoML PySpark pipeline.  SHAP values offer a way to
understand how each feature contributes to the predictions of a model.

The implementation here uses SHAP's `KernelExplainer`, which works
with any black‑box model.  Because the AutoML pipeline relies on
PySpark ML models, we wrap the Spark pipeline and model into a
prediction function that can be called by SHAP.  The prediction
function converts incoming NumPy arrays to pandas DataFrames, then
creates a Spark DataFrame, applies the preprocessing pipeline and
finally generates predictions via the trained model.  To keep
computation manageable, a small number of rows from the training
dataset should be passed into ``compute_shap_values``.  The
resulting SHAP values and summary plot are saved to the specified
output directory.

Note: SHAP computation can be slow for large models or datasets.  To
control execution time, the number of samples used for explanation can
be adjusted via the ``max_samples`` argument.  If SHAP or its
dependencies are not installed, or if an error occurs during
computation, the function will catch the exception and log a warning
without halting the AutoML pipeline.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore

try:
    import shap  # type: ignore
except ImportError as e:
    # If shap is not installed, users will see a message when attempting
    # to compute SHAP values.  The import is deferred here so that
    # pipelines can still run without explainability support.
    shap = None  # type: ignore


def compute_shap_values(
    spark,
    pipeline_model,
    model,
    sample_df,
    feature_cols: List[str],
    output_dir: str,
    model_type: str = "classification",
    max_samples: int = 50,
) -> None:
    """Compute SHAP values for a trained model and save results.

    Parameters
    ----------
    spark : SparkSession
        The active SparkSession used to create DataFrames.
    pipeline_model : pyspark.ml.PipelineModel
        The preprocessing pipeline that prepares raw features and outputs
        a ``features`` vector.
    model : pyspark.ml.Model
        The trained model whose predictions will be explained.
    sample_df : pyspark.sql.DataFrame
        A Spark DataFrame containing raw feature columns from which a
        sample will be drawn for explanation.  This should be the
        training dataset or a subset thereof.
    feature_cols : List[str]
        The list of feature column names to include in the SHAP
        explanation.  Only these columns will be passed into the
        prediction function.
    output_dir : str
        Directory where the SHAP summary plot and SHAP values CSV will
        be saved.  The directory will be created if it does not exist.
    model_type : str, default ``"classification"``
        A label used in filenames indicating the type of model being
        explained (e.g., 'classification', 'regression', 'clustering').
    max_samples : int, default ``50``
        The maximum number of rows from ``sample_df`` to use when
        computing SHAP values.  Fewer samples reduce computation time.

    Notes
    -----
    - SHAP values provide a local explanation for individual
      predictions.  Kernel SHAP can be computationally intensive; the
      number of samples and ``nsamples`` passed to ``shap_values`` are
      deliberately limited here.
    - For classification models, SHAP returns a list of arrays (one
      per output class).  This implementation uses the first array,
      which corresponds to the first class.  For binary classifiers,
      this is typically sufficient.
    """
    # If shap is unavailable, skip explainability
    if shap is None:
        print("⚠️ SHAP library not installed. Skipping explainability.")
        return

    # Collect a small sample of the data to a pandas DataFrame
    try:
        # Use collect() instead of toPandas() to avoid Arrow conversion issues
        sample_data = sample_df.select(*feature_cols).limit(max_samples).collect()
        pd_data = pd.DataFrame([row.asDict() for row in sample_data])
    except Exception as e:
        print(f"⚠️ Could not collect sample for SHAP computation: {e}")
        return

    if pd_data.empty:
        print("⚠️ No data available for SHAP computation.")
        return

    # Define a prediction function compatible with SHAP
    def predict_fn(input_array):
        """Predict function that accepts a 2D NumPy array and returns
        predictions as a 1D NumPy array.  The function handles the
        conversion from NumPy -> pandas -> Spark DataFrame -> model
        predictions.
        """
        import pandas as pd  # local import

        # Build a pandas DataFrame with the correct feature columns
        rows = pd.DataFrame(input_array, columns=feature_cols)
        # Create a Spark DataFrame from the pandas DataFrame
        sdf = spark.createDataFrame(rows)
        # Apply the preprocessing pipeline to obtain the 'features' vector
        processed = pipeline_model.transform(sdf)
        # Generate predictions using the trained model
        preds = model.transform(processed)
        # Convert predictions to pandas using collect() to avoid Arrow issues
        preds_rows = preds.select("prediction").collect()
        preds_pd = pd.DataFrame([row.asDict() for row in preds_rows])
        # Return predictions as a NumPy array
        return preds_pd["prediction"].values

    try:
        # Create the SHAP explainer using KernelExplainer.  The
        # background dataset is the sample itself to limit computation.
        explainer = shap.KernelExplainer(predict_fn, pd_data, keep_index=False)
        # Compute SHAP values.  nsamples controls the number of
        # evaluations of the prediction function; using ``max_samples``
        # here keeps runtime reasonable.
        shap_values = explainer.shap_values(pd_data, nsamples=max_samples)

        # For classification models with multiple outputs, shap_values is
        # a list of arrays (one per class).  Use the first by default.
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate SHAP summary plot without displaying it
        shap.summary_plot(shap_vals, pd_data, show=False)
        summary_path = os.path.join(output_dir, f"shap_summary_{model_type}.png")
        plt.tight_layout()
        plt.savefig(summary_path)
        plt.close()

        # Save SHAP values to a CSV file for further analysis
        shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
        shap_csv_path = os.path.join(output_dir, f"shap_values_{model_type}.csv")
        shap_df.to_csv(shap_csv_path, index=False)

        print(f"✅ SHAP summary plot saved to {summary_path}")
        print(f"✅ SHAP values saved to {shap_csv_path}")

    except Exception as e:
        # Catch any errors during SHAP computation and proceed silently
        print(f"⚠️ Failed to compute SHAP values: {e}")
