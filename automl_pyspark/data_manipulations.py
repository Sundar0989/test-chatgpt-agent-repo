"""
data_manipulations.py

Core data manipulation utilities for AutoML PySpark package.
Includes:
    1. Missing value calculation
    2. Variable type identification
    3. Categorical encoding
    4. Numerical imputation
    5. Column renaming
    6. Feature/target joining
    7. Train/valid/test splitting
    8. Vector assembly
    9. Feature scaling
"""

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.functions import col


def missing_value_calculation(X, miss_per=0.75):
    """
    Select columns with missing value percentage below threshold.
    """
    missing = X.select([
                F.count(F.when(F.col(c).isNull() | (F.col(c) == ''), c)).alias(c)
                if dict(X.dtypes)[c] == 'string'
                else F.count(F.when(F.col(c).isNull() | F.isnan(c), c)).alias(c)
                for c in X.columns
             ])
    missing_len = X.count()
    final_missing = missing.toPandas().transpose()
    final_missing.reset_index(inplace=True)
    final_missing.rename(columns={0: 'missing_count'}, inplace=True)
    final_missing['missing_percentage'] = final_missing['missing_count'] / missing_len
    vars_selected = final_missing['index'][final_missing['missing_percentage'] <= miss_per].tolist()
    return vars_selected


def identify_variable_type(X):
    """
    Identify categorical (string) and numerical variables.
    """
    l = X.dtypes
    char_vars = []
    num_vars = []
    for i in l:
        if i[1] == 'string':
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    return char_vars, num_vars


def analyze_categorical_cardinality(X, char_vars, max_categorical_cardinality=50):
    """
    Analyze categorical variables and identify which ones should be treated as numeric
    due to high cardinality.
    
    Args:
        X: Input DataFrame
        char_vars: List of categorical variable names
        max_categorical_cardinality: Maximum number of unique values for a categorical variable
                                   before converting to numeric (default: 50)
    
    Returns:
        Tuple of (categorical_vars_to_keep, categorical_vars_to_convert_numeric)
    """
    if not char_vars:
        return [], []
    
    print(f"🔍 Analyzing cardinality for {len(char_vars)} categorical variables...")
    
    categorical_vars_to_keep = []
    categorical_vars_to_convert_numeric = []
    
    for var in char_vars:
        # Count unique values for this categorical variable
        unique_count = X.select(var).distinct().count()
        
        if unique_count > max_categorical_cardinality:
            print(f"   📊 {var}: {unique_count} unique values → Converting to NUMERIC (exceeds threshold of {max_categorical_cardinality})")
            categorical_vars_to_convert_numeric.append(var)
        else:
            print(f"   📊 {var}: {unique_count} unique values → Keeping as CATEGORICAL")
            categorical_vars_to_keep.append(var)
    
    if categorical_vars_to_convert_numeric:
        print(f"✅ Converting {len(categorical_vars_to_convert_numeric)} high-cardinality categorical variables to numeric")
        print(f"✅ Keeping {len(categorical_vars_to_keep)} low-cardinality categorical variables")
    else:
        print(f"✅ All categorical variables have acceptable cardinality (≤ {max_categorical_cardinality})")
    
    return categorical_vars_to_keep, categorical_vars_to_convert_numeric


def convert_categorical_to_numeric(X, categorical_vars_to_convert):
    """
    Convert high-cardinality categorical variables to numeric using hash-based encoding.
    
    Args:
        X: Input DataFrame
        categorical_vars_to_convert: List of categorical variable names to convert
    
    Returns:
        DataFrame with converted variables
    """
    if not categorical_vars_to_convert:
        return X
    
    print(f"🔄 Converting {len(categorical_vars_to_convert)} categorical variables to numeric...")
    
    from pyspark.sql.functions import hash, col, when, isnan, isnull
    
    for var in categorical_vars_to_convert:
        print(f"   Converting {var} to numeric using hash encoding...")
        
        # Create a hash-based numeric encoding
        # Use abs(hash()) to ensure positive values, then take modulo to control range
        X = X.withColumn(
            var + "_numeric", 
            when(col(var).isNotNull() & (col(var) != ""), 
                 (hash(col(var)) % 1000000).cast("integer"))  # Modulo to control range
            .otherwise(-999)  # Handle nulls/empty strings with special value
        )
        
        # Drop the original categorical column
        X = X.drop(var)
        
        # Rename the numeric column back to original name
        X = X.withColumnRenamed(var + "_numeric", var)
    
    print(f"✅ Categorical to numeric conversion completed")
    return X


def categorical_to_index(X, char_vars):
    """
    Encode categorical variables using StringIndexer.
    Returns transformed DataFrame and fitted PipelineModel.
    """
    if not char_vars:
        return X, None
    chars = X.select(char_vars)
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_encoded", handleInvalid="keep") for column in chars.columns]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(chars)
    X = char_labels.transform(X)
    return X, char_labels


def numerical_imputation(X, num_vars, impute_with=0):
    """
    Impute missing values in numerical columns.
    """
    X = X.fillna(impute_with, subset=num_vars)
    return X


def rename_columns(X, char_vars):
    """
    Rename indexed columns to encoded format for VectorAssembler compatibility.
    """
    # Rename _index to _encoded to match VectorAssembler expectations
    for var in char_vars:
        if f"{var}_index" in X.columns:
            X = X.withColumnRenamed(f"{var}_index", f"{var}_encoded")
    return X


def join_features_and_target(X, Y):
    """
    Join features and target DataFrames on a generated row id.
    """
    X = X.withColumn('id', F.monotonically_increasing_id())
    Y = Y.withColumn('id', F.monotonically_increasing_id())
    joinedDF = X.join(Y, 'id', 'inner').drop('id')
    return joinedDF


def train_valid_test_split(df, train_size=0.4, valid_size=0.3, seed=12345):
    """
    Split DataFrame into train, valid, and test sets.
    """
    train, valid, test = df.randomSplit([train_size, valid_size, 1 - train_size - valid_size], seed=seed)
    return train, valid, test


def assembled_vectors(df, list_of_features_to_scale, target_column_name):
    """
    Assemble feature columns into a single vector column.
    """
    assembler = VectorAssembler(inputCols=list_of_features_to_scale, outputCol='features')
    pipeline = Pipeline(stages=[assembler])
    assembleModel = pipeline.fit(df)
    selectedCols = [target_column_name, 'features'] + list_of_features_to_scale
    df = assembleModel.transform(df).select(selectedCols)
    return df


def scaled_dataframes(train, valid, test, list_of_features_to_scale, target_column_name):
    """
    Scale features in train, valid, and test DataFrames.
    Returns transformed DataFrames and fitted PipelineModel.
    """
    assembler = VectorAssembler(inputCols=list_of_features_to_scale, outputCol='assembled_features')
    scaler = StandardScaler(inputCol='assembled_features', outputCol='features')
    pipeline = Pipeline(stages=[assembler, scaler])
    pipelineModel = pipeline.fit(train)
    selectedCols = [target_column_name, 'features'] + list_of_features_to_scale
    train = pipelineModel.transform(train).select(selectedCols)
    valid = pipelineModel.transform(valid).select(selectedCols)
    test = pipelineModel.transform(test).select(selectedCols)
    return train, valid, test, pipelineModel 