"""
metrics_calculator.py

Calculate the metrics like ROC, Accuracy and KS using the code below
"""

from pyspark.sql.types import DoubleType
from pyspark.sql import Window
from pyspark.sql.functions import desc, udf
from pyspark.sql import functions as F
import time
import builtins
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
import pandas as pd

# Note: Do NOT create a SparkSession or SparkContext at the module level in a package
# Users should pass their own SparkSession if needed

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''), index=data.index, columns=data.columns)

def calculate_metrics(predictions, y, data_type):
    start_time4 = time.time()

    # Calculate ROC
    evaluator = BinaryClassificationEvaluator(labelCol=y, rawPredictionCol='probability')
    auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    print('AUC calculated', auroc)

    selectedCols = predictions.select(F.col("probability"), F.col('prediction'), F.col(y)).rdd.map(lambda row: (float(row['probability'][1]), float(row['prediction']), float(row[y]))).collect()
    y_score, y_pred, y_true = zip(*selectedCols)

    # Calculate Accuracy
    accuracydf = predictions.withColumn('acc', F.when(predictions.prediction == predictions[y], 1).otherwise(0))
    accuracy = accuracydf.agg(F.avg('acc')).collect()[0][0]
    print('Accuracy calculated', accuracy)

    # Build KS Table - Optimized Spark-native approach
    # Pull the positive‑class probability as a scalar
    print('KS calculation starting')
    split1_udf = F.udf(lambda v: float(v[1]), DoubleType())
    prob_col = split1_udf('probability') if data_type in {'train','valid','test','oot1','oot2'} \
                                         else F.col('probability').cast(DoubleType())

    decile_df = predictions.select(
        F.col(y).cast('int').alias('label'),
        prob_col.alias('probability')
    ).withColumn('non_target', 1 - F.col('label'))

    # Totals for normalisation
    totals = decile_df.agg(
        F.sum('label').alias('total_target'),
        F.sum('non_target').alias('total_non_target')
    ).first()
    total_target = totals['total_target'] or 0.0
    total_non_target = totals['total_non_target'] or 0.0

    
    
    # Assign a deterministic row number to break ties
    rn_window = Window.partitionBy(F.lit(1)).orderBy(F.desc('probability'))
    df_with_rn = decile_df.withColumn('rownum', F.row_number().over(rn_window))
    
    # Then order by probability and rownum for decile assignment
    ntile_window = Window.partitionBy(F.lit(1)).orderBy(F.desc('probability'), F.col('rownum'))
    with_decile = df_with_rn.withColumn('decile', F.ntile(10).over(ntile_window))

    # Aggregate once per decile
    agg_df = (with_decile
              .groupBy('decile')
              .agg(F.count('*').alias('count'),
                   F.sum('label').alias('target'),
                   F.sum('non_target').alias('non_target'))
              .orderBy('decile'))

    # Cumulative sums within Spark
    decile_window = Window.partitionBy(F.lit(1)).orderBy('decile').rowsBetween(Window.unboundedPreceding, 0)
    agg_df = (agg_df
              .withColumn('cum_target', F.sum('target').over(decile_window))
              .withColumn('cum_non_target', F.sum('non_target').over(decile_window)))

    # KS statistic: max difference between the two cumulative distributions
    ks_col = F.abs(agg_df.cum_target/total_target - agg_df.cum_non_target/total_non_target)
    ks_value = agg_df.select(F.max(ks_col)).first()[0]

    # Convert *only the aggregated table* to pandas for plotting and round numbers
    decile_table = (agg_df
                    .withColumn('Pct_target', (agg_df.target/agg_df['count'])*100)
                    .withColumn('%Dist_Target', (agg_df.cum_target/total_target)*100)
                    .withColumn('%Dist_non_Target', (agg_df.cum_non_target/total_non_target)*100)
                    .withColumn('spread', ks_col*100)
                    .toPandas()
                    .round(2))
    
    print("KS_Value =", builtins.round(ks_value*100, 2))
    print("Metrics calculation process Completed in : " + " %s seconds" % (time.time() - start_time4))
    return auroc, accuracy, builtins.round(ks_value*100, 2), y_score, y_pred, y_true, decile_table